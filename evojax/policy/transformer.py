from matplotlib.font_manager import list_fonts
from flax import linen as nn

import logging
import jax
import jax.numpy as jnp

import itertools
import functools

from typing import Tuple, Callable, List, Optional, Iterable, Any
from flax.struct import dataclass
from evojax.task.base import TaskState
from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.util import create_logger
from evojax.util import get_params_format_fn
import math
import numpy as np


class PositionalEncoding(nn.Module):
    d_model: int  # Hidden dimensionality of the input.
    max_len: int = 200  # Maximum length of a sequence to expect.

    def setup(self):
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32)[:, None]
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[None]
        self.pe = jax.device_put(pe)

    def __call__(self, x):
        x = x + self.pe[:, :, :x.shape[2]]
        return x


class transformer_layer(nn.Module):
    num_heads: int
    out_features: int
    qkv_features: int

    def setup(self):
        self.attention1 = nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.qkv_features,
                                           out_features=self.out_features)

        self.ln1 = nn.LayerNorm()

        self.dense1 = nn.Dense(self.out_features)

        self.dense2 = nn.Dense(self.out_features)

        self.ln2 = nn.LayerNorm()

    def __call__(self, inputs: jnp.ndarray, mask: jnp.ndarray):
        out_attention = inputs + self.attention1(inputs)

        out_attention = self.ln1(out_attention)

        out = self.dense1(out_attention)
        out = nn.activation.relu(out)
        out = self.dense2(out_attention)

        out = out + out_attention

        out = self.ln2(out)

        return out


class Transformer(nn.Module):
    output_size: int
    hidden_layers: list
    encoder_size: int
    num_heads: int
    out_features: list
    max_len: int
    qkv_features: int

    def setup(self):
        self.encoder = nn.Dense(self.encoder_size)

        self.positional_encoding = PositionalEncoding(self.encoder_size, max_len=self.max_len)

        self.tf_layer1 = transformer_layer(num_heads=self.num_heads, qkv_features=self.qkv_features,
                                           out_features=self.out_features[0])
        self.tf_layer2 = transformer_layer(num_heads=self.num_heads, qkv_features=self.qkv_features,
                                           out_features=self.out_features[1])

        self._hiddens = [(nn.Dense(size)) for size in self.hidden_layers]
        # self._encoder=nn.Dense(64)
        self._output_proj = nn.Dense(self.output_size)

    def __call__(self, inputs: jnp.ndarray, mask: jnp.ndarray, timestep: int):
        encoded = self.encoder(inputs)

        x = self.positional_encoding(encoded)
        x = self.tf_layer1(inputs=x, mask=mask)

        x = self.tf_layer2(inputs=x, mask=mask)

        x = x[0, timestep]

        for layer in self._hiddens:
            x = jax.nn.tanh(layer(x))
        x = self._output_proj(x)

        return x


@dataclass
class transformer_state(PolicyState):
    mask: jnp.array
    history: jnp.array
    timesteps: jnp.array
    keys: jnp.array


class TransformerPolicy(PolicyNetwork):

    def __init__(self, input_dim: int,
                 qkv_features: int,
                 output_dim: int,
                 hidden_layers: list = [32],
                 num_heads: int = 4,
                 encoder_size: int = 32,
                 max_len: int = 100,
                 logger: logging.Logger = None):

        if logger is None:
            self._logger = create_logger(name='MetaRNNolicy')
        else:
            self._logger = logger
        model = Transformer(output_size=output_dim, hidden_layers=hidden_layers, encoder_size=encoder_size,
                            qkv_features=qkv_features, num_heads=num_heads, out_features=[encoder_size, encoder_size],
                            max_len=max_len)
        self.params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, max_len, input_dim)), jnp.ones((1, max_len)),
                                 timestep=0)

        self.num_params, format_params_fn = get_params_format_fn(self.params)
        self._logger.info('Transformer.num_params = {}'.format(self.num_params))
        self.max_len = max_len
        self.input_dim = input_dim
        self._format_params_fn = (jax.vmap(format_params_fn))
        self._forward_fn = (jax.vmap(model.apply))

    def reset(self, states: TaskState) -> PolicyState:
        """Reset the policy.
        Args:
            TaskState - Initial observations.
        Returns:
            PolicyState. Policy internal states.
        """
        keys = jax.random.split(jax.random.PRNGKey(0), states.obs.shape[0])
        history = jnp.zeros((states.obs.shape[0], 1, self.max_len, self.input_dim))
        mask = jnp.zeros((states.obs.shape[0], 1,  self.max_len))

        return transformer_state(keys=keys, history=history, mask=mask,
                                 timesteps=jnp.zeros((states.obs.shape[0],), dtype=jnp.int8))

    def get_actions(self, t_states: TaskState, params: jnp.ndarray, p_states: PolicyState):
        params = self._format_params_fn(params)
        new_inp = jnp.concatenate([t_states.obs, t_states.last_action, t_states.reward], axis=-1)
        timesteps = p_states.timesteps
        history=p_states.history
        history=jnp.where(timesteps[0]>=max_len,jnp.roll(history,1,axis=-2),history)
        history = history.at[:, 0, jnp.maximum(timesteps[0],self.max_len-1)].set(new_inp)
        mask = p_states.mask.at[:, 0, jnp.maximum(timesteps[0],self.max_len-1)].set(1)

        out = self._forward_fn(params, inputs=history, mask=mask, timestep=timesteps)
        timesteps = timesteps + 1
        return out, transformer_state(keys=p_states.keys, history=history, mask=mask, timesteps=timesteps)
