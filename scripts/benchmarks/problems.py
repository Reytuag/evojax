import ast
import re
import yaml
import json
import numpy as np
from evojax.policy import MLPPolicy
from evojax.policy import MLPPolicy_b
from evojax.policy import MetaRnnPolicy
from evojax.policy import MetaRnnPolicy_b
from evojax.policy import MetaRnnPolicy_t
from evojax.policy import MetaRnnPolicy_b2
from evojax.policy import MetaRnnPolicy_bcppr
from evojax.policy.convnet import ConvNetPolicy
from evojax.policy import SymLA_Policy
from evojax.policy import TransformerPolicy


def setup_problem(config, logger):
    if config["problem_type"] == "cartpole_easy":
        return setup_cartpole(config, False)
    elif config["problem_type"] == "cartpole_hard":
        return setup_cartpole(config, True)
    elif config["problem_type"] == "brax":
        return setup_brax(config)
    elif config["problem_type"] == "mnist":
        return setup_mnist(config, logger)
    elif config["problem_type"] == "waterworld":
        return setup_waterworld(config)
    elif config["problem_type"] == "waterworld_ma":
        return setup_waterworld_ma(config)
    elif config["problem_type"] == "gridworld_meta":
        return setup_gridworld(config)
    elif config["problem_type"] == "gridworld_meta_b":
        return setup_gridworld_b(config)
    elif config["problem_type"] == "gridworld_recipes":
        return setup_gridworld_recipes(config)
    elif config["problem_type"] == "gridworld_recipes_reward_item":
        return setup_gridworld_recipes_reward_item(config)
    elif config["problem_type"] == "vector_recipes_3":
        return setup_vector_recipes_3(config)
    elif config["problem_type"] == "vector_recipes":
        return setup_vector_recipes(config)
    elif config["problem_type"] == "CPPR":
        return setup_CPPR(config)
    elif config["problem_type"] == "vector_recipes_changes":
        return setup_vector_recipes_changes(config)
    elif config["problem_type"] == "vector_recipes_no_remove":
        return setup_vector_recipes_no_remove(config)
    elif config["problem_type"] == "vector_recipes_2tree":
        return setup_vector_recipes_2tree(config)
    elif config["problem_type"] == "vector_recipes_2tree_bis":
        return setup_vector_recipes_2tree_bis(config)


def setup_vector_recipes_2tree_bis(config):
    from evojax.task.vector_recipe_2tree_bis import Gridworld

    train_task = Gridworld(test=False, nb_items=config["nb_items"], max_steps=config["episode_len"])
    test_task = Gridworld(test=True, nb_items=config["nb_items"], max_steps=config["episode_len"])
    if (config["policy"] == 'SymLA'):
        policy = SymLA_Policy(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            msg_dim=config["msg_size"],
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            num_micro_ticks=config['num_micro_ticks'],
            hidden_layers=config["hidden_layers"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN'):
        policy = MetaRnnPolicy_b(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN2'):
        policy = MetaRnnPolicy_b2(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim_1=config["hidden_size"],
            hidden_dim_2=config["hidden_size_2"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN_t'):
        policy = MetaRnnPolicy_t(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    else:
        policy = MLPPolicy_b(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dims=[config["hidden_size"]] * 2,

            output_dim=train_task.act_shape[0],
            output_act_fn="categorical"
        )

    return train_task, test_task, policy
def setup_vector_recipes_2tree(config):
    from evojax.task.vector_recipe_2tree import Gridworld

    train_task = Gridworld(test=False, nb_items=config["nb_items"], max_steps=config["episode_len"])
    test_task = Gridworld(test=True, nb_items=config["nb_items"], max_steps=config["episode_len"])
    if (config["policy"] == 'SymLA'):
        policy = SymLA_Policy(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            msg_dim=config["msg_size"],
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            num_micro_ticks=config['num_micro_ticks'],
            hidden_layers=config["hidden_layers"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN'):
        policy = MetaRnnPolicy_b(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN2'):
        policy = MetaRnnPolicy_b2(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim_1=config["hidden_size"],
            hidden_dim_2=config["hidden_size_2"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN_t'):
        policy = MetaRnnPolicy_t(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    else:
        policy = MLPPolicy_b(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dims=[config["hidden_size"]] * 2,

            output_dim=train_task.act_shape[0],
            output_act_fn="categorical"
        )

    return train_task, test_task, policy

def setup_vector_recipes_no_remove(config):
    from evojax.task.vector_recipe_no_remove import Gridworld

    train_task = Gridworld(test=False, nb_items=config["nb_items"], max_steps=config["episode_len"])
    test_task = Gridworld(test=True, nb_items=config["nb_items"], max_steps=config["episode_len"])
    if (config["policy"] == 'SymLA'):
        policy = SymLA_Policy(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            msg_dim=config["msg_size"],
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            num_micro_ticks=config['num_micro_ticks'],
            hidden_layers=config["hidden_layers"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN'):
        policy = MetaRnnPolicy_b(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN2'):
        policy = MetaRnnPolicy_b2(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim_1=config["hidden_size"],
            hidden_dim_2=config["hidden_size_2"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN_t'):
        policy = MetaRnnPolicy_t(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    else:
        policy = MLPPolicy_b(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dims=[config["hidden_size"]] * 2,

            output_dim=train_task.act_shape[0],
            output_act_fn="categorical"
        )

    return train_task, test_task, policy


def setup_vector_recipes_changes(config):
    from evojax.task.vector_recipe_drastic_change import Gridworld

    train_task = Gridworld(test=False, nb_items=config["nb_items"], max_steps=config["episode_len"])
    test_task = Gridworld(test=True, nb_items=config["nb_items"], max_steps=config["episode_len"])
    if (config["policy"] == 'SymLA'):
        policy = SymLA_Policy(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            msg_dim=config["msg_size"],
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            num_micro_ticks=config['num_micro_ticks'],
            hidden_layers=config["hidden_layers"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN'):
        policy = MetaRnnPolicy_b(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN2'):
        policy = MetaRnnPolicy_b2(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim_1=config["hidden_size"],
            hidden_dim_2=config["hidden_size_2"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN_t'):
        policy = MetaRnnPolicy_t(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    else:
        policy = MLPPolicy_b(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dims=[config["hidden_size"]] * 2,

            output_dim=train_task.act_shape[0],
            output_act_fn="categorical"
        )

    return train_task, test_task, policy


def setup_CPPR(config):
    from evojax.task.CPPR_1agent import Gridworld

    train_task = Gridworld(test=False, SX=config["SX"], SY=config["SY"], max_steps=config["episode_len"])
    test_task = Gridworld(test=True, SX=config["SX"], SY=config["SY"], max_steps=config["episode_len"])
    if (config["policy"] == 'SymLA'):
        policy = SymLA_Policy(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            msg_dim=config["msg_size"],
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            num_micro_ticks=config['num_micro_ticks'],
            hidden_layers=config["hidden_layers"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN'):
        policy = MetaRnnPolicy_b(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN_bcppr'):
        policy = MetaRnnPolicy_bcppr(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_layers=config["encoder_layers"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN2'):
        policy = MetaRnnPolicy_b2(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim_1=config["hidden_size"],
            hidden_dim_2=config["hidden_size_2"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN_t'):
        policy = MetaRnnPolicy_t(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    else:
        policy = MLPPolicy_b(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dims=[config["hidden_size"]] * 2,

            output_dim=train_task.act_shape[0],
            output_act_fn="categorical"
        )

    return train_task, test_task, policy


def setup_vector_recipes(config):
    from evojax.task.vector_recipe import Gridworld

    train_task = Gridworld(test=False, nb_items=config["nb_items"], max_steps=config["episode_len"])
    test_task = Gridworld(test=True, nb_items=config["nb_items"], max_steps=config["episode_len"])
    if (config["policy"] == 'SymLA'):
        policy = SymLA_Policy(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            msg_dim=config["msg_size"],
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            num_micro_ticks=config['num_micro_ticks'],
            hidden_layers=config["hidden_layers"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN'):
        policy = MetaRnnPolicy_b(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN2'):
        policy = MetaRnnPolicy_b2(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim_1=config["hidden_size"],
            hidden_dim_2=config["hidden_size_2"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN_t'):
        policy = MetaRnnPolicy_t(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")

    elif (config["policy"] == 'Transformer'):
        policy = TransformerPolicy(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            qkv_features= config["qkv_features"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            num_heads=config["num_heads"],
            max_len=config["max_len"],
            causal=config["causal"],
            encoder_size=config["encoder_size"],)
    else:
        policy = MLPPolicy_b(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dims=[config["hidden_size"]] * 2,

            output_dim=train_task.act_shape[0],
            output_act_fn="categorical"
        )

    return train_task, test_task, policy


def setup_vector_recipes_3(config):
    from evojax.task.vector_recipe_3 import Gridworld

    train_task = Gridworld(test=False, spawn_prob=config["spawn_prob"], max_steps=config["episode_len"])
    test_task = Gridworld(test=True, spawn_prob=config["spawn_prob"], max_steps=config["episode_len"])
    if (config["policy"] == 'SymLA'):
        policy = SymLA_Policy(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            msg_dim=config["msg_size"],
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            num_micro_ticks=config['num_micro_ticks'],
            hidden_layers=config["hidden_layers"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN'):
        policy = MetaRnnPolicy_b(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN2'):
        policy = MetaRnnPolicy_b2(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim_1=config["hidden_size"],
            hidden_dim_2=config["hidden_size_2"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN_t'):
        policy = MetaRnnPolicy_t(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    else:
        policy = MLPPolicy_b(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dims=[config["hidden_size"]] * 2,

            output_dim=train_task.act_shape[0],
            output_act_fn="categorical"
        )

    return train_task, test_task, policy


def setup_gridworld_recipes_reward_item(config):
    from evojax.task.gridworld_recipe_reward_item import Gridworld

    train_task = Gridworld(test=False, spawn_prob=config["spawn_prob"])
    test_task = Gridworld(test=True, spawn_prob=config["spawn_prob"])
    if (config["policy"] == 'SymLA'):
        policy = SymLA_Policy(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            msg_dim=config["msg_size"],
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            num_micro_ticks=config['num_micro_ticks'],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN'):
        policy = MetaRnnPolicy_b(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim=config["hidden_size"],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_dim=train_task.act_shape[0],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN_t'):
        policy = MetaRnnPolicy_t(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN2'):
        policy = MetaRnnPolicy_b2(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim_1=config["hidden_size"],
            hidden_dim_2=config["hidden_size_2"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    else:
        policy = MLPPolicy_b(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dims=[config["hidden_size"]] * 2,

            output_dim=train_task.act_shape[0],
            output_act_fn="categorical"
        )

    return train_task, test_task, policy


def setup_gridworld_recipes(config):
    if (config["nb_items"] == 4):
        from evojax.task.gridworld_recipe_4 import Gridworld
    else:
        from evojax.task.gridworld_recipe_3 import Gridworld

    train_task = Gridworld(test=False, spawn_prob=config["spawn_prob"], max_steps=config["episode_len"])
    test_task = Gridworld(test=True, spawn_prob=config["spawn_prob"], max_steps=config["episode_len"])
    if (config["policy"] == 'SymLA'):
        policy = SymLA_Policy(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            msg_dim=config["msg_size"],
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            num_micro_ticks=config['num_micro_ticks'],
            hidden_layers=config["hidden_layers"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN'):
        policy = MetaRnnPolicy_b(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN2'):
        policy = MetaRnnPolicy_b2(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim_1=config["hidden_size"],
            hidden_dim_2=config["hidden_size_2"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN_t'):
        policy = MetaRnnPolicy_t(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            hidden_layers=config["hidden_layers"],
            encoder=config["encoder"],
            encoder_size=config["encoder_size"],
            output_act_fn="categorical")
    else:
        policy = MLPPolicy_b(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dims=[config["hidden_size"]] * 2,

            output_dim=train_task.act_shape[0],
            output_act_fn="categorical"
        )

    return train_task, test_task, policy


def setup_gridworld_b(config):
    from evojax.task.gridworld_repop_bis import Gridworld

    train_task = Gridworld(test=False, spawn_prob=config["spawn_prob"])
    test_task = Gridworld(test=True, spawn_prob=config["spawn_prob"])
    if (config["policy"] == 'SymLA'):
        policy = SymLA_Policy(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            msg_dim=config["msg_size"],
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            num_micro_ticks=config['num_micro_ticks'],
            output_act_fn="categorical")
    elif (config["policy"] == 'MetaRNN'):
        policy = MetaRnnPolicy_b(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            output_act_fn="categorical")
    else:
        policy = MLPPolicy_b(
            input_dim=train_task.obs_shape[0] + train_task.act_shape[0] + 1,
            hidden_dims=[config["hidden_size"]] * 2,

            output_dim=train_task.act_shape[0],
            output_act_fn="categorical"
        )

    return train_task, test_task, policy


def setup_gridworld(config):
    from evojax.task.gridworld_repop import Gridworld

    train_task = Gridworld(test=False, spawn_prob=config["spawn_prob"])
    test_task = Gridworld(test=True, spawn_prob=config["spawn_prob"])
    if (config["policy"] == 'MetaRNN'):
        policy = MetaRnnPolicy(
            input_dim=train_task.obs_shape[0],
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0],
            output_act_fn="categorical")


    else:
        policy = MLPPolicy(
            input_dim=train_task.obs_shape[0],
            hidden_dims=[config["hidden_size"]] * 2,
            output_dim=train_task.act_shape[0],
            output_act_fn="categorical"
        )

    return train_task, test_task, policy


def setup_cartpole(config, hard=False):
    if (config["policy"] == 'MetaRNN'):
        from evojax.task.cartpole_meta import CartPoleSwingUp

        train_task = CartPoleSwingUp(test=False, harder=hard)
        test_task = CartPoleSwingUp(test=True, harder=hard)
        policy = MetaRnnPolicy(
            input_dim=train_task.obs_shape[0],
            hidden_dim=config["hidden_size"],
            output_dim=train_task.act_shape[0], )

    else:
        from evojax.task.cartpole import CartPoleSwingUp

        train_task = CartPoleSwingUp(test=False, harder=hard)
        test_task = CartPoleSwingUp(test=True, harder=hard)
        policy = MLPPolicy(
            input_dim=train_task.obs_shape[0],
            hidden_dims=[config["hidden_size"]] * 2,
            output_dim=train_task.act_shape[0],
        )
    return train_task, test_task, policy


def setup_brax(config):
    from evojax.task.brax_task import BraxTask

    train_task = BraxTask(env_name=config["env_name"], test=False)
    test_task = BraxTask(env_name=config["env_name"], test=True)
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[0],
        output_dim=train_task.act_shape[0],
        hidden_dims=[32, 32, 32, 32],
    )
    return train_task, test_task, policy


def setup_mnist(config, logger):
    from evojax.task.mnist import MNIST

    policy = ConvNetPolicy(logger=logger)
    train_task = MNIST(batch_size=config["batch_size"], test=False)
    test_task = MNIST(batch_size=config["batch_size"], test=True)
    return train_task, test_task, policy


def setup_waterworld(config, max_steps=500):
    from evojax.task.waterworld import WaterWorld

    train_task = WaterWorld(test=False, max_steps=max_steps)
    test_task = WaterWorld(test=True, max_steps=max_steps)
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[0],
        hidden_dims=[
            config["hidden_size"],
        ],
        output_dim=train_task.act_shape[0],
        output_act_fn="softmax",
    )
    return train_task, test_task, policy


def setup_waterworld_ma(config, num_agents=16, max_steps=500):
    from evojax.task.ma_waterworld import MultiAgentWaterWorld

    train_task = MultiAgentWaterWorld(
        num_agents=num_agents, test=False, max_steps=max_steps
    )
    test_task = MultiAgentWaterWorld(
        num_agents=num_agents, test=True, max_steps=max_steps
    )
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[-1],
        hidden_dims=[
            config["hidden_size"],
        ],
        output_dim=train_task.act_shape[-1],
        output_act_fn="softmax",
    )
    return train_task, test_task, policy


def convert(obj):
    """Conversion helper instead of JSON encoder for handling booleans."""
    if isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (list, tuple)):
        return [convert(item) for item in obj]
    if isinstance(obj, dict):
        return {convert(key): convert(value) for key, value in obj.items()}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return convert(obj.tolist())
    if isinstance(obj, np.bool_):
        return int(obj)
    return obj


def save_yaml(obj: dict, filename: str) -> None:
    """Save object as yaml file."""
    data = json.dumps(convert(obj), indent=1)
    data_dump = ast.literal_eval(data)
    with open(filename, "w") as f:
        yaml.safe_dump(data_dump, f, default_flow_style=False)


def load_yaml(config_fname: str) -> dict:
    """Load in YAML config file."""
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
    with open(config_fname) as file:
        yaml_config = yaml.load(file, Loader=loader)
    return yaml_config



