# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple
from PIL import Image
from PIL import ImageDraw
import numpy as np

import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass

from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask



SIZE_GRID=4
AGENT_VIEW=2








@dataclass
class AgentState(object):
    posx: jnp.int32
    posy: jnp.int32
    inventory: jnp.int32

@dataclass
class State(TaskState):
    obs: jnp.ndarray
    last_action:jnp.ndarray
    reward:jnp.ndarray
    state: jnp.ndarray
    agent: AgentState
    steps: jnp.int32
    permutation_recipe:jnp.ndarray
    key: jnp.ndarray






def get_obs(state: jnp.ndarray,posx:jnp.int32,posy:jnp.int32) -> jnp.ndarray:
    obs=jnp.ravel(jax.lax.dynamic_slice(jnp.pad(state,((AGENT_VIEW,AGENT_VIEW),(AGENT_VIEW,AGENT_VIEW),(0,0))),(posx-AGENT_VIEW+AGENT_VIEW,posy-AGENT_VIEW+AGENT_VIEW,1),(2*AGENT_VIEW+1,2*AGENT_VIEW+1,5)))
    return obs

def get_init_state_fn(key: jnp.ndarray) -> jnp.ndarray:
    grid=jnp.zeros((SIZE_GRID,SIZE_GRID,6))
    posx,posy=(0,0)
    grid=grid.at[posx,posy,0].set(1)
    pos_obj=jax.random.randint(key,(6,2),0,SIZE_GRID)
	
    grid=grid.at[pos_obj[0,0],pos_obj[0,1],1].add(1)
    grid=grid.at[pos_obj[1,0],pos_obj[1,1],2].add(1)
    grid=grid.at[pos_obj[2,0],pos_obj[2,1],3].add(1)
    grid=grid.at[pos_obj[3,0],pos_obj[3,1],1].add(1)
    grid=grid.at[pos_obj[4,0],pos_obj[4,1],2].add(1)
    grid=grid.at[pos_obj[5,0],pos_obj[5,1],3].add(1)
    

    return (grid)
def test_recipes(items,recipes):

	recipe_done=jnp.where(items[recipes[0]]*items[recipes[1]]>0,jnp.array([recipes[0],recipes[1],4]),jnp.zeros(3,jnp.int32))
	recipe_done=jnp.where(items[recipes[2]]*items[4]>0,jnp.array([recipes[2],4,5]),recipe_done)
	return recipe_done
	


def drop(grid,posx,posy,inventory,recipes):
       grid=grid.at[posx,posy,inventory].add(1)
       inventory=0
       reward=0
       #test recipe
       recipe_done=jax.lax.cond(grid[posx,posy,1:].sum()==2,test_recipes,lambda x,y:jnp.zeros(3,jnp.int32),*(grid[posx,posy,:],recipes))
       grid=jnp.where(recipe_done[2]>0,grid.at[posx,posy,recipe_done[0]].set(0).at[posx,posy,recipe_done[1]].set(0).at[posx,posy,recipe_done[2]].set(1),grid)
       reward=recipe_done[2]
       return grid,inventory,reward
       
def collect(grid,posx,posy,inventory):
	inventory=jnp.where(grid[posx,posy,1:].sum()>0,jnp.argmax(grid[posx,posy,1:])+1,0)
	grid=jnp.where(inventory>0,grid.at[posx,posy,inventory].add(-1),grid)
	return grid,inventory


class Gridworld(VectorizedTask):
    """gridworld task."""

    def __init__(self,
                 max_steps: int = 80,
                 test: bool = False,spawn_prob=0.005):

        self.max_steps = max_steps
        self.obs_shape = tuple([(AGENT_VIEW*2+1)*(AGENT_VIEW*2+1)*5+1, ])
        self.act_shape = tuple([7, ])
        self.test = test


        def reset_fn(key):
            next_key, key = random.split(key)
            posx,posy=(0,0)
            agent=AgentState(posx=posx,posy=posy,inventory=0)
            grid=get_init_state_fn(key)
            next_key, key = random.split(next_key)
            rand=jax.random.uniform(key)
            grid=jnp.where(rand>0.5,grid.at[1,4,1].set(1),grid.at[4,1,1].set(1))
            next_key, key = random.split(next_key)
            permutation_recipe=jax.random.permutation(key,3)+1
            return State(state=grid, obs=jnp.concatenate([get_obs(state=grid,posx=posx,posy=posy),jnp.zeros(1)]),last_action=jnp.zeros((7,)),reward=jnp.zeros((1,)),agent=agent,
                         steps=jnp.zeros((), dtype=int),permutation_recipe=permutation_recipe, key=next_key)
        self._reset_fn = jax.jit(jax.vmap(reset_fn))


        def step_fn(state, action):
            #spawn food
            grid=state.state
            reward=0


            	

            #move agent
            key, subkey = random.split(state.key)
            #maybe later make the agent to output the one hot categorical
            action=jax.random.categorical(subkey,action)
            action=jax.nn.one_hot(action,7)

            action_int=action.astype(jnp.int32)

            posx=state.agent.posx-action_int[1]+action_int[3]
            posy=state.agent.posy-action_int[2]+action_int[4]
            posx=jnp.clip(posx,0,SIZE_GRID-1)
            posy=jnp.clip(posy,0,SIZE_GRID-1)
            grid=grid.at[state.agent.posx,state.agent.posy,0].set(0)
            grid=grid.at[posx,posy,0].set(1)
            #collect or drop
            inventory=state.agent.inventory
            grid,inventory,reward=jax.lax.cond(jnp.logical_and(action[5]>0,inventory>0),drop,(lambda a,b,c,d,e:(a,d,0)),*(grid,posx,posy,inventory,state.permutation_recipe))
            grid,inventory=jax.lax.cond(jnp.logical_and(action[6]>0,inventory==0),collect,(lambda a,b,c,d: (a,d)),*(grid,posx,posy,inventory))
            

          


            steps = state.steps + 1
            done = False
            steps = jnp.where(done, jnp.zeros((), jnp.int32), steps)
            key, sub_key = random.split(key)

            #keep it in case we let agent several trials
            grid = jax.lax.cond(
                done, lambda x: get_init_state_fn(sub_key), lambda x: x, grid)

            return State(state=grid, obs=jnp.concatenate([get_obs(state=grid,posx=posx,posy=posy),jnp.zeros(1)*inventory]),last_action=action,reward=jnp.ones((1,))*reward,agent=AgentState(posx=posx,posy=posy,inventory=inventory),
                         steps=steps,permutation_recipe=state.permutation_recipe, key=key), reward, done
        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: State,
             action: jnp.ndarray) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state, action)



