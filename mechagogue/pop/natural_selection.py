'''
The natural selection algorithm simulates a population of players forward
over multiple time steps.  Each step uses a model function to compute an
action for each agent, then uses those actions to compute environment
dynamics.  In addition to a state and observation, the environment should
produce a list of players that currently exist, and their parents.  This
information is then used to construct the weights of new players.

This algorithm does not have an optimization objective, but instead relies
on the environment dynamics to update the population over time.
'''

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.static_dataclass import static_dataclass
from mechagogue.arg_wrappers import ignore_unused_args

@static_dataclass
class NaturalSelectionConfig:
    pass
    #rollout_steps : int = 256
    #children_per_chunk = None

@static_dataclass
class NaturalSelectionState:
    env_state : Any = None
    obs : Any = None
    model_params : Any = None

def natural_selection(
    config,
    reset_env,
    step_env,
    init_model_state,
    model,
    breed,
):
    
    # wrap the provided functions
    reset_env = ignore_unused_args(reset_env,
        ('key',))
    step_env = ignore_unused_args(step_env,
        ('key', 'state', 'action'))
    init_model_state = ignore_unused_args(init_model_state,
        ('key',))
    init_model_state = jax.vmap(init_model_state)
    model = ignore_unused_args(model,
        ('key', 'x', 'state'))
    model = jax.vmap(model)
    breed = ignore_unused_args(breed,
        ('key', 'state'))
    breed = jax.vmap(breed)
    
    def init(key):
        
        # generate keys
        env_key, model_key = jrng.split(key)
        
        # reset the environment
        env_state, obs, players = reset_env(env_key)
        
        # build the model_params
        max_population_size, = players.shape
        model_keys = jrng.split(model_key, max_population_size)
        model_params = init_model_state(model_keys)
        
        return NaturalSelectionState(env_state, obs, model_params)
    
    def step(key, state):
        
        # generate keys
        action_key, env_key, breed_key = jrng.split(key, 3)
        
        # compute actions
        max_population_size = state.obs.shape[0]
        action_keys = jrng.split(action_key, max_population_size)
        actions = model(action_keys, state.obs, state.model_params)
        
        # step the environment
        env_state, obs, players, parents, children = step_env(
            env_key, state.env_state, actions)
        
        # update the model params
        max_num_children, = children.shape
        breed_keys = jrng.split(breed_key, max_num_children)
        child_data = breed(breed_keys, state.model_params[parents])
        model_params = state.model_params.at[children].set(child_data)
        
        return NaturalSelectionState(env_state, obs, model_params)
    
    return init, step
