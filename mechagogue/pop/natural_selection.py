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
from mechagogue.tree import tree_getitem, tree_setitem

@static_dataclass
class NaturalSelectionParams:
    max_population : int

@static_dataclass
class NaturalSelectionState:
    env_state : Any
    obs : Any
    model_state : Any

def natural_selection(
    params,
    reset_env,
    step_env,
    init_model_state,
    model,
    breed,
    make_report = lambda : None,
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
    make_report = ignore_unused_args(make_report, (
        'state',
        'actions',
        'next_state',
        'players',
        'parent_locations',
        'child_locations',
    ))
    
    def init(key):
        
        # generate keys
        env_key, model_key = jrng.split(key)
        
        # reset the environment
        env_state, obs, players = reset_env(env_key)
        
        # build the model_state
        model_keys = jrng.split(model_key, params.max_population)
        model_state = init_model_state(model_keys)
        
        return NaturalSelectionState(env_state, obs, model_state), players
    
    def step(key, state):
        
        # generate keys
        action_key, env_key, breed_key = jrng.split(key, 3)
        
        # compute actions
        action_keys = jrng.split(action_key, params.max_population)
        actions = model(action_keys, state.obs, state.model_state)

        # step the environment
        env_state, obs, players, parent_locations, child_locations = step_env(
            env_key, state.env_state, actions)
        
        # update the model state
        max_num_children, = child_locations.shape
        breed_keys = jrng.split(breed_key, max_num_children)
        parent_state = tree_getitem(state.model_state, parent_locations)
        child_state = breed(breed_keys, parent_state)
        model_state = tree_setitem(
            state.model_state, child_locations, child_state)
        
        # build the next state
        next_state = state.replace(
            env_state=env_state, obs=obs, model_state=model_state)
        
        # log
        report = make_report(
            state,
            actions,
            next_state,
            players,
            parent_locations,
            child_locations,
        )
        
        return next_state, report
    
    return init, step
