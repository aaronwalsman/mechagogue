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
    init_population_state,
    player_traits,
    model,
    breed,
    adapt,
):
    '''
    reset_env(key) -> (
        env_state, obs, players)
    step_env(key, state, action, traits) -> (
        env_state, obs, players, parents, children)
    player_traits(model_state) -> (
        traits)
    init_population_state(key, population_size, max_population_size) -> (
        model_state)
    model(key, x, model_state) -> (
        action, adaptation)
    breed(key, parent_state) -> (
        child_state)
    adapt(key, adaptation, model_state) -> (
        model_state)
        Allows for model-controlled updates to the model parameters in order
        to simulate memory and physical changes over the course of a player's
        lifetime.  Also allows for Lamarckianism if the heritable components
        of model_state are modified here.
    '''
    # wrap the provided functions
    reset_env = ignore_unused_args(reset_env,
        ('key',))
    step_env = ignore_unused_args(step_env,
        ('key', 'state', 'action', 'traits'))
    init_population_state = ignore_unused_args(init_population_state,
        ('key', 'population_size', 'max_population_size'))
    player_traits = ignore_unused_args(player_traits,
        ('state',))
    model = ignore_unused_args(model,
        ('key', 'x', 'state'))
    model = jax.vmap(model)
    breed = ignore_unused_args(breed,
        ('key', 'state'))
    breed = jax.vmap(breed)
    adapt = ignore_unused_args(adapt,
        ('key', 'adaptation', 'state'))
    adapt = jax.vmap(adapt)
    
    def init(key):
        
        # generate keys
        env_key, model_key = jrng.split(key)
        
        # reset the environment
        env_state, obs, players = reset_env(env_key)
        
        # build the model_state
        population_size = jnp.sum(players)
        model_state = init_population_state(
            model_key, population_size, params.max_population)
        
        return NaturalSelectionState(env_state, obs, model_state), players
    
    def step(key, state):
        
        # generate keys
        action_key, adapt_key, env_key, breed_key = jrng.split(key, 4)
        
        # compute the traits that will be passed to the environment
        traits = player_traits(state.model_state)
        
        # compute actions
        action_keys = jrng.split(action_key, params.max_population)
        actions, adaptations = model(action_keys, state.obs, state.model_state)
        
        # modify the model state according to the adaptations signal
        adapt_keys = jrng.split(adapt_key, params.max_population)
        model_state = adapt(adapt_keys, adaptations, state.model_state)
        
        # step the environment
        env_state, obs, players, parents, children = step_env(
            env_key, state.env_state, actions, traits)
        
        # update the model state
        max_num_children, = children.shape
        breed_keys = jrng.split(breed_key, max_num_children)
        parent_state = tree_getitem(model_state, parents)
        child_state = breed(breed_keys, parent_state)
        model_state = tree_setitem(model_state, children, child_state)
        
        # build the next state
        next_state = state.replace(
            env_state=env_state, obs=obs, model_state=model_state)
        
        return (
            next_state,
            players,
            parents,
            children,
            actions,
            traits,
            adaptations,
        )
    
    return init, step
