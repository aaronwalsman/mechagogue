from typing import Any, Callable

import jax.random as jrng

def population_game(
    params: Any,
    initialize_fn: Callable,
    transition_fn: Callable,
    observe_fn: Callable,
    players_fn: Callable,
    parents_fn: Callable
):
    '''
    Bundles the component functions of a population game into reset and step
    functions.  The components are:
    
    params: configuration parameters that do not change during evaluation.
    initialize_fn: a function which constructs a new environment state given
        a random key and params
    transition_fn: a function which constructs a new environment state given
        a random key, params, a previous state and actions for each player
    observe_fn: a function which constructs observations for each player
        given a random key, params and a state
    players_fn: a function which lists the current players given params and
        a state
    parents_fn: a function which lists the parents of the current players
        given params and a state
    '''
    def reset(key):
        initialize_key, observe_key = jrng.split(key, 2)
        state = initialize_fn(initialize_key, params)
        players = players_fn(params, state)
        parents = parents_fn(params, state)
        obs = observe_fn(observe_key, params, state)
        return state, obs, players, parents
    
    def step(key, state, action):
        transition_key, observe_key = jrng.split(key, 2)
        next_state = transition_fn(transition_key, params, state, action)
        players = players_fn(params, next_state)
        parents = parents_fn(params, next_state)
        obs = observe_fn(key, params, next_state)
        return next_state, obs, players, parents
    
    return reset, step
