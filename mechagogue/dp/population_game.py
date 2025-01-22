from typing import Any, Callable

import jax.random as jrng

def population_game(
    config: Any,
    initialize_fn: Callable,
    transition_fn: Callable,
    observe_fn: Callable,
    players_fn: Callable,
    parents_fn: Callable
):
    '''
    Bundles the component functions of a population game into reset and step
    functions.  The components are:
    
    config: configuration parameters that do not change during evaluation.
    initialize_fn: a function which constructs a new environment state given
        a random key and config
    transition_fn: a function which constructs a new environment state given
        a random key, config, a previous state and actions for each player
    observe_fn: a function which constructs observations for each player
        given a random key, config and a state
    players_fn: a function which lists the current players given config and
        a state
    parents_fn: a function which lists the parents of the current players
        given config and a state
    '''
    def reset(key):
        initialize_key, observe_key = jrng.split(key, 2)
        state = initialize_fn(initialize_key, config)
        players = players_fn(config, state)
        parents = parents_fn(config, state)
        obs = observe_fn(observe_key, config, state)
        return state, obs, players, parents
    
    def step(key, state, action):
        transition_key, observe_key = jrng.split(key, 2)
        next_state = transition_fn(transition_key, config, state, action)
        players = players_fn(config, next_state)
        parents = parents_fn(config, next_state)
        obs = observe_fn(key, config, next_state)
        return next_state, obs, players, parents
    
    return reset, step
