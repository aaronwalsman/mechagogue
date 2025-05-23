from typing import Any, Callable

import jax.random as jrng

from mechagogue.arg_wrappers import ignore_unused_args

def poeg(
    init_state: Callable,
    transition: Callable,
    observe: Callable,
    active_players: Callable,
    family_info: Callable,
):
    '''
    Bundles the component functions of a partially observable ecological game
    into reset and step functions.  The components are:
    
    init_state: a function which constructs a new environment state given
        a random key
    transition: a function which constructs a new environment state given
        a random key, a previous state and actions for each player
    observe: a function which constructs observations for each player
        given a random key and a state
    '''
    
    init_state = ignore_unused_args(init_state,
        ('key',))
    transition = ignore_unused_args(transition,
        ('key', 'state', 'action', 'traits'))
    observe = ignore_unused_args(observe,
        ('key', 'state'))
    active_players = ignore_unused_args(active_players,
        ('state',))
    family_info = ignore_unused_args(family_info,
        ('state', 'action', 'next_state'))
    
    def reset(key):
        # generate new keys
        state_key, observe_key = jrng.split(key)
        
        # generate the first state, observation and live vector
        state = init_state(state_key)
        obs = observe(observe_key, state)
        players = active_players(state)
        
        # return
        return state, obs, players
    
    def step(key, state, action, traits):
        # generate new keys
        transition_key, observe_key = jrng.split(key)
        
        # generate the next state and observation
        next_state = transition(transition_key, state, action, traits)
        obs = observe(observe_key, next_state)
        players = active_players(next_state)
        parents, children = family_info(state, action, next_state)
        
        # return
        return next_state, obs, players, parents, children
    
    return reset, step
