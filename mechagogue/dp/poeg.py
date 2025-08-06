from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.standardize import standardize_args, standardize_interface
from mechagogue.static import static_functions, static_data

default_init = lambda : None
default_step = lambda state : state

def standardize_poeg(poeg):
    return standardize_interface(
        poeg,
        init = (('key',), default_init),
        step = (('key', 'state', 'action', 'traits'), default_step),
    )

def make_poeg(
    init_state: Callable,
    transition: Callable,
    observe: Callable,
    active_players: Callable,
    family_info: Callable,
    **members,
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
    
    init_state = standardize_args(init_state,
        ('key',))
    transition = standardize_args(transition,
        ('key', 'state', 'action', 'traits'))
    observe = standardize_args(observe,
        ('key', 'state'))
    active_players = standardize_args(active_players,
        ('state',))
    family_info = standardize_args(family_info,
        ('state', 'action', 'next_state'))
    
    @static_functions
    class POEG:
        def init(key):
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
        
        def population(state):
            return jnp.sum(active_players(state))
        
        def extinct(state):
            #n = active_players(state).sum()
            return not jnp.any(active_players(state))
    
    setattr(POEG, 'init_state', init_state)
    setattr(POEG, 'transition', transition)
    setattr(POEG, 'observe', observe)
    setattr(POEG, 'active_players', active_players)
    setattr(POEG, 'family_info', family_info)
    
    for name, member in members.items():
        if callable(member):
            member = staticmethod(member)
        setattr(POEG, name, member)
    
    return POEG
