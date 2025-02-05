from typing import Any, Callable

import jax.random as jrng

from mechagogue.arg_wrappers import ignore_unused_args

def population_game(
    init_state: Callable,
    transition: Callable,
    observe: Callable,
    breed: Callable,
    alive: Callable,
    children: Callable
):
    '''
    Bundles the component functions of a population game into reset and step
    functions.  The components are:
    
    init_state: a function which constructs a new environment state given
        a random key
    transition: a function which constructs a new environment state given
        a random key, a previous state and actions for each player
    observe: a function which constructs observations for each player
        given a random key and a state
    alive: a function which returns a boolean vector indicating which
        players are currently alive given a state
    children: a function which lists the parents of new children that
        were generated during the transition from one state to another
    '''
    
    init_state = ignore_unused_args(init_state,
        ('key',))
    transition = ignore_unused_args(transition,
        ('key', 'state', 'action'))
    observe = ignore_unused_args(observe,
        ('key', 'state'))
    alive = ignore_unused_args(alive,
        ('state',))
    children = ignore_unused_args(children,
        ('state', 'next_state'))
    
    def reset(key):
        # generate new keys
        state_key, observe_key = jrng.split(key)
        
        # generate the first state, observation and live vector
        state = init_state(state_key)
        obs = observe(observe_key, state)
        
        # determine the live players
        live = alive(state)
        
        # return
        return state, obs, live
    
    def step(key, state, action):
        # generate new keys
        transition_key, observe_key = jrng.split(key)
        
        # generate the next state and observation
        next_state = transition(transition_key, state, action)
        obs = observe(observe_key, next_state)
        
        # determine the live players and new children
        live = alive(next_state)
        child = children(state, next_state)
        
        # return
        return next_state, obs, live, child
    
    return reset, step
