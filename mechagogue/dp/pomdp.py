'''
Partially Observable Markov Decision Process (POMDP) framework.

Builder for POMDP environments from component functions including
state transitions, observations, rewards, and terminal conditions.
'''

from typing import Any, Callable, Tuple

import jax.random as jrng

from mechagogue.standardize import standardize_args, standardize_interface
from mechagogue.static import static_functions, static_data

default_init = lambda : None
default_step = lambda state : state

def standardize_pomdp(pomdp):
    return standardize_interface(
        pomdp,
        init = (('key',), default_init),
        step = (('key', 'state', 'action'), default_step),
    )

def make_pomdp(
    init_state : Callable,
    transition : Callable,
    observe : Callable,
    terminal : Callable = lambda : False,
    reward : Callable = lambda : 0.,
):
    '''
    Builds a partially observable markov decision
    process (POMDP) from its various components.
    
    [wikipedia](www.wikipedia.com/pomdp)
    
    The components of a POMDP are:
    
    init_state(key) -> state:
        A stochastic function that samples an initial state.
    
    transition(key, state, action) -> state:
        A stochastic function that maps a state and action to a new state.
    
    observe(key, state) -> obs:
        A stochastic function that maps a state to an observation.
    
    terminal(state) -> bool:
        A deterministic function mapping state to a boolean indicating
        that the current episode has terminated.
    
    reward(key, state, action, next_state) -> float:
        A stochastic function mapping some combination of state, action and
        next state to reward.
    
    The returned environment will have the following static methods:
    
    init(key) -> state, obs, done:
        Samples an initial state and observation.
    
    step(key, state, action) -> state, obs, done, reward:
        Samples a next_state, observation, reward and done given a current
        state and action.
    '''
    
    init_state = standardize_args(init_state, ('key',))
    transition = standardize_args(transition, ('key', 'state', 'action'))
    observe = standardize_args(observe, ('key', 'state'))
    terminal = standardize_args(terminal, ('state',))
    reward = standardize_args(reward, ('key', 'state', 'action', 'next_state'))
    
    @static_functions
    class POMDP:
        def init(key):
            initialize_key, observe_key = jrng.split(key, 2)
            
            # generate the first state and observation
            state = init_state(initialize_key)
            obs = observe(observe_key, state)
            done = terminal(state)
            
            return state, obs, done
        
        def step(key, state, action):
            transition_key, observe_key, reward_key = jrng.split(key, 3)
            
            # generate the next state and observation
            next_state = transition(transition_key, state, action)
            obs = observe(observe_key, next_state)
            
            # compute the reward and done
            rew = reward(reward_key, state, action, next_state)
            done = terminal(next_state)
            
            return next_state, obs, done, rew
    
    return POMDP
