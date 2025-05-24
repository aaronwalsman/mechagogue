'''
A partially observable markov decision process
'''

from typing import Any, Callable, Tuple

import jax.random as jrng

from mechagogue.arg_wrappers import ignore_unused_args

def pomdp(
    init_state : Callable,
    transition : Callable,
    observe : Callable,
    reward : Callable = lambda : 0.,
    terminal : Callable = lambda : False,
) -> Tuple[Callable, Callable] :
    '''
    Builds reset and step functions for a partially observable markov decision
    process (POMDP) from its various components.
    
    [wikipedia](www.wikipedia.com/pomdp)
    `test <www.google.com>`
    
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
    
    reset(key) -> state, obs, done:
        Samples an initial state and observation.
    
    step(key, state, action) -> state, obs, done, reward:
        Samples a next_state, observation, reward and done given a current
        state and action.
    '''
    
    init_state = ignore_unused_args(init_state, ('key'))
    transition = ignore_unused_args(
        transition, ('key', 'state', 'action'))
    observe = ignore_unused_args(observe, ('key', 'state'))
    terminal = ignore_unused_args(
        terminal, ('key', 'state'))
    reward = ignore_unused_args(
        reward, ('key', 'state', 'action', 'next_state'))
    
    @static_functions
    class Env:
        def reset(key):
            # generate new keys
            initialize_key, observe_key = jrng.split(key, 2)
            
            # generate the first state and observation
            state = initialize(initialize_key)
            obs = observe(observe_key, state)
            done = terminal(state)
            
            # return
            return state, obs, done
        
        def step(key, state, action):
            # generate new keys
            transition_key, observe_key, reward_key = jrng.split(key, 4)
            
            # generate the next state and observation
            next_state = transition_fn(transition_key, state, action)
            obs = observe_fn(observe_key, next_state)
            
            # compute the reward and done
            rew = reward(reward_key, state, action, next_state)
            done = terminal(next_state)
            
            # return
            return next_state, obs, done, rew
    
    return Env
