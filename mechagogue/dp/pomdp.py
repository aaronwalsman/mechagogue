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
    
    The components of a POMDP are:
    
    init_state(key) -> state:
        A stochastic function that samples an initial state.
    
    transition(key, state, action) -> state:
        A stochastic function that maps a state and action to a new state.
    
    observe(key, state) -> obs:
        A stochastic function that maps a state to an observation.
    
    reward(key, state, action, next_state) -> float:
        A stochastic function mapping some combination of state, action and
        next state to reward.
    
    terminal(key, state, action, next_state) -> bool:
        A stochastic function mapping some combination of state, action and
        next state to a boolean indicating that the current episode has
        terminated.
    
    Any of the functions above may require only a subset of the listed
        arguments.  The ignore_unused_args wrapper will filter out those
        which are not required.
    
    The returned functions are:
    
    reset(key) -> state, obs:
        Samples an initial state and observation.
    
    step(key, state, action) -> state, obs, reward, done:
        Samples a next_state, observation, reward and done given a current
        state and action.
    '''
    
    init_state = ignore_unused_args(
        init_state, ('key',))
    transition = ignore_unused_args(
        transition, ('key', 'state', 'action'))
    observe = ignore_unused_args(
        observe, ('key', 'state'))
    terminal = ignore_unused_args(
        terminal, ('state',))
    reward = ignore_unused_args(
        reward, ('key', 'state', 'action', 'next_state'))
    
    def reset(key):
        # generate new keys
        state_key, observe_key = jrng.split(key)
        
        # generate the first state and observation
        state = init_state(state_key)
        obs = observe(observe_key, state)
        done = terminal(state)
        
        # return
        return state, obs, done
    
    def step(key, state, action):
        # generate new keys
        transition_key, observe_key, done_key, reward_key = jrng.split(key, 4)
        
        next_state = transition(transition_key, state, action)
        obs = observe(observe_key, next_state)
        done = terminal(next_state)
        rew = reward(reward_key, state, action, next_state)
        
        # return
        return next_state, obs, done, rew
    
    return reset, step
