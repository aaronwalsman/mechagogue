'''
A partially observable markov decision process
'''

from typing import Any, Callable, Tuple

import jax.random as jrng

from decisionprocess.filter_args import filter_args

def pomdp(
    params : Any,
    initialize_fn : Callable,
    transition_fn : Callable,
    observe_fn : Callable,
    reward_fn : Callable = lambda key, state, action, next_state: 0.,
    done_fn : Callable = lambda key, state, action, next_state: False,
    reward_format : str = 'san',
    done_format : str = 'san',
) -> Tuple[Callable, Callable] :
    '''
    Builds reset and step functions for a partially observable markov decision
    process (POMDP) from its various components.
    
    [wikipedia](www.wikipedia.com/pomdp)
    `test <www.google.com>`
    
    The components of a POMDP are:
    
    params:
        A static set of environment-specific configuration parameters defining
        properties of the environment.
    
    initialize_fn(key, params) -> state:
        A stochastic function that samples an initial state.
    
    transition_fn(key, params, state, action) -> state:
        A stochastic function that maps a state and action to a new state.
    
    observe_fn(key, params, state) -> obs:
        A stochastic function that maps a state to an observation.
    
    reward_fn(key, [state, action, next_state]) -> float:
        A stochastic function mapping some combination of state, action and
        next state to reward.  The 'reward_format' argument controls which
        of [state, action, next_state] should be used to compute the reward.
    
    done_fn(key, [state, action, next_state]) -> bool:
        A stochastic function mapping some combination of state, action and
        next state to a boolean indicating that the current episode has
        terminated.  The 'done_format' argument controls which of
        [state, action, next_state] should be used to compute done.
    
    The returned functions are:
    
    reset(key) -> state, obs:
        Samples an initial state and observation.
    
    step(key, state, action) -> state, obs, reward, done:
        Samples a next_state, observation, reward and done given a current
        state and action.
    '''
    
    def reset(key):
        # generate new keys
        initialize_key, observe_key = jrng.split(key, 2)
        
        # generate the first state and observation
        state = initialize_fn(initialize_key, params)
        obs = observe_fn(observe_key, params, state)
        
        # return
        return state, obs
    
    def step(key, state, action):
        # generate new keys
        transition_key, observe_key, reward_key, done_key = jrng.split(key, 4)
        
        # generate the next state and observation
        next_state = transition_fn(transition_key, params, state, action)
        obs = observe_fn(key, params, next_state)
        
        # compute the reward and done
        reward_args = filter_args(reward_format, state, action, next_state)
        reward = reward_fn(reward_key, params, *reward_args)
        done_args = filter_args(done_format, state, action, next_state)
        done = done_fn(done_key, params, *done_args)
        
        # return
        return next_state, obs, reward, done
    
    return reset, step