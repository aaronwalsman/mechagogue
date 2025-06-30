'''
A partially observable markov decision process
'''

from typing import Any, Callable, Tuple

import jax.random as jrng

from mechagogue.arg_wrappers import ignore_unused_args

def contextual_bandit(
    sample_context : Callable,
    reward : Callable = lambda : 0.,
) -> Tuple[Callable, Callable] :
    
    sample_context = ignore_unused_args(
        sample_context, ('key'))
    reward = ignore_unused_args(
        reward, ('key', 'context', 'action'))
    
    def step(key, context, action):
        # generate new keys
        reward_key, context_key = jrng.split(key, 4)
        
        rew = reward(reward_key, context, action)
        
        next_context = sample_context(context_key)
        
        # return
        return next_context, rew
    
    return sample_context, step
