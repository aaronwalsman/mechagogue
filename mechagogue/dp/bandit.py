'''
A partially observable markov decision process
'''

from typing import Any, Callable, Tuple

import jax.random as jrng

from mechagogue.arg_wrappers import ignore_unused_args

def contextual_bandit(
    reward : Callable,
) -> Tuple[Callable, Callable] :
    
    reward = ignore_unused_args(
        reward, ('key', 'action'))
    
    return lambda: None, reward
