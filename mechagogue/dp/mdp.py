from typing import Any, Callable

from mechagogue.dp.pomdp import pomdp

def mdp(
    params : Any,
    initialize_fn : Callable,
    transition_fn : Callable,
    reward_fn : Callable,
    done_fn : Callable,
    reward_format='san',
    done_format='san',
):
    return pomdp(
        params,
        initialize_fn,
        transition_fn,
        lambda key, params, state : state,
        reward_fn,
        done_fn,
        reward_format=reward_format,
        done_format=done_format,
    )
