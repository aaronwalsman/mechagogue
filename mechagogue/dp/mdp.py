from typing import Any, Callable

from mechagogue.dp.pomdp import pomdp

def mdp(
    initialize_fn : Callable,
    transition_fn : Callable,
    reward_fn : Callable,
    done_fn : Callable,
    config : Any = None,
):
    return pomdp(
        initialize_fn,
        transition_fn,
        lambda key, config, state : state,
        reward_fn,
        done_fn,
        config=config,
    )
