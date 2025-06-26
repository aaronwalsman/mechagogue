from typing import Any, Callable

from mechagogue.dp.pomdp import pomdp

def mdp(
    init_state : Callable,
    transition : Callable,
    reward : Callable,
    terminal : Callable,
):
    return pomdp(
        init_state,
        transition,
        lambda state : state,
        reward,
        terminal,
    )
