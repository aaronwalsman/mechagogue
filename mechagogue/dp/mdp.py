from typing import Any, Callable

from mechagogue.dp.pomdp import make_pomdp

def mdp(
    init_state : Callable,
    transition : Callable,
    reward : Callable,
    terminal : Callable,
):
    return make_pomdp(
        init_state,
        transition,
        lambda state : state,
        terminal,
        reward,
    )
