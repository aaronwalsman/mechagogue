"""
Simple Nim (take 1 or 2) turn-based game.
"""

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.dp.aec import make_aec
from mechagogue.static import static_data, static_functions


@static_data
class NimState:
    pile: jnp.int32
    to_play: jnp.int32
    done: jnp.bool_
    winner: jnp.int32
    step_count: jnp.int32


ACTION_TAKE1 = jnp.int32(0)
ACTION_TAKE2 = jnp.int32(1)
N_ACTIONS = 2


def make_nim(
    max_pile=7,
    min_pile=None,
):
    if min_pile is None:
        min_pile = max_pile
    if min_pile < 1 or min_pile > max_pile:
        raise ValueError("min_pile must be in [1, max_pile].")

    @static_functions
    class Nim:
        num_players = 2

        def init_state(key):
            if min_pile == max_pile:
                pile = jnp.int32(max_pile)
            else:
                pile = jrng.randint(
                    key,
                    (),
                    min_pile,
                    max_pile + 1,
                    dtype=jnp.int32,
                )
            return NimState(
                pile=pile,
                to_play=jnp.int32(0),
                done=jnp.bool_(False),
                winner=jnp.int32(-1),
                step_count=jnp.int32(0),
            )

        def transition(state, action):
            take = jnp.where(action == ACTION_TAKE2, jnp.int32(2),
                             jnp.int32(1))
            pile = state.pile - take
            done = pile <= 0
            winner = jnp.where(done, state.to_play, jnp.int32(-1))
            next_state = state.replace(
                pile=jnp.maximum(pile, 0),
                to_play=jnp.int32(1 - state.to_play),
                done=done,
                winner=winner,
                step_count=state.step_count + 1,
            )
            return next_state

        def observation(state, player):
            return {
                "pile": state.pile,
                "to_play": state.to_play,
                "player": player,
                "done": state.done,
                "action_mask": Nim.action_legal_mask(state),
            }

        def current_player(state):
            return state.to_play

        def terminal(state):
            return state.done

        def reward(key, state, action, next_state):
            del key, action
            win = next_state.winner == state.to_play
            return jnp.where(next_state.done, jnp.where(win, 1.0, -1.0), 0.0)

        def action_legal_mask(state):
            can_take2 = state.pile >= 2
            return jnp.array([True, can_take2], dtype=jnp.bool_)

    aec = make_aec(
        Nim.init_state,
        Nim.transition,
        Nim.observation,
        Nim.current_player,
        Nim.terminal,
        Nim.reward,
        num_players=Nim.num_players,
    )
    Nim.init = aec.init
    Nim.step = aec.step

    return Nim
