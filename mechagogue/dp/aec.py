"""
Agent-environment cycle (AEC) environment builder.

Builds turn-based multi-agent environments from component functions.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.standardize import standardize_args, standardize_interface
from mechagogue.static import static_functions

default_init = lambda: None
default_step = lambda state: state


def standardize_aec(aec):
    return standardize_interface(
        aec,
        init=(("key",), default_init),
        step=(("key", "state", "action"), default_step),
    )


def make_aec(
    init_state: Callable,
    transition: Callable,
    observe: Callable,
    current_player: Callable,
    terminal: Callable,
    reward: Callable,
    num_players: int,
):
    """
    Builds an AEC environment from component functions.

    init_state(key) -> state:
        Samples an initial state.

    transition(key, state, action) -> state:
        Maps state and action to a next state.

    observe(key, state, player) -> obs:
        Maps a state and player to an observation.

    current_player(state) -> player:
        Returns the player id for the current turn.

    terminal(state) -> bool:
        Returns True when the episode is over.

    reward(key, state, action, next_state) -> reward:
        Computes a reward signal for the transition. Prefer returning a
        vector shaped (num_players,) so all players observe outcomes. If
        a scalar is returned, it is assigned to the current player and
        other players receive zero reward.

    The returned object exposes:
      init(key) -> state, obs, player, done
      step(key, state, action) -> state, obs, player, done, reward
    where obs is for the current player and action is the action for
    that player.
    """
    init_state = standardize_args(init_state, ("key",))
    transition = standardize_args(transition, ("key", "state", "action"))
    observe = standardize_args(observe, ("key", "state", "player"))
    current_player = standardize_args(current_player, ("state",))
    terminal = standardize_args(terminal, ("state",))
    reward = standardize_args(
        reward, ("key", "state", "action", "next_state")
    )
    num_players_value = num_players

    @static_functions
    class AEC:
        num_players = num_players_value

        def init(key):
            init_key, observe_key = jrng.split(key, 2)
            state = init_state(init_key)
            player = current_player(state)
            obs = observe(observe_key, state, player)
            done = terminal(state)
            return state, obs, player, done

        def step(key, state, action):
            transition_key, observe_key, reward_key = jrng.split(key, 3)
            done = terminal(state)

            def _step(_):
                prev_player = current_player(state)
                next_state = transition(transition_key, state, action)
                player = current_player(next_state)
                obs = observe(observe_key, next_state, player)
                players = jnp.arange(num_players, dtype=jnp.int32)
                rew = reward(reward_key, state, action, next_state)
                is_scalar = rew.ndim == 0
                rew = jnp.where(is_scalar, jnp.asarray(rew), rew)
                if is_scalar:
                    mask = players == prev_player
                    rew = jnp.where(mask, rew, jnp.zeros_like(rew))
                done_next = terminal(next_state)
                return next_state, obs, player, done_next, rew

            def _stay(_):
                player = current_player(state)
                obs = observe(observe_key, state, player)
                reward = jnp.zeros((num_players,), dtype=jnp.float32)
                return state, obs, player, done, reward

            return jax.lax.cond(done, _stay, _step, operand=None)

    return AEC
