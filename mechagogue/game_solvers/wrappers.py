"""Wrappers for game solver inputs."""

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.standardize import standardize_args
from mechagogue.static import static_functions


def vectorize_aec(aec, num_envs):
    """
    Vectorize an AEC-style turn-based game across environments.
    """
    init_fn = standardize_args(aec.init, ("key",))
    step_fn = standardize_args(aec.step, ("key", "state", "action"))

    def _select_by_done(done, x_keep, x_reset):
        mask = done
        while mask.ndim < x_keep.ndim:
            mask = jnp.expand_dims(mask, axis=-1)
        return jnp.where(mask, x_reset, x_keep)

    def init_env(key):
        keys = jrng.split(key, num_envs)
        return jax.vmap(init_fn)(keys)

    def step_env(key, state, action):
        def step_one(key, state, action):
            step_key, reset_key = jrng.split(key)
            next_state, next_obs, next_player, done, reward = step_fn(
                step_key, state, action
            )
            reset_state, reset_obs, reset_player, _ = init_fn(reset_key)
            next_state = jax.tree.map(
                lambda xk, xr: _select_by_done(done, xk, xr),
                next_state,
                reset_state,
            )
            next_obs = jax.tree.map(
                lambda xk, xr: _select_by_done(done, xk, xr),
                next_obs,
                reset_obs,
            )
            next_player = _select_by_done(done, next_player, reset_player)
            return next_state, next_obs, next_player, done, reward

        keys = jrng.split(key, num_envs)
        return jax.vmap(step_one)(keys, state, action)

    @static_functions
    class VectorizedAEC:
        num_players = aec.num_players
        init = init_env
        step = step_env

        if hasattr(aec, "action_legal_mask"):
            _mask_fn = standardize_args(aec.action_legal_mask, ("state",))
            _mask_fn = jax.vmap(_mask_fn)
            action_legal_mask = _mask_fn

    return VectorizedAEC
