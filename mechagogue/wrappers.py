from typing import Tuple, Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jrng

import chex

def step_auto_reset(
    key : chex.PRNGKey,
    reset_env : Callable,
    step_env : Callable,
    params : Any,
    state : Any,
    action : Any,
):

    # generate rng keys
    key, step_key, reset_key = jrng.split(key, 3)

    # compute the step observation, state, reward and done
    obs_step, state_step, reward, done, *other = step_env(
        step_key, params, state, action)

    # compute the reset observation and state
    obs_reset, state_reset = reset_env(reset_key, params)

    # select the observation and state from either the step or reset
    # observation/state depending on done
    obs, state = jax.tree.map(
        lambda r, s : jnp.where(jnp.expand_dims(
            done, axis=tuple(range(1, len(r.shape)))), r, s
        ),
        (obs_reset, state_reset),
        (obs_step, state_step),
    )
    return (obs, state, reward, done, *other)
