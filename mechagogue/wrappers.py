from typing import Tuple, Any, Callable
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrng

import chex

def step_auto_reset_OLD(
    key : chex.PRNGKey,
    initial_state_distribution : Callable,
    forward : Callable,
    state : Any,
    action : Any,
):

    # generate rng keys
    key, initial_state_key, forward_key = jrng.split(key, 3)

    # compute the step observation, state, reward and done
    obs_step, state_step, reward, done = step_env(
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

def step_auto_reset(
    key: chex.PRNGKey,
    state: Any,
    action: Any,
    reset: Callable,
    step: Callable,
):
    step_key, reset_key = jrng.split(key)
    step_state, step_obs, reward, done = step(step_key, state, action)
    reset_state, reset_obs = reset(reset_key)
    state, obs = jax.tree.map(
        lambda r, s : jnp.where(jnp.expand_dims(
            done, axis=tuple(range(len(r.shape)))), r, s  
        ),
        (reset_state, reset_obs),
        (step_state, step_obs),
    )
    return state, obs, reward, done

def auto_reset_wrapper(
    reset: Callable,
    step: Callable,
):
    return reset, partial(step_auto_reset, reset=reset, step=step)

def episode_return_wrapper(
    reset: Callable,
    step: Callable,
):
    def wrapped_reset(key):
        state, obs = reset(key)
        return (0, state), obs
    
    def wrapped_step(key, state, action):
        r, state = state
        state, obs, reward, done = step(key, state, action)
        return (r+reward, state), obs, reward, done
    
    return wrapped_reset, wrapped_step

def parallel_env_wrapper(
    reset: Callable,
    step: Callable,
):
    return jax.vmap(reset, in_axes=(0,)), jax.vmap(step, in_axes=(0,0,0))
