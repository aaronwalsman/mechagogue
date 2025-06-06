from functools import partial
from typing import Tuple, Any, Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrng

import chex

from mechagogue.arg_wrappers import ignore_unused_args, split_random_keys

def auto_reset_wrapper(
    reset: Callable,
    step: Callable,
):
    
    def wrapped_reset(
        key: chex.PRNGKey,
    ):
        wrapped_state, obs, done = reset(key)
        return (wrapped_state, False), obs, done
    
    def wrapped_step(
        key: chex.PRNGKey,
        state: Any,
        action: Any,
    ):
        wrapped_state, previous_done = state
        
        # generate keys
        step_key, reset_key = jrng.split(key)
        
        # step and reset the wrapped environment
        step_state, step_obs, done, reward = step(
            step_key, wrapped_state, action)
        reset_state, reset_obs, reset_done = reset(reset_key)
        reset_reward = jnp.zeros_like(reward)
        
        # If previous_done was True, discard the `step` results and substitute the `reset` results
        # Otherwise keep the `step` results
        wrapped_state, obs, done, reward = jax.tree.map(
            lambda r, s : jnp.where(previous_done, r, s),
            (reset_state, reset_obs, reset_done, reset_reward),
            (step_state, step_obs, done, reward),
        )
        return (wrapped_state, done), obs, done, reward
    
    return wrapped_reset, wrapped_step

# ---------------------------------------------------------------------------
# Sticky-action wrapper, matching MinAtar: repeat previous action with prob p
# ---------------------------------------------------------------------------
def sticky_action_wrapper(
    reset: Callable,
    step: Callable,
    prob: float = 0.1,
):
    """
    Wrap the environment `step` so that the effective action is
    either the one the agent proposed (`action`) or, with probability `prob`,
    the previous effective action.
    Works transparently with vectorized envs (extra batch dim).
    """

    def _init_last_action(first_action):
        # Ensure same dtype/shape as action space (scalar or batched)
        return jnp.zeros_like(first_action)

    def wrapped_step(
        key: chex.PRNGKey,
        state: Any,
        action: Any,
    ):
        env_state, last_action = state
        key, sticky_key = jrng.split(key)

        # decide which action to feed into the base env
        repeat_prev = jrng.uniform(sticky_key, shape=()) < prob
        eff_action  = jnp.where(repeat_prev, last_action, action)

        # step the base environment
        next_env_state, obs, done, reward = step(key, env_state, eff_action)

        # when an episode terminates, reset last_action to zero
        next_last_action = jnp.where(done, _init_last_action(eff_action), eff_action)

        return (next_env_state, next_last_action), obs, done, reward

    # expose an initializer so callers can build the composite state
    def wrapped_reset(key):
        env_state, obs, done = reset(key)
        return (env_state, _init_last_action(0)), obs, done

    # we only modify `step`, keep original `reset` unchanged
    return wrapped_reset, wrapped_step

def auto_reset_wrapper_overwrite_terminal(
    reset: Callable,
    step: Callable,
):
    @dataclass
    class AutoResetState:
        next_state: Any
        final_state: Any
    
    def wrapped_reset(key):
        wrapped_state, obs, done = reset(key)
        return (wrapped_state, wrapped_state), obs, done
    
    def wrapped_step(
        key: chex.PRNGKey,
        state: Any,
        action: Any,
    ):
        # generate keys
        step_key, reset_key = jrng.split(key)
        
        # step and reset the wrapped environment
        env_state, _ = state
        step_state, step_obs, done, reward = step(step_key, env_state, action)
        reset_state, reset_obs = reset(reset_key)
        
        state, obs = jax.tree.map(
            lambda r, s : jnp.where(jnp.expand_dims(
                done, axis=tuple(range(len(r.shape)))), r, s  
            ),
            (reset_state, reset_obs),
            (step_state, step_obs),
        )
        return (state, step_state), obs, done, reward
    
    return wrapped_reset, wrapped_step

def episode_return_wrapper(
    reset: Callable,
    step: Callable,
):
    def wrapped_reset(key):
        wrapped_state, obs, done = reset(key)
        return (wrapped_state, 0.), obs, done
    
    def wrapped_step(key, state, action):
        wrapped_state, returns = state
        wrapped_state, obs, done, reward = step(
            key, wrapped_state, action)
        state = (wrapped_state, returns+reward)
        return state, obs, done, reward
    
    return wrapped_reset, wrapped_step

def parallel_env_wrapper(
    reset: Callable,
    step: Callable,
    num_parallel_envs: int,
):
    reset = ignore_unused_args(reset, ('key',))
    reset = jax.vmap(reset)
    reset = split_random_keys(reset, num_parallel_envs)
    
    step = ignore_unused_args(step, ('key', 'state', 'action'))
    step = jax.vmap(step)
    step = split_random_keys(step, num_parallel_envs)
    
    return reset, step
