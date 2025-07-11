from typing import Any, Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.standardize import standardize_args
from mechagogue.standardize import split_random_keys
from mechagogue.static import static_functions


def auto_reset_wrapper(env):
    @static_functions
    class AutoResetWrapper:
        def init(key):
            wrapped_state, obs, done = env.init(key)
            return (wrapped_state, False), obs, done
        
        def step(key, state, action):
            wrapped_state, previous_done = state
            
            # generate keys
            step_key, reset_key = jrng.split(key)
            
            # step and reset the wrapped environment
            step_state, step_obs, done, reward = env.step(
                step_key, wrapped_state, action)
            reset_state, reset_obs, reset_done = env.init(reset_key)
            reset_reward = jnp.zeros_like(reward)
            
            # If previous_done was True, discard the `step` results and substitute the `reset` results
            # Otherwise keep the `step` results
            wrapped_state, obs, done, reward = jax.tree.map(
                lambda r, s : jnp.where(previous_done, r, s),
                (reset_state, reset_obs, reset_done, reset_reward),
                (step_state, step_obs, done, reward),
            )
            return (wrapped_state, done), obs, done, reward
    
    return AutoResetWrapper


# ---------------------------------------------------------------------------
# Sticky-action wrapper, matching MinAtar: repeat previous action with prob p
# ---------------------------------------------------------------------------
def sticky_action_wrapper(env, prob=0.1):
    @static_functions
    class StickyActionWrapper:
        def _init_last_action_like(arr):
            return jnp.zeros_like(arr)

        def init(key):
            env_state, obs, done = env.init(key)
            # we don’t know the action dtype until we see one; keep scalar 0
            return (env_state, jnp.array(0, dtype=jnp.int32)), obs, done

        def step(key, state, action):
            env_state, last_action = state
            key, sk = jrng.split(key)

            # choose per-environment whether to repeat
            repeat_prev = jrng.uniform(sk, shape=action.shape) < prob
            eff_action  = jnp.where(repeat_prev, last_action, action)

            next_env_state, obs, done, rew = env.step(key, env_state, eff_action)

            next_last_action = jnp.where(done,
                                        StickyActionWrapper._init_last_action_like(eff_action),
                                        eff_action)
            return (next_env_state, next_last_action), obs, done, rew

    return StickyActionWrapper


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
    
    def wrapped_step(key, state, action):
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


def parallel_env_wrapper(env, num_parallel_envs: int):
    reset = standardize_args(env.init, ('key',))
    reset = jax.vmap(reset)
    reset = split_random_keys(reset, num_parallel_envs)
    
    step = standardize_args(env.step, ('key', 'state', 'action'))
    step = jax.vmap(step)
    step = split_random_keys(step, num_parallel_envs)
    
    return reset, step
