from functools import partial
from typing import Tuple, Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jrng

import chex

def auto_reset_wrapper(
    reset: Callable,
    step: Callable,
):
    #@dataclass
    #class AutoResetState:
    #    wrapped_state: Any
    #    terminal: bool = False
    
    def wrapped_reset(
        key: chex.PRNGKey,
    ):
        wrapped_state, obs = reset(key)
        return (wrapped_state, False), obs
    
    def wrapped_step(
        key: chex.PRNGKey,
        state: Any,
        action: Any,
    ):
        wrapped_state, previous_done = state
        
        # generate keys
        step_key, reset_key = jrng.split(key)
        
        step_state, step_obs, reward, done = step(
            step_key, wrapped_state, action)
        reset_state, reset_obs = reset(reset_key)
        
        wrapped_state, obs, reward, done = jax.tree.map(
            lambda r, s : jnp.where(jnp.expand_dims(
                previous_done, axis=tuple(range(len(r.shape)))), r, s  
            ),
            (reset_state, reset_obs, reward, done),
            (step_state, step_obs, 0., False),
        )
        return (wrapped_state, done), obs, reward, done
    
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
        state, obs = reset(key)
        return (state, state), obs
    
    def wrapped_step(
        key: chex.PRNGKey,
        state: Any,
        action: Any,
    ):
        # generate keys
        step_key, reset_key = jrng.split(key)
        
        # step and reset the wrapped environment
        env_state, _ = state
        step_state, step_obs, reward, done = step(step_key, env_state, action)
        reset_state, reset_obs = reset(reset_key)
        
        state, obs = jax.tree.map(
            lambda r, s : jnp.where(jnp.expand_dims(
                done, axis=tuple(range(len(r.shape)))), r, s  
            ),
            (reset_state, reset_obs),
            (step_state, step_obs),
        )
        return (state, step_state), obs, reward, done
    
    #return reset, partial(step_auto_reset, reset=reset, step=step)
    return wrapped_reset, wrapped_step

def episode_return_wrapper(
    reset: Callable,
    step: Callable,
):
    #@dataclass
    #class EpisodeReturnState:
    #    wrapped_state: Any
    #    returns: float = 0.
    
    def wrapped_reset(key):
        wrapped_state, obs = reset(key)
        return (wrapped_state, 0.), obs
    
    def wrapped_step(key, state, action):
        wrapped_state, returns = state
        wrapped_state, obs, reward, done = step(
            key, wrapped_state, action)
        state = (wrapped_state, returns+reward)
        return state, obs, reward, done
    
    return wrapped_reset, wrapped_step

def parallel_env_wrapper(
    reset: Callable,
    step: Callable,
):
    return jax.vmap(reset, in_axes=(0,)), jax.vmap(step, in_axes=(0,0,0))
