import jax
import jax.numpy as jnp
import jax.random as jrng

import mechagogue.envs.maxatar.breakout as breakout
from mechagogue.wrappers import auto_reset_wrapper

breakout_reset, breakout_step = auto_reset_wrapper(
    breakout.reset, breakout.step)

def test_breakout(key):
    
    n = 1000
    
    key, init_key = jrng.split(key)
    
    state, obs, done = breakout_reset(init_key)
    
    def scan_step(key_state_obs_done, _):
        key, state, obs, done = key_state_obs_done
        key, action_key, step_key = jrng.split(key, 3)
        action = jrng.randint(key, shape=(), minval=0, maxval=6)
        next_state, next_obs, next_done, rew = breakout_step(
            step_key, state, action)
        
        return (key, next_state, next_obs, next_done), (
            state, obs, done, action, rew)
    
    key_state_obs, trajectories = jax.lax.scan(
        scan_step, (key, state, obs, done), None, length=n)
    
    return trajectories

test_breakout = jax.jit(test_breakout)

state, obs, done, action, rew = test_breakout(jrng.key(1234))

breakpoint()

