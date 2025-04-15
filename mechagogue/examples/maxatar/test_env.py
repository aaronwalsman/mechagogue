import jax
import jax.numpy as jnp
import jax.random as jrng
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

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

state, obs, done, action, rew = test_breakout(jrng.key(5678))

cmap = sns.color_palette("cubehelix", breakout.TOTAL_CHANNELS)
cmap.insert(0, (0,0,0))
cmap = colors.ListedColormap(cmap)
bounds = [i for i in range(breakout.TOTAL_CHANNELS+2)]
norm = colors.BoundaryNorm(bounds, breakout.TOTAL_CHANNELS+1)
_, ax = plt.subplots(1,1)
plt.show(block=False)

for i, obs_i in enumerate(obs):
    assert obs_i.shape == (10,10,4)
    numerical_state = jnp.amax(
        obs_i * jnp.reshape(jnp.arange(breakout.TOTAL_CHANNELS) + 1, (1, 1, -1)), axis=2) + 0.5
    ax.imshow(
        numerical_state, cmap=cmap, norm=norm, interpolation='none')
    plt.pause(1)
    plt.cla()
    
plt.close()

breakpoint()

