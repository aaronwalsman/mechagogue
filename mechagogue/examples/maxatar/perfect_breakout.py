'''
Generate and visualize a perfect-action policy playing Breakout.
'''

import jax
import jax.numpy as jnp
import jax.random as jrng
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import imageio.v3 as iio

import mechagogue.envs.maxatar.breakout as breakout
from mechagogue.wrappers import auto_reset_wrapper, sticky_action_wrapper

# Initialize the environment
env = breakout.make_env(ramping=True, perfect=True)
env = auto_reset_wrapper(env)  # sticky_action_wrapper(env, prob=0.1)

def perfect_breakout(key):
    '''
    Generate n frames with a perfect-action policy
    Returns:
        (state, obs, done, action, reward) stacked along axis 0
    '''
    
    # number of frames to generate
    n = 100
    
    key, init_key = jrng.split(key)
    state, obs, done = env.init(init_key)
    
    def scan_step(key_state_obs_done, _):
        key, state, obs, done = key_state_obs_done
        key, action_key, step_key = jrng.split(key, 3)
        action = jnp.array(0, dtype=jnp.int32)  # no-op; perfect = True will move the paddle below the ball
        next_state, next_obs, next_done, rew = env.step(
            step_key, state, action)
        
        return (key, next_state, next_obs, next_done), (
            state, obs, done, action, rew)
    
    key_state_obs, trajectories = jax.lax.scan(
        scan_step, (key, state, obs, done), None, length=n)
    
    return trajectories

perfect_breakout = jax.jit(perfect_breakout)

state, obs, done, action, rew = perfect_breakout(jrng.key(5678))

# visualize and save GIF
cmap   = sns.color_palette("cubehelix", breakout.TOTAL_CHANNELS)
cmap.insert(0, (0,0,0))
cmap   = colors.ListedColormap(cmap)
bounds = list(range(breakout.TOTAL_CHANNELS + 2))
norm   = colors.BoundaryNorm(bounds, breakout.TOTAL_CHANNELS + 1)

fig, ax = plt.subplots(1, 1)
frames = []
pause = 0.5  # s per frame

for i, obs_i in enumerate(obs):
    numeric = jnp.amax(
        obs_i * jnp.reshape(jnp.arange(breakout.TOTAL_CHANNELS) + 1, (1, 1, -1)), axis=2) + 0.5

    ax.imshow(numeric, cmap=cmap, norm=norm, interpolation="none")
    ax.set_title(f"Frame {i}")
    ax.axis("off")
    fig.canvas.draw()

    # grab the pixel buffer & store it
    frame = jnp.array(fig.canvas.buffer_rgba()).copy()
    frames.append(frame)
    ax.cla()

plt.close(fig)

out_path = "out/breakout_perfect.gif"
iio.imwrite(out_path, frames, duration=pause)
