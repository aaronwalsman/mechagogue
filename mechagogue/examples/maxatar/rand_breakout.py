import jax
import jax.numpy as jnp
import jax.random as jrng
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import imageio.v3 as iio

import mechagogue.envs.maxatar.breakout as breakout
from mechagogue.wrappers import auto_reset_wrapper, sticky_action_wrapper

# Get the reset and step functions from make_env
reset_env, step_env = breakout.make_env(ramping=True)
breakout_reset, breakout_step = auto_reset_wrapper(reset_env, step_env)  # sticky_action_wrapper(reset_env, step_env, prob=0.1)

def rand_breakout(key):
    """
        Generate n frames with a random-action policy; returns
       (state, obs, done, action, reward) stacked along axis 0.
    """
    n = 100  # number of frames to generate
    
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

rand_breakout = jax.jit(rand_breakout)

state, obs, done, action, rew = rand_breakout(jrng.key(5678))

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

iio.imwrite("out/breakout_random.gif", frames, duration=pause)
