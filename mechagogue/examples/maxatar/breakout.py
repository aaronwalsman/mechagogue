import jax
import jax.numpy as jnp
import jax.random as jrng
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import imageio.v3 as iio
from pathlib import Path

from mechagogue.optim.sgd import sgd
from mechagogue.rl.dqn import DQNConfig, dqn
import mechagogue.envs.maxatar.breakout as breakout
from mechagogue.wrappers import auto_reset_wrapper
from mechagogue.nn.mlp import mlp
from mechagogue.nn.sequence import layer_sequence
from mechagogue.tree import tree_getitem


# DQN config
BATCH_SIZE = 32
PARALLEL_ENVS = 8
REPLAY_BUFFER_SIZE = 100000  # 32*10000
REPLAY_START_SIZE = 5000
DISCOUNT = 0.99
START_EPSILON = 1.0
END_EPSILON = 0.1
FIRST_N_FRAMES = 100000
TARGET_UPDATE = 0.1
TARGET_UPDATE_FREQUENCY = 1000
BATCHES_PER_STEP = 1

LEARNING_RATE = 3e-3 # 1e-2
MOMENTUM = 0.9  # dampens oscillations, parly mimics RMSProp's running-average effect. 'poor-man's RMSProp'
NUM_EPOCHS = 500000


def generate_trajectories(
    key,
    model_state,
    model,
    reset_env,
    step_env,
    num_frames: int = 500,
):
    """
    Roll out a trajectory using the given model in the given environment, for the given number of steps.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Used for seeding the random number generators.
    model_state : PyTree
        The parameters of the model to use.
    model : Callable
        The function representing the model. It should take a key, observation, and model_state as arguments.
    reset_env : Callable
        The function used to reset the environment. It should take a key as an argument.
    step_env : Callable
        The function used to take a step in the environment. It should take a key, state, and action as arguments.
    num_frames : int, default=500
        The maximum number of steps to take in the environment.

    Returns
    -------
    trajectories : tuple of ndarrays
        Contains the state, observation, done, action, and reward for each step of the rollout.
    """
    key, init_key = jrng.split(key)
    state, obs, done = reset_env(init_key)
    
    def scan_step(key_state_obs_done, _):
        key, state, obs, done = key_state_obs_done
        key, model_key, step_key = jrng.split(key, 3)
        
        q_values = model(model_key, obs, model_state)  # (1, 6)
        q_values = q_values.min(axis=0)                # (6,) – collapse ensemble
        action = jnp.argmax(q_values, axis=-1)         # scalar
        
        next_state, next_obs, next_done, rew = step_env(
            step_key, state, action)
        
        return (key, next_state, next_obs, next_done), (
            state, obs, done, action, rew)
    
    key_state_obs, trajectories = jax.lax.scan(
        scan_step, (key, state, obs, done), None, length=num_frames)
    
    return trajectories


def breakout_dqn(key, num_epochs: int = 500000, num_frames: int = 100):
    """
    Train a DQN to play Breakout, then visualize the trained agent.

    Parameters
    ----------
    key : jax.random.PRNGKey
    num_epochs : int, default=500000
        Number of epochs to train the DQN for.
    num_frames : int, default=100
        Number of frames to generate for the visualization.

    Returns
    -------
    trajectories : tuple of ndarrays
        Contains the state, observation, done, action, and reward for each step
        of the evaluation rollouts.
    """
    breakout_reset, breakout_step = auto_reset_wrapper(breakout.reset, breakout.step)
    
    dqn_config = DQNConfig(
        batch_size=BATCH_SIZE,
        parallel_envs=PARALLEL_ENVS,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        replay_start_size=REPLAY_START_SIZE,
        discount=DISCOUNT,
        start_epsilon=START_EPSILON,
        end_epsilon=END_EPSILON,
        first_n_frames=FIRST_N_FRAMES,
        target_update=TARGET_UPDATE,
        target_update_frequency=TARGET_UPDATE_FREQUENCY,
        batches_per_step=BATCHES_PER_STEP,
    )
    
    # build Q‑network
    in_channels = 400  # 10×10×4 flattened (size of grid x TOTAL_CHANNELS)
    hidden_layers = 1
    hidden_channels = 256
    out_channels = 6   # action space size
    
    init_model_params, model = layer_sequence(
        (
            (lambda: None, lambda x: x.reshape(-1, in_channels)),  # flatten
            mlp(
                hidden_layers=hidden_layers,
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                p_dropout=0.1,
            ),
        )
    )
    
    init_optimizer_params, optimize = sgd(learning_rate=LEARNING_RATE, momentum=MOMENTUM)
    
    def random_action(key):
        return jrng.randint(key, shape=(), minval=0, maxval=6)
    
    init_dqn, step_dqn = dqn(
        dqn_config,
        breakout_reset,
        breakout_step,
        init_model_params,
        model,
        init_optimizer_params,
        optimize,
        random_action,
    )

    init_key, step_key = jrng.split(key)
    dqn_state = init_dqn(init_key)

    def train_epoch(dqn_state, key):
        dqn_state, loss = step_dqn(key, dqn_state)
        return dqn_state, loss

    dqn_state, losses = jax.lax.scan(
        train_epoch,
        dqn_state,
        jrng.split(step_key, num_epochs),
    )

    print("Training finished. Final 10 losses:", losses[-10:])
    
    # Extract a single ensemble member for action selection
    single_model_state = tree_getitem(dqn_state.model_state, 0)
    key, traj_key = jrng.split(key)
    
    # Generate evaluation rollouts (NO exploration)
    trajectories = generate_trajectories(
        key=traj_key,
        model_state=single_model_state,
        model=model,
        reset_env=breakout_reset,
        step_env=breakout_step,
        num_frames=num_frames,
    )

    return trajectories


def show_episode(
    traj,
    pause: float = 0.1,
    block: bool = False,
    save_path: str | Path | None = None,
):
    """
    Visualize episodes of Breakout observations.

    Args
    ----
    traj         : array of observations
    pause        : seconds between frames (≈ fps⁻¹)
    block        : whether plt.show() should block at the end
    save_path    : optional path. Writes an animated GIF, e.g. "ep0.gif"
    """

    cmap = sns.color_palette("cubehelix", breakout.TOTAL_CHANNELS)
    cmap.insert(0, (0, 0, 0))
    cmap = colors.ListedColormap(cmap)
    bounds = [i for i in range(breakout.TOTAL_CHANNELS+2)]
    norm = colors.BoundaryNorm(bounds, breakout.TOTAL_CHANNELS + 1)

    frames = []

    fig, ax = plt.subplots(1, 1)
    plt.show(block=False)

    for i, obs_i in enumerate(traj):
        assert obs_i.shape == (10,10,4)
        numerical_state = jnp.amax(
            obs_i * jnp.reshape(jnp.arange(breakout.TOTAL_CHANNELS) + 1, (1, 1, -1)), axis=2) + 0.5
        ax.imshow(
            numerical_state,
            cmap=cmap,
            norm=norm,
            interpolation="none",
            animated=True
        )
        ax.set_title(f"Frame {i}")
        ax.axis("off")
        plt.pause(pause)

        if save_path:
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba()).copy()
            frames.append(frame)

        ax.cla()

    if save_path:
        iio.imwrite(save_path, frames, duration=pause)
        print(f"Wrote {save_path}")

    plt.show(block=block)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)  # higher precision
    jnp.set_printoptions(precision=15, linewidth=120)

    # train Q-network and generate trajectories using learned greedy policy
    trajs = breakout_dqn(jrng.key(1234), num_epochs=NUM_EPOCHS, num_frames=500)
    traj_shapes = jax.tree_util.tree_map(lambda x: x.shape, trajs)
    print("Generated trajectories with shape:", traj_shapes)
    
    state, obs, done, action, rew = trajs

    # visualize trajectories
    show_episode(obs, pause=0.5, save_path="out/traj_1M_epochs_old_config.gif")
