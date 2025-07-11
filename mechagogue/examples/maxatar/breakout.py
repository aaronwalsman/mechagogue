"""
    Train a DQN agent on the MaxAtar Breakout environment and (optionally) visualize its
    behavior.
    
    python breakout.py
        -e, --epochs <number of training epochs>
        -n, --episodes <number of greedy evaluation episodes after training>
        --delay <delay between frames in seconds>
        --gif <filename for GIF output>

    Example:
        python breakout.py -e 1000000 --gif breakout_1M_epochs.gif
        
        python breakout.py -e 1000000 --gif breakout_1M_epochs_cnn_tuned.gif
        
        python breakout.py -e 5000000 --gif breakout_5M_epochs_cnn.gif
        
        Trains for 1M epochs then generates 10 episodes of gameplay and saves a GIF to 
        breakout_1M_epochs.gif with 0.5s between frames
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import jax.random as jrng
import numpy as np
import seaborn as sns

from mechagogue.optim.sgd import sgd
from mechagogue.optim.rmsprop import rmsprop
from mechagogue.rl.dqn import DQNConfig, dqn
import mechagogue.envs.maxatar.breakout as breakout
from mechagogue.wrappers import auto_reset_wrapper, sticky_action_wrapper
from mechagogue.nn.q_networks import q_network_mlp, q_network_cnn
from mechagogue.tree import tree_getitem


# DQN hyper‑parameters
NUM_EPOCHS_DEFAULT = 500_000

# tuned config 2 (following MinAtar)
BATCH_SIZE = 32
PARALLEL_ENVS = 1
REPLAY_BUFFER_SIZE = 100_000
REPLAY_START_SIZE = 5_000
DISCOUNT = 0.99
START_EPSILON = 1.0
END_EPSILON = 0.1
FIRST_N_FRAMES = 100_000
TARGET_UPDATE_FREQUENCY = 1_000
BATCHES_PER_STEP = 1

STEP_SIZE = 0.00025
RMS_ALPHA = 0.95
RMS_EPS = 0.01
RMS_CENTERED = True

STICKY_P = 0.10        # ← set 0.0 to disable
RAMPING = True

NUM_Q_MODELS = 1

# tuned config 1 (following MinAtar)
# BATCH_SIZE = 32
# PARALLEL_ENVS = 8
# REPLAY_BUFFER_SIZE = 100_000
# REPLAY_START_SIZE = 5_000
# DISCOUNT = 0.99
# START_EPSILON = 1.0
# END_EPSILON = 0.1
# FIRST_N_FRAMES = 100_000
# TARGET_UPDATE = 0.1
# TARGET_UPDATE_FREQUENCY = 1_000
# BATCHES_PER_STEP = 1

# LEARNING_RATE = 3e-3
# MOMENTUM = 0.9

# STICKY_P = 0.0
# RAMPING = False

# NUM_Q_MODELS = 2

# old config
# BATCH_SIZE = 32
# PARALLEL_ENVS = 8
# REPLAY_BUFFER_SIZE = 32*10000
# REPLAY_START_SIZE = 5000
# DISCOUNT = 0.9
# START_EPSILON = 0.1
# END_EPSILON = 0.1
# FIRST_N_FRAMES = 100000
# TARGET_UPDATE = 0.1
# TARGET_UPDATE_FREQUENCY = 1000
# BATCHES_PER_STEP = 1

# LEARNING_RATE = 1e-2
# MOMENTUM = 0

# STICKY_P = 0.0
# RAMPING = False

# NUM_Q_MODELS = 2

# Visualization helpers
def build_palette(n_channels: int) -> np.ndarray:
    pal = sns.color_palette("cubehelix", n_channels)
    pal = [(0, 0, 0)] + [
        (int(r * 255), int(g * 255), int(b * 255)) for r, g, b in pal
    ]
    return np.asarray(pal, dtype=np.uint8)


def obs_to_rgb(obs: np.ndarray, palette: np.ndarray, scale: int = 40) -> np.ndarray:
    has_obj = obs.any(axis=2)
    idx = obs.argmax(axis=2) + 1
    idx[~has_obj] = 0  # background
    rgb_small = palette[idx]  # 10×10×3
    return np.repeat(np.repeat(rgb_small, scale, axis=0), scale, axis=1)


PALETTE = build_palette(breakout.TOTAL_CHANNELS)


def generate_trajectories(
    key: jax.random.PRNGKey,
    model_state,
    model,
    env,
    *,
    episodes: int,
    max_steps: int = 1_000,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Greedily rollout `episodes` full episodes with at most `max_steps` frames each using the `model` policy."""
    def run_episode(key, _):
        key, reset_key = jrng.split(key)
        state, obs, done = env.init(reset_key)
        
        def step_fn(carry, _):
            key, state, obs, done, G = carry
            current_obs = obs
            
            # Only take action if not done
            key, mk, sk = jrng.split(key, 3)
            q = model.forward(mk, obs, model_state).min(axis=0)
            action = jnp.argmax(q)
            
            # Always call env.step (for JAX tracing), but conditionally use results
            next_state, next_obs, next_done, reward = env.step(sk, state, action)
            
            # Update everything conditionally based on current done status
            state = jax.tree.map(lambda old, new: jnp.where(done, old, new), state, next_state)
            obs   = jax.tree.map(lambda old, new: jnp.where(done, old, new), obs, next_obs)
            G     = jnp.where(done, G, G + reward)  # Only add reward if not done
            done  = jnp.logical_or(done, next_done)
            
            return (key, state, obs, done, G), (current_obs, reward, done)
        
        # Run episode
        initial_carry = (key, state, obs, done, 0.0)
        final_carry, (obs_seq, rew_seq, done_seq) = jax.lax.scan(
            step_fn, initial_carry, None, length=max_steps
        )
        
        episode_return = final_carry[4]
        return key, (obs_seq, episode_return)
    
    # Run all episodes
    key, (all_obs, returns) = jax.lax.scan(run_episode, key, None, length=episodes)
    
    return all_obs, returns


def write_gif(obs_frames: jnp.ndarray, gif_path: str, delay: float) -> None:
    """Write observation frames to a GIF file."""
    gif_path = str(Path(gif_path).expanduser())
    print(f"Recording gameplay to '{gif_path}'...")
    
    # Flatten episodes and steps into single sequence
    if obs_frames.ndim > 3:  # If shape is (episodes, steps, ...)
        obs_frames = obs_frames.reshape(-1, *obs_frames.shape[2:])
    
    # Convert all frames to RGB
    frames = [
        obs_to_rgb(np.asarray(obs, dtype=np.uint8), PALETTE)
        for obs in obs_frames
    ]
    # Write all frames at once
    iio.imwrite(gif_path, frames, duration=delay)
    print(f"GIF saved: {gif_path}")


# Training + evaluation wrapper
def breakout_dqn(
    key: jax.random.PRNGKey,
    *,
    num_epochs: int,
    episodes: int,
    gif_path: str | None,
    delay: float,
):
    env = breakout.make_env(ramping=RAMPING)

    if STICKY_P > 0.0:
        env = sticky_action_wrapper(env, prob=STICKY_P)

    cfg = DQNConfig(
        batch_size=BATCH_SIZE,
        parallel_envs=PARALLEL_ENVS,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        replay_start_size=REPLAY_START_SIZE,
        discount=DISCOUNT,
        start_epsilon=START_EPSILON,
        end_epsilon=END_EPSILON,
        first_n_frames=FIRST_N_FRAMES,
        target_update_frequency=TARGET_UPDATE_FREQUENCY,
        batches_per_step=BATCHES_PER_STEP,
        step_size=STEP_SIZE,
        rms_alpha=RMS_ALPHA,
        rms_eps=RMS_EPS,
        rms_centered=RMS_CENTERED,
        num_q_models=NUM_Q_MODELS,
    )

    # Q‑network
    num_actions = 6
    
    # model = q_network_mlp(10*10*4, num_actions)
    model = q_network_cnn(breakout.TOTAL_CHANNELS, num_actions)

    # optimizer = sgd(learning_rate=LEARNING_RATE, momentum=MOMENTUM)
    optimizer = rmsprop(
        learning_rate=cfg.step_size,
        alpha=cfg.rms_alpha,
        eps=cfg.rms_eps,
        centered=cfg.rms_centered,
    )


    def random_action(key):
        return jrng.randint(key, shape=(), minval=0, maxval=6)

    init_dqn, step_dqn = dqn(
        cfg, env, model, optimizer, random_action
    )

    # Training
    init_key, *scan_keys = jrng.split(key, num_epochs + 1)
    state = init_dqn(init_key)

    def epoch_fn(st, k):
        st, loss = step_dqn(k, st)
        return st, loss

    state, losses = jax.lax.scan(epoch_fn, state, jnp.stack(scan_keys))
    print("Training finished. Final 10 losses:", np.asarray(losses[-10:]))

    # Evaluation
    model_state = tree_getitem(state.model_state, 0)
    key, eval_key = jrng.split(key)
    obs_frames, returns = generate_trajectories(
        eval_key,
        model_state,
        model,
        env,
        episodes=episodes,
    )

    if gif_path is not None:
        write_gif(obs_frames, gif_path, delay)

    return returns


# CLI entry point
def main() -> None:
    parser = argparse.ArgumentParser(description="Train & evaluate a DQN on MaxAtar Breakout (GIF optional).")
    parser.add_argument("-e", "--epochs", type=int, default=NUM_EPOCHS_DEFAULT, help="Training epochs")
    parser.add_argument("-n", "--episodes", type=int, default=10, help="Greedy evaluation episodes")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds per frame in GIF")
    parser.add_argument("--gif", type=str, default=None, help="Output GIF filename; omit to disable recording")
    args = parser.parse_args()

    # jax.config.update("jax_enable_x64", True)
    jnp.set_printoptions(precision=12, linewidth=120, suppress=True)

    # 1234
    # 2134
    master_key = jrng.key(2134)
    returns = breakout_dqn(
        master_key,
        num_epochs=args.epochs,
        episodes=args.episodes,
        gif_path=args.gif,
        delay=args.delay,
    )
    
    print("Returns:", returns)

    mean_ret = float(np.mean(returns))
    stderr_ret = float(np.std(returns) / np.sqrt(len(returns)))
    print("-" * 60)
    print(f"Average return over {args.episodes} episodes: {mean_ret:.2f} ± {stderr_ret:.2f}")


if __name__ == "__main__":
    main()
