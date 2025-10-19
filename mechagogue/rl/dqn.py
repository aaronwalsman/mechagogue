'''
Deep Q-Network (DQN) with experience replay and target networks.

Implements the DQN algorithm for discrete action spaces with epsilon-greedy
exploration, replay buffer, and periodic target network updates.
'''

from typing import Any

import jax
import jax.random as jrng
import jax.numpy as jnp

from mechagogue.static import static_data
from mechagogue.standardize import standardize_args
from mechagogue.standardize import split_random_keys
from mechagogue.wrappers import (
    episode_return_wrapper,
    auto_reset_wrapper,
    parallel_env_wrapper,
)
from mechagogue.tree import tree_getitem, tree_setitem, ravel_tree

@static_data
class DQNConfig:
    # data collection
    batch_size: int = 32
    parallel_envs: int = 32
    replay_buffer_size: int = 32*5000
    replay_start_size: int = 5000
    rollout_steps: int = 1

    # initial_random_data: int = 1000*32,
    
    # learning hyperparameters
    discount: float = 0.95
    step_size: float = 0.00025        # alpha / lr parameter for RMSProp
    rms_alpha: float = 0.95           # smoothing constant (minatar: SQUARED_GRAD_MOMENTUM)
    rms_eps: float = 0.01             # term added to denominator (minatar: MIN_SQUARED_GRAD)
    rms_centered: bool = True         # use centered RMSProp

    # exploration
    start_epsilon: float = 1.0
    end_epsilon: float = 0.1
    first_n_frames: int = 100000  # anneal length
    
    # target network
    target_update_frequency: int = 1000
    
    num_q_models: int = 2
    batches_per_step: int = 1  # frequency of gradient steps
    
@static_data
class DQNState:
    env_state : Any = None
    obs : Any = None
    done : Any = False
    model_state : Any = None
    target_state : Any = None
    optim_state : Any = None
    replay_buffer : Any = None
    current_step : int = 0
    policy_steps : int = 0  # count of gradient updates so far

def huber_loss(error: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    '''Compute element-wise Huber loss.'''
    abs_err = jnp.abs(error)
    is_small = abs_err <= delta
    # quadratic part: 0.5 * error^2    for |error| <= delta  
    # linear part: delta*(|error| - 0.5*delta) for |error| > delta
    return jnp.where(is_small,
                     0.5 * error**2,
                     delta * (abs_err - 0.5 * delta))

def dqn(
    config,
    env,
    model,
    optimizer,
    random_action,
):
    assert (
        config.replay_buffer_size %
        (config.parallel_envs * config.rollout_steps)
    ) == 0
    
    # auto-reset and parallelize the environment
    env = auto_reset_wrapper(env)
    reset_env, step_env = parallel_env_wrapper(env, config.parallel_envs)
    
    # parallelize the model over the number of q functions
    init_model = standardize_args(model.init, ('key',))
    init_model = jax.vmap(init_model)
    init_model = split_random_keys(
        init_model, config.num_q_models)
    single_model = standardize_args(model.forward, ('key', 'x', 'state'))
    model = jax.vmap(single_model, in_axes=(0, None, 0))
    model = split_random_keys(model, config.num_q_models)
    
    # wrap the optimizer
    init_optim = standardize_args(
        optimizer.init, ('key', 'model_state'))
    optim = standardize_args(
        optimizer.optimize, ('key', 'grad', 'model_state', 'optim_state'))
    
    # parallelize the action sampler
    random_action = standardize_args(random_action, ('key',))
    random_action = jax.vmap(random_action)
    random_action = split_random_keys(random_action, config.parallel_envs)
    
    def init(key):
        reset_key, model_key, optim_key, replay_key = jrng.split(key, 4)
        
        env_state, obs, done = reset_env(reset_key)
        
        # build the model, target and optimizer state
        model_state = init_model(model_key)
        target_state = model_state
        optim_state = init_optim(optim_key, model_state)
        
        # fill the replay buffer with data from the random policy initially
        def fill_replay_buffer(state_obs_done, key):
            env_state, obs, done = state_obs_done
            action_key, step_key = jrng.split(key)
            
            # sample a random action
            action = random_action(action_key)
            
            # take an environment step
            next_env_state, next_obs, next_done, reward = step_env(
                step_key, env_state, action)
            
            return (
                (next_env_state, next_obs, next_done),
                (obs, action, reward, next_obs, done, next_done),
            )
        
        fill_steps = (
            config.replay_buffer_size // config.parallel_envs)
        (env_state, obs, done), replay_buffer = jax.lax.scan(
            fill_replay_buffer,
            (env_state, obs, done),
            jrng.split(replay_key, fill_steps),
        )
        
        replay_buffer = ravel_tree(replay_buffer, 0, 2)
        
        return DQNState(
            env_state,
            obs,
            done,
            model_state,
            target_state,
            optim_state,
            replay_buffer,
            0,
        )
    
    def step(key, state):
        # number of environment frames we have seen so far
        frames_seen = (
            state.current_step * config.rollout_steps * config.parallel_envs
        )
        
        # ε-greedy schedule
        delta = frames_seen - config.replay_start_size
        frac  = jnp.clip(delta / config.first_n_frames, 0.0, 1.0)
        epsilon = jnp.where(
            frames_seen < config.replay_start_size,
            config.start_epsilon,
            (1.0 - frac) * config.start_epsilon + frac * config.end_epsilon,
        )
        
        single_model_state = tree_getitem(state.model_state, 0)
        
        # generate new data
        def rollout(state_obs_done, key):
            env_state, obs, done = state_obs_done
            model_key, random_key, selector_key, step_key = jrng.split(key, 4)
            
            # sample an action according to the epsilon-greedy policy
            q = single_model(model_key, obs, single_model_state)
            greedy = jnp.argmax(q, axis=-1)
            random = random_action(random_key)
            selector = jrng.bernoulli(
                selector_key, shape=greedy.shape, p=epsilon)
            action = jnp.where(selector, random, greedy)
            
            # take an environment step
            next_env_state, next_obs, next_done, reward = step_env(
                step_key, env_state, action)
            
            return (
                (next_env_state, next_obs, next_done),
                (obs, action, reward, next_obs, done, next_done),
            )
        
        key, rollout_key = jrng.split(key)
        (env_state, obs, done), trajectories = jax.lax.scan(
            rollout,
            (state.env_state, state.obs, state.done),
            jrng.split(rollout_key, config.rollout_steps),
        )
        
        # write to the replay buffer
        trajectories = ravel_tree(trajectories, 0, 2)
        replay_offset = config.rollout_steps * config.parallel_envs
        examples_per_step = config.rollout_steps * config.parallel_envs
        replay_start = (state.current_step * examples_per_step)
        replay_start = replay_start % config.replay_buffer_size
        indices = jnp.arange(examples_per_step) + replay_start
        replay_buffer = tree_setitem(state.replay_buffer, indices, trajectories)
        
        # train
        def train_batch(model_target_optim_state, key):
            model_state, target_state, optim_state = model_target_optim_state
            
            # sample data from the replay buffer
            data_key, target_key, model_key, optim_key = jrng.split(key, 4)
            replay_buffer_indices = jrng.choice(
                data_key,
                config.replay_buffer_size,
                shape=(config.batch_size,),
                replace=False,
            )
            obs, action, reward, next_obs, done, next_done = tree_getitem(
                replay_buffer, replay_buffer_indices)
            
            # compute the q targets
            next_q = model(target_key, next_obs, target_state)
            next_q = next_q.min(axis=0)
            next_v = next_q.max(axis=-1)
            target = reward + (~next_done) * config.discount * next_v
            
            # compute the loss and gradients
            def q_loss(model_key, model_state, obs, action, done, target):
                q = model(model_key, obs, model_state)
                q = q[:,jnp.arange(config.batch_size), action]
                TD_error = q - target
                # Huber loss per element, then mask out terminal transitions
                loss = huber_loss(TD_error) * (~done)
                return loss.mean()
            
            loss, grad = jax.value_and_grad(q_loss, argnums=1)(
                model_key, model_state, obs, action, done, target)
            
            # apply the gradients
            model_state, optim_state = optim(
                optim_key, grad, model_state, optim_state)
            
            return (model_state, target_state, optim_state), loss
        
        key, batch_key = jrng.split(key)
        model_target_optim_state, batch_losses = jax.lax.scan(
            train_batch,
            (state.model_state, state.target_state, state.optim_state),
            jrng.split(batch_key, config.batches_per_step),
        )
        model_state, target_state_old, optim_state = model_target_optim_state
        
        # Hard‐update target every TARGET_UPDATE_FREQUENCY gradient steps
        policy_steps_new = state.policy_steps + config.batches_per_step
        should_update = (policy_steps_new % config.target_update_frequency) == 0
        # if should_update, copy model → target, else keep old
        target_state = jax.lax.cond(
            should_update,
            lambda: model_state,
            lambda: target_state_old
        )
        
        return DQNState(
            env_state,
            obs,
            done,
            model_state,
            target_state,
            optim_state,
            replay_buffer,
            state.current_step + 1,
            policy_steps_new,
        ), batch_losses
    
    return init, step
