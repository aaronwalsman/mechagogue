from typing import Sequence

import jax
import jax.random as jrng
import jax.numpy as jnp

from mechagogue.static_dataclass import static_dataclass
from mechagogue.tree import ravel_tree, shuffle_tree, batch_tree
from mechagogue.arg_wrappers import (
    ignore_unused_args,
)
from mechagogue.wrappers import (
    episode_return_wrapper,
    auto_reset_wrapper,
    parallel_env_wrapper,
)

@static_dataclass
class VPGConfig:
    #num_steps: int = 81920 # <- how many steps
    parallel_envs: int = 32
    rollout_steps: int = 256
    training_epochs: int = 4
    
    discount: float = 0.95
    
    batch_size: int = 256
    
    epsilon: float = 1e-5

def vpg(
    config,
    reset_env,
    step_env,
    init_model,
    model,
    init_optim,
    optim,
):
    
    # wrap the environment
    reset_env, step_env = auto_reset_wrapper(reset_env, step_env)
    reset_env, step_env = episode_return_wrapper(reset_env, step_env)
    reset_env, step_env = parallel_env_wrapper(
        reset_env, step_env, config.parallel_envs)
    
    init_model = ignore_unused_args(init_model,
        ('key',))
    model = ignore_unused_args(model,
        ('key', 'x', 'state'))
    init_optim = ignore_unused_args(init_optim,
        ('key', 'model_state'))
    optim = ignore_unused_args(optim,
        ('key', 'grad', 'model_state', 'optim_state'))
    
    def init(key):
        # generate keys
        reset_key, model_key, optim_key = jrng.split(key, 3)
        
        # reset the environment
        #reset_keys = jrng.split(reset_key, config.parallel_envs)
        state, obs = reset_env(reset_key)
        done = jnp.zeros(config.parallel_envs, dtype=jnp.bool)
        
        # generate new model state
        model_state = init_model(model_key)
        
        # generate new optimizer state
        optim_state = init_optim(optim_key, model_state)
        
        return state, obs, done, model_state, optim_state

    def step(key, state, obs, done, model_state, optim_state):
        # rollout trajectories
        def rollout(state_obs_done, key):
            
            # unpack
            state, obs, done = state_obs_done
            episode_returns = state[0]
            
            model_key, action_key, step_key = jrng.split(key, 3)
            
            # sample an action
            action_sampler, _ = model(model_key, obs, model_state)
            action = action_sampler(action_key)
            
            # take an environment step
            #step_keys = jrng.split(step_key, config.parallel_envs)
            next_state, next_obs, reward, next_done = step_env(
                step_key, state, action)
            
            return (
                (next_state, next_obs, next_done),
                (obs, action, reward, done),
            )
        
        # scan rollout_step to accumulate trajectories
        key, rollout_key = jrng.split(key)
        (state, obs, done), trajectories = jax.lax.scan(
            rollout,
            (state, obs, done),
            jrng.split(rollout_key, config.rollout_steps),
        )
        
        # unpack trajectories
        traj_obs, traj_action, traj_reward, traj_done = trajectories
        
        # compute returns
        def compute_returns(running_returns, reward_done):
            reward, done = reward_done
            returns = running_returns + reward
            running_returns = returns * (1. - done) * config.discount
            return running_returns, returns
        
        _, traj_returns = jax.lax.scan(
            compute_returns,
            jnp.zeros(config.parallel_envs),
            (traj_reward, traj_done),
            reverse=True,
        )
        
        # normalize returns
        traj_returns = traj_returns - jnp.mean(traj_returns)
        traj_returns = traj_returns / (jnp.std(traj_returns) + config.epsilon)
        
        # train on the trajectory data
        def train_epoch(model_optim_state, key):
            
            model_state, optim_state = model_optim_state
            
            # shuffle and batch the data
            # generate a shuffle permutation
            key, shuffle_key = jrng.split(key)
            o, a, r, d = ravel_tree(
                (traj_obs, traj_action, traj_returns, traj_done), 0, 2)
            o, a, r, d = shuffle_tree(shuffle_key, (o,a,r,d))
            o, a, r, d = batch_tree((o,a,r,d), config.batch_size)
            
            '''
            num_transitions = config.parallel_envs * config.rollout_steps
            shuffle_permutation = jrng.permutation(
                shuffle_key, num_transitions)
            
            # apply the shuffle and batch the data
            num_batches = num_transitions // config.batch_size
            def shuffle_and_batch(x):
                # first reshape to (num_transitions, ...)
                s = x.shape[2:]
                x = x.reshape(num_transitions, *s)
                # second apply the permutation
                x = x[shuffle_permutation]
                # third reshape into batches
                x = x.reshape(num_batches, config.batch_size, *s)
                return x
            key, batch_key = jrng.split(key)
            batch_keys = jrng.split(batch_key, num_batches)
            obs_batches = jax.tree.map(shuffle_and_batch, traj_obs)
            action_batches = jax.tree.map(shuffle_and_batch, traj_action)
            return_batches = shuffle_and_batch(traj_returns)
            done_batches = shuffle_and_batch(traj_done)
            batches = (
                batch_keys,
                obs_batches,
                action_batches,
                return_batches,
                done_batches,
            )
            '''
            
            # train the model on all batches
            def train_batch(model_optim_state, key_obs_action_returns_done):
                
                # unpack
                model_state, optim_state = model_optim_state
                key, obs, action, returns, done = key_obs_action_returns_done
                
                # compute the loss
                def vpg_loss(
                    model_key, model_state, obs, action, returns, done
                ):
                    _, action_logp = model(model_key, obs, model_state)
                    logp = action_logp(action)
                    loss = jnp.mean(-logp * returns * ~done)
                    jax.debug.print('lp {lp} r {r} d {d} a {a}', lp=logp, r=returns, d=done, a=action)
                    return loss
                
                model_key, optim_key = jrng.split(key)
                loss, grad = jax.value_and_grad(vpg_loss, argnums=1)(
                    model_key, model_state, obs, action, returns, done)
                
                jax.debug.print('mp {mp}', mp=model_state)
                
                # apply the gradients
                model_state, optim_state = optim(
                    optim_key, grad, model_state, optim_state)
                
                return (model_state, optim_state), loss
            
            # scan to train on all shuffled batches
            num_batches, _ = r.shape
            key, batch_key = jrng.split(key)
            k = jrng.split(batch_key, num_batches)
            (model_state, optim_state), batch_losses = jax.lax.scan(
                train_batch,
                (model_state, optim_state),
                (k, o, a, r, d),
            )
            
            return (model_state, optim_state), batch_losses
        
        # scan to train multiple epochs
        key, epoch_key = jrng.split(key)
        epoch_keys = jrng.split(epoch_key, config.training_epochs)
        (model_state, optim_state), epoch_losses = jax.lax.scan(
            train_epoch,
            (model_state, optim_state),
            epoch_keys,
            config.training_epochs,
        )
        
        losses = epoch_losses.reshape(-1)
        
        return (state, obs, done, model_state, optim_state), losses
    
    return init, step
