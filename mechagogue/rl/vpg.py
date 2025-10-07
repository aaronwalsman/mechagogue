'''
Vanilla Policy Gradient (VPG) algorithm for policy optimization.

Implements on-policy reinforcement learning with Monte Carlo returns
and policy gradient estimation.
'''

from typing import Any, Sequence

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
    #num_steps: int = 81920
    parallel_envs: int = 32
    rollout_steps: int = 256
    training_epochs: int = 4
    
    discount: float = 0.95
    
    batch_size: int = 256
    
    epsilon: float = 1e-5

@static_dataclass
class VPGState:
    env_state : Any = None
    obs : Any = None
    done : Any = False
    model_state : Any = None
    optim_state : Any = None

def vpg(
    params,
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
        reset_env, step_env, params.parallel_envs)
    
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
        state, obs, done = reset_env(reset_key)
        
        # generate new model state
        model_state = init_model(model_key)
        
        # generate new optimizer state
        optim_state = init_optim(optim_key, model_state)
        
        return VPGState(state, obs, done, model_state, optim_state)

    def step(key, state):
        # rollout trajectories
        def rollout(state_obs_done, key):
            
            # unpack
            env_state, obs, done = state_obs_done
            episode_returns = env_state[0]
            
            model_key, action_key, step_key = jrng.split(key, 3)
            
            # sample an action
            action_sampler, _ = model(model_key, obs, state.model_state)
            action = action_sampler(action_key)
            
            # take an environment step
            next_env_state, next_obs, next_done, reward = step_env(
                step_key, env_state, action)
            
            return (
                (next_env_state, next_obs, next_done),
                (obs, action, done, reward),
            )
        
        # scan rollout_step to accumulate trajectories
        key, rollout_key = jrng.split(key)
        (env_state, obs, done), trajectories = jax.lax.scan(
            rollout,
            (state.env_state, state.obs, state.done),
            jrng.split(rollout_key, config.rollout_steps),
        )
        
        # unpack trajectories
        traj_obs, traj_action, traj_done, traj_reward = trajectories
        
        # compute returns
        def compute_returns(running_returns, reward_done):
            reward, done = reward_done
            returns = running_returns + reward
            running_returns = returns * (1. - done) * params.discount
            return running_returns, returns
        
        _, traj_returns = jax.lax.scan(
            compute_returns,
            jnp.zeros(params.parallel_envs),
            (traj_reward, traj_done),
            reverse=True,
        )
        
        # normalize returns
        traj_returns = traj_returns - jnp.mean(traj_returns)
        traj_returns = traj_returns / (jnp.std(traj_returns) + params.epsilon)
        
        # train on the trajectory data
        def train_epoch(model_optim_state, key):
            
            model_state, optim_state = model_optim_state
            
            # shuffle and batch the data
            # generate a shuffle permutation
            key, shuffle_key = jrng.split(key)
            o, a, r, d = ravel_tree(
                (traj_obs, traj_action, traj_returns, traj_done), 0, 2)
            o, a, r, d = shuffle_tree(shuffle_key, (o,a,r,d))
            o, a, r, d = batch_tree((o,a,r,d), params.batch_size)
            
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
                    return loss
                
                model_key, optim_key = jrng.split(key)
                loss, grad = jax.value_and_grad(vpg_loss, argnums=1)(
                    model_key, model_state, obs, action, returns, done)
                
                jax.debug.print('grad {g} returns {r}', g=grad, r=returns)
                
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
        epoch_keys = jrng.split(epoch_key, params.training_epochs)
        (model_state, optim_state), epoch_losses = jax.lax.scan(
            train_epoch,
            (state.model_state, state.optim_state),
            epoch_keys,
            params.training_epochs,
        )
        
        losses = epoch_losses.reshape(-1)
        
        return VPGState(env_state, obs, done, model_state, optim_state), losses
    
    return init, step
