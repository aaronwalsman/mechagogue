from typing import Sequence

import jax
import jax.random as jrng
import jax.numpy as jnp

from flax.struct import dataclass
from flax.training.train_state import TrainState
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import optax
import distrax

from mechagogue.wrappers import (
    episode_return_wrapper,
    auto_reset_wrapper,
    parallel_env_wrapper,
)

@dataclass
class VPGParams:
    total_steps: int = 81920
    parallel_envs: int = 32
    rollout_steps: int = 256
    training_passes: int = 4
    
    discount: float = 0.95
    #learning_rate : float = 3e-4
    #max_grad_norm : float = 0.5
    #adam_eps : float = 1e-5
    
    @property
    def num_epochs(self):
        return self.total_steps // (self.parallel_envs * self.rollout_steps)

def vpg(
    key,
    params,
    reset,
    step,
    policy,
    weights,
    train,
    train_parameters,
):
    
    # vectorize and auto-reset
    # wait, this isn't right, if we're doing episode returns we need
    # something different for auto_resets because otherwise the final
    # and only important return will be thrown away
    reset, step = episode_return_wrapper(reset, step)
    reset, step = auto_reset_wrapper(reset, step)
    reset, step = parallel_env_wrapper(reset, step)
    
    # reset to get the initial environment state and observation
    key, reset_key = jrng.split(key)
    reset_keys = jrng.split(reset_key, params.parallel_envs)
    state, obs = reset(reset_keys)
    done = jnp.zeros(params.parallel_envs, dtype=jnp.bool)
    
    # this function will be scanned in order to train each epoch
    def epoch_step(epoch_state, _):
        
        # unpack
        key, state, obs, done, train_state = epoch_state
        
        # rollout trajectories
        def rollout(rollout_state, _):
            
            # unpack
            key, state, obs, done = rollout_state
            
            # sample an action
            key, action_key = jrng.split(key)
            action_keys = jrng.split(action_key, train_params.parallel_envs)
            action_sampler, _ = policy(weights, obs)
            action = action_sampler(action_keys)
            
            # take an environment step
            key, step_key = jrng.split(key)
            step_keys = jrng.split(step_key, params.parallel_envs)
            next_state, next_obs, reward, next_done = step(
                step_keys, state, action)
            
            #action_keys = jrng.split(action_key, train_params.parallel_envs)

            #action_distribution = policy.apply(train_state.params, obs)
            #action = action_distribution.sample(seed=action_key)
            #
            ## take an environment step
            #key, step_key = jrng.split(key)
            #step_keys = jrng.split(step_key, train_params.parallel_envs)
            ## import ipdb; ipdb.set_trace
            #obs, state, reward, done = jax.vmap(step_env, in_axes=(0,None,0,0))(
            #    step_keys, env_params, state, action)
            
            # pack
            rollout_state = (key, next_state, next_obs, next_done)
            transition = (obs, action, reward, done)
            
            return rollout_state, transition
        
        # scan rollout_step to accumulate trajectories
        (key, state, obs, done), trajectories = jax.lax.scan(
            rollout,
            (key, state, obs, done),
            None,
            params.rollout_steps,
        )
        
        # compute returns
        def compute_returns(running_returns, reward_done):
            reward, done = reward_done
            returns = running_returns + reward
            running_returns = returns * (1. - done) * params.discount
            return running_returns, returns
        
        # scan to accumulate returns
        _, _, traj_reward, traj_done = trajectories
        _, returns = jax.lax.scan(
            compute_returns,
            jnp.zeros(params.parallel_envs),
            (traj_reward, traj_done),
            reverse=True,
        )
        
        # TODO: train policy; two scans over train_epoch and train_batch
        # TODO: loss func?
        # train the policy
        def train_epoch(train_state, _):
            
            # shuffle the data
            key, shuffle_key = jrng.split(key)
            total_transitions = (
                params.parallel_envs * params.rollout_steps)
            shuffle_permutation = jrng.permutation(
                shuffle_key, total_transitions)
            
            
            def train_batch(train_state, batch):
                
                # unpack
                weights, = train_state
                obs, action, returns = batch
                
                def vpg_loss(weights, obs, action, returns):
                    _, action_logp = policy(weights, obs)
                    logp = action_logp(action)
                    policy_loss = jnp.mean(-logp * returns)
                    return policy_loss
                
                loss, grad = jax.value_and_grad(vpg_loss, obs, action, returns)
                
                # apply the gradients
                weights = train(weights, grad)
                
                return (weights,), loss
            
            jax.lax.scan(
                train_batch,
                (weights,),
                batch_data,
                STEPS,
            )
        
        jax.lax.scan(
            train_epoch,
            params.training_passes
        )
        
        # pack
        epoch_state = key, state, obs, done, train_state
        
        return epoch_state
    
    train_state = None
    
    # scan to train each epoch
    epoch_state = (key, state, obs, done, train_state)
    (key, state, obs, done, train_state), _ = jax.lax.scan(
        epoch_step, epoch_state, None, params.num_epochs)
    
    breakpoint()


def vpg_loss(policy, params, obs, action, returns):
    action_distribution = policy.apply(train_state.params, obs)
    log_prob = action_distribution.log_prob(action)
    
    return jnp.mean(-log_prob * returns)
            

if __name__ == '__main__':
    from dirt.examples.nom import (
        reset, step, NomParams, NomObservation, NomAction
    )
    key = jrng.key(1234)
    train_params = VPGParams()
    env_params = NomParams()
    
    class NomPolicy(nn.Module):
        activation: str = "tanh"

        @nn.compact
        def __call__(self, observation):
            if self.activation == 'relu':
                activation = nn.relu
            elif self.activation == 'tanh':
                activation = nn.tanh
            else:
                raise NotImplementedError
            
            view = observation.view
            *b,h,w = view.shape
            
            view = nn.Embed(
                2,
                8,
                embedding_init=orthogonal(jnp.sqrt(2))
            )(view)
            view = jnp.reshape(view, (*b,h*w*8,))
            
            view = nn.Dense(
                64,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.)
            )(view)
            
            health = nn.Dense(
                64,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.)
            )(observation.health)
            
            x = view + health
            
            x = nn.Dense(
                64,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.)
            )(x)
            x = activation(x)
            x = nn.Dense(
                64,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.)
            )(x)
            x = activation(x)
            
            forward_logits = nn.Dense(
                2,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.)
            )(x)
            forward_distribution = distrax.Categorical(logits=forward_logits)
            
            rotate_logits = nn.Dense(
                4,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.)
            )(x)
            rotate_distribution = distrax.Categorical(logits=rotate_logits)
            
            action_distribution = distrax.Joint(
                NomAction(forward_distribution, rotate_distribution))
            
            return action_distribution
    
    policy = NomPolicy()
    key, weight_key = jrng.split(key)
    obs = NomObservation.zero(env_params)
    weights = policy.init(weight_key, obs)
    import ipdb; ipdb.set_trace()
    train(key, train_params, env_params, reset, step, policy, weights)
