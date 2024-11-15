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

#from mechagogue.wrappers import step_auto_reset
from mechagogue.joint import JointDistribution

@dataclass
class VPGParams:
    total_steps : int = 50000
    parallel_envs : int = 32
    rollout_steps : int = 256
    
    learning_rate : float = 3e-4
    max_grad_norm : float = 0.5
    adam_eps : float = 1e-5
    
    @property
    def num_epochs(self):
        return self.total_steps // (self.parallel_envs * self.rollout_steps)

def vpg_loss(policy, params, obs, action, returns):
    action_distribution = policy.apply(train_state.params, obs)
    log_prob = action_distribution.log_prob(action)
    
    breakpoint()
    
    return -log_prob * returns

vpg_loss_and_grad = jax.value_and_grad(vpg_loss)

def train(
    key,
    train_params,
    env_params,
    reset_env,
    step_env,
    policy,
    weights
):
    
    # reset to get the initial observation and env_state
    key, reset_key = jrng.split(key)
    reset_keys = jrng.split(reset_key, train_params.parallel_envs)
    obs, state = jax.vmap(reset_env, in_axes=(0,None))(reset_keys, env_params)
    
    def rollout_train_epoch(epoch_state, _):
        
        key, obs, state, train_state = epoch_state
        
        # rollout trajectories
        def rollout_step(rollout_state, _):
            
            # unpack
            key, obs, state = rollout_state
            
            # sample an action
            key, action_key = jrng.split(key)
            action_keys = jrng.split(action_key, train_params.parallel_envs)

            action_distribution = policy.apply(train_state.params, obs)
            action = action_distribution.sample(seed=action_key)
            
            # take an environment step
            key, step_key = jrng.split(key)
            step_keys = jrng.split(step_key, train_params.parallel_envs)
            # import ipdb; ipdb.set_trace
            obs, state, reward, done = jax.vmap(step_env, in_axes=(0,None,0,0))(
                step_keys, env_params, state, action)
            
            # pack
            rollout_state = (key, obs, state)
            transition = (obs, action, reward, done)
            
            return rollout_state, transition
        
        # scan to accumulate trajectories
        (key, obs, state), trajectories = jax.lax.scan(
            rollout_step, (key, obs, state), None, train_params.rollout_steps)
        
        # TODO: compute the normalized returns
        
        # TODO: train policy; two scans over train_epoch and train_batch
        # TODO: loss func?
        # train the policy
        def train_epoch():
            
            # shuffle the data
            key, shuffle_key = jrng.split(key)
            total_transitions = (
                train_params.parallel_envs * train_params.rollout_steps)
            shuffle_permutation = jrng.permutation(
                shuffle_key, total_transitions)
            
            
            def train_batch(train_state, batch):
                obs, action, returns = batch
                loss, grads = vpg_loss_and_grad(
                    policy, train_state.params, obs, action, returns)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, loss
    
    # build the optimizer operator chain
    chain = optax.chain(
        optax.clip_by_global_norm(train_params.max_grad_norm),
        optax.adam(
            learning_rate=train_params.learning_rate,
            eps=train_params.adam_eps
        ),
    )
    
    # initialize the train state
    train_state = TrainState.create(
        apply_fn=policy.apply,
        params=weights,
        tx=chain,
    )
    
    # scan to train each epoch
    epoch_state = (key, obs, state, train_state)
    (key, obs, state, train_state), _ = jax.lax.scan(
        rollout_train_epoch, epoch_state, None, train_params.num_epochs)

if __name__ == '__main__':
    from dirt.examples.nom import reset, step, NomParams, NomObservation
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
            h,w = view.shape[-2:]
            
            view = nn.Embed(
                2,
                8,
                embedding_init=orthogonal(jnp.sqrt(2))
            )(view)
            view = jnp.reshape(view, (h*w*8,))
            
            view = nn.Dense(
                64,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.)
            )(view)
            
            stomach = nn.Dense(
                64,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.)
            )(observation.stomach)
            
            x = view + stomach
            
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
            
            forward = nn.Dense(
                2,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.)
            )(x)
            forward_distribution = distrax.Categorical(logits=forward)
            
            rotate = nn.Dense(
                4,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.)
            )(x)
            rotate_distribution = distrax.Categorical(logits=rotate)
            
            action_distribution = JointDistribution(
                (forward_distribution, rotate_distribution))
            
            return action_distribution
    
    policy = NomPolicy()
    key, weight_key = jrng.split(key)
    obs = NomObservation.zero(env_params)
    weights = policy.init(weight_key, obs)
    import ipdb; ipdb.set_trace()
    train(key, train_params, env_params, reset, step, policy, weights)
