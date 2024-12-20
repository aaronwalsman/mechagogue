from typing import Sequence

import jax
import jax.random as jrng
import jax.numpy as jnp

from flax.struct import dataclass

from mechagogue.wrappers import (
    episode_return_wrapper,
    auto_reset_wrapper,
    parallel_env_wrapper,
)

@dataclass
class VPGParams:
    #num_steps: int = 81920 # <- how many steps
    parallel_envs: int = 32
    rollout_steps: int = 256
    training_epochs: int = 4
    
    discount: float = 0.95
    
    batch_size: int = 256
    
    eps: float = 1e-5

def vpg(
    key,
    params,
    reset_env,
    step_env,
    policy,
    initialize_weights,
    train_weights,
):
    
    # wrap the environment and policy
    reset_env, step_env = auto_reset_wrapper(reset_env, step_env)
    reset_env, step_env = episode_return_wrapper(reset_env, step_env)
    reset_env, step_env = parallel_env_wrapper(reset_env, step_env)
    
    def reset_vpg(
        key,
    ):
        # generate keys
        reset_key, weight_key = jrng.split(key)
        
        # reset the environment
        reset_keys = jrng.split(reset_key, params.parallel_envs)
        state, obs = reset_env(reset_keys)
        done = jnp.zeros(params.parallel_envs, dtype=jnp.bool)
        
        # generate new weights
        weights = initialize_weights(key)
        
        return state, obs, done, weights

    def step_vpg(
        key,
        state,
        obs,
        done,
        weights,
    ):
        # rollout trajectories
        def rollout(rollout_state, key):
            
            # unpack
            state, obs, done = rollout_state
            episode_returns = state[0]
            
            # sample an action
            key, action_key = jrng.split(key)
            action_sampler, _ = policy(weights, obs)
            action = action_sampler(action_key)
            
            # take an environment step
            key, step_key = jrng.split(key)
            step_keys = jrng.split(step_key, params.parallel_envs)
            next_state, next_obs, reward, next_done = step_env(
                step_keys, state, action)
            
            # pack
            rollout_state = (next_state, next_obs, next_done)
            transition = (obs, action, reward, done)
            
            return rollout_state, transition
        
        # scan rollout_step to accumulate trajectories
        key, rollout_key = jrng.split(key)
        (state, obs, done), trajectories = jax.lax.scan(
            rollout,
            (state, obs, done),
            jrng.split(rollout_key, params.rollout_steps),
            params.rollout_steps,
        )
        
        # DAPHNE TODO:
        # Can you help verify that these rollouts are correct?  The simple
        # target environment that we are testing with now should produce data
        # that is very easy to interpret.
        # Note that our reset wrapper includes the final "terminal" state in
        # the trajectory (rather than skipping over it like the default gym
        # wrapper), below is me trying to make sure my book-keeping
        # is all correct.  Each set of brackets [...] is one transition,
        # o is observation, a is actino, r is reward and the last number is
        # done.  In the example below there are five transitions, where the
        # agent first makes three decisions, the last one causes termination,
        # then a new trajectory starts.  Note that we have one transition
        # [oT aT 0 (1)] which represents the transition from the last terminal
        # state to the next initial state which the agent has no control over.
        # this means we will turn off the loss for these transitions later
        # (I already have a stab at implementing this later in the loss fn,
        # but not sure if it's
        # right:
        # [o1 a1 r1 (0)] [o2 a2 r2 (0)] [o3 a3 r3 (0)] [oT aT 0 (1)] [o1 a1...]
        #  g1 = r1        g2 = r1+r2     g3 = r1+r2+r3  gT = 0
        #  loss = 1       loss = 1       loss = 1       loss = 0
        
        # unpack trajectories
        traj_obs, traj_action, traj_reward, traj_done = trajectories
        
        # compute returns
        def compute_returns(running_returns, reward_done):
            reward, done = reward_done
            returns = running_returns + reward
            running_returns = returns * (1. - done) * params.discount
            return running_returns, returns
        
        # scan to accumulate returns
        _, traj_returns = jax.lax.scan(
            compute_returns,
            jnp.zeros(params.parallel_envs),
            (traj_reward, traj_done),
            reverse=True,
        )
        
        # normalize returns
        traj_returns = traj_returns - jnp.mean(traj_returns)
        traj_returns = traj_returns / (jnp.std(traj_returns) + params.eps)
        
        # train on the trajectory data
        def train_epoch(weights, key):
            
            # shuffle and batch the data
            # generate a shuffle permutation
            key, shuffle_key = jrng.split(key)
            num_transitions = params.parallel_envs * params.rollout_steps
            shuffle_permutation = jrng.permutation(
                shuffle_key, num_transitions)
            
            # apply the shuffle and batch the data
            num_batches = num_transitions // params.batch_size
            def shuffle_and_batch(x):
                # first reshape to (num_transitions, ...)
                s = x.shape[2:]
                x = x.reshape(num_transitions, *s)
                # second apply the permutation
                x = x[shuffle_permutation]
                # third reshape into batches
                x = x.reshape(num_batches, params.batch_size, *s)
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
            
            # train the policy on all batches
            def train_batch(weights, batch):
                
                # unpack
                key, obs, action, returns, done = batch
                
                # compute the loss
                def vpg_loss(weights, obs, action, returns, done):
                    _, action_logp = policy(weights, obs)
                    logp = action_logp(action)
                    policy_loss = jnp.mean(-logp * returns * ~done)
                    return policy_loss
                
                # DAPHNE TODO: Help, help!  I need an adult!
                # I can't figure out what's breaking the line below here.
                # JAX is mad because it can't figure out the shape of something
                # for some reason.
                
                loss, grad = jax.value_and_grad(
                    vpg_loss, obs, action, returns, done)
                
                # apply the gradients
                weights = train_weights(key, weights, grad)
                
                return weights, loss
            
            # scan to train on all shuffled batches
            weights, batch_losses = jax.lax.scan(
                train_batch,
                weights,
                batches,
                num_batches,
            )
            
            return weights, batch_losses
        
        # scan to train multiple epochs
        key, epoch_key = jrng.split(key)
        epoch_keys = jrng.split(epoch_key, params.training_epochs)
        weights, epoch_losses = jax.lax.scan(
            train_epoch,
            weights,
            epoch_keys,
            params.training_epochs,
        )
        
        losses = epoch_losses.reshape(-1)
        
        return state, obs, done, weights, losses
    
    return reset_vpg, step_vpg

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
