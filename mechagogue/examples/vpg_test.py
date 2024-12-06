import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.dp.mdp import mdp
from mechagogue.rl.vpg import vpg, VPGParams

def initialize_target_env(key, target):
    return jnp.array([0,0], dtype=jnp.int32)

def transition_target_env(key, target, position, action):
    position = position + action
    return position

def reward_target_env(key, target, position):
    return -jnp.sum(jnp.abs(target - position))

def done_target_env(key, target, position):
    return jnp.all(target == position)

reset_env, step_env = mdp(
    jnp.zeros(2, dtype=jnp.int32),
    initialize_target_env,
    transition_target_env,
    reward_target_env,
    done_target_env,
    reward_format='__n',
    done_format='__n',
)

def uniform_policy(weights, obs):
    n = obs.shape[0]
    def sample(key):
        return jrng.randint(key, (n,2), -1, 2)
    
    def logp(action):
        return jnp.log(jnp.full(n, 3)) + jnp.log(jnp.full(n, 3))
    
    return sample, logp

# this shouldn't be necessary with a neural network policy that takes
# batched input
#uniform_policy = jax.vmap(uniform_policy, in_axes=(None,0))

def train(key, params):
    
    vpg_reset, vpg_step = vpg(
        key,
        params,
        reset_env,
        step_env,
        uniform_policy,
        lambda key : None,
        lambda key : None,
    )
    
    key, reset_key = jrng.split(key)
    vpg_state = vpg_reset(reset_key)
    
    def step(vpg_state, key):
        vpg_state = vpg_step(key, *vpg_state)
    
    num_epochs = 12
    step_keys = jrng.split(key, num_epochs)
    jax.lax.scan(step, vpg_state, step_keys, num_epochs)

if __name__ == '__main__':
    key = jrng.key(1234)
    params = VPGParams(
        parallel_envs=1024,
        rollout_steps=256,
        training_epochs=4,
        discount=0.9,
    )
    jax.jit(train, static_argnums=(1,))(key, params)
