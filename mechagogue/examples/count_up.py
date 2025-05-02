import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.nn as jnn

from mechagogue.dp.mdp import mdp
from mechagogue.nn.linear import embedding_layer
from mechagogue.nn.distributions import categorical
from mechagogue.optim.sgd import sgd
from mechagogue.rl.dqn import DQNConfig, dqn
from mechagogue.rl.vpg import VPGConfig, vpg

def countup(n, reward_good=1.0, reward_bad=-1.0):
    """
        Environment where the agent must take action s+1 to advance from state s.

        States: 0 ... n
        Actions: 0 ... n
        Reward: reward_good for the correct action, reward_bad otherwise.
        Episode terminates when state == n.
    """
    
    def init_state():
        # start in state 0
        return jnp.zeros((), dtype=jnp.int32)
    
    def transition(state, action):
        # always just execute the action:
        return jnp.array(action, dtype=jnp.int32)
        
        # only execute the action if it's 'correct'; otherwise remain in current state
        # return jnp.where(action == state+1,
        #                 jnp.array(action, dtype=jnp.int32),
        #                 jnp.array(state, dtype=jnp.int32))
        
        # go to the next state if the action is correct, otherwise terminate the game
        # return jnp.where(action == state+1,
        #                  jnp.array(action, dtype=jnp.int32),
        #                  jnp.array(n, dtype=jnp.int32))
    
    def reward(state, action):
        return jnp.where(action == state + 1,
                         jnp.array(reward_good,  dtype=jnp.float32),
                         jnp.array(reward_bad,   dtype=jnp.float32))
    
    def terminal(state):
        return state == n
    
    return mdp(init_state, transition, reward, terminal)  # returns reset_fn, step_fn

def countup_dqn(key, n=4):
    reset_env, step_env = countup(n)
    dqn_config = DQNConfig(
        batch_size=32,
        parallel_envs=8,
        replay_buffer_size=32*10000,
        discount=0.0, # 0.9,
        target_update=0.1,
        epsilon=0.1
    )
    
    init_model_params, model = embedding_layer(n+1, n+1)
    
    init_optimizer_params, optimize = sgd(learning_rate=1e-2)
    
    def random_action(key):
        return jrng.randint(key, minval=0, maxval=(n+1), shape=())
    
    init_dqn, step_dqn = dqn(
        dqn_config,
        reset_env,
        step_env,
        init_model_params,
        model,
        init_optimizer_params,
        optimize,
        random_action,
    )
    
    init_key, step_key = jrng.split(key)
    dqn_state = init_dqn(init_key)
    
    print('before training')
    print(dqn_state.model_state[0])
    
    def train_epoch(dqn_state, key):
        dqn_state, loss = step_dqn(key, dqn_state)
        return dqn_state, loss
    
    dqn_state, losses = jax.lax.scan(
        train_epoch,
        dqn_state,
        jrng.split(step_key, 100000),
    )
    
    print('after training')
    print(dqn_state.model_state[0])
    print(dqn_state.target_state[0])

def countup_vpg(key, n=3):
    reset_env, step_env = countup(n)
    vpg_config = VPGConfig(
        batch_size=16,
        rollout_steps=16,
    )
    
    init_model_params, model_embedding = embedding_layer(n, n+1)
    def model(x, params):
        x = model_embedding(x, params)
        return categorical(x)
    
    init_optimizer_params, optimize = sgd(learning_rate=0.01)
    
    init_vpg, step_vpg = vpg(
        vpg_config,
        reset_env,
        step_env,
        init_model_params,
        model,
        init_optimizer_params,
        optimize,
    )
    
    init_key, step_key = jrng.split(key)
    vpg_state = init_vpg(init_key)
    
    def train_epoch(vpg_state, key):
        vpg_state, loss = step_vpg(key, *vpg_state)
        return vpg_state, loss
    
    vpg_state, losses = jax.lax.scan(
        train_epoch,
        vpg_state,
        jrng.split(step_key, 1000),
    )
    
    model_params = vpg_state[3]
    print(model_params)

if __name__ == '__main__':
    jax.config.update("jax_enable_x64", True)   # higher precision
    jnp.set_printoptions(precision=15, linewidth=120)
    countup_dqn(jrng.key(1234))
