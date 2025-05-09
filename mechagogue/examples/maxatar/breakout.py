import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.nn as jnn

from mechagogue.dp.mdp import mdp
from mechagogue.nn.linear import embedding_layer
from mechagogue.nn.distributions import categorical
from mechagogue.optim.sgd import sgd
from mechagogue.rl.dqn import DQNConfig, dqn
import mechagogue.envs.maxatar.breakout as breakout
from mechagogue.wrappers import auto_reset_wrapper
from mechagogue.nn.mlp import mlp
from mechagogue.nn.sequence import layer_sequence


def breakout_dqn(key):
    breakout_reset, breakout_step = auto_reset_wrapper(
    breakout.reset, breakout.step)
    
    dqn_config = DQNConfig(
        batch_size=32,
        parallel_envs=8,
        replay_buffer_size=32*10000,
        discount=0.9,
        target_update=0.1,
        epsilon=0.1
    )
    
    # build the model
    in_channels = 400  # 10x10x4  (size of grid x TOTAL_CHANNELS)
    hidden_layers = 1
    hidden_channels = 256
    out_channels = 6
    
    init_model_params, model = layer_sequence((
        (lambda : None, lambda x : x.reshape(-1,in_channels)), # flatten
        mlp(
            hidden_layers=hidden_layers,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            p_dropout=0.1,
        ),
    ))
    
    init_optimizer_params, optimize = sgd(learning_rate=1e-2)
    
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
    
    num_epochs = 100000
    dqn_state, losses = jax.lax.scan(
        train_epoch,
        dqn_state,
        jrng.split(step_key, num_epochs),
    )
    
    print(losses)
    
if __name__ == '__main__':
    jax.config.update("jax_enable_x64", True)   # higher precision
    jnp.set_printoptions(precision=15, linewidth=120)
    breakout_dqn(jrng.key(1234))
