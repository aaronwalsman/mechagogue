import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.nn as jnn

from mechagogue.dp.mdp import mdp

def count_up(n):
    def init_state():
        return jnp.zeros((), dtype=jnp.int32)
    
    def transition(state, action):
        return jnp.where(action == state+1, jnp.array(action), jnp.array(n))
    
    def reward(state, action):
        r = (action == (state + 1)).astype(jnp.float32)
        return r
    
    def terminal(state):
        return state == n
    
    return mdp(init_state, transition, reward, terminal)
