import jax
import jax.numpy as jnp
import jax.random as jrng

def dropout_layer(p_drop):
    def model(key, x):
        def drop_leaf(leaf):
            drop = jrng.bernoulli(key, p_drop, leaf.shape)
            return jnp.where(drop, 0, leaf)
        return jax.tree.map(drop_leaf, x)
    
    return lambda : None, model
