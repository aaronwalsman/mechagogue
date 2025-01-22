import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.tree import tree_key

def normal_mutate(
    learning_rate=3e-4,
    auto_scale=False
):
    
    def init():
        return None
    
    def mutate(key, model_params):
        def mutate_leaf(key, leaf):
            num_parents = leaf.shape[0]
            leaf = jnp.sum(leaf, axis=0) / num_parents
            delta = jrng.normal(key, shape=leaf.shape, dtype=leaf.dtype)
            return leaf + learning_rate * delta
        
        keys = tree_key(key, jax.tree.structure(model_params))
        model_params = jax.tree.map(mutate_leaf, keys, model_params)
        
        return model_params, None
    
    return init, mutate
