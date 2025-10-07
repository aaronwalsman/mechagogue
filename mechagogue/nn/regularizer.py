'''
Regularization layers including dropout.
'''

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.static import static_functions

def dropout_layer(p_drop):
    @static_functions
    class DropoutLayer:
        def forward(key, x):
            def drop_leaf(leaf):
                drop = jrng.bernoulli(key, p_drop, leaf.shape)
                return jnp.where(drop, 0, leaf)
            return jax.tree.map(drop_leaf, x)
    
    return DropoutLayer
