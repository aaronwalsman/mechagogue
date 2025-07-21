from functools import partial

import jax
import jax.numpy as jnp

from mechagogue.static import static_functions

def pool_layer(
    reduce,
    init_value,
    post_pool=None,
    kernel_size=(2,2),
    stride=None,
    axis=(-3,-2),
    padding='VALID',
):
    
    if stride is None:
        stride = kernel_size
    
    @static_functions
    class PoolLayer:
        def forward(x):
            full_kernel_size = [1 for _ in x.shape]
            full_stride = [1 for _ in x.shape]
            for k,s,a in zip(kernel_size, stride, axis):
                full_kernel_size[a] = k
                full_stride[a] = s
            
            x = jax.lax.reduce_window(
                x, init_value, reduce, full_kernel_size, full_stride, padding)
            
            if post_pool is not None:
                x = post_pool(x)
            
            return x
    
    return PoolLayer

maxpool_layer = partial(pool_layer, reduce=jax.lax.max, init_value=-jnp.inf)

def avgpool_layer(
    kernel_size=(2,2),
    **kwargs
):
    return pool_layer(
        jax.lax.add,
        0,
        post_pool=lambda x : x/jnp.prod(jnp.array(kernel_size)),
        kernel_size=kernel_size,
        **kwargs
    )
