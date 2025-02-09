from functools import partial

import jax
import jax.numpy as jnp

def pool_layer(
    reduce,
    init_value,
    kernel_size=(2,2),
    stride=None,
    axis=(-3,-2),
    padding='VALID',
):
    
    if stride is None:
        stride = kernel_size
    
    def model(x):
        full_kernel_size = [1 for _ in x.shape]
        full_stride = [1 for _ in x.shape]
        for k,s,a in zip(kernel_size, stride, axis):
            full_kernel_size[a] = k
            full_stride[a] = s
        
        x = jax.lax.reduce_window(
            x, init_value, reduce, full_kernel_size, full_stride, padding)
        
        return x
    
    return lambda key : None, model

maxpool_layer = partial(pool_layer, reduce=jax.lax.max, init_value=-jnp.inf)

def avgpool_layer(
    kernel_size=(2,2),
    **kwargs
):
    kernel_elements = 1
    for k in kernel_size:
        kernel_elements *= k
    
    init_pool, model_pool = pool_layer(
        jax.lax.add, 0, kernel_size=kernel_size, **kwargs)
    
    def model(x):
        x = model_pool(x)
        return x / kernel_elements
    
    return init_pool, model
