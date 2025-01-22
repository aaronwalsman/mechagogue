from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.nn as jnn

from mechagogue.arg_wrappers import ignore_unused_args

def layer_sequence(
    layers,
    accumulate=lambda x0, x1 : x1,
):
    num_layers = len(layers)
    
    init_layers = [
        ignore_unused_args(init_layer, ('key',))
        for init_layer, _ in layers
    ]
    
    model_layers = [
        ignore_unused_args(model_layer, ('key', 'x', 'params'))
        for _, model_layer in layers
    ]
    
    def init_params(key):
        layer_keys = jrng.split(key, num_layers)
        params = []
        for layer_key, init_layer in zip(layer_keys, init_layers):
            #init_layer_params, _ = layer
            layer_params = init_layer(layer_key)
            params.append(layer_params)
        return params
    
    def model(key, x, params):
        layer_keys = jrng.split(key, num_layers)
        x0 = x
        for layer_key, model_layer, layer_params in zip(
            layer_keys, model_layers, params
        ):
            x1 = model_layer(layer_key, x0, layer_params)
            x0 = accumulate(x0, x1)
        
        return x0
    
    return init_params, model

residual_layer_sequence = partial(layer_sequence, accumulate=jax.lax.add)

def repeat_layer(
    layer,
    repeat,
    accumulate=lambda x0, x1 : x1,
):
    init_layer_params, model_layer = layer
    init_layer_params = ignore_unused_args(init_layer_params, ('key',))
    model_layer = ignore_unused_args(model_layer, ('key', 'x', 'params'))
    
    def init_params(key):
        keys = jrng.split(key, repeat)
        return jax.vmap(init_layer)(keys)
    
    def model(key, x, params):
        def layer_step(x0, key_params):
            key, params = key_params
            x1 = layer_model(key, x0, params)
            x1 = accumulate(x0, x1)
            return x1, None
        
        keys = jrng.split(key, repeat)
        x, _ = jax.lax.scan(layer_step, x, (keys, params))
        return x
    
    return init_params, model

repeat_residual_layer = partial(repeat_layer, accumulate=jax.lax.add)

def repeat_shared_layer(
    layer,
    repeat,
    accumulate=lambda x0, x1 : x1,
):
    init_params, model_layer = layer
    model_layer = ignore_unused_args(model_layer, ('key', 'x', 'params'))
    
    def model(key, x, params):
        def layer_step(x0, key):
            x1 = layer_model(key, x0, params)
            x1 = accumulate(x0, x1)
            return x1, None
        
        keys = jrng.split(key, repeat)
        x, _ = jax.lax.scan(layer_step, x, keys)
        return x
    
    return init_params, model

repeat_shared_residual_layer = partial(
    repeat_shared_layer, accumulate=jax.lax.add)
