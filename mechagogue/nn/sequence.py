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
        ignore_unused_args(model_layer, ('key', 'x', 'state'))
        for _, model_layer in layers
    ]
    
    def init(key):
        layer_keys = jrng.split(key, num_layers)
        state = []
        for layer_key, init_layer in zip(layer_keys, init_layers):
            layer_state = init_layer(layer_key)
            state.append(layer_state)
        return state
    
    def model(key, x, state):
        layer_keys = jrng.split(key, num_layers)
        x0 = x
        for layer_key, model_layer, layer_state in zip(
            layer_keys, model_layers, state
        ):
            x1 = model_layer(layer_key, x0, layer_state)
            x0 = accumulate(x0, x1)
        
        return x0
    
    return init, model

residual_layer_sequence = partial(layer_sequence, accumulate=jax.lax.add)

def repeat_layer(
    layer,
    repeat,
    accumulate=lambda x0, x1 : x1,
):
    init_layer, model_layer = layer
    init_layer = ignore_unused_args(init_layer, ('key',))
    model_layer = ignore_unused_args(model_layer, ('key', 'x', 'state'))
    
    def init(key):
        keys = jrng.split(key, repeat)
        return jax.vmap(init_layer)(keys)
    
    def model(key, x, state):
        def layer_step(x0, key_state):
            key, state = key_state
            x1 = layer_model(key, x0, state)
            x1 = accumulate(x0, x1)
            return x1, None
        
        keys = jrng.split(key, repeat)
        x, _ = jax.lax.scan(layer_step, x, (keys, state))
        return x
    
    return init, model

repeat_residual_layer = partial(repeat_layer, accumulate=jax.lax.add)

def repeat_shared_layer(
    layer,
    repeat,
    accumulate=lambda x0, x1 : x1,
):
    init_layer, model_layer = layer
    model_layer = ignore_unused_args(model_layer, ('key', 'x', 'state'))
    
    def model(key, x, state):
        def layer_step(x0, key):
            x1 = layer_model(key, x0, state)
            x1 = accumulate(x0, x1)
            return x1, None
        
        keys = jrng.split(key, repeat)
        x, _ = jax.lax.scan(layer_step, x, keys)
        return x
    
    return init_layer, model

repeat_shared_residual_layer = partial(
    repeat_shared_layer, accumulate=jax.lax.add)
