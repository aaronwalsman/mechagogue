from functools import partial
from typing import Tuple, Any

import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.nn as jnn

from mechagogue.static import static_functions, static_data
from mechagogue.standardize import standardize_args
from mechagogue.nn.layer import standardize_layer


def layer_sequence(
    layers,
    accumulate=lambda x0, x1 : x1,
):
    num_layers = len(layers)
    layers = [standardize_layer(layer) for layer in layers]
    
    @static_functions
    class LayerSequence:
        @static_data
        class LayerSequenceState:
            layer_states : Tuple[Any]
        
        def init(key):
            layer_keys = jrng.split(key, num_layers)
            layer_states = []
            for layer_key, layer in zip(layer_keys, layers):
                layer_state = layer.init(layer_key)
                layer_states.append(layer_state)
                
            return LayerSequence.LayerSequenceState(tuple(layer_states))
        
        def forward(key, x, state):
            layer_keys = jrng.split(key, num_layers)
            x0 = x
            # for key_layer_state in zip(layer_keys, layers, state.layer_states):
            for k, l, s in zip(layer_keys, layers, state.layer_states):
                # layer_key, layer, layer_state = key_layer_state
                # x1 = layer.forward(layer_key, x0, layer_state)
                x1 = l.forward(k, x0, s)
                x0 = accumulate(x0, x1)
            
            return x0
    
    return LayerSequence


residual_layer_sequence = partial(layer_sequence, accumulate=jax.lax.add)


def repeat_layer(
    layer,
    repeat,
    accumulate=lambda x0, x1 : x1,
):
    layer = standardize_layer(layer)
    
    @static_functions
    class RepeatLayer:
        def init(key):
            keys = jrng.split(key, repeat)
            return jax.vmap(layer.init)(keys)
        
        def forward(key, x, state):
            def layer_step(x0, key_state):
                key, state = key_state
                x1 = layer.forward(key, x0, state)
                x1 = accumulate(x0, x1)
                return x1, None
            
            keys = jrng.split(key, repeat)
            x, _ = jax.lax.scan(layer_step, x, (keys, state))
            return x
    
    return RepeatLayer


repeat_residual_layer = partial(repeat_layer, accumulate=jax.lax.add)


def repeat_shared_layer(
    layer,
    repeat,
    accumulate=lambda x0, x1 : x1,
):
    layer = standardize_layer(layer)
    
    @static_functions
    class RepeatSharedLayer:
        
        init = layer.init
        
        def forward(key, x, state):
            def layer_step(x0, key):
                x1 = layer.forward(key, x0, state)
                x1 = accumulate(x0, x1)
                return x1, None
            
            keys = jrng.split(key, repeat)
            x, _ = jax.lax.scan(layer_step, x, keys)
            return x
    
    return RepeatSharedLayer


repeat_shared_residual_layer = partial(
    repeat_shared_layer, accumulate=jax.lax.add)
