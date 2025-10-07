'''
Structured layer operations for tuples and dictionaries of layers.
'''

from typing import Tuple, Dict, Any

import jax.random as jrng

from mechagogue.static import static_functions
from mechagogue.standardize import standardize_args
from mechagogue.nn.layer import standardize_layer, make_layer

def split_tuple(n):
    return make_layer(forward=lambda x : (x,)*n)

def split_dict(*keys):
    return make_layer(forward=lambda x : {key:x for key in keys}

def tuple_layer(layers):
    num_layers = len(layers)
    layers = [standardize_layer(layer) for layer in layers]
    
    @static_functions
    class TupleLayer:
        
        @static_data
        class TupleState:
            layer_states : Tuple[Any]
        
        def init(key):
            layer_keys = jrng.split(key, num_layers)
            return TupleState(tuple(layer.init(layer_key)
                for layer, layer_key
                in zip(layers, layer_keys)
            ))
        
        def forward(key, x, state):
            layer_keys = jrng.split(key, num_layers)
            return (
                layer.forward(layer_key, xi, layer_state)
                for xi, layer, layer_key, layer_state
                in zip(x, layers, layer_keys, state.layer_states)
            )
    
    return TupleLayer

def dict_layer(layers):
    layers = {key:standardize_layer(layer) for key, layer in layers.items()}
    
    @static_functions
    class DictLayer:
        
        @static_data
        class DictState:
            layer_states : Dict[Any, Any]
        
        def init(key):
            layer_keys = jrng.split(key, len(layers))
            return DictState({
                name : layer.init(layer_key)
                for (name, layer), layer_key
                in zip(layers.items(), layer_keys)
            })
        
        def forward(key, x, state):
            layer_keys = jrng.split(key, len(layers))
            return {
                name : layer.forward(
                    layer_key, x[name], state.layer_states[name])
                for (name, layer), layer_key
                in zip(layers.items(), layer_keys)
            }
    
    return DictLayer
