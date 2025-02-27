import jax.random as jrng

from mechagogue.arg_wrappers import ignore_unused_args

def parallel_tuple_layer(layers):
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
        head_keys = jrng.split(key, num_layers)
        return (init_layer(head_key)
            for init_layer, head_key
            in zip(init_layers, head_keys)
        )
    
    def model(key, x, state):
        head_keys = jrng.split(key, num_layers)
        return (
            model_layer(head_key, xi, head_state)
            for xi, model_layer, head_key, head_state
            in zip(x, model_layers, head_keys, state)
        )
    
    return init, model

def parallel_dict_layer(layers):
    init_layers = {
        name : ignore_unused_args(init_layer, ('key',))
        for name, (init_layer, _) in layers.items()
    }
    model_layers = {
        name : ignore_unused_args(model_layer, ('key', 'x', 'state'))
        for name, (_, model_layer) in layers.items()
    }
    
    def init(key):
        head_keys = jrng.split(key, len(layers))
        return {name : init_layer(head_key)
            for (name, init_layer), head_key
            in zip(init_layers.items(), head_keys)
        }
    
    def model(key, x, state):
        head_keys = jrng.split(key, len(layers))
        return {
            name : model_layer(head_key, x[name], state[name])
            for (name, model_layer), head_key
            in zip(model_layers.items(), head_keys)
        }
    
    return init, model
