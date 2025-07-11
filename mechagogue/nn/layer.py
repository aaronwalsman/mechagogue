from mechagogue.standardize import standardize_interface
from mechagogue.static import static_functions


default_init = lambda : None
default_forward = lambda x : x


def standardize_layer(layer):
    return standardize_interface(
        layer,
        init = (('key',), default_init),
        forward = (('key', 'x', 'state'), default_forward),
    )


def make_layer(init=default_init, forward=default_forward):
    init_fn = init
    forward_fn = forward
    
    @static_functions
    class Layer:
        init = init_fn
        forward = forward_fn
    
    return Layer
