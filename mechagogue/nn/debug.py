'''
Debugging utility layers for printing activations, shapes, and breakpoints.
'''

import jax
import jax.numpy as jnp

from mechagogue.static import static_functions

def print_activations_layer(prefix):
    @static_functions
    class PrintActivationsLayer:
        def forward(x):
            jax.debug.print(prefix + '{x}', x=x)
            return x
    
    return PrintActivationsLayer

def print_max_activations_layer(prefix):
    @static_functions
    class PrintMaxActivationsLayer:
        def forward(x):
            jax.debug.print(prefix + '{x}', x=jnp.max(x))
            return x
    
    return PrintMaxActivationsLayer

def print_shape_layer(prefix):
    @static_functions
    class PrintShapeLayer:
        def forward(x):
            jax.debug.print(prefix + '{x}', x=x.shape)
    
    return PrintShapeLayer

def breakpoint_layer():
    @static_functions
    class BreakpointLayer:
        def forward(key, x):
            breakpoint()
            return x
    
    return BreakpointLayer
