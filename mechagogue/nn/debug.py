import jax

from mechagogue.static import static_functions

def print_activations_layer(prefix):
    @static_functions
    class PrintActivationsLayer:
        def forward(x):
            jax.debug.print(prefix + '{x}', x=x)
            return x
    
    return PrintActivationsLayer

def breakpoint_layer():
    @static_functions
    class BreakpointLayer:
        def forward(key, x):
            breakpoint()
            return x
    
    return BreakpointLayer
