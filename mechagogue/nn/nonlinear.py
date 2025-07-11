import jax.nn as jnn

from mechagogue.static import static_functions


def relu_layer():
    @static_functions
    class ReluLayer:
        def forward(x):
            return jnn.relu(x)

    return ReluLayer
