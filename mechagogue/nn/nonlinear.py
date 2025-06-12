import jax.nn as jnn

from mechagogue.static import static_functions

@static_functions
class ReluLayer:
    def forward(x):
        return jnn.relu(x)
