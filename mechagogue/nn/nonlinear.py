'''
Nonlinear activation function layers.
'''

import jax.nn as jnn

from mechagogue.nn.layer import make_layer

def relu_layer():
    return make_layer(forward=lambda x : jnn.relu(x))
