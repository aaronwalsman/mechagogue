import jax.nn as jnn

#from mechagogue.static import static_functions
from mechagogue.nn.layer import make_layer

def relu_layer():
    #@static_functions
    #class ReluLayer:
    #    def forward(x):
    #        return jnn.relu(x)

    #return ReluLayer
    return make_layer(forward=lambda x : jnn.relu(x))
