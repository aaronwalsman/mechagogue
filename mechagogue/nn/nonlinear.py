import jax.nn as jnn

def relu_layer():
    def model(x):
        return jnn.relu(x)
    
    return lambda key : None, model
