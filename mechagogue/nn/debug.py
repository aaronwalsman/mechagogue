import jax

def print_activations_layer(prefix):
    def model(x):
        jax.debug.print(prefix + '{x}', x=x)
        return x
    
    return lambda : None, model
