def permute_layer(permutation):
    def model(x):
        return x[...,permutation]
    
    return lambda: None, model
