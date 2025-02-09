import jax.lax as lax
import jax.numpy as jnp

from mechagogue.nn.initializers import kaiming, zero

def linear_layer(
    in_channels,
    out_channels,
    use_bias=False,
    init_weights=kaiming,
    init_bias=zero,
    dtype=jnp.float32,
):
    def init(key):
        weight_shape = (in_channels, out_channels)
        weight = init_weights(key, shape=weight_shape, dtype=dtype)
        if use_bias:
            bias_shape = (out_channels,)
            bias = init_bias(key, shape=bias_shape, dtype=dtype)
        else:
            bias = None
        return weight, bias
    
    def model(x, state):
        weight, bias = state
        x = x @ weight
        if bias is not None:
            x = x + bias
        return x
    
    return init, model

def embedding_layer(
    num_embeddings,
    channels,
    init_weight=kaiming,
    dtype=jnp.float32,
):
    def init(key):
        weight_shape = (num_embeddings, channels)
        weight = init_weight(key, shape=weight_shape, dtype=dtype)
        return weight
    
    def model(x, state):
        weight = state
        x = weight[x]
        return x
    
    return init, model

def conv_layer(
    in_channels,
    out_channels,
    kernel_size=(3,3),
    stride=(1,1),
    padding='SAME',
    use_bias=False,
    init_weight=kaiming,
    init_bias=zero,
    dtype=jnp.float32,
):
    def init(key):
        weight_shape = kernel_size + (in_channels, out_channels)
        weight = init_weight(key, shape=weight_shape, dtype=dtype)
        if use_bias:
            bias_shape = (out_channels,)
            bias = init_bias(key, shape=bias_shape, dtype=dtype)
        else:
            bias = None
        return weight, bias
    
    def model(x, state):
        weight, bias = state
        x = lax.conv_general_dilated(
            x,
            weight,
            window_strides=stride,
            padding=padding,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        )
        x = x + bias
        return x
    
    return init, model
