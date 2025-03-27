import string

import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrng

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
        weight_key, bias_key = jrng.split(key)
        weight_shape = (in_channels, out_channels)
        weight = init_weights(weight_key, shape=weight_shape, dtype=dtype)
        if use_bias:
            bias_shape = (out_channels,)
            bias = init_bias(bias_key, shape=bias_shape, dtype=dtype)
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

def grouped_linear_layer(
    in_channels,
    out_channels,
    groups,
    use_bias=False,
    init_weights=kaiming,
    init_bias=zero,
    dtype=jnp.float32,
):
    assert in_channels % groups == 0
    assert out_channels % groups == 0
    in_group_channels = in_channels // groups
    out_group_channels = out_channels // groups
    
    def init(key):
        weight_key, bias_key = jrng.split(key)
        weight_shape = (groups, in_group_channels, out_group_channels)
        weight = init_weights(weight_key, shape=weight_shape, dtype=dtype)
        if use_bias:
            bias_shape = (out_channels,)
            bias = init_bias(key, shape=bias_shape, dtype=dtype)
        else:
            bias = None
        return weight, bias
    
    def  model(x, state):
        weight, bias = state
        *b, c = x.shape
        x = x.reshape(*b, groups, in_group_channels)
        abc = string.ascii_lowercase[:len(b)]
        x = jnp.einsum(f'{abc}xy,xyz->{abc}xz', x, weight)
        x = x.reshape(*b, -1)
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
        
        num_dims = len(x.shape)
        if num_dims == 3:
            x = x[None,:,:,:]
        
        x = lax.conv_general_dilated(
            x,
            weight,
            window_strides=stride,
            padding=padding,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        )
        
        if num_dims == 3:
            x = x[0]
        
        if bias is not None:
            x = x + bias
        return x
    
    return init, model
