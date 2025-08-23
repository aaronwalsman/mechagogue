import string
from typing import Optional, Any

import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.standardize import standardize_args
from mechagogue.static import static_functions, static_data
from mechagogue.nn.initializers import kaiming, zero


@static_data
class LinearState:
    weight : Optional[jnp.ndarray] = None
    bias : Optional[jnp.ndarray] = None

def linear_layer(
    in_channels,
    out_channels,
    use_weight=True,
    use_bias=False,
    init_weights=kaiming,
    init_bias=zero,
    dtype=jnp.float32,
):
    init_weights = standardize_args(init_weights, ('key', 'shape', 'dtype'))
    init_bias = standardize_args(init_bias, ('key', 'shape', 'dtype'))
    
    @static_functions
    class LinearLayer:
        
        def init(key):
            weight_key, bias_key = jrng.split(key)
            if use_weight:
                weight_shape = (in_channels, out_channels)
                weight = init_weights(
                    weight_key, shape=weight_shape, dtype=dtype)
            else:
                weight = None
            if use_bias:
                bias_shape = (out_channels,)
                bias = init_bias(bias_key, shape=bias_shape, dtype=dtype)
            else:
                bias = None
            return LinearState(weight, bias)
        
        def forward(x, state):
            if state.weight is not None:
                x = x @ state.weight
            if state.bias is not None:
                x = x + state.bias
            return x
        
        def state_statistics(name, state):
            datapoint = {}
            if state.weight is not None:
                datapoint[f'{name}_weight_magnitude_mean'] = jnp.abs(
                    state.weight).mean()
            if state.bias is not None:
                datapoint[f'{name}_bias_magnitude_mean'] = jnp.abs(
                    state.bias).mean()
            return datapoint
    
    return LinearLayer

def grouped_linear_layer(
    in_channels,
    out_channels,
    groups,
    use_bias=False,
    init_weights=kaiming,
    init_bias=zero,
    dtype=jnp.float32,
):
    init_weights = standardize_args(init_weights, ('key', 'shape', 'dtype'))
    init_bias = standardize_args(init_bias, ('key', 'shape', 'dtype'))
    
    assert in_channels % groups == 0
    assert out_channels % groups == 0
    in_group_channels = in_channels // groups
    out_group_channels = out_channels // groups
    
    @static_functions
    class GroupedLinearLayer:
        def init(key):
            weight_key, bias_key = jrng.split(key)
            weight_shape = (groups, in_group_channels, out_group_channels)
            weight = init_weights(weight_key, shape=weight_shape, dtype=dtype)
            if use_bias:
                bias_shape = (out_channels,)
                bias = init_bias(key, shape=bias_shape, dtype=dtype)
            else:
                bias = None
            return LinearState(weight, bias)
        
        def forward(x, state):
            *b, c = x.shape
            x = x.reshape(*b, groups, in_group_channels)
            abc = string.ascii_lowercase[:len(b)]
            x = jnp.einsum(f'{abc}xy,xyz->{abc}xz', x, state.weight)
            x = x.reshape(*b, -1)
            if state.bias is not None:
                x = x + state.bias
            return x
        
    return GroupedLinearLayer

def embedding_layer(
    num_embeddings,
    channels,
    init_weights=kaiming,
    dtype=jnp.float32,
):
    init_weights = standardize_args(init_weights, ('key', 'shape', 'dtype'))
    
    @static_functions
    class EmbeddingLayer:
        def init(key):
            weight_shape = (num_embeddings, channels)
            weight = init_weights(key, shape=weight_shape, dtype=dtype)
            return LinearState(weight)
        
        def forward(x, state):
            x = state.weight[x]
            return x
    
    return EmbeddingLayer

def conv_layer(
    in_channels,
    out_channels,
    kernel_size=(3,3),
    stride=(1,1),
    padding='SAME',
    use_bias=False,
    init_weights=kaiming,
    init_bias=zero,
    dtype=jnp.float32,
):
    init_weights = standardize_args(init_weights, ('key', 'shape', 'dtype'))
    init_bias = standardize_args(init_bias, ('key', 'shape', 'dtype'))
    
    @static_functions
    class ConvLayer:
        def init(key):
            weight_shape = kernel_size + (in_channels, out_channels)
            weight = init_weights(key, shape=weight_shape, dtype=dtype)
            if use_bias:
                bias_shape = (out_channels,)
                bias = init_bias(key, shape=bias_shape, dtype=dtype)
            else:
                bias = None
            return LinearState(weight, bias)
        
        def forward(x, state):
            num_dims = len(x.shape)
            x = x.astype(state.weight.dtype)
            if num_dims == 3:
                x = x[None,:,:,:]
            
            x = lax.conv_general_dilated(
                x,
                state.weight,
                window_strides=stride,
                padding=padding,
                dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            )
            
            if num_dims == 3:
                x = x[0]
            
            if state.bias is not None:
                x = x + state.bias
            return x
        
        def state_statistics(name, state):
            datapoint = {}
            if state.weight is not None:
                datapoint[f'{name}_weight_magnitude_mean'] = jnp.abs(
                    state.weight).mean()
            if state.bias is not None:
                datapoint[f'{name}_bias_magnitude_mean'] = jnp.abs(
                    state.bias).mean()
            return datapoint
    
    return ConvLayer
