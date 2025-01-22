import jax.numpy as jnp

from mechagogue.nn.linear import linear_layer
from mechagogue.nn.regularizer import dropout_layer
from mechagogue.nn.nonlinear import relu_layer
from mechagogue.nn.sequence import layer_sequence
from mechagogue.nn.initializers import kaiming, zero

def mlp(
    hidden_layers,
    in_channels,
    hidden_channels,
    out_channels=None,
    use_bias=False,
    p_dropout=0,
    init_weight=kaiming,
    init_bias=zero,
    dtype=jnp.float32
):
    
    layers = []
    in_c = in_channels
    for _ in range(hidden_layers):
        layers.append(linear_layer(
            in_c, hidden_channels, use_bias=use_bias, dtype=dtype))
        
        if p_dropout:
            layers.append(dropout_layer(p_dropout))
        
        layers.append(relu_layer())
        in_c = hidden_channels
    
    if out_channels is not None:
        layers.append(linear_layer(
            in_c, out_channels, use_bias=False, dtype=dtype))
    
    return layer_sequence(layers)
