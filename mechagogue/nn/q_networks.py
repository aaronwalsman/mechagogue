"""
    Q-network architectures, including a simple MLP and a MinAtar-style CNN.
    e.g. can be used for Breakout DQN training
"""

from mechagogue.nn.mlp import mlp
from mechagogue.nn.linear import linear_layer, conv_layer
from mechagogue.nn.nonlinear import relu_layer
from mechagogue.nn.sequence import layer_sequence


def q_network_mlp(in_channels: int, num_actions: int):
    """
        flatten → MLP
        
        e.g. MinAtar breakout:
        400 * 256 + 256 * 6 = 102400 + 1536 = 103936 params
        treats every pixel independently
        
        returns a LayerSequence object with init and forward static methods
    """
    hidden_layers = 1
    hidden_channels = 256
    
    mlp = layer_sequence(
        (
            (lambda: None, lambda x: x.reshape(-1, in_channels)),
            mlp(
                hidden_layers=hidden_layers,
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=num_actions,
                p_dropout=0.1,
            ),
        )
    )
    
    return mlp


def q_network_cnn(in_channels: int, num_actions: int):
    """
        3x3 convolution, 16 filters → flatten → FC-128 → FC-num_actions,
        with ReLU after each hidden layer. Dimensions and padding convention
        are the same as the MinAtar reference network.
        
        e.g. MinAtar breakout:
        3*3*4*16 + 8*8*16*128 + 128*6 = 576 + 131072 + 768 = 132416 params
        shares weights spatially, providing inductive bias for grid inputs
        
        returns a LayerSequence object with init and forward static methods
    """
    board_dim = 10  # input is 10x10
    conv_filters = 16
    
    def dim_after_convolution(size, kernel_size=3, stride=1):
        return (size - (kernel_size - 1) - 1) // stride + 1
    
    conv_out_h = conv_out_w = dim_after_convolution(board_dim)  # 8x8 feature map after applying 3x3 'VALID' conv
    fc_in = conv_out_h * conv_out_w * conv_filters  # 8×8×16 = 1024
    fc_out = 128

    cnn = layer_sequence(
        (
            # 3×3 Conv2D, stride 1, VALID padding
            conv_layer(
                in_channels=in_channels,
                out_channels=conv_filters,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding='VALID',
                use_bias=True,
            ),
            relu_layer(),
            (lambda: None, lambda x: x.reshape(-1, fc_in)),  # flatten NHWC -> (…, 1024)
            linear_layer(fc_in, fc_out, use_bias=True),  # fully-connected 128
            relu_layer(),
            linear_layer(fc_out, num_actions, use_bias=True),  # output layer
        )
    )

    return cnn
