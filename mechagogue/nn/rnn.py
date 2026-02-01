"""
Recurrent neural network layers.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.static import static_data, static_functions
from mechagogue.standardize import standardize_args
from mechagogue.nn.initializers import kaiming, zero


@static_data
class GRUState:
    w_ir: Optional[jnp.ndarray] = None
    w_iz: Optional[jnp.ndarray] = None
    w_in: Optional[jnp.ndarray] = None
    w_hr: Optional[jnp.ndarray] = None
    w_hz: Optional[jnp.ndarray] = None
    w_hn: Optional[jnp.ndarray] = None
    b_ir: Optional[jnp.ndarray] = None
    b_iz: Optional[jnp.ndarray] = None
    b_in: Optional[jnp.ndarray] = None
    b_hr: Optional[jnp.ndarray] = None
    b_hz: Optional[jnp.ndarray] = None
    b_hn: Optional[jnp.ndarray] = None


def gru_cell(
    in_channels: int,
    hidden_channels: int,
    use_bias: bool = True,
    init_weights=kaiming,
    init_bias=zero,
    dtype=jnp.float32,
):
    init_weights = standardize_args(init_weights, ("key", "shape", "dtype"))
    init_bias = standardize_args(init_bias, ("key", "shape", "dtype"))

    @static_functions
    class GRUCell:
        def init(key):
            k = jrng.split(key, 12)
            w_ir = init_weights(k[0], (in_channels, hidden_channels), dtype)
            w_iz = init_weights(k[1], (in_channels, hidden_channels), dtype)
            w_in = init_weights(k[2], (in_channels, hidden_channels), dtype)
            w_hr = init_weights(k[3], (hidden_channels, hidden_channels), dtype)
            w_hz = init_weights(k[4], (hidden_channels, hidden_channels), dtype)
            w_hn = init_weights(k[5], (hidden_channels, hidden_channels), dtype)

            if use_bias:
                b_ir = init_bias(k[6], (hidden_channels,), dtype)
                b_iz = init_bias(k[7], (hidden_channels,), dtype)
                b_in = init_bias(k[8], (hidden_channels,), dtype)
                b_hr = init_bias(k[9], (hidden_channels,), dtype)
                b_hz = init_bias(k[10], (hidden_channels,), dtype)
                b_hn = init_bias(k[11], (hidden_channels,), dtype)
            else:
                b_ir = b_iz = b_in = None
                b_hr = b_hz = b_hn = None

            return GRUState(
                w_ir=w_ir,
                w_iz=w_iz,
                w_in=w_in,
                w_hr=w_hr,
                w_hz=w_hz,
                w_hn=w_hn,
                b_ir=b_ir,
                b_iz=b_iz,
                b_in=b_in,
                b_hr=b_hr,
                b_hz=b_hz,
                b_hn=b_hn,
            )

        def forward(x, h, state):
            x = x.astype(dtype)
            h = h.astype(dtype)

            x_r = x @ state.w_ir
            x_z = x @ state.w_iz
            x_n = x @ state.w_in
            h_r = h @ state.w_hr
            h_z = h @ state.w_hz
            h_n = h @ state.w_hn

            if state.b_ir is not None:
                x_r = x_r + state.b_ir
                x_z = x_z + state.b_iz
                x_n = x_n + state.b_in
                h_r = h_r + state.b_hr
                h_z = h_z + state.b_hz
                h_n = h_n + state.b_hn

            r = jax.nn.sigmoid(x_r + h_r)
            z = jax.nn.sigmoid(x_z + h_z)
            n = jax.nn.tanh(x_n + r * h_n)
            h_next = (1.0 - z) * n + z * h
            return h_next

    return GRUCell
