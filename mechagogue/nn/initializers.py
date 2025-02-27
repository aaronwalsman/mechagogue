import jax
import jax.numpy as jnp
import jax.random as jrng

def kaiming(key, shape, dtype=jnp.float32):
    fan_in = shape[-2]
    std = jnp.sqrt(2/fan_in)
    return jrng.normal(key, shape, dtype=dtype) * std

def xavier(key, shape, dtype=jnp.float32):
    fan_in = shape[-2]
    fan_out = shape[-1]
    b = (6/(fan_in + fan_out))**0.5
    return jrng.uniform(key, shape, minval=-b, maxval=b, dtype=dtype)

def zero(key, shape, dtype=jnp.float32):
    return jnp.zeros(shape, dtype=dtype)
