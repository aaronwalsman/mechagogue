import jax
import jax.numpy as jnp
import jax.random as jrng

def kaiming_std(fan_in):
    return jnp.sqrt(2/fan_in)

def kaiming(key, shape, dtype=jnp.float32):
    fan_in = shape[-2]
    return jrng.normal(key, shape, dtype=dtype) * kaiming_std(fan_in)

def xavier(key, shape, dtype=jnp.float32):
    fan_in = shape[-2]
    fan_out = shape[-1]
    b = (6/(fan_in + fan_out))**0.5
    return jrng.uniform(key, shape, minval=-b, maxval=b, dtype=dtype)

def zero(shape, dtype=jnp.float32):
    return jnp.zeros(shape, dtype=dtype)

def eye(shape, dtype=jnp.float32):
    return jnp.eye(shape[0], dtype=dtype)
