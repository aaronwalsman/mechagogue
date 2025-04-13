import jax.numpy as jnp
import jax.random as jrng

key = jrng.key(1234)

key, a_key = jrng.split(key)
print(jrng.uniform(a_key, (), minval=0, maxval=1))

key, b_key = jrng.split(key)
print(jrng.uniform(b_key, (), minval=0, maxval=1))

