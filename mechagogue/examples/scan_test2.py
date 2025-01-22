import jax.numpy as jnp
from jax import make_jaxpr
import jax.lax as lax

def func11(arr, extra):
    ones = jnp.ones(arr.shape)
    def body(carry, aelems):
        ae1, ae2 = aelems
        return (carry + ae1 * ae2 + extra, carry)
    return lax.scan(body, 0., (arr, ones))
make_jaxpr(func11)(jnp.arange(16), 5.)
