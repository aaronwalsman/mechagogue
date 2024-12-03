import jax
import jax.numpy as jnp

gamma = 0.9

def compute_returns(ret, rd):
    r, d = rd
    ret = ret + r
    return ret*(1.-d)*gamma, ret

r = jnp.zeros(10)
r = r.at[4].set(1)
r = r.at[8].set(1)
d = jnp.zeros(10, dtype=jnp.bool)
d = d.at[9].set(1)
d = d.at[5].set(1)
_, ret = jax.lax.scan(compute_returns, 0, (r, d), reverse=True)

breakpoint()
