import jax
import jax.numpy as jnp

def a(my_var):
    my_var2 = my_var*2
    def b(carry, x):
        y = carry+x+my_var2
        return y, y
    _, y = jax.lax.scan(b, 0, jnp.zeros(12), 12)
    return y

jit_a = jax.jit(a)

print(jit_a(1))
print(jit_a(3))
