import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.nn as jnn

def categorical(logits, temp=1.):
    def sampler(key):
        p = jnn.softmax(logits / temp, axis=-1)
        b, c = logits.shape
        # TODO: is there really no better way than vmap?
        def single_sample(key, p):
            return jrng.choice(key, c, p=p)
        return jax.vmap(single_sample)(jrng.split(key, b), p)
    
    def logprob(y):
        logp = jnn.log_softmax(logits, axis=-1)
        b, c = logp.shape
        return logp[jnp.arange(b), y]
    
    return sampler, logprob
