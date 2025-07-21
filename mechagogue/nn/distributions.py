from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.nn as jnn

from mechagogue.static import static_functions

def categorical(logits, temp=1., choices=None):
    @static_functions
    class Categorical:
        def sample(key):
            u = jax.random.uniform(key, logits.shape, minval=1e-6, maxval=1.)
            gumbel = -jnp.log(-jnp.log(u))
            y = jnp.argmax(logits + gumbel, axis=-1)
            if choices is not None:
                y = choices[y]
            return y
        
        def logp(y):
            logp = jnn.log_softmax(logits, axis=-1)
            return jnp.take_along_axis(logp, y[..., None], axis=-1)[..., 0]
        
    return Categorical

'''
def categorical(logits, temp=1., choices=None):
    @static_functions
    class Categorical:
        def sample(key):
            p = jnn.softmax(logits / temp, axis=-1)
            *b, c = logits.shape
            n = 1
            for bb in b:
                n *= bb
            
            if choices is None:
                sampler_choices = c
            else:
                assert choices.shape == (c,)
                sampler_choices = choices
            
            # TODO: is there really no better way than vmap?
            def single_sample(key, p):
                return jrng.choice(key, sampler_choices, p=p)
            p = p.reshape(n, c)
            samples = jax.vmap(single_sample)(jrng.split(key, n), p)
            samples = samples.reshape(*b)
            return samples
        
        def logp(y):
            logp = jnn.log_softmax(logits, axis=-1)
            *b, c = logp.shape
            n = 1
            for bb in b:
                n *= bb
            logp = logp.reshape(n, c)
            y = y.reshape(n)
            logp_y = logp[jnp.arange(n), y]
            return logp_y
    
    return Categorical
'''

def sampler_layer(distribution):
    @static_functions
    class SamplerLayer:
        def forward(key, x):
            d = distribution(x)
            value = d.sample(key)
            return value
    
    return SamplerLayer

def categorical_sampler_layer(**kwargs):
    return sampler_layer(partial(categorical, **kwargs))
