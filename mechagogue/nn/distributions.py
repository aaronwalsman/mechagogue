from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.nn as jnn

def categorical(logits, temp=1., choices=None):
    def sampler(key):
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
    
    def logprob(y):
        logp = jnn.log_softmax(logits, axis=-1)
        *b, c = logp.shape
        n = 1
        for bb in b:
            n *= bb
        logp = logp.reshape(n, c)
        y = y.reshape(n)
        logp_y = logp[jnp.arange(n), y]
        return logp_y
    
    return sampler, logprob

def sampler_layer(distribution):
    def model(key, x):
        sampler, _ = distribution(x)
        value = sampler(key)
        return value
    
    return lambda : None, model

def categorical_sampler_layer(**kwargs):
    return sampler_layer(partial(categorical, **kwargs))
