'''
Probability distributions and sampling layers for stochastic neural networks.
'''

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
        
        def entropy():
            logp = jnn.log_softmax(logits, axis=-1)
            p = jnp.exp(logp)
            h = (-p*logp).sum(axis=-1)
            return h
        
    return Categorical

def sampler_layer(
    distribution,
    include_sample=True,
    include_logp=False,
    include_entropy=False,
):
    @static_functions
    class SamplerLayer:
        def forward(key, x):
            d = distribution(x)
            result = []
            if include_sample:
                sample = d.sample(key)
                result.append(sample)
            if include_logp:
                logp = d.logp(sample)
                result.append(logp)
            if include_entropy:
                entropy = d.entropy()
                result.append(entropy)
            if len(result) == 1:
                return result[0]
            else:
                return tuple(result)
    
    return SamplerLayer

def categorical_sampler_layer(
    include_sample=True,
    include_logp=False,
    include_entropy=False,
    **kwargs
):
    return sampler_layer(
        partial(categorical, **kwargs),
        include_sample=include_sample,
        include_logp=include_logp,
        include_entropy=include_entropy,
    )
