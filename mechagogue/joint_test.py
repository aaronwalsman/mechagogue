from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrng

from flax.struct import dataclass

from dirt.examples.nom import NomAction

import distrax

#a = NomAction(1,0)

class MyThing(NamedTuple):
    a : int
    b : int

thang = MyThing(distrax.Categorical(probs=jnp.array([0.5, 0.5])), distrax.Categorical(probs=jnp.array([0.25, 0.75])))

thing_d = distrax.Joint(thang)

key = jrng.key(1234)

thing = thing_d.sample(seed=key)

@dataclass
class MyBlang:
    c : int
    d : int = 42

blang_d = distrax.Joint((distrax.Categorical(probs=jnp.array([0.5, 0.5])), distrax.Categorical(probs=jnp.array([0.25,0.75]))))

def converter(ungh):
    print(ungh)
    return MyBlang(ungh)

#blang_td = distrax.Transformed(blang_d, converter)

#bling = blang_td.sample(seed=key)

breakpoint()

'''
class DistributionWrapper:
    def __init__(self, distribution, Wrapper):
        self.Wrapper = Wrapper
        self.distribution = distribution
    
    def sample(
'''

class DistributionWrapper:
    def __init__(self, distribution, forward, inverse):
        self.distribution = distribution
        self.forward = forward
        self.inverse = inverse
    
    def sample(self, *args, **kwargs):
        return self.forward(self.distribution.sample(*args, **kwargs))
    
    def log_prob(self, *args, **kwargs):
        return self.distribution.log_prob(self.inverse(*args, **kwargs))

class DataClassDistribution:
    @classmethod
    def sample(cls, key, distribution):
        distributions, structure = jax.tree.flatten(distribution)
        keys = jrng.split(key, len(distributions)
        samples = [distribution.sample(seed=key)


