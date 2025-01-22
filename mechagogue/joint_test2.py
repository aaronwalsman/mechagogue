import jax.numpy as jnp

from flax.struct import PyTreeNode, dataclass

@dataclass
class MyThing:
    a : int = 2
    b : int = 3



import distrax

class MyDistributionBundler:
    def __init__(self, distribution, Class):
        self.distribution = distribution
        self.Class = Class

    def sample(self, *args, **kwargs):
        sample = self.distribution.sample(*args, **kwargs)
        return self.Class(sample)
    
    def log_prob(self, sample):
        return self.distribution.log_prob(sample.__dict__)

my_distribution = MyThing(
    distrax.Categorical(probs=jnp.array([0.1, 0.1, 0.5, 0.3])),
    distrax.Categorical(probs=jnp.array([0.9, 0.05, 0.04, 0.01])),
)

my_thing = distrax.Joint()

