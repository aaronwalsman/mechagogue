import jax.random as jrng

class JointDistribution:
    def __init__(self, distributions):
        self.distributions = distributions
    
    def sample(self, seed):
        keys = jrng.split(seed, len(self.distributions))
        return tuple(d.sample(seed=k) for d,k in zip(self.distributions, keys))
    
    def log_prob(self, sample):
        return sum(d.log_prob(s) for d, s in zip(self.distributions, sample))
