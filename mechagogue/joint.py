import jax.random as jrng

raise Exception('DEPRECATED')

class JointDistribution:
    def __init__(self, distributions):
        self.distributions = distributions
    
    def sample(self, seed):
        keys = jrng.split(seed, len(self.distributions))
        return tuple(d.sample(seed=k) for d,k in zip(self.distributions, keys))
    
    def log_prob(self, sample):
        return sum(d.log_prob(s) for d, s in zip(self.distributions, sample))

class JointDistribution:
    def __init__(self, distributions):
        self.distributions = distributions
    
    def sample(self, seed):
        # make keys for everything
        sample_keys = SOMETHING
        return jax.tree.map(
            lambda key, distribution : distribution.sample(key),
            sample_keys,
            self.distributions,
        )

def sample_tree_joint(key, tree_joint):
    tree_structure = jax.tree.structure(tree_joint)
    distributions = jax.tree.flatten(tree_joint)
    keys = jrng.split(key, len(distributions))
    samples = jax.tree.map(lambda k, d : d.sample(k), keys, distributions)
    tree_sample = jax.tree.unflatten(tree_structure, samples)
    return tree_sample
