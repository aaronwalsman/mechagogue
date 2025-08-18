def num_parameters(model):
    def is_parameter(leaf):
        return hasattr(leaf, 'shape') and hasattr(leaf, 'dtype')
    def leaf_size(leaf):
        return int(np.prod(leaf.shape, dtype=np.int64))
    leaves = jax.tree_util.tree_leaves(tree)
    return sum(leaf_size(leaf) for leaf in leaves if is_parameter(leaf))
