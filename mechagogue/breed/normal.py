import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.tree import tree_key

def normal_mutate(
    learning_rate=3e-4,
    auto_scale=False
):
    
    def mutate(key, state):
        def mutate_leaf(key, leaf):
            num_parents = leaf.shape[0]
            leaf = jnp.sum(leaf, axis=0) / num_parents
            delta = jrng.normal(key, shape=leaf.shape, dtype=leaf.dtype)
            if auto_scale and leaf.ndim > 1:  # ignores 0-d scalars, e.g. mutable_channels_state and layer_switch_states for 'resizable' MLPs
                fan_in = leaf.shape[-2]
                alpha = (1-(learning_rate**2)*fan_in/2)**0.5
                leaf = leaf * alpha
            return leaf + learning_rate * delta
        
        keys = tree_key(key, jax.tree.structure(state))
        state = jax.tree.map(mutate_leaf, keys, state)
        
        return state
    
    return mutate

def test_auto_scale():
    mutate = normal_mutate(learning_rate=0.01, auto_scale=True)
    from mechagogue.nn.initializers import kaiming
    key = jrng.key(1234)
    key, init_key = jrng.split(key)
    weight = kaiming(init_key, (256,256))
    
    std = jnp.std(weight.reshape(-1))
    print(f'std before: {std}')
    
    for i in range(512):
        key, mutate_key = jrng.split(key)
        weight = mutate(mutate_key, weight[None,...])
    
    std = jnp.std(weight.reshape(-1))
    print(f'std after: {std}')

if __name__ == '__main__':
    test_auto_scale()
