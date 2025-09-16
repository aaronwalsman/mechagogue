import jax
import jax.numpy as jnp
import jax.random as jrng
 
from mechagogue.tree import tree_key

def normal_mutate(
    learning_rate=3e-4,
    update_density=None,
    auto_scale=False,
    average_over_parents=False,
):
    
    def mutate(key, state):
        def mutate_leaf(key, leaf):
            if average_over_parents:
                num_parents = leaf.shape[0]
                leaf = jnp.sum(leaf, axis=0) / num_parents
            delta_key, mask_key = jrng.split(key)
            delta = jrng.normal(delta_key, shape=leaf.shape, dtype=leaf.dtype)
            if update_density is not None:
                delta_mask = jrng.bernoulli(
                    mask_key, update_density, delta.shape)
                delta = delta * delta_mask
            if auto_scale and leaf.ndim > 1:
                fan_in = leaf.shape[-2]
                alpha = (1-(learning_rate**2)*fan_in/2)**0.5
                leaf = leaf * alpha
            #num = jnp.sum(jnp.ones_like(delta)).astype(jnp.int32)
            #num_zero = jnp.sum(delta * learning_rate == 0.)
            #jax.debug.print('num zero {nz}/{n}', nz=num_zero, n=num)
            #jax.debug.print('num zero {nz}/{n}', nz=num_zero, n=num)
            #return leaf + learning_rate * delta
            new_leaf = leaf + learning_rate * delta
            
            # XXX
            #new_leaf = leaf.astype(jnp.bfloat16) + learning_rate * delta.astype(jnp.bfloat16)
            #new_leaf = new_leaf.astype(leaf.dtype)
            #new_leaf = new_leaf.astype(jnp.bfloat16)
            #new_leaf = new_leaf.astype(jnp.float32)
            # XXX
            
            #unchanged = jnp.sum(new_leaf == leaf)
            #jax.debug.print('num zero {nz}/{n}', nz=unchanged, n=num)
            return new_leaf
        
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
