import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrng

import msgpack

def save_leaf_data(item, file_path):
    leaves, _ = jax.tree.flatten(item)
    
    def dehydrate_leaf(leaf):
        if isinstance(leaf, jnp.ndarray):
            if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
                leaf = jrng.key_data(leaf)
            leaf = np.array(leaf)
        if isinstance(leaf, np.ndarray):
            leaf = leaf.tobytes()
        
        return leaf
    
    leaves = jax.tree.map(dehydrate_leaf, leaves)
    with open(file_path, 'wb') as f:
        msgpack.pack(leaves, f)

def load_leaf_data(file_path):
    with open(file_path, 'rb') as f:
        return msgpack.unpack(f)

def load_from_example(example, file_path):
    leaves = load_leaf_data(file_path)
    example_leaves, treedef = jax.tree.flatten(example)
    
    def rehydrate_leaf(leaf, example_leaf):
        if isinstance(example_leaf, (np.ndarray, jnp.ndarray)):
            np_dtype = example_leaf.dtype
            np_shape = example_leaf.shape
            if jnp.issubdtype(example_leaf.dtype, jax.dtypes.prng_key):
                key_data = jrng.key_data(example_leaf)
                np_dtype = key_data.dtype
                np_shape = key_data.shape
            leaf = np.frombuffer(leaf, dtype=np_dtype)
            leaf = leaf.reshape(np_shape)
            if isinstance(example_leaf, jnp.ndarray):
                leaf = jnp.array(leaf)
                if jnp.issubdtype(example_leaf.dtype, jax.dtypes.prng_key):
                    leaf = jrng.wrap_key_data(leaf)
        
        return leaf
    
    leaves = jax.tree.map(rehydrate_leaf, leaves, example_leaves)
    return jax.tree.unflatten(treedef, leaves)

def test_serial():
    from mechagogue.sup.sup import SupParams
    
    key = jrng.key(1234)
    obj = SupParams(batch_size=12, shuffle=False)
    thing = (key, {'a':obj, 'b':'text', 'c':12, 'd':12.5, 'e':[key, key]})
    
    save_leaf_data(thing, './tmp.data')
    
    key2 = jrng.key(2345)
    example = (
        key2, {'a':SupParams(), 'b':'hello', 'c':18, 'd':1.2, 'e':[key2, key2]})
    loaded_obj = load(example, './tmp.data')
    
    breakpoint()

if __name__ == '__main__':
    test_serial()
