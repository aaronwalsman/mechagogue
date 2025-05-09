import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrng

import msgpack

'''
Built using flax's serialization module as reference, but added support for
PRNG keys.
'''

JNP_ARRAY_EXT_ID = 1
NP_ARRAY_EXT_ID = 2
NP_GENERIC_EXT_ID = 3
COMPLEX_EXT_ID = 4

def dtype_from_name(name: str):
    if name == 'bfloat16':
        return jnp.bfloat16
    elif name == 'key<fry>':
        return np.int32
    else:
        return np.dtype(name)

def np_to_packable(leaf):
    return (leaf.dtype.name, leaf.shape, leaf.tobytes())

def jnp_to_packable(leaf):
    is_prng_key = False
    if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
        is_prng_key = True
        leaf = jrng.key_data(leaf)
    
    return (is_prng_key,) + np_to_packable(np.array(leaf))

def leaf_to_msgpack(leaf):
    if isinstance(leaf, jnp.ndarray):
        data = msgpack.packb(jnp_to_packable(leaf))
        return msgpack.ExtType(JNP_ARRAY_EXT_ID, data)
    elif isinstance(leaf, np.ndarray):
        data = msgpack.packb(np_to_packable(leaf))
        return msgpack.ExtType(NP_ARRAY_EXT_ID, data)
    elif isinstance(leaf, np.generic):
        data = msgpack.packb(np_to_packable(np.asarray(leaf)))
        return msgpack.ExtType(NP_GENERIC_EXT_ID, data)
    elif isinstance(leaf, complex):
        data = msgpack.packb((leaf.real, leaf.imag))
        return msgpack.ExtType(COMPLEX_EXT_ID, data)
    
    return leaf

def pack_leaf_data(item):
    leaves, _ = jax.tree.flatten(item)
    return msgpack.packb(leaves, default=leaf_to_msgpack, strict_types=True)

def save_leaf_data(item, destination):
    if isinstance(destination, str):
        with open(destination, 'wb') as f:
            return save_leaf_data(item, f)
    
    data = pack_leaf_data(item)
    destination.write(data)

def packable_to_np(data):
    dtype_name, shape, buffer = data
    return np.frombuffer(
        buffer,
        dtype=dtype_from_name(dtype_name),
        count=-1,
        offset=0,
    ).reshape(shape)

def packable_to_jnp(data):
    is_prng_key, *data = data
    np_leaf = packable_to_np(data)
    leaf = jnp.array(np_leaf)
    if is_prng_key:
        leaf = jrng.wrap_key_data(leaf)
    return leaf

def msgpack_to_leaf(code, data):
    if code == JNP_ARRAY_EXT_ID:
        data = msgpack.unpackb(data)
        leaf = packable_to_jnp(data)
        return leaf
    elif code == NP_ARRAY_EXT_ID:
        data = msgpack.unpackb(data)
        leaf = packable_to_np(data)
        return leaf
    elif code == NP_GENERIC_EXT_ID:
        data = msgpack.unpackb(data)
        leaf = packable_to_np(data)[()]
        return leaf
    elif code == COMPLEX_EXT_ID:
        data = msgpack.unpackb(data)
        return complex(*data)
    
    return msgpack.ExtType(code, data)

def unpack_leaf_data(data):
    leaves = msgpack.unpackb(data, ext_hook=msgpack_to_leaf)
    return leaves

def unpack_tree_data(treedef, data):
    leaves = unpack_leaf_data(data)
    return jax.tree.unflatten(treedef, leaves)

def load_tree_data(treedef, source):
    if isinstance(source, str):
        with open(source, 'rb') as f:
            return load_tree_data(treedef, f)
    
    data = source.read()
    return unpack_tree_data(treedef, data)

def load_example_data(tree, source):
    treedef = jax.tree.structure(tree)
    return load_tree_data(treedef, source)

def test_serial():
    from mechagogue.sup.sup import SupParams
    
    key = jrng.key(1234)
    obj = SupParams(batch_size=12, shuffle=False)
    thing = (
        jrng.split(key, 4),
        {
            'a':obj,
            'b':'text',
            'c':12,
            'd':12.5,
            'e':complex(1,-2.),
            'f':jnp.arange(24).reshape(3,4,2).astype(jnp.bfloat16),
            'g':np.arange(18).reshape(3,3,2),
            'h':np.array([12], dtype=np.int32)[0],
        },
    )
    
    data = pack_leaf_data(thing)
    
    leaves, treedef = jax.tree.flatten(thing)
    
    unpacked_thing = unpack_tree(treedef, data)
    
    breakpoint()

if __name__ == '__main__':
    test_serial()
