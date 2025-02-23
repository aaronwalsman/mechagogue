import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrng

import msgpack

def dtype_from_name(name: str):
    if name == 'bfloat16':
        return jax.numpy.bfloat16
    elif name == 'key<fry>':
        return np.int32
    else:
        return np.dtype(name)

'''
def jnp_to_np(leaf):
    if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
        leaf = jrng.key_data(leaf)
    leaf = np.array(leaf)
    return leaf

def dtype_data(leaf):
    jnp_dtype = None
    np_dtype = None
    if isinstance(leaf, jnp.ndarray):
        jnp_dtype = leaf.dtype
'''
'''
def save_leaf_data(item, file_path):
    leaves, _ = jax.tree.flatten(item)
    
    def default(obj):
        np_dtype = None
        jnp_dtype = None
        if isinstance(obj, jnp.ndarray):
            if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
                dtype = dtype_to_str(
                leaf = jrng.key_data(leaf)
            leaf = np.array(leaf)
        if isinstance(leaf, np.ndarray):
            shape = leaf.shape
            leaf = leaf.tobytes()
'''

JNP_ARRAY_EXT_ID = 1
NP_ARRAY_EXT_ID = 2
NP_GENERIC_EXT_ID = 3
COMPLEX_EXT_ID = 4

def np_to_tuple(leaf):
    return (leaf.dtype.name, leaf.shape, leaf.tobytes())

def jnp_to_tuple(leaf):
    is_prng_key = False
    if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
        is_prng_key = True
        leaf = jrng.key_data(leaf)
    
    return (is_prng_key,) + np_to_tuple(np.array(leaf))

def leaf_to_msgpack(leaf):
    if isinstance(leaf, jnp.ndarray):
        data = msgpack.packb(jnp_to_tuple(leaf))
        return msgpack.ExtType(JNP_ARRAY_EXT_ID, data)
    elif isinstance(leaf, np.ndarray):
        data = msgpack.packb(np_to_tuple(leaf))
        return msgpack.ExtType(NP_ARRAY_EXT_ID, data)
    elif isinstance(leaf, np.generic):
        data = msgpack.packb(np_to_tuple(np.asarray(leaf)))
        return msgpack.ExtType(NP_GENERIC_EXT_ID, data)
    elif isinstance(leaf, complex):
        data = msgpack.packb((leaf.real, leaf.imag))
        return msgpack.ExtType(COMPLEX_EXT_ID, data)
    
    return leaf

def pack_leaf_data(item):
    leaves, _ = jax.tree.flatten(item)
    return msgpack.packb(leaves, default=leaf_to_msgpack, strict_types=True)

#def save_leaf_data(item, file_path):
#    leaves, _ = jax.tree.flatten(item)
#    
#    def dehydrate_leaf(leaf):
#        if isinstance(leaf, jnp.ndarray):
#            if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
#                leaf = jrng.key_data(leaf)
#            leaf = np.array(leaf)
#        if isinstance(leaf, np.ndarray):
#            leaf = leaf.tobytes()
#        
#        return leaf
#    
#    leaves = jax.tree.map(dehydrate_leaf, leaves)
#    with open(file_path, 'wb') as f:
#        msgpack.pack(leaves, f)

def tuple_to_np(data):
    dtype_name, shape, buffer = data
    return np.frombuffer(
        buffer,
        dtype=dtype_from_name(dtype_name),
        count=-1,
        offset=0,
    ).reshape(shape)

def msgpack_to_leaf(code, data):
    if code == JNP_ARRAY_EXT_ID:
        data = msgpack.unpackb(data)
        is_prng_key, *data = data
        np_leaf = tuple_to_np(data)
        leaf = jnp.array(np_leaf)
        if is_prng_key:
            leaf = jrng.wrap_key_data(leaf)
        return leaf
    elif code == NP_ARRAY_EXT_ID:
        data = msgpack.unpackb(data)
        leaf = tuple_to_np(data)
        return leaf
    elif code == NP_GENERIC_EXT_ID:
        data = msgpack.unpackb(data)
        leaf = tuple_to_np(data)
        leaf = leaf[()]
        return leaf
    elif code == COMPLEX_EXT_ID:
        data = msgpack.unpackb(data)
        return complex(*data)
    
    return msgpack.ExtType(code, data)

def unpack_leaf_data(data):
    leaves = msgpack.unpackb(data, ext_hook=msgpack_to_leaf)
    return leaves

def unpack_tree(treedef, data):
    leaves = unpack_leaf_data(data)
    return jax.tree.unflatten(treedef, leaves)

#def load_leaf_data(file_path):
#    with open(file_path, 'rb') as f:
#        return msgpack.unpack(f)
#
#def load_from_example(example, file_path):
#    leaves = load_leaf_data(file_path)
#    example_leaves, treedef = jax.tree.flatten(example)
#    
#    def rehydrate_leaf(leaf, example_leaf):
#        if isinstance(example_leaf, (np.ndarray, jnp.ndarray)):
#            np_dtype = example_leaf.dtype
#            np_shape = example_leaf.shape
#            if jnp.issubdtype(example_leaf.dtype, jax.dtypes.prng_key):
#                key_data = jrng.key_data(example_leaf)
#                np_dtype = key_data.dtype
#                np_shape = key_data.shape
#            leaf = np.frombuffer(leaf, dtype=np_dtype)
#            leaf = leaf.reshape(np_shape)
#            if isinstance(example_leaf, jnp.ndarray):
#                leaf = jnp.array(leaf)
#                if jnp.issubdtype(example_leaf.dtype, jax.dtypes.prng_key):
#                    leaf = jrng.wrap_key_data(leaf)
#        
#        return leaf
#    
#    leaves = jax.tree.map(rehydrate_leaf, leaves, example_leaves)
#    return jax.tree.unflatten(treedef, leaves)

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
            'f':jnp.arange(24).reshape(3,4,2),
            'g':np.arange(18).reshape(3,3,2),
            'h':np.array([12], dtype=np.int32)[0],
        },
    )
    
    data = pack_leaf_data(thing)
    
    leaves, treedef = jax.tree.flatten(thing)
    
    unpacked_thing = unpack_tree(treedef, data)
    
    #key2 = jrng.key(2345)
    #example = (
    #    key2, {'a':SupParams(), 'b':'hello', 'c':18, 'd':1.2, 'e':[key2, key2]})
    #loaded_obj = load(example, './tmp.data')
    
    breakpoint()

if __name__ == '__main__':
    test_serial()
