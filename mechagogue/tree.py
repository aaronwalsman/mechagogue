import math

import jax
import jax.numpy as jnp
import jax.random as jrng

def tree_len(tree, axis=0):
    return jax.tree.leaves(tree)[0].shape[axis]

def tree_getitem(tree, index):
    def leaf_getitem(leaf):
        return leaf[index]
    return jax.tree.map(leaf_getitem, tree)

def tree_setitem(tree, index, tree_value):
    def leaf_setitem(leaf, leaf_value):
        return leaf.at[index].set(leaf_value)
    return jax.tree.map(leaf_setitem, tree, tree_value)

def tree_additem(tree, index, tree_value):
    def leaf_additem(leaf, leaf_value):
        return leaf.at[index].add(leaf_value)
    return jax.tree.map(leaf_setitem, tree)

def ravel_tree(tree, start_axis=0, end_axis=None):
    def leaf_ravel(leaf):
        c = leaf.shape
        e = end_axis
        if e is None:
            e = len(c)
        return leaf.reshape(*c[:start_axis], -1, *c[e:])
    return jax.tree.map(leaf_ravel, tree)

def tree_key(key, tree_structure):
    keys = tuple(jrng.split(key, tree_structure.num_leaves))
    key_tree = jax.tree.unflatten(tree_structure, keys)
    return key_tree

def shuffle_tree(key, tree, axis=0):
    def shuffle_leaf(leaf):
        return jrng.permutation(key, leaf, axis=axis)
    return jax.tree.map(shuffle_leaf, tree)

def pad_tree(tree, target_elements, axis=0, pad_value=0):
    num_elements = tree_len(tree, axis=axis)
    assert num_elements <= target_elements
    extra_elements = target_elements - num_elements
    def pad_leaf(leaf):
        extra_shape = list(leaf.shape)
        extra_shape[axis] = extra_elements
        extra_data = jnp.full(extra_shape, pad_value)
        return jnp.concatenate((leaf, extra_data), axis=axis)
    
    padded_tree = jax.tree.map(pad_leaf, tree)
    valid = jnp.ones(target_elements, dtype=jnp.bool)
    valid = valid.at[num_elements:].set(False)
    return padded_tree, valid

def pad_tree_batch_size(tree, batch_size, axis=0, pad_value=0):
    num_elements = tree_len(tree, axis=axis)
    padded_num_elements = math.ceil(num_elements / batch_size) * batch_size
    return pad_tree(
        tree, padded_num_elements, axis=axis, pad_value=pad_value)

def clip_tree(tree, target_elements, axis=0):
    num_elements = tree_len(tree, axis=axis)
    assert num_elements >= target_elements
    def clip_leaf(leaf):
        index = [slice(None) for _ in leaf.shape]
        index[axis] = slice(None, target_elements)
        index = tuple(index)
        return leaf[index]
    
    return jax.tree.map(clip_leaf, tree)

def clip_tree_batch_size(tree, batch_size, axis=0):
    num_elements = tree_len(tree, axis=axis)
    clipped_num_elements = math.floor(num_elements / batch_size) * batch_size
    return clip_tree(tree, clipped_num_elements, axis=axis)

def batch_tree(tree, batch_size, axis=0):
    num_elements = tree_len(tree, axis=axis)
    assert num_elements % batch_size == 0, (
        'num_elements is not divisable by batch_size, '
        'use pad_tree_batch_size or clip_tree_batch_size'
    )
    num_batches = num_elements // batch_size
    def batch_leaf(leaf):
        assert leaf.shape[axis] == num_elements
        leading_shape = leaf.shape[:axis]
        trailing_shape = leaf.shape[axis+1:]
        batched_shape = (
            leading_shape + (num_batches, batch_size) + trailing_shape)
        return leaf.reshape(batched_shape)
    
    return jax.tree.map(batch_leaf, tree)
