import itertools
import argparse
from typing import Any
from dataclasses import dataclass, fields, is_dataclass

import jax

def static_dataclass(cls):
    cls = dataclass(frozen=True)(cls)

    def tree_flatten(obj):
        children = tuple(getattr(obj, field.name) for field in fields(obj))
        aux_data = None  # No auxiliary data needed
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        if not is_dataclass(cls):
            raise TypeError("class is not a dataclass")
        
        field_names = (field.name for field in fields(cls))
        return cls(**dict(zip(field_names, children)))
    
    def replace(obj, **kwargs):
        field_dict = {
            field.name : getattr(obj, field.name)
            for field in fields(obj)
        }
        field_dict.update(kwargs)
        return cls(**field_dict)
    
    def sweep(obj, **kwargs):
        results = []
        for value_combination in zip(*kwargs.values()):
            replace_kwargs = dict(zip(kwargs.keys(), value_combination))
            results.append(obj.replace(**replace_kwargs))
        
        return results
    
    def sweep_all_combinations(obj, **kwargs):
        results = []
        for value_combination in itertools.product(kwargs.values()):
            replace_kwargs = dict(zip(kwargs.keys(), value_combination))
            results.append(obj.replace(**replace_kwargs))
        
        return results
    
    cls.tree_flatten = tree_flatten
    cls.tree_unflatten = tree_unflatten
    cls.replace = replace
    
    jax.tree_util.register_pytree_node_class(cls)

    return cls
