from functools import partial
import itertools
import argparse
from typing import Any
from dataclasses import dataclass, fields, is_dataclass

import jax

def is_static_data(obj):
    return getattr(obj, 'STATIC_DATA', False)

def static_data(cls):
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
    
    def override_children(obj, recurse=False):
        updates = {}
        for field in fields(obj):
            child = getattr(obj, field.name)
            if is_static_data(child):
                child_updates = {}
                for child_field in fields(child):
                    if hasattr(obj, child_field.name):
                        child_updates[child_field.name] = getattr(
                            obj, child_field.name)
                        
                updated_child = child.replace(**child_updates)
                if recurse:
                    updated_child = updated_child.override_children(
                        recurse=True)
                updates[field.name] = updated_child
        
        return obj.replace(**updates)
    
    def override_descendants(obj):
        return obj.override_children(recurse=True)
    
    def sweep(obj, **kwargs):
        results = []
        for value_combination in zip(*kwargs.values()):
            replace_kwargs = dict(zip(kwargs.keys(), value_combination))
            results.append(obj.replace(**replace_kwargs))
        
        return results
    
    def sweep_combos(obj, **kwargs):
        results = []
        for value_combination in itertools.product(kwargs.values()):
            replace_kwargs = dict(zip(kwargs.keys(), value_combination))
            results.append(obj.replace(**replace_kwargs))
        
        return results
    
    cls.STATIC_DATA = True
    cls.tree_flatten = tree_flatten
    cls.tree_unflatten = tree_unflatten
    cls.replace = replace
    cls.override_children = override_children
    cls.override_descendants = override_descendants
    cls.sweep = sweep_list
    cls.sweep_combos = sweep_combos
    
    jax.tree_util.register_pytree_node_class(cls)

    return cls

def static_functions(cls):
    for name, value in cls.__dict__.items():
        if callable(value):
            setattr(cls, name, staticmethod(value))
    
    return cls
