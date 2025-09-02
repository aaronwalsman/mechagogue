from functools import partial
import itertools
import argparse
from typing import Any
import dataclasses
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
        '''
        field_dict = {
            field.name : getattr(obj, field.name)
            for field in fields(obj)
        }
        field_dict.update(kwargs)
        return cls(**field_dict)
        '''
        return dataclasses.replace(obj, **kwargs)
    
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
    
    cls.STATIC_DATA = True
    cls.tree_flatten = tree_flatten
    cls.tree_unflatten = tree_unflatten
    cls.replace = replace
    cls.override_children = override_children
    cls.override_descendants = override_descendants
    
    jax.tree_util.register_pytree_node_class(cls)

    return cls

def is_static_functions(obj):
    return getattr(obj, 'STATIC_FUNCTIONS', False)

def static_functions(cls):
    
    for name, value in cls.__dict__.items():
        if callable(value):
            setattr(cls, name, staticmethod(value))
    
    cls.STATIC_FUNCTIONS = True
    
    return cls
