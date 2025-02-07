from dataclasses import dataclass, fields, is_dataclass

from jax import tree_util

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
    
    cls.tree_flatten = tree_flatten
    cls.tree_unflatten = tree_unflatten
    
    tree_util.register_pytree_node_class(cls)

    return cls
