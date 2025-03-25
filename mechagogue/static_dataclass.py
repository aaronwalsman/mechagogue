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
    
    def add_commandline_args(obj, parser, prefixes=()):
        cls = obj.__class__
        for field in fields(cls):
            name_components = prefixes + (field.name,)
            argname = f'--{"-".join(name_components)}'
            default = getattr(obj, field.name)
            if hasattr(default, 'add_commandline_args'):
                extended_prefixes = prefixes + (field.name,)
                default.add_commandline_args(parser, prefixes=extended_prefixes)
            else:
                parser.add_argument(argname, type=field.type, default=default)
         
        return parser
    
    def update_from_commandline(obj, args, prefixes=()):
        cls = obj.__class__
        constructor_args = {}
        for field in fields(cls):
            name_components = prefixes + (field.name,)
            attrname = "_".join(name_components)
            default = getattr(obj, field.name)
            if hasattr(default, 'update_from_commandline'):
                constructor_args[field.name] = default.update_from_commandline(
                    args, prefixes=name_components)
            else:
                constructor_args[field.name] = getattr(args, attrname)
        
        return cls(**constructor_args)
    
    @classmethod
    def from_commandline(cls, *args, **kwargs):
        obj = cls(*args, **kwargs)
        parser = argparse.ArgumentParser()
        obj.add_commandline_args(parser)
        args = parser.parse_args()
        obj = obj.update_from_commandline(args)
        return obj
       
    cls.tree_flatten = tree_flatten
    cls.tree_unflatten = tree_unflatten
    cls.replace = replace
    cls.add_commandline_args = add_commandline_args
    cls.update_from_commandline = update_from_commandline
    cls.from_commandline = from_commandline
    
    jax.tree_util.register_pytree_node_class(cls)

    return cls

if __name__ == '__main__':
    
    import argparse
    
    @static_dataclass
    class A:
        hello : str = 'world'
    
    @static_dataclass
    class B:
        help_me : str = 'jon_keto'
        int_field : int = 1
        float_field : float = 2.
        a : Any = A('earth')
    
    b = B()
    parser = argparse.ArgumentParser()
    b.add_commandline_args(parser)
    args = parser.parse_args()
    b_commandline = b.from_commandline_args(args)
    
    print('Original:', b)
    print('Commandline:', b_commandline)
