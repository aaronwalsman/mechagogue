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
        
        #def parse_tuple(stype, dtype=str):
        #    def parse(s):
        #        return stype(dtype(item) for item in s.split(','))
        #    
        #    return parse
        
        #def parse_dict(dtype, kdtype=str, vdtype=str):
        #    def parse(s):
        #        try:
        #            return dtype(
        #                (kdtype(k), vdtype(v))
        #                for k,v in item.split(':') for item in s.split(',')
        #            )
        #        except ValueError:
        #            raise argparse.ArgumentTypeError(
        #                f'Dictionary must be in the format k1:v1,k2:v2... '
        #                f'where k1 has type {kdtype} and v1 has type {vdtype}'
        #            )
        #    
        #    return parse
        
        for field in fields(cls):
            name_components = prefixes + (field.name,)
            argname = f'--{"-".join(name_components)}'
            default = getattr(obj, field.name)
            if hasattr(default, 'add_commandline_args'):
                extended_prefixes = prefixes + (field.name,)
                default.add_commandline_args(parser, prefixes=extended_prefixes)
            else:
                #if field.type in (str, int, float):
                #    dtype = field.type
                #elif field.type.__origin__ is tuple:
                #    stype = field.type.__origin__
                #    dtype = parse_tuple(stype, *field.type.__args__)
                #elif field.type.__origin__ is dict:
                #    stype = field.type.__origin__
                #    dtype = parse_sequence(stype, *field.dtype.__args__)
                parser.add_argument(argname, type=field.type, default=default)
         
        return parser
    
    def from_args(obj, args, prefixes=()):
        cls = obj.__class__
        constructor_args = {}
        for field in fields(cls):
            name_components = prefixes + (field.name,)
            attrname = "_".join(name_components)
            default = getattr(obj, field.name)
            if hasattr(default, 'from_args'):
                constructor_args[field.name] = default.from_args(
                    args, prefixes=name_components)
            else:
                constructor_args[field.name] = getattr(args, attrname)
        
        return cls(**constructor_args)
       
    cls.tree_flatten = tree_flatten
    cls.tree_unflatten = tree_unflatten
    cls.replace = replace
    cls.add_commandline_args = add_commandline_args
    cls.from_args = from_args
    
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
    b_commandline = b.from_args(args)
    
    print('Original:', b)
    print('Commandline:', b_commandline)
