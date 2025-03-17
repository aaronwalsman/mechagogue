import argparse
from dataclasses import fields, is_dataclass

from mechagogue.serial import load_example_data

def _add_commandline_args(obj, parser, prefixes=()):
    cls = obj.__class__
    assert is_dataclass(cls)
    for field in fields(cls):
        name_components = prefixes + (field.name,)
        argname = f'--{"-".join(name_components)}'
        default = getattr(obj, field.name)
        if is_dataclass(default.__class__):
            extended_prefixes = prefixes + (field.name,)
            _add_commandline_args(default, parser, prefixes=extended_prefixes)
        else:
            # this causes problems if the bool is True by default
            #if field.type == bool:
            #    parser.add_argument(
            #        argname, action='store_true')
            #else:
            parser.add_argument(argname, type=field.type, default=default)
    
    return parser

def _update_from_commandline(obj, args, prefixes=()):
    cls = obj.__class__
    assert is_dataclass(cls)
    constructor_args = {}
    for field in fields(cls):
        name_components = prefixes + (field.name,)
        attrname = "_".join(name_components)
        default = getattr(obj, field.name)
        if is_dataclass(default.__class__):
            constructor_args[field.name] = _update_from_commandline(
                default, args, prefixes=name_components)
        else:
            constructor_args[field.name] = getattr(args, attrname)

    return cls(**constructor_args)

def commandline_interface(cls):
    def from_commandline(obj):
        # make the parser and add the load argument
        parser = argparse.ArgumentParser()
        parser.add_argument('--load', type=str, default=None)
        
        # if --load was specified, load the specified file
        load_args, other_args = parser.parse_known_args()
        if load_args.load is not None:
            obj = load_example_data(obj, load_args.load)
        
        # add the other commandline args and parse them
        _add_commandline_args(obj, parser)
        commandline_args = parser.parse_args()
        
        # update the (possibly loaded) parameters from the commandline
        obj = _update_from_commandline(obj, commandline_args)
        return obj
    
    cls.from_commandline = from_commandline
    
    return cls
