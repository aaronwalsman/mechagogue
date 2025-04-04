from typing import Tuple, get_origin, get_args

import argparse
from dataclasses import fields, is_dataclass

from mechagogue.serial import load_example_data

def _annotation_type_parser(annotation):
    if get_origin(annotation) is tuple:
        return _tuple_parser(annotation)
    elif (
        annotation is str or
        annotation is int or
        annotation is float or
        annotation is bool,
    ):
        return annotation
    else:
        raise Exception(f'Unsupported commandline type: {annotation}')

def _tuple_parser(annotation):
    args = get_args(annotation)
    arg_parsers = [_annotation_type_parser(arg) for arg in args]
    def parser(value):
        components = value.split(',')
        assert len(components) == len(args)
        return tuple(
            arg_parser(component)
            for arg_parser, component in zip(arg_parsers, components)
        )
    
    return parser

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
            #if isinstance(field.type, Tuple):
            #    breakpoint()
            #if get_origin(field.type) is tuple:
            #    parser_type = _tuple_parser
            #else:
            #    parser_type = field.type
            type_parser = _annotation_type_parser(field.type)
            parser.add_argument(argname, type=type_parser, default=default)
    
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
