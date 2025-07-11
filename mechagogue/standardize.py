import copy
import inspect
from functools import wraps
from makefun import with_signature

import jax.random as jrng


def standardize_interface(obj, **functions):
    # Handle tuples by creating a new object with the tuple's functions
    if isinstance(obj, tuple):
        # Simple object to hold the functions
        class StandardizedObject:
            pass
        new_obj = StandardizedObject()
        
        # If it's a tuple of (init, forward) functions, extract them
        if len(obj) == 2:
            init_func, forward_func = obj
            new_obj.init = init_func
            new_obj.forward = forward_func
        else:
            # For other tuple types, just use the default functions
            pass
        
        obj = new_obj
    
    obj = copy.deepcopy(obj)
    for function_name, (argnames, function) in functions.items():
        if hasattr(obj, function_name):
            function = getattr(obj, function_name)
            update_function = False
        else:
            assert function is not None, (
                f'standardize_interface speficies argnames '
                f'for "{function_name}", but it does not exist.'
            )
            update_function = True
        if argnames is not None:
            function = standardize_args(function, argnames)
            update_function = True
        if update_function:
            setattr(obj, function_name, staticmethod(function))
    
    return obj


def standardize_args(function, argnames):
    if isinstance(argnames, str):
        argnames = (argnames,)
    signature = inspect.signature(function)
    existing_argnames = set()
    for name, param in signature.parameters.items():
        if (param.kind == inspect.Parameter.VAR_POSITIONAL or
            param.kind == inspect.Parameter.VAR_KEYWORD
        ):
            assert name not in argnames
        elif name not in argnames:
            assert param.default is not inspect.Parameter.empty, (
                f'argument "{name}" is required by {function}')
        else:
            existing_argnames.add(name)
    
    new_signature = ', '.join(argnames)
    @with_signature(f'({new_signature})', func=function)
    #@wraps(function)
    #def wrapped_function(*args, **kwargs):
    def wrapped_function(**kwargs):
        assert all(argname in kwargs for argname in argnames), (
            f'Expected: {argnames} Receieved: {tuple(kwargs.keys())}')
        #assert len(args) == len(argnames), (
        #    f'Received:{args} Expected:{argnames}')
        
        #wrapped_args = {
        #    a:arg for a, arg in zip(argnames, args)
        #    if a in existing_argnames
        #}
        #overlapping_keys = set(wrapped_args.keys()) & set(kwargs.keys())
        #assert not len(overlapping_keys) (
        #    f'args and kwargs contain overlaps {tuple(overlapping_keys)}')
        #wrapped_args.update(kwargs)
        wrapped_args = {
            argname:kwargs[argname] for argname in existing_argnames}
        
        return function(**wrapped_args)
    
    return wrapped_function


def split_random_keys(function, n):
    def wrapped_function(key, *args, **kwargs):
        keys = jrng.split(key, n)
        return function(keys, *args, **kwargs)
    
    return wrapped_function


'''
def ignore_unused_args(function, argnames):
    signature = inspect.signature(function)
    existing_argnames = set()
    for name, param in signature.parameters.items():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            assert name not in argnames
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            assert name not in argnames
            continue
        if name not in argnames:
            assert param.default is not inspect.Parameter.empty, (
                f'argument "{name}" was not provided and has no default!')
        else:
            existing_argnames.add(name)
    
    def wrapped_function(*args, **kwargs):
        assert len(args) == len(argnames)
        wrapped_args = {
            a:arg for a, arg in zip(argnames, args) if a in existing_argnames}
        wrapped_args.update(kwargs)
        return function(**wrapped_args)
    
    return wrapped_function
'''
