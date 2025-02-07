import inspect

import jax.random as jrng

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
            assert param.default is not inspect.Parameter.empty
        else:
            existing_argnames.add(name)
    
    def wrapped_function(*args, **kwargs):
        assert len(args) == len(argnames)
        wrapped_args = {
            a:arg for a, arg in zip(argnames, args) if a in existing_argnames}
        wrapped_args.update(kwargs)
        return function(**wrapped_args)
    
    return wrapped_function

def split_random_keys(function, n):
    def wrapped_function(key, *args, **kwargs):
        keys = jrng.split(key, n)
        return function(keys, *args, **kwargs)
    
    return wrapped_function
