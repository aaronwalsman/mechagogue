def filter_args(arg_format, *args):
    return tuple(arg for f, arg in zip(arg_format, args) if f != '_')
