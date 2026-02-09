import inspect


def init_from_dict(cls, data: dict):
    """Construct cls from dict, filtering to valid __init__ params."""
    keys = list(inspect.signature(cls.__init__).parameters.keys())[1:]
    return cls(**{k: v for k, v in data.items() if k in keys})
