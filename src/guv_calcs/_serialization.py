import inspect


def init_from_dict(cls, data: dict):
    """Construct cls from dict, filtering to valid __init__ params."""
    keys = list(inspect.signature(cls.__init__).parameters.keys())[1:]
    return cls(**{k: v for k, v in data.items() if k in keys})


def migrate_lamp_dict(data: dict) -> dict:
    """Apply legacy migrations to a lamp dict."""
    data = dict(data)
    if "spectra" in data and "spectrum" not in data:
        data["spectrum"] = data.pop("spectra")
    if "depth" in data and "fixture" not in data and "height" not in data:
        legacy_depth = data.pop("depth", 0.0)
        data.setdefault("housing_width", data.get("width"))
        data.setdefault("housing_length", data.get("length"))
        data.setdefault("housing_height", legacy_depth)
    return data


def deserialize_geometry(data: dict, polygon_cls, rect_cls) -> dict:
    """Deserialize a nested 'geometry' dict into a typed grid object."""
    data = dict(data)
    geom_data = data.pop("geometry", None)
    if geom_data is not None:
        cls = polygon_cls if "polygon" in geom_data else rect_cls
        data["geometry"] = cls.from_dict(geom_data)
    return data
