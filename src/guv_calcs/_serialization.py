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


def migrate_room_dict(data: dict, saved_version: str) -> dict:
    """Apply legacy migrations to a room dict."""
    from packaging.version import Version

    data = dict(data)

    if Version(saved_version) < Version("0.4.33"):
        from .geometry import Polygon2D
        from .reflectance import RoomDimensions, init_room_surfaces

        dims = RoomDimensions(
            polygon=Polygon2D.rectangle(
                data.get("x", 6.0),
                data.get("y", 4.0),
            ),
            z=data.get("z", 2.7),
        )

        # Extract reflectances: try new key first, then legacy individual keys
        reflectances = data.get("reflectances")
        if reflectances is None:
            legacy = {}
            for surface in ["ceiling", "north", "east", "south", "west", "floor"]:
                key = f"reflectance_{surface}"
                if key in data:
                    legacy[surface] = data[key]
            if legacy:
                reflectances = legacy

        surfaces = init_room_surfaces(
            dims=dims,
            reflectances=reflectances,
            x_spacings=data.get("x_spacings"),
            y_spacings=data.get("y_spacings"),
        )
        data["surfaces"] = {k: v.to_dict() for k, v in surfaces.items()}

    return data


def migrate_zone_dict(data: dict) -> dict:
    """Migrate old 'hours' key to 'seconds' for CalcZone init."""
    data = dict(data)
    if "hours" in data and "exposure_time" not in data:
        data["seconds"] = data.pop("hours") * 3600
    elif "exposure_time" in data:
        data["seconds"] = data.pop("exposure_time")
    return data


def deserialize_geometry(data: dict, polygon_cls, rect_cls) -> dict:
    """Deserialize a nested 'geometry' dict into a typed grid object."""
    data = dict(data)
    geom_data = data.pop("geometry", None)
    if geom_data is not None:
        cls = polygon_cls if "polygon" in geom_data else rect_cls
        data["geometry"] = cls.from_dict(geom_data)
    return data
