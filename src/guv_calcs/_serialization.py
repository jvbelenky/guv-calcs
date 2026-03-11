import inspect
from importlib import resources

from photompy import IESFile

from .lamp.lamp_configs import LAMP_CONFIGS


def init_from_dict(cls, data: dict):
    """Construct cls from dict, filtering to valid __init__ params."""
    keys = list(inspect.signature(cls.__init__).parameters.keys())[1:]
    return cls(**{k: v for k, v in data.items() if k in keys})


# ---------------------------------------------------------------------------
# Preset identification from IES data (for legacy save migration)
# ---------------------------------------------------------------------------

def _build_preset_lookups():
    """Build LUMCAT/LUMINAIRE -> preset and fingerprint -> preset maps."""
    header_map: dict[str, str] = {}
    fp_map: dict[bytes, str] = {}

    for preset_id, config in LAMP_CONFIGS.items():
        try:
            path = resources.files("guv_calcs.data.lamp_data").joinpath(config["ies_file"])
            ies = IESFile.read(path)
            kw = getattr(ies.header, "keywords", {})
            for field in ("LUMCAT", "LUMINAIRE"):
                val = kw.get(field)
                if val and val != "Unknown":
                    header_map[val] = preset_id
            fp_map[ies.photometry.to_fingerprint()] = preset_id
        except Exception:
            pass

    return header_map, fp_map


_HEADER_TO_PRESET, _FINGERPRINT_TO_PRESET = _build_preset_lookups()


def identify_preset(lamp) -> str | None:
    """Try to identify a preset from a Lamp's loaded IES data.

    Checks header keywords (LUMCAT, LUMINAIRE) first, then falls back
    to photometry fingerprint matching.
    """
    if lamp.ies is None:
        return None

    kw = getattr(lamp.ies.header, "keywords", {})
    for field in ("LUMCAT", "LUMINAIRE"):
        val = kw.get(field)
        if val and val in _HEADER_TO_PRESET:
            return _HEADER_TO_PRESET[val]

    try:
        fp = lamp.ies.photometry.to_fingerprint()
        if fp in _FINGERPRINT_TO_PRESET:
            return _FINGERPRINT_TO_PRESET[fp]
    except Exception:
        pass

    return None


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
        data["surfaces"] = {
            k: {"R": v.R, "T": v.T,
                "x_spacing": v.plane.x_spacing, "y_spacing": v.plane.y_spacing}
            for k, v in surfaces.items()
        }

    return data


def migrate_zone_dict(data: dict) -> dict:
    """Apply legacy migrations to a zone dict."""
    data = dict(data)
    if "hours" in data and "exposure_time" not in data:
        data["seconds"] = data.pop("hours") * 3600
    elif "exposure_time" in data:
        data["seconds"] = data.pop("exposure_time")
    # Migrate legacy show_values -> display_mode
    if "show_values" in data and "display_mode" not in data:
        data["display_mode"] = "heatmap" if data["show_values"] else "none"
    data.pop("show_values", None)
    # colormap is no longer a zone param (room-level only)
    data.pop("colormap", None)
    return data


def deserialize_geometry(data: dict, grid_cls) -> dict:
    """Deserialize a nested 'geometry' dict into a typed grid object."""
    data = dict(data)
    geom_data = data.pop("geometry", None)
    if geom_data is not None:
        if isinstance(geom_data, dict):
            data["geometry"] = grid_cls.from_dict(geom_data)
        else:
            data["geometry"] = geom_data
    return data


def migrate_legacy_zone_geometry(data: dict, grid_cls) -> dict:
    """Construct a geometry object from legacy flat zone params.

    Legacy .guv files (pre-0.5) store spatial params (x1, x2, y1, y2,
    height, offset, etc.) directly on the zone dict rather than in a
    nested 'geometry' dict.  This function detects that case and builds
    the appropriate grid object so that ``from_dict`` can proceed.
    """
    from .geometry import SurfaceGrid, VolumeGrid

    if "geometry" in data:
        return data

    # Only migrate if we have legacy spatial keys
    if "x1" not in data:
        return data

    data = dict(data)

    if grid_cls is SurfaceGrid:
        x1 = data.pop("x1", 0.0)
        x2 = data.pop("x2", 6.0)
        y1 = data.pop("y1", 0.0)
        y2 = data.pop("y2", 4.0)
        height = data.pop("height", 0.0)
        ref_surface = data.pop("ref_surface", "xy")
        direction = data.pop("direction", 1)
        offset = data.pop("offset", True)

        grid_kwargs = {"offset": offset}

        num_x = data.pop("num_x", None)
        num_y = data.pop("num_y", None)
        x_spacing = data.pop("x_spacing", None)
        y_spacing = data.pop("y_spacing", None)

        if num_x is not None and num_y is not None:
            grid_kwargs["num_points_init"] = (num_x, num_y)
        elif x_spacing is not None and y_spacing is not None:
            grid_kwargs["spacing_init"] = (x_spacing, y_spacing)

        data["geometry"] = SurfaceGrid.from_legacy(
            mins=(x1, y1),
            maxs=(x2, y2),
            height=height,
            ref_surface=ref_surface,
            direction=direction if direction is not None else 1,
            **grid_kwargs,
        )

    elif grid_cls is VolumeGrid:
        x1 = data.pop("x1", 0.0)
        x2 = data.pop("x2", 6.0)
        y1 = data.pop("y1", 0.0)
        y2 = data.pop("y2", 4.0)
        z1 = data.pop("z1", 0.0)
        z2 = data.pop("z2", 2.7)
        offset = data.pop("offset", True)

        grid_kwargs = {"offset": offset}

        num_x = data.pop("num_x", None)
        num_y = data.pop("num_y", None)
        num_z = data.pop("num_z", None)
        x_spacing = data.pop("x_spacing", None)
        y_spacing = data.pop("y_spacing", None)
        z_spacing = data.pop("z_spacing", None)

        if num_x is not None and num_y is not None and num_z is not None:
            grid_kwargs["num_points_init"] = (num_x, num_y, num_z)
        elif x_spacing is not None and y_spacing is not None and z_spacing is not None:
            grid_kwargs["spacing_init"] = (x_spacing, y_spacing, z_spacing)

        data["geometry"] = VolumeGrid.from_legacy(
            mins=(x1, y1, z1),
            maxs=(x2, y2, z2),
            **grid_kwargs,
        )

    return data


def migrate_surface_grid_dict(data: dict) -> dict:
    """Migrate legacy PlaneGrid dicts to SurfaceGrid format."""
    from .geometry import Polygon2D

    data = dict(data)
    if "polygon" not in data and "spans" in data:
        spans = data.pop("spans")
        data["polygon"] = Polygon2D.rectangle(spans[0], spans[1])
    elif "polygon" in data and isinstance(data["polygon"], dict):
        data["polygon"] = Polygon2D.from_dict(data["polygon"])
    data.pop("spans", None)
    data.pop("height", None)
    data.pop("direction", None)
    # Rename old serialization key
    if "spacing" not in data and "spacing_init" in data:
        data["spacing_init"] = data.pop("spacing_init")  # already correct field name
    elif "spacing" in data and "spacing_init" not in data:
        data["spacing_init"] = data.pop("spacing")
    return data


def migrate_volume_grid_dict(data: dict) -> dict:
    """Migrate legacy VolGrid dicts to VolumeGrid format."""
    from .geometry import Polygon2D

    data = dict(data)
    if "polygon" not in data and "spans" in data:
        spans = data.pop("spans")
        data["polygon"] = Polygon2D.rectangle(spans[0], spans[1])
        data["depth"] = spans[2]
    elif "polygon" not in data and "origin" in data:
        data["polygon"] = Polygon2D.rectangle(1.0, 1.0)
        data.setdefault("depth", 2.7)
    elif "polygon" in data and isinstance(data["polygon"], dict):
        data["polygon"] = Polygon2D.from_dict(data["polygon"])
    data.pop("spans", None)
    # Rename old serialization key
    if "spacing" not in data and "spacing_init" in data:
        data["spacing_init"] = data.pop("spacing_init")  # already correct field name
    elif "spacing" in data and "spacing_init" not in data:
        data["spacing_init"] = data.pop("spacing")
    return data
