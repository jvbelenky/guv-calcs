"""Factory functions for standard calculation zones."""

from .calc_zone import CalcPlane, CalcVol
from .units import convert_length

# Canonical zone IDs
WHOLE_ROOM_FLUENCE = "WholeRoomFluence"
EYE_LIMITS = "EyeLimits"
SKIN_LIMITS = "SkinLimits"
STANDARD_ZONE_IDS = [WHOLE_ROOM_FLUENCE, EYE_LIMITS, SKIN_LIMITS]

# Cap standard zone grids at 200 points per dimension (200×200 = 40K pts per plane)
MAX_POINTS_PER_DIM = 200


def _standard_zone_spacing(span, base_spacing):
    """Compute zone spacing for a given span, capping at MAX_POINTS_PER_DIM."""
    if span <= 0:
        return base_spacing
    if span < base_spacing:
        return span / 10
    if span / base_spacing + 1 <= MAX_POINTS_PER_DIM:
        return base_spacing
    return span / (MAX_POINTS_PER_DIM - 1)


def create_standard_zones(standard, dims):
    """Create the standard calculation zones for a room."""
    flags = standard.flags(dims.units)
    base_spacing = convert_length("meters", dims.units, 0.1)
    x_min, y_min, x_max, y_max = dims.polygon.bounding_box
    spacing = (
        _standard_zone_spacing(x_max - x_min, base_spacing),
        _standard_zone_spacing(y_max - y_min, base_spacing),
    )
    return [
        _whole_room_fluence(dims),
        _eye_limits(dims, flags, spacing),
        _skin_limits(dims, flags, spacing),
    ]


def update_standard_zones(standard, calc_zones, dims):
    """Update existing standard zones for changed standard or dimensions.

    Always recomputes correct spacing for the current dimensions to prevent
    grid explosion when rooms are resized.
    """
    flags = standard.flags(dims.units)
    x_min, y_min, x_max, y_max = dims.polygon.bounding_box
    base_spacing = convert_length("meters", dims.units, 0.1)
    x_sp = _standard_zone_spacing(x_max - x_min, base_spacing)
    y_sp = _standard_zone_spacing(y_max - y_min, base_spacing)

    for zone_id in [SKIN_LIMITS, EYE_LIMITS]:
        if zone_id not in calc_zones:
            continue
        zone = calc_zones[zone_id]
        # preserve_spacing=False preserves current num_points (safe, no explosion),
        # then set_spacing applies the correct spacing for the new dimensions.
        zone.set_dimensions(
            x1=x_min, x2=x_max, y1=y_min, y2=y_max,
            preserve_spacing=False,
        )
        zone.set_spacing(x_spacing=x_sp, y_spacing=y_sp)
        zone.set_height(height=flags["height"])
        if zone_id == EYE_LIMITS:
            zone.fov_vert = flags["fov_vert"]
            zone.vert = flags["eye_vert"]
        else:
            zone.horiz = flags["skin_horiz"]

    if WHOLE_ROOM_FLUENCE in calc_zones:
        zone = calc_zones[WHOLE_ROOM_FLUENCE]
        zone.set_dimensions(
            x1=x_min, x2=x_max,
            y1=y_min, y2=y_max,
            z2=dims.z,
            preserve_spacing=False,
        )
        # Reset to fixed 25×25×25 grid.  Must clear spacing_init because
        # it takes priority over num_points_init in Axis1D.
        zone.geometry = zone.geometry.update(
            num_points_init=(25, 25, 25), spacing_init=None
        )


# --- individual zone builders ---

def _whole_room_fluence(dims):
    return CalcVol.from_dims(
        dims=dims, zone_id=WHOLE_ROOM_FLUENCE, name="Whole Room Fluence",
        show_values=False, num_points=(25, 25, 25),
    )


def _eye_limits(dims, flags, spacing):
    return CalcPlane.from_face(
        dims=dims, wall="floor", normal_offset=flags["height"],
        zone_id=EYE_LIMITS, name="Eye Dose (8 Hours)",
        dose=True, hours=8, use_normal=False,
        vert=flags["eye_vert"], fov_vert=flags["fov_vert"], spacing=spacing,
    )


def _skin_limits(dims, flags, spacing):
    return CalcPlane.from_face(
        dims=dims, wall="floor", normal_offset=flags["height"],
        zone_id=SKIN_LIMITS, name="Skin Dose (8 Hours)",
        dose=True, hours=8, use_normal=False,
        horiz=flags["skin_horiz"], spacing=spacing,
    )
