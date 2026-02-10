"""Factory functions for standard calculation zones."""

from .calc_zone import CalcPlane, CalcVol
from .units import convert_length

# Canonical zone IDs
WHOLE_ROOM_FLUENCE = "WholeRoomFluence"
EYE_LIMITS = "EyeLimits"
SKIN_LIMITS = "SkinLimits"
STANDARD_ZONE_IDS = [WHOLE_ROOM_FLUENCE, EYE_LIMITS, SKIN_LIMITS]


def create_standard_zones(standard, dims):
    """Create the standard calculation zones for a room."""
    flags = standard.flags(dims.units)
    spacing = convert_length("meters", dims.units, 0.1, 0.1)
    return [
        _whole_room_fluence(dims),
        _eye_limits(dims, flags, spacing),
        _skin_limits(dims, flags, spacing),
    ]


def update_standard_zones(standard, calc_zones, dims, preserve_spacing=True):
    """Update existing standard zones for changed standard or dimensions."""
    flags = standard.flags(dims.units)
    x_min, y_min, x_max, y_max = dims.polygon.bounding_box

    if SKIN_LIMITS in calc_zones:
        zone = calc_zones[SKIN_LIMITS]
        zone.set_dimensions(
            x1=x_min, x2=x_max, y1=y_min, y2=y_max,
            preserve_spacing=preserve_spacing,
        )
        zone.set_height(height=flags["height"])
        zone.horiz = flags["skin_horiz"]

    if EYE_LIMITS in calc_zones:
        zone = calc_zones[EYE_LIMITS]
        zone.set_dimensions(
            x1=x_min, x2=x_max, y1=y_min, y2=y_max,
            preserve_spacing=preserve_spacing,
        )
        zone.set_height(height=flags["height"])
        zone.fov_vert = flags["fov_vert"]
        zone.vert = flags["eye_vert"]

    if WHOLE_ROOM_FLUENCE in calc_zones:
        zone = calc_zones[WHOLE_ROOM_FLUENCE]
        zone.set_dimensions(
            x1=x_min, x2=x_max,
            y1=y_min, y2=y_max,
            z2=dims.z,
            preserve_spacing=preserve_spacing,
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
