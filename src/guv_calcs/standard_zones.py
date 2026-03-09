from dataclasses import dataclass

from .calc_zone import CalcPlane, CalcVol
from .units import convert_length

WHOLE_ROOM_FLUENCE = "WholeRoomFluence"
EYE_LIMITS = "EyeLimits"
SKIN_LIMITS = "SkinLimits"
STANDARD_ZONE_IDS = [WHOLE_ROOM_FLUENCE, EYE_LIMITS, SKIN_LIMITS]
MAX_POINTS_PER_DIM = 200


@dataclass(frozen=True)
class StandardZoneConfig:
    """Per-standard zone configuration. Only values that vary between standards."""
    height_m: float
    height_ft: float
    eye_vert: bool
    eye_fov_vert: float
    skin_horiz: bool


# height_ft values are deliberately rounded to user-friendly numbers
# (e.g. 6.25 ft = 6'3") rather than exact conversions from height_m
_ZONE_CONFIGS = {
    "ul8802": StandardZoneConfig(height_m=1.9, height_ft=6.25, eye_vert=False, eye_fov_vert=180, skin_horiz=False),
}
_DEFAULT_CONFIG = StandardZoneConfig(height_m=1.8, height_ft=5.9, eye_vert=True, eye_fov_vert=80, skin_horiz=True)


def get_zone_config(standard):
    """Get per-standard zone configuration."""
    return _ZONE_CONFIGS.get(standard.value, _DEFAULT_CONFIG)


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
    cfg = get_zone_config(standard)
    if dims.units == "feet":
        height = cfg.height_ft
    else:
        height = convert_length("meters", dims.units, cfg.height_m)
    base_spacing = convert_length("meters", dims.units, 0.1)
    x_min, y_min, x_max, y_max = dims.polygon.bounding_box
    spacing = (
        _standard_zone_spacing(x_max - x_min, base_spacing),
        _standard_zone_spacing(y_max - y_min, base_spacing),
    )
    return [
        CalcVol.from_dims(
            dims=dims, zone_id=WHOLE_ROOM_FLUENCE, name="Whole Room Fluence",
            num_points=(25, 25, 25), display_mode="none",
        ),
        CalcPlane.from_face(
            dims=dims, wall="floor", normal_offset=height,
            zone_id=EYE_LIMITS, name="Eye Dose (8 Hours)",
            dose=True, hours=8, use_normal=False, spacing=spacing,
            vert=cfg.eye_vert, horiz=False,
            fov_vert=cfg.eye_fov_vert, fov_horiz=180,
        ),
        CalcPlane.from_face(
            dims=dims, wall="floor", normal_offset=height,
            zone_id=SKIN_LIMITS, name="Skin Dose (8 Hours)",
            dose=True, hours=8, use_normal=False, spacing=spacing,
            vert=False, horiz=cfg.skin_horiz,
            fov_vert=180, fov_horiz=360,
        ),
    ]
