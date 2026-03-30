from dataclasses import dataclass

from .calc_zone import CalcPlane, CalcVol
from .units import convert_length

WHOLE_ROOM_FLUENCE = "WholeRoomFluence"
EYE_LIMITS = "EyeLimits"
SKIN_LIMITS = "SkinLimits"
STANDARD_ZONE_IDS = [WHOLE_ROOM_FLUENCE, EYE_LIMITS, SKIN_LIMITS]

# Default grid density for standard plane zones (per axis).
DEFAULT_PLANE_NUM_POINTS = 50


@dataclass(frozen=True)
class StandardZoneConfig:
    """Per-standard zone configuration. Only values that vary between standards."""
    height_m: float
    height_ft: float
    eye_calc_mode: str = "eye_worst_case"
    eye_fov_horiz: float | None = None  # Override for eye zone FOV horiz
    skin_calc_mode: str = "planar_normal"


# height_ft values are deliberately rounded to user-friendly numbers
# (e.g. 6.25 ft = 6'3") rather than exact conversions from height_m
_ZONE_CONFIGS = {
    "ul8802": StandardZoneConfig(
        height_m=1.9, height_ft=6.25,
        eye_calc_mode="fluence_rate", eye_fov_horiz=180,
        skin_calc_mode="planar_max",
    ),
}
_DEFAULT_CONFIG = StandardZoneConfig(
    height_m=1.8, height_ft=5.9,
    eye_calc_mode="eye_worst_case",
    skin_calc_mode="planar_normal",
)


def get_zone_config(standard):
    """Get per-standard zone configuration."""
    return _ZONE_CONFIGS.get(standard.value, _DEFAULT_CONFIG)


def create_standard_zones(standard, dims):
    """Create the standard calculation zones for a room."""
    cfg = get_zone_config(standard)
    if dims.units == "feet":
        height = cfg.height_ft
    else:
        height = convert_length("meters", dims.units, cfg.height_m)

    n = DEFAULT_PLANE_NUM_POINTS

    eye_kwargs = {}
    if cfg.eye_fov_horiz is not None:
        eye_kwargs["fov_horiz"] = cfg.eye_fov_horiz

    return [
        CalcVol.from_dims(
            dims=dims, zone_id=WHOLE_ROOM_FLUENCE, name="Whole Room Fluence",
            num_points=(25, 25, 25), display_mode="none",
        ),
        CalcPlane.from_face(
            dims=dims, wall="floor", normal_offset=height,
            zone_id=EYE_LIMITS, name="Eye Dose (8 Hours)",
            calc_mode=cfg.eye_calc_mode,
            dose=True, hours=8, num_points=(n, n),
            **eye_kwargs,
        ),
        CalcPlane.from_face(
            dims=dims, wall="floor", normal_offset=height,
            zone_id=SKIN_LIMITS, name="Skin Dose (8 Hours)",
            calc_mode=cfg.skin_calc_mode,
            dose=True, hours=8, num_points=(n, n),
        ),
    ]
