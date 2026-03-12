from enum import StrEnum
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class PlaneCalcSpec:
    horiz: bool
    vert: bool
    use_normal: bool
    fov_vert: float
    fov_horiz: float
    view_direction: tuple | None = None
    view_target: tuple | None = None

    def matches(self, other: "PlaneCalcSpec") -> bool:
        """Check if flags match. For view params, only checks None vs not-None."""
        return (
            self.horiz == other.horiz
            and self.vert == other.vert
            and self.use_normal == other.use_normal
            and self.fov_vert == other.fov_vert
            and self.fov_horiz == other.fov_horiz
            and (self.view_direction is None) == (other.view_direction is None)
            and (self.view_target is None) == (other.view_target is None)
        )


class PlaneCalcMode(StrEnum):
    FLUENCE_RATE = "fluence_rate"
    PLANAR_NORMAL = "planar_normal"
    PLANAR_MAX = "planar_max"
    VERTICAL = "vertical"
    VERTICAL_DIR = "vertical_dir"
    EYE_WORST_CASE = "eye_worst_case"
    EYE_DIRECTIONAL = "eye_directional"
    EYE_TARGET = "eye_target"
    CUSTOM = "custom"

    @classmethod
    def from_token(cls, token: str) -> "PlaneCalcMode":
        token = re.sub(r"[\s_\-]+", "_", str(token).strip().lower())
        aliases = {
            "fluence": cls.FLUENCE_RATE,
            "fluence_rate": cls.FLUENCE_RATE,
            "all_angles": cls.FLUENCE_RATE,
            "normal": cls.PLANAR_NORMAL,
            "planar_normal": cls.PLANAR_NORMAL,
            "planar_max": cls.PLANAR_MAX,
            "max": cls.PLANAR_MAX,
            "vertical": cls.VERTICAL,
            "vertical_dir": cls.VERTICAL_DIR,
            "eye": cls.EYE_WORST_CASE,
            "eye_worst_case": cls.EYE_WORST_CASE,
            "eye_directional": cls.EYE_DIRECTIONAL,
            "directional": cls.EYE_DIRECTIONAL,
            "eye_target": cls.EYE_TARGET,
            "target": cls.EYE_TARGET,
            "custom": cls.CUSTOM,
        }
        try:
            return aliases[token]
        except KeyError:
            raise ValueError(f"Unknown plane calc_mode {token!r}")

    @classmethod
    def from_flags(
        cls,
        horiz: bool,
        vert: bool,
        use_normal: bool,
        fov_vert: float,
        fov_horiz: float,
        view_direction: tuple | None = None,
        view_target: tuple | None = None,
    ) -> "PlaneCalcMode":
        current = PlaneCalcSpec(
            horiz=bool(horiz),
            vert=bool(vert),
            use_normal=bool(use_normal),
            fov_vert=float(fov_vert),
            fov_horiz=float(fov_horiz),
            view_direction=view_direction,
            view_target=view_target,
        )
        for pct in cls:
            if pct is cls.CUSTOM:
                continue
            if pct.spec.matches(current):
                return pct
        return cls.CUSTOM

    @property
    def spec(self) -> PlaneCalcSpec:
        specs = {
            PlaneCalcMode.FLUENCE_RATE: PlaneCalcSpec(
                horiz=False, vert=False, use_normal=False,
                fov_vert=180.0, fov_horiz=360.0,
            ),
            PlaneCalcMode.PLANAR_NORMAL: PlaneCalcSpec(
                horiz=True, vert=False, use_normal=True,
                fov_vert=180.0, fov_horiz=360.0,
            ),
            PlaneCalcMode.PLANAR_MAX: PlaneCalcSpec(
                horiz=False, vert=False, use_normal=True,
                fov_vert=180.0, fov_horiz=360.0,
            ),
            PlaneCalcMode.VERTICAL: PlaneCalcSpec(
                horiz=False, vert=True, use_normal=False,
                fov_vert=180.0, fov_horiz=360.0,
            ),
            PlaneCalcMode.VERTICAL_DIR: PlaneCalcSpec(
                horiz=False, vert=True, use_normal=True,
                fov_vert=180.0, fov_horiz=360.0,
            ),
            PlaneCalcMode.EYE_WORST_CASE: PlaneCalcSpec(
                horiz=False, vert=True, use_normal=False,
                fov_vert=80.0, fov_horiz=120.0,
            ),
            PlaneCalcMode.EYE_DIRECTIONAL: PlaneCalcSpec(
                horiz=True, vert=False, use_normal=True,
                fov_vert=80.0, fov_horiz=120.0,
                view_direction=(0.0, 1.0, 0.0),
            ),
            PlaneCalcMode.EYE_TARGET: PlaneCalcSpec(
                horiz=True, vert=False, use_normal=True,
                fov_vert=80.0, fov_horiz=120.0,
                view_target=(0.0, 0.0, 0.0),
            ),
        }
        return specs.get(self, PlaneCalcSpec(
            horiz=False, vert=False, use_normal=True,
            fov_vert=180.0, fov_horiz=360.0,
        ))
