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


class PlaneCalcType(StrEnum):
    FLUENCE = "fluence"
    PLANAR_NORMAL = "planar_normal"
    PLANAR_MAX = "planar_max"
    EYE_EXPOSURE = "eye_exposure"
    CUSTOM = "custom"

    @classmethod
    def from_token(cls, token: str) -> "PlaneCalcType":
        token = re.sub(r"[\s_\-]+", "_", str(token).strip().lower())
        aliases = {
            "fluence": cls.FLUENCE,
            "all_angles": cls.FLUENCE,
            "normal": cls.PLANAR_NORMAL,
            "planar_normal": cls.PLANAR_NORMAL,
            "planar_max": cls.PLANAR_MAX,
            "max": cls.PLANAR_MAX,
            "eye": cls.EYE_EXPOSURE,
            "eye_exposure": cls.EYE_EXPOSURE,
            "custom": cls.CUSTOM,
        }
        try:
            return aliases[token]
        except KeyError:
            raise ValueError(f"Unknown plane calc_type {token!r}")

    @property
    def spec(self) -> PlaneCalcSpec:
        if self is PlaneCalcType.FLUENCE:
            return PlaneCalcSpec(
                horiz=False,
                vert=False,
                use_normal=False,
                fov_vert=180.0,
                fov_horiz=360.0,
            )
        if self is PlaneCalcType.PLANAR_NORMAL:
            return PlaneCalcSpec(
                horiz=True,
                vert=False,
                use_normal=True,
                fov_vert=180.0,
                fov_horiz=360.0,
            )
        if self is PlaneCalcType.PLANAR_MAX:
            return PlaneCalcSpec(
                horiz=False,
                vert=False,
                use_normal=True,
                fov_vert=180.0,
                fov_horiz=360.0,
            )
        if self is PlaneCalcType.EYE_EXPOSURE:
            return PlaneCalcSpec(
                horiz=False,
                vert=True,
                use_normal=False,
                fov_vert=80.0,
                fov_horiz=180.0,
            )
        # CUSTOM has no spec
        return PlaneCalcSpec(
            horiz=False, vert=False, use_normal=True, fov_vert=180.0, fov_horiz=360.0
        )
