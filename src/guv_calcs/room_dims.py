from dataclasses import dataclass, replace
import numpy as np
from .units import LengthUnits


@dataclass(frozen=True, slots=True)
class RoomDimensions:
    x: float
    y: float
    z: float
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    units: "LengthUnits" = LengthUnits.METERS

    @property
    def volume(self) -> float:
        return self.x * self.y * self.z

    @property
    def dimensions(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def faces(self) -> dict:
        # x1, x2, y1, y2, height, ref_surface, direction
        return {
            "floor": (0, self.x, 0, self.y, 0, "xy", 1),
            "ceiling": (0, self.x, 0, self.y, self.z, "xy", -1),
            "south": (0, self.x, 0, self.z, 0, "xz", 1),
            "north": (0, self.x, 0, self.z, self.y, "xz", -1),
            "west": (0, self.y, 0, self.z, 0, "yz", 1),
            "east": (0, self.y, 0, self.z, self.x, "yz", -1),
        }

    def with_(self, *, x=None, y=None, z=None, units=None):
        return replace(
            self,
            x=self.x if x is None else x,
            y=self.y if y is None else y,
            z=self.z if z is None else z,
            units=self.units if units is None else LengthUnits.from_any(units),
        )
