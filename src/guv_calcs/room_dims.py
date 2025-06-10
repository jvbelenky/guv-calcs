from dataclasses import dataclass, replace
import numpy as np


@dataclass(frozen=True, slots=True)
class RoomDimensions:
    x: float
    y: float
    z: float
    units: str = "meters"

    def volume(self) -> float:
        return self.x * self.y * self.z

    def dimensions(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def with_(self, *, x=None, y=None, z=None, units=None):
        return replace(
            self,
            x=self.x if x is None else x,
            y=self.y if y is None else y,
            z=self.z if z is None else z,
            units=self.units if units is None else units.lower(),
        )
