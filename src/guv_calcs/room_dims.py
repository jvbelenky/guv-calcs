from dataclasses import dataclass, replace, field
import numpy as np
from .units import LengthUnits, convert_length
from .polygon import Polygon2D


@dataclass(frozen=True, slots=True)
class RoomDimensions:
    x: float
    y: float
    z: float
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    units: "LengthUnits" = LengthUnits.METERS

    @property
    def dimensions(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def volume(self) -> float:
        return self.x * self.y * self.z

    @property
    def cubic_meters(self) -> float:
        x, y, z = convert_length(self.units, LengthUnits.METERS, self.dimensions)
        return x * y * z

    @property
    def cubic_feet(self) -> float:
        x, y, z = convert_length(self.units, LengthUnits.FEET, self.dimensions)
        return x * y * z

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

    @property
    def is_polygon(self) -> bool:
        return False


@dataclass(frozen=True)
class PolygonRoomDimensions:
    """
    Room dimensions defined by a 2D polygon floor plan with uniform ceiling height.

    Walls are automatically generated from polygon edges.
    """

    polygon: Polygon2D
    z: float
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    units: LengthUnits = LengthUnits.METERS
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.polygon, Polygon2D):
            # Allow list/tuple of vertices as input
            poly = Polygon2D(vertices=tuple(tuple(v) for v in self.polygon))
            object.__setattr__(self, "polygon", poly)

    @property
    def is_polygon(self) -> bool:
        return True

    @property
    def x(self) -> float:
        """Bounding box x dimension."""
        return self.polygon.x

    @property
    def y(self) -> float:
        """Bounding box y dimension."""
        return self.polygon.y

    @property
    def dimensions(self) -> np.ndarray:
        """Bounding box dimensions."""
        return np.array([self.x, self.y, self.z])

    @property
    def volume(self) -> float:
        """Volume based on polygon area times height."""
        return self.polygon.area * self.z

    @property
    def cubic_meters(self) -> float:
        # Convert polygon area and height to meters
        dims = convert_length(self.units, LengthUnits.METERS, self.x, self.y, self.z)
        # Calculate the scale factor squared for area
        scale = dims[0] / self.x if self.x != 0 else 1.0
        area_m2 = self.polygon.area * scale * scale
        return area_m2 * dims[2]

    @property
    def cubic_feet(self) -> float:
        dims = convert_length(self.units, LengthUnits.FEET, self.x, self.y, self.z)
        scale = dims[0] / self.x if self.x != 0 else 1.0
        area_ft2 = self.polygon.area * scale * scale
        return area_ft2 * dims[2]

    @property
    def faces(self) -> dict:
        """
        Returns face definitions for floor, ceiling, and numbered walls.

        Each wall is defined by:
            (edge_start, edge_end, height, outward_normal_2d)

        Floor and ceiling use the same format as RoomDimensions for compatibility
        with axis-aligned operations, but also include polygon reference.
        """
        if "faces" in self._cache:
            return self._cache["faces"]

        x_min, y_min, x_max, y_max = self.polygon.bounding_box

        result = {
            # Floor and ceiling use bounding box for backward compatibility
            # but actual geometry should use polygon
            "floor": (x_min, x_max, y_min, y_max, 0, "xy", 1, self.polygon),
            "ceiling": (x_min, x_max, y_min, y_max, self.z, "xy", -1, self.polygon),
        }

        # Add numbered walls from polygon edges
        for i, ((x1, y1), (x2, y2)) in enumerate(self.polygon.edges):
            edge_length = self.polygon.edge_lengths[i]
            normal_2d = self.polygon.edge_normals[i]
            # Wall definition: (x1, y1, x2, y2, edge_length, z_height, normal_2d)
            result[f"wall_{i}"] = (x1, y1, x2, y2, edge_length, self.z, normal_2d)

        self._cache["faces"] = result
        return result

    @property
    def wall_ids(self) -> list[str]:
        """List of wall IDs (wall_0, wall_1, etc.)."""
        return [f"wall_{i}" for i in range(len(self.polygon.edges))]

    def with_(self, *, z=None, units=None, polygon=None):
        return PolygonRoomDimensions(
            polygon=self.polygon if polygon is None else polygon,
            z=self.z if z is None else z,
            origin=self.origin,
            units=self.units if units is None else LengthUnits.from_any(units),
        )

    def to_dict(self) -> dict:
        return {
            "polygon": self.polygon.to_dict(),
            "z": self.z,
            "origin": list(self.origin),
            "units": str(self.units.value),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PolygonRoomDimensions":
        polygon = Polygon2D.from_dict(data["polygon"])
        return cls(
            polygon=polygon,
            z=data["z"],
            origin=tuple(data.get("origin", (0.0, 0.0, 0.0))),
            units=LengthUnits.from_any(data.get("units", "meters")),
        )
