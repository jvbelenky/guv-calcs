from dataclasses import dataclass, field
from typing import NamedTuple
import numpy as np
from .units import LengthUnits, convert_length
from .polygon import Polygon2D


class RectFace(NamedTuple):
    """Face data for axis-aligned rectangular rooms (floor, ceiling, walls)."""
    x1: float
    x2: float
    y1: float
    y2: float
    height: float
    ref_surface: str
    direction: int


class PolygonFloorCeilingFace(NamedTuple):
    """Face data for polygon room floor/ceiling."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    height: float
    ref_surface: str
    direction: int
    polygon: object


class WallFace(NamedTuple):
    """Face data for polygon room walls."""
    x1: float
    y1: float
    x2: float
    y2: float
    edge_length: float
    z_height: float
    normal_2d: tuple


@dataclass(frozen=True)
class RoomDimensions:
    """Room dimensions defined by a 2D polygon floor plan with uniform ceiling height.

    All rooms use polygon representation internally. Rectangular rooms are
    stored as 4-vertex polygons. Use the ``x``, ``y``, ``z`` constructor
    kwargs on Room for convenience; they create a rectangle polygon.
    """

    polygon: Polygon2D
    z: float
    units: LengthUnits = LengthUnits.METERS
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.polygon, Polygon2D):
            poly = Polygon2D(vertices=tuple(tuple(v) for v in self.polygon))
            object.__setattr__(self, "polygon", poly)

    @property
    def is_polygon(self) -> bool:
        """True if this room has a non-rectangular floor plan."""
        return self.polygon.n_vertices != 4 or not self._is_axis_aligned_rect()

    @property
    def x(self) -> float:
        """Bounding box width."""
        return self.polygon.x

    @property
    def y(self) -> float:
        """Bounding box height."""
        return self.polygon.y

    @property
    def dimensions(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def volume(self) -> float:
        return self.polygon.area * self.z

    @property
    def cubic_meters(self) -> float:
        dims = convert_length(self.units, LengthUnits.METERS, self.x, self.y, self.z)
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
        """Face definitions for floor, ceiling, and walls.

        Non-rectangular rooms get numbered walls (wall_0, wall_1, ...).
        Rectangular rooms get cardinal walls (south, north, west, east) for
        backward compatibility.
        """
        if "faces" in self._cache:
            return self._cache["faces"]

        x_min, y_min, x_max, y_max = self.polygon.bounding_box

        if self.is_polygon:
            result = {
                "floor": PolygonFloorCeilingFace(x_min, x_max, y_min, y_max, 0, "xy", 1, self.polygon),
                "ceiling": PolygonFloorCeilingFace(x_min, x_max, y_min, y_max, self.z, "xy", -1, self.polygon),
            }
            for i, ((x1, y1), (x2, y2)) in enumerate(self.polygon.edges):
                edge_length = self.polygon.edge_lengths[i]
                normal_2d = self.polygon.edge_normals[i]
                result[f"wall_{i}"] = WallFace(x1, y1, x2, y2, edge_length, self.z, normal_2d)
        else:
            # Axis-aligned rectangle: use actual bounding box values
            result = {
                "floor": RectFace(x_min, x_max, y_min, y_max, 0, "xy", 1),
                "ceiling": RectFace(x_min, x_max, y_min, y_max, self.z, "xy", -1),
                "south": RectFace(x_min, x_max, 0, self.z, y_min, "xz", 1),
                "north": RectFace(x_min, x_max, 0, self.z, y_max, "xz", -1),
                "west": RectFace(y_min, y_max, 0, self.z, x_min, "yz", 1),
                "east": RectFace(y_min, y_max, 0, self.z, x_max, "yz", -1),
            }

        self._cache["faces"] = result
        return result

    @property
    def wall_ids(self) -> list[str]:
        """List of wall IDs."""
        if self.is_polygon:
            return [f"wall_{i}" for i in range(len(self.polygon.edges))]
        return ["south", "north", "west", "east"]

    def contains_point(self, x: float, y: float) -> bool:
        """Check if (x, y) is within the floor polygon."""
        return self.polygon.contains_point_inclusive(x, y)

    def with_(self, *, x=None, y=None, z=None, units=None, polygon=None):
        """Return a new RoomDimensions with updated values."""
        new_units = self.units if units is None else LengthUnits.from_any(units)
        new_z = self.z if z is None else z

        if polygon is not None:
            new_polygon = polygon
        elif (x is not None or y is not None) and not self.is_polygon:
            bb = self.polygon.bounding_box  # (x_min, y_min, x_max, y_max)

            if isinstance(x, (tuple, list)):
                new_x_min, new_x_max = float(x[0]), float(x[1])
            elif x is not None:
                new_x_min, new_x_max = bb[0], bb[0] + float(x)
            else:
                new_x_min, new_x_max = bb[0], bb[2]

            if isinstance(y, (tuple, list)):
                new_y_min, new_y_max = float(y[0]), float(y[1])
            elif y is not None:
                new_y_min, new_y_max = bb[1], bb[1] + float(y)
            else:
                new_y_min, new_y_max = bb[1], bb[3]

            new_polygon = Polygon2D(vertices=(
                (new_x_min, new_y_min), (new_x_max, new_y_min),
                (new_x_max, new_y_max), (new_x_min, new_y_max)
            ))
        else:
            new_polygon = self.polygon

        return RoomDimensions(
            polygon=new_polygon,
            z=new_z,
            units=new_units,
        )

    def _is_axis_aligned_rect(self) -> bool:
        """Check if this is an axis-aligned rectangle."""
        if self.polygon.n_vertices != 4:
            return False
        xs = sorted(set(v[0] for v in self.polygon.vertices))
        ys = sorted(set(v[1] for v in self.polygon.vertices))
        return len(xs) == 2 and len(ys) == 2

