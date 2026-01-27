from dataclasses import dataclass, replace, field
import numpy as np
from .axis import Axis1D
from .polygon import Polygon2D


@dataclass(frozen=True, eq=False)
class PolygonGridBase:
    """Base class for polygon-constrained grids."""

    polygon: Polygon2D
    spacing_init: tuple | None = None
    num_points_init: tuple | None = None
    offset: bool = True
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.polygon, Polygon2D):
            poly = Polygon2D(vertices=tuple(tuple(v) for v in self.polygon))
            object.__setattr__(self, "polygon", poly)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    @property
    def _spans(self) -> tuple:
        """Override in subclasses to provide dimension spans."""
        raise NotImplementedError

    @property
    def axes(self):
        if self._cache.get("axes") is not None:
            return self._cache["axes"]
        spacing = self.spacing_init or (None,) * len(self._spans)
        num_points = self.num_points_init or (None,) * len(self._spans)
        axes = [
            Axis1D(span=abs(s), spacing_init=sp, num_points_init=n, offset=self.offset)
            for s, sp, n in zip(self._spans, spacing, num_points)
        ]
        self._cache["axes"] = axes
        return axes

    @property
    def spacing(self) -> tuple:
        return tuple(float(axis.spacing) for axis in self.axes)

    @property
    def x_spacing(self) -> float:
        return self.spacing[0]

    @property
    def y_spacing(self) -> float:
        return self.spacing[1]

    @property
    def num_x(self) -> int:
        return len(self.axes[0].points)

    @property
    def num_y(self) -> int:
        return len(self.axes[1].points)

    @property
    def _xy_grid_points(self) -> tuple[np.ndarray, np.ndarray]:
        """X and Y grid points offset by bounding box origin."""
        if self._cache.get("_xy_grid_points") is not None:
            return self._cache["_xy_grid_points"]
        x_min, y_min, _, _ = self.polygon.bounding_box
        xp = self.axes[0].points + x_min
        yp = self.axes[1].points + y_min
        self._cache["_xy_grid_points"] = (xp, yp)
        return xp, yp

    @property
    def _xy_mask(self) -> np.ndarray:
        """Boolean mask for points inside polygon in xy plane."""
        if self._cache.get("_xy_mask") is not None:
            return self._cache["_xy_mask"]
        xp, yp = self._xy_grid_points
        xx, yy = np.meshgrid(xp, yp, indexing="ij")
        points_2d = np.column_stack([xx.ravel(), yy.ravel()])
        mask = self.polygon.contains_points(points_2d)
        self._cache["_xy_mask"] = mask
        return mask

    def update(self, **changes):
        new = replace(self, **changes)
        object.__setattr__(new, "_cache", {})
        return new

    @property
    def update_state(self) -> tuple:
        return ()


@dataclass(frozen=True, eq=False)
class PolygonGrid(PolygonGridBase):
    """
    A 2D grid of points constrained to lie within a polygon boundary.

    Used for floor/ceiling discretization in polygon-based rooms.
    Creates a bounding box grid and filters to points inside the polygon.
    """

    height: float = 0.0
    direction: int = 1  # +1 for floor (normal up), -1 for ceiling (normal down)

    @property
    def _spans(self) -> tuple:
        x_min, y_min, x_max, y_max = self.polygon.bounding_box
        return (x_max - x_min, y_max - y_min)

    def __repr__(self):
        return (
            f"PolygonGrid(polygon={self.polygon.n_vertices} vertices, "
            f"height={self.height}, "
            f"spacing={self.spacing}, "
            f"num_points={self.num_points}, "
            f"offset={self.offset})"
        )

    @property
    def coords(self) -> np.ndarray:
        """3D coordinates of points inside the polygon at the specified height."""
        if self._cache.get("coords") is not None:
            return self._cache["coords"]

        xp, yp = self._xy_grid_points
        xx, yy = np.meshgrid(xp, yp, indexing="ij")
        points_2d = np.column_stack([xx.ravel(), yy.ravel()])

        # Filter to points inside polygon
        inside = self._xy_mask
        x_inside = points_2d[inside, 0]
        y_inside = points_2d[inside, 1]
        z_inside = np.full_like(x_inside, self.height)

        coords = np.column_stack([x_inside, y_inside, z_inside])
        self._cache["coords"] = coords
        return coords

    @property
    def num_points(self) -> tuple[int, ...]:
        """Number of points (single value since polygon grid is irregular)."""
        return (len(self.coords),)

    @property
    def origin(self) -> tuple[float, float, float]:
        x_min, y_min, _, _ = self.polygon.bounding_box
        return (x_min, y_min, self.height)

    @property
    def normal(self) -> np.ndarray:
        """Surface normal (up for floor, down for ceiling)."""
        return np.array([0.0, 0.0, float(self.direction)])

    @property
    def u_hat(self) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0])

    @property
    def v_hat(self) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0])

    @property
    def basis(self) -> np.ndarray:
        """Return an orthonormal basis (u, v, n) for the surface."""
        if self._cache.get("basis") is not None:
            return self._cache["basis"]
        basis = np.stack([self.u_hat, self.v_hat, self.normal], axis=1)
        self._cache["basis"] = basis
        return basis

    @property
    def calc_state(self) -> tuple:
        return (
            self.polygon.vertices,
            self.height,
            self.spacing,
            self.offset,
            self.direction,
        )

    def update_dimensions(self, polygon=None, height=None, preserve_spacing=True):
        """Update with new polygon or height."""
        new_poly = polygon if polygon is not None else self.polygon
        new_height = height if height is not None else self.height
        if preserve_spacing:
            return self.update(
                polygon=new_poly, height=new_height, spacing_init=self.spacing
            )
        else:
            return self.update(
                polygon=new_poly,
                height=new_height,
                num_points_init=self.num_points_init,
            )

    def to_dict(self) -> dict:
        return {
            "polygon": self.polygon.to_dict(),
            "height": self.height,
            "spacing_init": self.spacing,
            "offset": self.offset,
            "direction": self.direction,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PolygonGrid":
        polygon = Polygon2D.from_dict(data["polygon"])
        return cls(
            polygon=polygon,
            height=data.get("height", 0.0),
            spacing_init=data.get("spacing_init"),
            num_points_init=data.get("num_points_init"),
            offset=data.get("offset", True),
            direction=data.get("direction", 1),
        )


@dataclass(frozen=True, eq=False)
class PolygonVolGrid(PolygonGridBase):
    """
    A 3D grid of points constrained to lie within a polygon boundary in x-y.

    Used for volumetric calculations in polygon-based rooms.
    Creates a bounding box grid and filters to points where (x,y) is inside the polygon.
    """

    z_height: float = 2.7  # Room height (z goes from 0 to z_height)

    @property
    def _spans(self) -> tuple:
        x_min, y_min, x_max, y_max = self.polygon.bounding_box
        return (x_max - x_min, y_max - y_min, self.z_height)

    def __repr__(self):
        return (
            f"PolygonVolGrid(polygon={self.polygon.n_vertices} vertices, "
            f"z_height={self.z_height}, "
            f"spacing={self.spacing}, "
            f"num_points={self.num_points}, "
            f"offset={self.offset})"
        )

    @property
    def z_spacing(self) -> float:
        return self.spacing[2]

    @property
    def num_z(self) -> int:
        return len(self.axes[2].points)

    @property
    def _grid_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Raw grid points including z axis."""
        if self._cache.get("_grid_points") is not None:
            return self._cache["_grid_points"]
        xp, yp = self._xy_grid_points
        zp = self.axes[2].points  # z starts at 0
        self._cache["_grid_points"] = (xp, yp, zp)
        return xp, yp, zp

    @property
    def coords(self) -> np.ndarray:
        """3D coordinates of points inside the polygon boundary."""
        if self._cache.get("coords") is not None:
            return self._cache["coords"]

        xp, yp, zp = self._grid_points

        # Create 2D mask for x-y plane
        xx_2d, yy_2d = np.meshgrid(xp, yp, indexing="ij")
        points_2d = np.column_stack([xx_2d.ravel(), yy_2d.ravel()])
        mask_2d = self._xy_mask

        # Get inside points
        x_inside = points_2d[mask_2d, 0]
        y_inside = points_2d[mask_2d, 1]
        n_xy = len(x_inside)

        # Build 3D coords with z varying fastest (matches VolGrid/Plotly ordering)
        x_rep = np.repeat(x_inside, len(zp))
        y_rep = np.repeat(y_inside, len(zp))
        z_tiled = np.tile(zp, n_xy)
        coords = np.column_stack([x_rep, y_rep, z_tiled])

        self._cache["coords"] = coords
        return coords

    @property
    def coords_full(self) -> np.ndarray:
        """Full bounding box coordinates (including points outside polygon)."""
        if self._cache.get("coords_full") is not None:
            return self._cache["coords_full"]

        xp, yp, zp = self._grid_points
        mesh = np.meshgrid(xp, yp, zp, indexing="ij")
        coords_full = np.column_stack([m.ravel() for m in mesh])
        self._cache["coords_full"] = coords_full
        return coords_full

    @property
    def _mask_full(self) -> np.ndarray:
        """Boolean mask for full 3D grid (True = inside polygon)."""
        if self._cache.get("_mask_full") is not None:
            return self._cache["_mask_full"]

        _, _, zp = self._grid_points
        mask_2d = self._xy_mask
        # Repeat each mask value for all z levels (z varies fastest)
        mask_full = np.repeat(mask_2d, len(zp))
        self._cache["_mask_full"] = mask_full
        return mask_full

    def values_to_full_grid(self, values: np.ndarray) -> np.ndarray:
        """Map filtered values back to full grid, with -inf outside polygon."""
        full_values = np.full(len(self.coords_full), -np.inf)
        full_values[self._mask_full] = values.flatten()
        return full_values

    @property
    def num_points(self) -> tuple[int, ...]:
        """Total number of points (irregular due to polygon filtering)."""
        return (len(self.coords),)

    @property
    def origin(self) -> tuple[float, float, float]:
        x_min, y_min, _, _ = self.polygon.bounding_box
        return (x_min, y_min, 0.0)

    @property
    def mins(self) -> tuple[float, float, float]:
        x_min, y_min, _, _ = self.polygon.bounding_box
        return (x_min, y_min, 0.0)

    @property
    def maxs(self) -> tuple[float, float, float]:
        _, _, x_max, y_max = self.polygon.bounding_box
        return (x_max, y_max, self.z_height)

    @property
    def x1(self) -> float:
        return self.mins[0]

    @property
    def x2(self) -> float:
        return self.maxs[0]

    @property
    def y1(self) -> float:
        return self.mins[1]

    @property
    def y2(self) -> float:
        return self.maxs[1]

    @property
    def z1(self) -> float:
        return 0.0

    @property
    def z2(self) -> float:
        return self.z_height

    @property
    def calc_state(self) -> tuple:
        return (
            self.polygon.vertices,
            self.z_height,
            self.spacing,
            self.offset,
        )

    def update_dimensions(self, mins=None, maxs=None, preserve_spacing=True):
        """Update with new z_height (polygon shape preserved)."""
        new_z = maxs[2] if maxs is not None else self.z_height
        if preserve_spacing:
            return self.update(z_height=new_z, spacing_init=self.spacing)
        else:
            return self.update(z_height=new_z, num_points_init=self.num_points_init)

    def to_dict(self) -> dict:
        return {
            "polygon": self.polygon.to_dict(),
            "z_height": self.z_height,
            "spacing_init": self.spacing,
            "offset": self.offset,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PolygonVolGrid":
        polygon = Polygon2D.from_dict(data["polygon"])
        return cls(
            polygon=polygon,
            z_height=data.get("z_height", 2.7),
            spacing_init=data.get("spacing_init"),
            num_points_init=data.get("num_points_init"),
            offset=data.get("offset", True),
        )
