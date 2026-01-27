from dataclasses import dataclass, replace, field
import numbers
import numpy as np
import inspect
import warnings
from .axis import Axis1D
from .polygon import Polygon2D


@dataclass(frozen=True, eq=False)
class RectGrid:
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    spans: tuple[float, ...] = (1.0, 1.0, 1.0)
    spacing_init: tuple[float, ...] | None = None
    num_points_init: tuple[int, ...] | None = None
    offset: bool = True
    _cache: dict = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self):

        if self.num_points_init is not None:
            if len(self.num_points_init) != len(self.spans):
                raise ValueError(
                    "num_points_init dimensions do not match min/max dimensions"
                )

        if self.spacing_init is not None:
            if len(self.spacing_init) != len(self.spans):
                raise ValueError(
                    "spacing_init dimensions do not match min/max dimensions"
                )

        if len(self.spans) > 3:
            raise ValueError("Maximum of three dimensions allowed")

        if len(self.spans) < 1:
            raise ValueError("Minimum of one dimension required")

        if type(self.offset) is not bool:
            raise TypeError("must be either True or False")

    def __eq__(self, other):
        if not isinstance(other, RectGrid):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    @classmethod
    def from_dict(cls, data):
        keys = list(inspect.signature(cls.__init__).parameters.keys())[1:]
        return cls(**{k: v for k, v in data.items() if k in keys})

    def to_dict(self):
        data = {
            "origin": tuple(self.origin),
            "spans": tuple(self.spans),
            "spacing_init": tuple(self.spacing),
        }
        data.update(self._extra_dict())
        return data

    def _extra_dict(self):
        return {}

    def update(self, **changes):
        new = replace(self, **changes)
        # clear cache
        object.__setattr__(new, "_cache", {})
        return new

    @property
    def axes(self):
        if self._cache.get("axes") is not None:
            return self._cache["axes"]

        axes = []
        spacing = self.spacing_init or (None,) * len(self.spans)
        num_points = self.num_points_init or (None,) * len(self.spans)
        for span, sp, n_pts in zip(self.spans, spacing, num_points):
            axis = Axis1D(
                span=abs(span),
                spacing_init=sp,
                num_points_init=n_pts,
                offset=self.offset,
            )
            axes.append(axis)
        self._cache["axes"] = axes
        return axes

    @property
    def points(self) -> list:
        if self._cache.get("points") is not None:
            return self._cache.get("points")
        points = [axis.points for axis, start in zip(self.axes, self.mins)]
        self._cache["points"] = points
        return points

    @property
    def mins(self):
        return tuple(
            min(orig, orig + span) for orig, span in zip(self.origin, self.spans)
        )

    @property
    def maxs(self):
        return tuple(
            max(orig, orig + span) for orig, span in zip(self.origin, self.spans)
        )

    @property
    def dimensions(self):
        return tuple((a, b) for a, b in zip(self.mins, self.maxs))

    # warning -- these will no longer function as intended for PlaneGrids if they are off-axis
    @property
    def x1(self) -> float:
        return self.mins[0] if len(self.mins) > 0 else None

    @property
    def x2(self) -> float:
        return self.maxs[0] if len(self.mins) > 0 else None

    @property
    def y1(self) -> float:
        return self.mins[1] if len(self.mins) > 1 else None

    @property
    def y2(self) -> float:
        return self.maxs[1] if len(self.mins) > 1 else None

    @property
    def z1(self) -> float:
        return self.mins[2] if len(self.mins) > 2 else None

    @property
    def z2(self) -> float:
        return self.maxs[2] if len(self.mins) > 2 else None

    @property
    def num_points(self) -> tuple:
        return tuple([int(len(pt)) for pt in self.points])

    @property
    def num_x(self) -> int:
        return self.num_points[0] if len(self.num_points) > 0 else None

    @property
    def num_y(self) -> int:
        return self.num_points[1] if len(self.num_points) > 1 else None

    @property
    def num_z(self) -> int:
        return self.num_points[2] if len(self.num_points) > 2 else None

    @property
    def spacing(self) -> tuple:
        return tuple([float(axis.spacing) for axis in self.axes])

    @property
    def x_spacing(self) -> float:
        return self.spacing[0] if len(self.spacing) > 0 else None

    @property
    def y_spacing(self) -> float:
        return self.spacing[1] if len(self.spacing) > 1 else None

    @property
    def z_spacing(self) -> float:
        return self.spacing[2] if len(self.spacing) > 2 else None

    @property
    def xp(self) -> np.ndarray:
        return self.points[0] if len(self.points) > 0 else None

    @property
    def yp(self) -> np.ndarray:
        return self.points[1] if len(self.points) > 1 else None

    @property
    def zp(self) -> np.ndarray:
        return self.points[2] if len(self.points) > 2 else None

    @property
    def calc_state(self) -> tuple:
        return tuple(self.origin) + (tuple(self.spans) + self.spacing + (self.offset,))

    @property
    def update_state(self) -> tuple:
        return ()


@dataclass(frozen=True, eq=False)
class VolGrid(RectGrid):
    def __repr__(self):
        return (
            f"VolGrid(dimensions={self.dimensions}, "
            f"spacing={self.spacing}, "
            f"num_points={self.num_points}, "
            f"offset={self.offset})"
        )

    @property
    def coords(self) -> np.ndarray:
        if self._cache.get("coords") is not None:
            return self._cache["coords"]
        mesh = np.meshgrid(*self.points, indexing="ij")
        X, Y, Z = [grid.reshape(-1) for grid in mesh]
        coords = np.array((X, Y, Z)).T + np.asarray(self.origin, float)
        coords = np.unique(coords, axis=0)
        self._cache["coords"] = coords
        return coords

    @classmethod
    def from_legacy(cls, *, mins, maxs, **kwargs):
        origin = np.asarray(mins, float)
        spans = np.asarray(maxs) - origin
        return cls(origin=tuple(origin), spans=tuple(spans), **kwargs)

    def update_dimensions(self, mins, maxs, preserve_spacing=True):
        """mostly to keep the pattern with PlaneGrid"""
        origin = np.asarray(mins, float)
        spans = np.asarray(maxs, float) - origin
        if preserve_spacing:
            return self.update(origin=origin, spans=spans, spacing_init=self.spacing)
        else:
            return self.update(
                origin=origin, spans=spans, num_points_init=self.num_points
            )


@dataclass(frozen=True, eq=False)
class PlaneGrid(RectGrid):
    u_vec: tuple[float, float, float] = (1.0, 0.0, 0.0)
    v_vec: tuple[float, float, float] = (0.0, 1.0, 0.0)

    def __post_init__(self):

        if len(self.spans) > 2:
            raise ValueError("Too many dimensions for a plane")

        u = np.array(self.u_vec, float)
        v = np.array(self.v_vec, float)

        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise ValueError("u_vec and v_vec must be finite numeric vectors.")

        if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0:
            raise ValueError("u_vec and v_vec must be non-zero.")

        # forbid (near-)parallel u and v; we want a proper plane basis
        if np.linalg.norm(np.cross(u, v)) == 0:
            raise ValueError("u_vec and v_vec must not be parallel.")

    def __repr__(self):
        return (
            f"PlaneGrid(dimensions={self.dimensions}, "
            f"spacing={self.spacing}, "
            f"num_points={self.num_points}, "
            f"offset={self.offset}, "
            f"normal={self.normal})"
        )

    def _extra_dict(self):
        data = super()._extra_dict()
        data.update({"u_vec": tuple(self.u_vec), "v_vec": tuple(self.v_vec)})
        return data

    @property
    def calc_state(self):
        return super().calc_state + tuple(self.u_vec) + tuple(self.v_vec)

    @property
    def extent(self):
        return np.array(
            np.asarray(self.origin, float)
            + self.u_hat * self.spans[0]
            + self.v_hat * self.spans[1]
        )

    @property
    def mins(self):
        return tuple(min(a, b) for a, b in zip(self.origin, self.extent) if a != b)

    @property
    def maxs(self):
        return tuple(max(a, b) for a, b in zip(self.origin, self.extent) if a != b)

    @property
    def u_hat(self):
        u = np.asarray(self.u_vec, float)
        return u / np.linalg.norm(u)

    @property
    def v_hat(self):
        u = self.u_hat
        v = np.asarray(self.v_vec, float)
        v = v - np.dot(u, v) * u
        return v / np.linalg.norm(v)

    @property
    def normal(self):
        n = np.cross(self.u_hat, self.v_hat)
        return n / np.linalg.norm(n)

    @property
    def basis(self):
        """
        Return an orthonormal basis (u, v, n) for the surface:
        - n is the normal vector (points outward from surface)
        - u, v span the surface plane
        """
        if self._cache.get("basis") is not None:
            return self._cache["basis"]
        basis = np.stack([self.u_hat, self.v_hat, self.normal], axis=1)
        self._cache["basis"] = basis
        return basis

    @property
    def coords(self) -> np.ndarray:
        if self._cache.get("coords") is not None:
            return self._cache["coords"]
        # 2D parameter grid in u/v space (same as your old X,Y)
        mesh = np.meshgrid(*self.points, indexing="ij")
        s_vals, t_vals = [grid.reshape(-1) for grid in mesh]
        origin = np.asarray(self.origin, float)
        # origin + s*u + t*v
        coords = origin + s_vals[:, None] * self.u_hat + t_vals[:, None] * self.v_hat
        self._cache["coords"] = coords
        return coords

    @classmethod
    def from_points(cls, *, p0, pU, pV, **kwargs):
        origin = np.asarray(p0, float)
        u = np.asarray(pU, float) - p0
        v = np.asarray(pV, float) - p0
        if np.dot(u, v) != 0:
            msg = f"point {pV} is not orthogonal to point {pV}"
            warnings.warn(msg)
        u_hat = u / np.linalg.norm(u)
        v_perp = v - np.dot(v, u_hat) * u_hat
        v_norm = np.linalg.norm(v_perp)
        if v_norm == 0:
            raise ValueError("from_points requires non-collinear points")
        v_hat = v_perp / v_norm
        s1, t1 = np.dot(u, u_hat), np.dot(u, v_hat)
        s2, t2 = np.dot(v, u_hat), np.dot(v, v_hat)
        spans = (max(s1, s2), max(t1, t2))
        return cls(spans=spans, origin=origin, u_vec=u, v_vec=v, **kwargs)

    def update_dimensions(self, mins, maxs, preserve_spacing=True):
        """keep current orientation, with different extents"""
        u1, u2, v1, v2 = mins[0], maxs[0], mins[1], maxs[1]
        span_u, span_v = (u2 - u1), (v2 - v1)
        origin = u1 * self.u_hat + v1 * self.v_hat
        if preserve_spacing:
            return self.update(
                origin=origin,
                spans=(span_u, span_v),
                spacing_init=self.spacing,
            )
        else:
            return self.update(
                origin=origin,
                spans=(span_u, span_v),
                num_points_init=self.num_points,
            )

    # ---------------- legacy properties for axis-aligned planes --------------------

    @classmethod
    def from_legacy(
        cls,
        *,
        mins: tuple,
        maxs: tuple,
        height: float = 0,
        ref_surface: str = "xy",
        direction: int = 1,
        **kwargs,
    ):
        if not isinstance(ref_surface, str):
            raise TypeError("ref_surface must be a string in [`xy`,`xz`,`yz`]")
        if ref_surface.lower() not in ["xy", "xz", "yz"]:
            raise ValueError("ref_surface must be a string in [`xy`,`xz`,`yz`]")
        if not isinstance(height, numbers.Number):
            raise TypeError("Height must be numeric")
        if direction is not None and direction not in [1, 0, -1]:
            raise ValueError("Direction must be in [1, 0, -1]")

        if ref_surface == "xy":
            if direction == 1:
                origin = (mins[0], mins[1], height)
                u_vec = (1, 0, 0)
                v_vec = (0, 1, 0)
            else:
                origin = (mins[1], maxs[1], height)
                u_vec = (1, 0, 0)
                v_vec = (0, -1, 0)

        elif ref_surface == "xz":
            if direction == 1:
                origin = (mins[1], height, maxs[1])
                u_vec = (1, 0, 0)
                v_vec = (0, 0, -1)
            else:
                origin = (mins[0], height, mins[1])
                u_vec = (1, 0, 0)
                v_vec = (0, 0, 1)

        elif ref_surface == "yz":
            if direction == 1:
                origin = (height, mins[0], mins[1])
                u_vec = (0, 1, 0)
                v_vec = (0, 0, 1)
            else:
                origin = (height, mins[1], maxs[1])
                u_vec = (0, 1, 0)
                v_vec = (0, 0, -1)

        spans = (maxs[0] - mins[0], maxs[1] - mins[1])

        return cls(
            spans=spans,
            origin=np.asarray(origin, float),
            u_vec=np.asarray(u_vec, float),
            v_vec=np.asarray(v_vec, float),
            **kwargs,
        )

    def update_legacy(self, height=None, ref_surface=None, direction=None):
        """keep current dimensions, update height/orientation/reference surface"""
        if self.ref_surface is None:
            raise ValueError(
                "update_from_legacy is only defined for axis-aligned planes"
            )
        return PlaneGrid.from_legacy(
            mins=self.mins,
            maxs=self.maxs,
            spacing_init=tuple(self.spacing),
            height=height or self.height,
            ref_surface=ref_surface or self.ref_surface,
            direction=direction or self.direction,
        )

    @property
    def ref_surface(self) -> str | None:
        """
        Return 'xy', 'xz', or 'yz' if the plane is axis-aligned.
        Otherwise return None.
        """
        n = self.normal  # unit normal, from existing property
        # axes = np.eye(3)
        labels = ["yz", "xz", "xy"]  # normals along +x,+y,+z ⇒ planes yz,xz,xy

        idx = int(np.argmax(np.abs(n)))
        if not np.isclose(abs(n[idx]), 1.0, atol=1e-6):
            return None  # not axis-aligned

        return labels[idx]

    @property
    def height(self) -> float | None:
        """
        For an axis-aligned plane, return the offset along the axis
        normal to the plane. Otherwise None.
        """
        rs = self.ref_surface
        if rs is None:
            return None

        # Could use origin only; using coords is more robust if origin isn’t on the plane,
        # but origin *should* be on the plane. This is the simple version:
        if rs == "xy":  # normal ±z
            return float(self.origin[2])
        elif rs == "xz":  # normal ±y
            return float(self.origin[1])
        elif rs == "yz":  # normal ±x
            return float(self.origin[0])

    @property
    def direction(self) -> int | None:
        """
        Sign of the normal relative to the axis orthogonal to the plane.
        +1 or -1 if axis-aligned, else None.
        """
        rs = self.ref_surface
        if rs is None:
            print("bork")
            return None

        n = self.normal
        if rs == "xy":  # normal ±z
            return 1 if n[2] > 0 else -1
        elif rs == "xz":  # normal ±y
            return 1 if n[1] > 0 else -1
        elif rs == "yz":  # normal ±x
            return 1 if n[0] > 0 else -1


@dataclass(frozen=True, eq=False)
class PolygonGrid:
    """
    A 2D grid of points constrained to lie within a polygon boundary.

    Used for floor/ceiling discretization in polygon-based rooms.
    Creates a bounding box grid and filters to points inside the polygon.
    """

    polygon: Polygon2D
    height: float = 0.0
    spacing_init: tuple[float, float] | None = None
    num_points_init: tuple[int, int] | None = None
    offset: bool = True
    direction: int = 1  # +1 for floor (normal up), -1 for ceiling (normal down)
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.polygon, Polygon2D):
            poly = Polygon2D(vertices=tuple(tuple(v) for v in self.polygon))
            object.__setattr__(self, "polygon", poly)

    def __eq__(self, other):
        if not isinstance(other, PolygonGrid):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self):
        return (
            f"PolygonGrid(polygon={self.polygon.n_vertices} vertices, "
            f"height={self.height}, "
            f"spacing={self.spacing}, "
            f"num_points={self.num_points}, "
            f"offset={self.offset})"
        )

    @property
    def axes(self):
        if self._cache.get("axes") is not None:
            return self._cache["axes"]

        x_min, y_min, x_max, y_max = self.polygon.bounding_box
        spans = (x_max - x_min, y_max - y_min)
        spacing = self.spacing_init or (None, None)
        num_points = self.num_points_init or (None, None)

        axes = []
        for span, sp, n_pts in zip(spans, spacing, num_points):
            axis = Axis1D(
                span=abs(span),
                spacing_init=sp,
                num_points_init=n_pts,
                offset=self.offset,
            )
            axes.append(axis)
        self._cache["axes"] = axes
        return axes

    @property
    def spacing(self) -> tuple[float, float]:
        return tuple(float(axis.spacing) for axis in self.axes)

    @property
    def x_spacing(self) -> float:
        return self.spacing[0]

    @property
    def y_spacing(self) -> float:
        return self.spacing[1]

    @property
    def _grid_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Raw grid points before polygon filtering."""
        if self._cache.get("_grid_points") is not None:
            return self._cache["_grid_points"]

        x_min, y_min, _, _ = self.polygon.bounding_box
        xp = self.axes[0].points + x_min
        yp = self.axes[1].points + y_min
        self._cache["_grid_points"] = (xp, yp)
        return xp, yp

    @property
    def _mask(self) -> np.ndarray:
        """Boolean mask for points inside polygon."""
        if self._cache.get("_mask") is not None:
            return self._cache["_mask"]

        xp, yp = self._grid_points
        xx, yy = np.meshgrid(xp, yp, indexing="ij")
        points_2d = np.column_stack([xx.ravel(), yy.ravel()])
        mask = self.polygon.contains_points(points_2d)
        self._cache["_mask"] = mask
        return mask

    @property
    def coords(self) -> np.ndarray:
        """3D coordinates of points inside the polygon at the specified height."""
        if self._cache.get("coords") is not None:
            return self._cache["coords"]

        xp, yp = self._grid_points
        xx, yy = np.meshgrid(xp, yp, indexing="ij")
        points_2d = np.column_stack([xx.ravel(), yy.ravel()])

        # Filter to points inside polygon
        inside = self._mask
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
    def num_x(self) -> int:
        return len(self.axes[0].points)

    @property
    def num_y(self) -> int:
        return len(self.axes[1].points)

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

    @property
    def update_state(self) -> tuple:
        return ()

    def update(self, **changes):
        new = replace(self, **changes)
        object.__setattr__(new, "_cache", {})
        return new

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
class PolygonVolGrid:
    """
    A 3D grid of points constrained to lie within a polygon boundary in x-y.

    Used for volumetric calculations in polygon-based rooms.
    Creates a bounding box grid and filters to points where (x,y) is inside the polygon.
    """

    polygon: Polygon2D
    z_height: float  # Room height (z goes from 0 to z_height)
    spacing_init: tuple[float, float, float] | None = None
    num_points_init: tuple[int, int, int] | None = None
    offset: bool = True
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.polygon, Polygon2D):
            poly = Polygon2D(vertices=tuple(tuple(v) for v in self.polygon))
            object.__setattr__(self, "polygon", poly)

    def __eq__(self, other):
        if not isinstance(other, PolygonVolGrid):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self):
        return (
            f"PolygonVolGrid(polygon={self.polygon.n_vertices} vertices, "
            f"z_height={self.z_height}, "
            f"spacing={self.spacing}, "
            f"num_points={self.num_points}, "
            f"offset={self.offset})"
        )

    @property
    def axes(self):
        if self._cache.get("axes") is not None:
            return self._cache["axes"]

        x_min, y_min, x_max, y_max = self.polygon.bounding_box
        spans = (x_max - x_min, y_max - y_min, self.z_height)
        spacing = self.spacing_init or (None, None, None)
        num_points = self.num_points_init or (None, None, None)

        axes = []
        for span, sp, n_pts in zip(spans, spacing, num_points):
            axis = Axis1D(
                span=abs(span),
                spacing_init=sp,
                num_points_init=n_pts,
                offset=self.offset,
            )
            axes.append(axis)
        self._cache["axes"] = axes
        return axes

    @property
    def spacing(self) -> tuple[float, float, float]:
        return tuple(float(axis.spacing) for axis in self.axes)

    @property
    def x_spacing(self) -> float:
        return self.spacing[0]

    @property
    def y_spacing(self) -> float:
        return self.spacing[1]

    @property
    def z_spacing(self) -> float:
        return self.spacing[2]

    @property
    def _grid_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Raw grid points before polygon filtering."""
        if self._cache.get("_grid_points") is not None:
            return self._cache["_grid_points"]

        x_min, y_min, _, _ = self.polygon.bounding_box
        xp = self.axes[0].points + x_min
        yp = self.axes[1].points + y_min
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
        mask_2d = self.polygon.contains_points(points_2d)

        # Get inside points
        x_inside = points_2d[mask_2d, 0]
        y_inside = points_2d[mask_2d, 1]
        n_xy = len(x_inside)

        # Build 3D coords with z varying fastest (matches VolGrid/Plotly ordering)
        # For each (x,y) inside point, repeat for all z values
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
        # Use meshgrid with z varying fastest (matches VolGrid/Plotly ordering)
        mesh = np.meshgrid(xp, yp, zp, indexing="ij")
        coords_full = np.column_stack([m.ravel() for m in mesh])
        self._cache["coords_full"] = coords_full
        return coords_full

    @property
    def _mask_full(self) -> np.ndarray:
        """Boolean mask for full 3D grid (True = inside polygon)."""
        if self._cache.get("_mask_full") is not None:
            return self._cache["_mask_full"]

        xp, yp, zp = self._grid_points
        xx, yy = np.meshgrid(xp, yp, indexing="ij")
        points_2d = np.column_stack([xx.ravel(), yy.ravel()])
        mask_2d = self.polygon.contains_points(points_2d)
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
    def num_x(self) -> int:
        return len(self.axes[0].points)

    @property
    def num_y(self) -> int:
        return len(self.axes[1].points)

    @property
    def num_z(self) -> int:
        return len(self.axes[2].points)

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

    @property
    def update_state(self) -> tuple:
        return ()

    def update(self, **changes):
        new = replace(self, **changes)
        object.__setattr__(new, "_cache", {})
        return new

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


@dataclass(frozen=True, eq=False)
class WallGrid:
    """
    A 2D grid for a wall segment defined by two 2D vertices and a height.

    Creates a rectangular grid in 3D space representing a wall panel.
    """

    p1: tuple[float, float]  # Start vertex (x, y) at z=0
    p2: tuple[float, float]  # End vertex (x, y) at z=0
    z_height: float
    normal_2d: tuple[float, float]  # Outward normal in xy plane
    spacing_init: tuple[float, float] | None = None
    num_points_init: tuple[int, int] | None = None
    offset: bool = True
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __eq__(self, other):
        if not isinstance(other, WallGrid):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self):
        return (
            f"WallGrid(p1={self.p1}, p2={self.p2}, "
            f"z_height={self.z_height}, "
            f"spacing={self.spacing})"
        )

    @property
    def edge_length(self) -> float:
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        return np.sqrt(dx * dx + dy * dy)

    @property
    def axes(self):
        if self._cache.get("axes") is not None:
            return self._cache["axes"]

        spans = (self.edge_length, self.z_height)
        spacing = self.spacing_init or (None, None)
        num_points = self.num_points_init or (None, None)

        axes = []
        for span, sp, n_pts in zip(spans, spacing, num_points):
            axis = Axis1D(
                span=abs(span),
                spacing_init=sp,
                num_points_init=n_pts,
                offset=self.offset,
            )
            axes.append(axis)
        self._cache["axes"] = axes
        return axes

    @property
    def spacing(self) -> tuple[float, float]:
        return tuple(float(axis.spacing) for axis in self.axes)

    @property
    def x_spacing(self) -> float:
        return self.spacing[0]

    @property
    def y_spacing(self) -> float:
        return self.spacing[1]

    @property
    def num_points(self) -> tuple[int, int]:
        return tuple(int(len(axis.points)) for axis in self.axes)

    @property
    def num_x(self) -> int:
        return self.num_points[0]

    @property
    def num_y(self) -> int:
        return self.num_points[1]

    @property
    def u_hat(self) -> np.ndarray:
        """Unit vector along the wall (from p1 to p2)."""
        if self._cache.get("u_hat") is not None:
            return self._cache["u_hat"]
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        length = self.edge_length
        u = np.array([dx / length, dy / length, 0.0])
        self._cache["u_hat"] = u
        return u

    @property
    def v_hat(self) -> np.ndarray:
        """Unit vector pointing up (z direction)."""
        return np.array([0.0, 0.0, 1.0])

    @property
    def normal(self) -> np.ndarray:
        """Outward-pointing unit normal."""
        return np.array([self.normal_2d[0], self.normal_2d[1], 0.0])

    @property
    def basis(self) -> np.ndarray:
        """Return an orthonormal basis (u, v, n) for the surface."""
        if self._cache.get("basis") is not None:
            return self._cache["basis"]
        basis = np.stack([self.u_hat, self.v_hat, self.normal], axis=1)
        self._cache["basis"] = basis
        return basis

    @property
    def origin(self) -> tuple[float, float, float]:
        return (self.p1[0], self.p1[1], 0.0)

    @property
    def coords(self) -> np.ndarray:
        """3D coordinates of grid points on the wall."""
        if self._cache.get("coords") is not None:
            return self._cache["coords"]

        # Local coordinates along wall (s) and up (t)
        s_vals = self.axes[0].points  # Along wall
        t_vals = self.axes[1].points  # Up (z)

        ss, tt = np.meshgrid(s_vals, t_vals, indexing="ij")
        s_flat = ss.ravel()
        t_flat = tt.ravel()

        # Convert to 3D: origin + s*u_hat + t*v_hat
        origin = np.array(self.origin)
        coords = (
            origin + s_flat[:, None] * self.u_hat + t_flat[:, None] * self.v_hat
        )

        self._cache["coords"] = coords
        return coords

    @property
    def calc_state(self) -> tuple:
        return (self.p1, self.p2, self.z_height, self.normal_2d, self.spacing, self.offset)

    @property
    def update_state(self) -> tuple:
        return ()

    def update(self, **changes):
        new = replace(self, **changes)
        object.__setattr__(new, "_cache", {})
        return new

    def to_dict(self) -> dict:
        return {
            "p1": list(self.p1),
            "p2": list(self.p2),
            "z_height": self.z_height,
            "normal_2d": list(self.normal_2d),
            "spacing_init": self.spacing,
            "offset": self.offset,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WallGrid":
        return cls(
            p1=tuple(data["p1"]),
            p2=tuple(data["p2"]),
            z_height=data["z_height"],
            normal_2d=tuple(data["normal_2d"]),
            spacing_init=data.get("spacing_init"),
            num_points_init=data.get("num_points_init"),
            offset=data.get("offset", True),
        )
