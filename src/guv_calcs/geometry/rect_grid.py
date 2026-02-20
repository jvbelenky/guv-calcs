from dataclasses import dataclass, replace, field
import numbers
import numpy as np
import warnings
from .axis import Axis1D
from .._serialization import init_from_dict


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
        return init_from_dict(cls, data)

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

    def _extra_dict(self):
        return super()._extra_dict()

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
            return self.update(origin=origin, spans=spans, num_points_init=self.num_points)


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
        # Preserve the fixed-axis component (e.g. height for axis-aligned planes)
        normal = self.normal
        origin = origin + np.dot(np.asarray(self.origin), normal) * normal
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
                num_points_init=self.num_points, spacing_init=None,
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
                origin = (mins[0], maxs[1], height)
                u_vec = (1, 0, 0)
                v_vec = (0, -1, 0)

        elif ref_surface == "xz":
            if direction == 1:
                origin = (mins[0], height, maxs[1])
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
                origin = (height, mins[0], maxs[1])
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
            return None

        n = self.normal
        if rs == "xy":  # normal ±z
            return 1 if n[2] > 0 else -1
        elif rs == "xz":  # normal ±y
            return 1 if n[1] > 0 else -1
        elif rs == "yz":  # normal ±x
            return 1 if n[0] > 0 else -1

    @classmethod
    def from_wall(cls, p1, p2, z_height, normal_2d=None, **kwargs):
        """Create a vertical wall plane from two 2D vertices."""
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        edge_length = np.sqrt(dx * dx + dy * dy)
        origin = (p1[0], p1[1], 0.0)
        u_vec = (dx, dy, 0.0)  # along wall
        v_vec = (0.0, 0.0, 1.0)  # up
        spans = (edge_length, z_height)
        return cls(
            origin=origin,
            spans=spans,
            u_vec=u_vec,
            v_vec=v_vec,
            **kwargs
        )
