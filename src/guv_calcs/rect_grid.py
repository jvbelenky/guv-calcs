from dataclasses import dataclass, replace, field
import numbers
import numpy as np
import inspect
from .axis import Axis1D


@dataclass(frozen=True)
class RectGrid:
    mins: tuple[float, ...]
    maxs: tuple[float, ...]
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

        if len(self.mins) != len(self.maxs):
            raise ValueError(
                f"number of min ({len(self.mins)}) and max ({len(self.maxs)}) dimensions do not match"
            )

        if self.num_points_init is not None:
            if len(self.num_points_init) != len(self.mins):
                raise ValueError(
                    "num_points_init dimensions do not match min/max dimensions"
                )

        if self.spacing_init is not None:
            if len(self.spacing_init) != len(self.mins):
                raise ValueError(
                    "spacing_init dimensions do not match min/max dimensions"
                )

        if len(self.mins) > 3:
            raise ValueError("Maximum of three dimensions allowed")

        if len(self.mins) < 1:
            raise ValueError("Minimum of one dimension required")

        if type(self.offset) is not bool:
            raise TypeError("must be either True or False")

    @classmethod
    def from_dict(cls, data):
        keys = list(inspect.signature(cls.__init__).parameters.keys())[1:]
        return cls(**{k: v for k, v in data.items() if k in keys})

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
        spacing = self.spacing_init or (None,) * len(self.mins)
        num_points = self.num_points_init or (None,) * len(self.mins)
        for lo, hi, sp, n_pts in zip(self.mins, self.maxs, spacing, num_points):
            axis = Axis1D(
                lo=lo,
                hi=hi,
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
        points = [axis.points for axis in self.axes]
        self._cache["points"] = points
        return points

    @property
    def dimensions(self) -> tuple:
        return np.array(
            tuple(np.array((float(axis.lo), float(axis.hi))) for axis in self.axes)
        )

    @property
    def spans(self) -> tuple:
        return tuple(abs(float(axis.hi - axis.lo)) for axis in self.axes)

    @property
    def x1(self) -> float:
        return self.dimensions[0][0] if len(self.dimensions) > 0 else None

    @property
    def x2(self) -> float:
        return self.dimensions[0][1] if len(self.dimensions) > 0 else None

    @property
    def y1(self) -> float:
        return self.dimensions[1][0] if len(self.dimensions) > 1 else None

    @property
    def y2(self) -> float:
        return self.dimensions[1][1] if len(self.dimensions) > 1 else None

    @property
    def z1(self) -> float:
        return self.dimensions[2][0] if len(self.dimensions) > 2 else None

    @property
    def z2(self) -> float:
        return self.dimensions[2][1] if len(self.dimensions) > 2 else None

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
        dims = tuple(val for dim in self.dimensions for val in dim)
        return dims + tuple(self.spacing) + (self.offset,)

    @property
    def update_state(self) -> tuple:
        return ()


@dataclass(frozen=True)
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
        coords = np.array((X, Y, Z)).T
        coords = np.unique(coords, axis=0)
        self._cache["coords"] = coords
        return coords

    def to_dict(self):
        return {"mins": self.mins, "maxs": self.maxs, "spacing": self.spacing}

    def update_dimensions(self, mins, maxs, preserve_spacing=True):
        """mostly to keep the pattern with PlaneGrid"""
        if preserve_spacing:
            return self.update(mins=mins, maxs=maxs, spacing_init=self.spacing)
        else:
            return self.update(mins=mins, maxs=maxs, num_points_init=self.num_points)


@dataclass(frozen=True)
class PlaneGrid(RectGrid):
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    u_vec: tuple[float, float, float] = (1.0, 0.0, 0.0)
    v_vec: tuple[float, float, float] = (0.0, 1.0, 0.0)

    def __post_init__(self):

        if len(self.mins) > 2:
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

    def to_dict(self):
        return {
            "mins": self.mins,
            "maxs": self.maxs,
            "spacing": self.spacing,
            "origin": self.origin,
            "u_vec": self.u_vec,
            "v_vec": self.v_vec,
        }

    @property
    def calc_state(self):
        return (
            super().calc_state
            + tuple(self.origin)
            + tuple(self.u_vec)
            + tuple(self.v_vec)
        )

    @property
    def update_state(self):
        return super().update_state

    @property
    def dimensions(self):
        """bespoke overwrite--for planes initialized with origin/uvec/vvec"""
        dim1 = np.asarray(self.origin, float)
        dim2 = np.array(
            np.array(self.origin)
            + np.array(self.u_hat) * self.maxs[0]
            + np.array(self.v_hat) * self.maxs[1]
        )
        return np.array(tuple(np.array((a, b)) for a, b in zip(dim1, dim2) if a != b))

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
    def from_vectors(cls, *, origin, u_vec, v_vec, **kwargs):
        origin = np.asarray(origin, float)
        u = np.asarray(u_vec, float)
        v = np.asarray(v_vec, float)
        if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0:
            raise ValueError("from_vectors requires non-zero u_vec and v_vec")
        # Orthonormalize u and v to get a clean 2D parameter frame
        u_hat = u / np.linalg.norm(u)
        v_perp = v - np.dot(v, u_hat) * u_hat
        v_norm = np.linalg.norm(v_perp)
        if v_norm == 0:
            raise ValueError("from_vectors requires non-collinear points")
        v_hat = v_perp / v_norm
        s1, t1 = np.dot(u, u_hat), np.dot(u, v_hat)
        s2, t2 = np.dot(v, u_hat), np.dot(v, v_hat)
        mins = (0, 0)
        maxs = (max(s1, s2), max(t1, t2))
        return cls(
            mins=mins, maxs=maxs, origin=origin, u_vec=u_vec, v_vec=v_vec, **kwargs
        )

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
            origin = (0.0, 0.0, float(height))
            u_vec = (1.0, 0.0, 0.0)
            v_vec = (0.0, 1.0, 0.0)
        elif ref_surface == "xz":
            origin = (0.0, float(height), 0.0)
            u_vec = (1.0, 0.0, 0.0)
            v_vec = (0.0, 0.0, 1.0)
        elif ref_surface == "yz":
            origin = (float(height), 0.0, 0.0)
            u_vec = (0.0, 1.0, 0.0)
            v_vec = (0.0, 0.0, 1.0)

        if direction == -1:
            v_vec = tuple(-np.asarray(v_vec))

        return cls(
            mins=mins, maxs=maxs, origin=origin, u_vec=u_vec, v_vec=v_vec, **kwargs
        )

    def update_from_legacy(self, height, ref_surface, direction):
        """keep current dimensions, update height/orientation/reference surface"""
        return PlaneGrid.from_legacy(
            mins=self.mins,
            maxs=self.maxs,
            spacing_init=tuple(self.spacing),
            height=height,
            ref_surface=ref_surface,
            direction=direction,
        )

    def update_dimensions(self, mins, maxs, preserve_spacing=True):
        """keep current orientation, with different extents"""
        u1, u2, v1, v2 = mins[0], maxs[0], mins[1], maxs[1]
        u_vec = np.array(self.u_hat) * (u2 - u1)
        v_vec = np.array(self.v_hat) * (v2 - v1)
        origin = u1 * self.u_hat + v1 * self.v_hat
        if preserve_spacing:
            return PlaneGrid.from_vectors(
                u_vec=u_vec,
                v_vec=v_vec,
                origin=origin,
                spacing_init=self.spacing,
            )
        else:
            return PlaneGrid.from_vectors(
                u_vec=u_vec,
                v_vec=v_vec,
                origin=origin,
                num_points_init=self.num_points,
            )
