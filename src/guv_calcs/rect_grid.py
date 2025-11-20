from dataclasses import dataclass, replace
import numbers
import numpy as np
from .axis import Axis1D


@dataclass(frozen=True, slots=True)
class RectGrid:
    mins: tuple
    maxs: tuple
    fallback_n_pts: int
    n_pts: tuple | None = None
    spacings: tuple | None = None
    offset: bool = True

    def __post_init__(self):

        if len(self.mins) != len(self.maxs):
            raise ValueError(
                f"number of min ({len(self.mins)}) and max ({len(self.maxs)}) dimensions do not match"
            )

        if self.n_pts is not None:
            if len(self.n_pts) != len(self.mins):
                raise ValueError("n_pts dimensions do not match min/max dimensions")

        if self.spacings is not None:
            if len(self.spacings) != len(self.mins):
                raise ValueError("spacing dimensions do not match min/max dimensions")

        if len(self.mins) > 3:
            raise ValueError("Maximum of three dimensions allowed")

        if len(self.mins) < 1:
            raise ValueError("Minimum of one dimension required")

        if type(self.offset) is not bool:
            raise TypeError("must be either True or False")

    def update(self, **changes):
        return replace(self, **changes)

    @property
    def axes(self):
        axes = []
        spacings = self.spacings or (None,) * len(self.mins)
        num_points = self.n_pts or (None,) * len(self.mins)
        for lo, hi, spacing, n_pts in zip(self.mins, self.maxs, spacings, num_points):
            axis = Axis1D(
                lo=lo,
                hi=hi,
                spacing_init=spacing,
                n_pts_init=n_pts,
                offset=self.offset,
                fallback_n_pts=self.fallback_n_pts,
            )
            axes.append(axis)
        return axes

    @property
    def points(self):
        return [axis.points for axis in self.axes]

    @property
    def num_points(self):
        return np.array([len(pt) for pt in self.points])

    @property
    def spacings(self):
        return np.array([axis.spacing for axis in self.axes])

    @property
    def dimensions(self):
        return tuple((axis.lo, axis.hi) for axis in self.axes)

    @property
    def x1(self):
        return self.dimensions[0][0] if len(self.dimensions) > 0 else None

    @property
    def x2(self):
        return self.dimensions[0][1] if len(self.dimensions) > 0 else None

    @property
    def y1(self):
        return self.dimensions[1][0] if len(self.dimensions) > 1 else None

    @property
    def y2(self):
        return self.dimensions[1][1] if len(self.dimensions) > 1 else None

    @property
    def z1(self):
        return self.dimensions[2][0] if len(self.dimensions) > 2 else None

    @property
    def z2(self):
        return self.dimensions[2][1] if len(self.dimensions) > 2 else None

    @property
    def num_x(self):
        return self.num_points[0] if len(self.num_points) > 0 else None

    @property
    def num_y(self):
        return self.num_points[1] if len(self.num_points) > 1 else None

    @property
    def num_z(self):
        return self.num_points[2] if len(self.num_points) > 2 else None

    @property
    def x_spacing(self):
        return self.spacings[0] if len(self.spacings) > 0 else None

    @property
    def y_spacing(self):
        return self.spacings[1] if len(self.spacings) > 1 else None

    @property
    def z_spacing(self):
        return self.spacings[2] if len(self.spacings) > 2 else None

    @property
    def xp(self):
        return self.points[0] if len(self.points) > 0 else None

    @property
    def yp(self):
        return self.points[1] if len(self.points) > 1 else None

    @property
    def zp(self):
        return self.points[2] if len(self.points) > 2 else None


@dataclass(frozen=True, slots=True)
class VolGrid(RectGrid):
    fallback_n_pts: int = 20

    @property
    def coords(self):
        X, Y, Z = [
            grid.reshape(-1) for grid in np.meshgrid(*self.points, indexing="ij")
        ]
        coords = np.array((X, Y, Z)).T

        return np.unique(coords, axis=0)


@dataclass(frozen=True, slots=True)
class PlaneGrid(RectGrid):
    fallback_n_pts: int = 50
    height: float = 0
    ref_surface: str = "xy"
    direction: int = 1

    def __post_init__(self):

        if len(self.mins) > 2:
            raise ValueError("Too many dimensions for a plane")

        if not isinstance(self.ref_surface, str):
            raise TypeError("ref_surface must be a string in [`xy`,`xz`,`yz`]")
        if self.ref_surface.lower() not in ["xy", "xz", "yz"]:
            raise ValueError("ref_surface must be a string in [`xy`,`xz`,`yz`]")

        if not isinstance(self.height, numbers.Number):
            raise TypeError("Height must be numeric")

        if self.direction is not None and self.direction not in [1, 0, -1]:
            raise ValueError("Direction must be in [1, 0, -1]")

    @property
    def coords(self):
        X, Y = [grid.reshape(-1) for grid in np.meshgrid(*self.points, indexing="ij")]

        if self.ref_surface.lower() in ["xy"]:
            Z = np.full(X.shape, self.height)
        elif self.ref_surface.lower() in ["xz"]:
            Z = Y
            Y = np.full(Y.shape, self.height)
        elif self.ref_surface.lower() in ["yz"]:
            Z = Y
            Y = X
            X = np.full(X.shape, self.height)

        return np.stack([X, Y, Z], axis=-1)

    @property
    def basis(self):
        """
        Return an orthonormal basis (u, v, n) for the surface:
        - n is the normal vector (points outward from surface)
        - u, v span the surface plane
        """

        if self.ref_surface == "xy":
            n = np.array([0, 0, 1])
        elif self.ref_surface == "xz":
            n = np.array([0, 1, 0])
        elif self.ref_surface == "yz":
            n = np.array([1, 0, 0])
        if self.direction != 0:
            n *= self.direction

        # Generate arbitrary vector not parallel to n
        tmp = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])

        u = np.cross(n, tmp)
        u = u / np.linalg.norm(u)

        v = np.cross(n, u)
        basis = np.stack([u, v, n], axis=1)

        return basis
