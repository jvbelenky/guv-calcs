import numpy as np
from enum import Enum
from dataclasses import dataclass, replace, field
from .geometry import PlanarGeometry

# @dataclass(frozen=True)
# class FaceSpec:
# u_axis: int          # 0 = x, 1 = y, 2 = z
# v_axis: int
# const_axis: int
# ref_plane: str
# normal_sign: int     # +1 or –1

# class Face(Enum):
# FLOOR   = FaceSpec(0, 1, 2, "xy", +1)   # xy plane, +z normal
# CEILING = FaceSpec(0, 1, 2, "xy", -1)   # xy plane, –z normal
# SOUTH   = FaceSpec(0, 2, 1, "xz", +1)   # xz plane, +y normal
# NORTH   = FaceSpec(0, 2, 1, "xz", -1)
# WEST    = FaceSpec(1, 2, 0, "yz", +1)   # yz plane, +x normal
# EAST    = FaceSpec(1, 2, 0, "yz", -1)

# @staticmethod
# def coerce(value: str | "Face") -> "Face":
# if isinstance(value, Face):
# return value
# return Face[value.strip().upper()]


@dataclass
class RectGrid(PlanarGeometry):

    """
    Axis-aligned rectangular *plane* geometry.
    """

    # -------- extents --------
    x1: float = 0.0
    x2: float = 6.0
    y1: float = 0.0
    y2: float = 4.0
    height: float = 1.8
    # -------- optional - filled in with defaults if not provided --------
    num_x: int | None = None
    num_y: int | None = None
    x_spacing: float | None = None
    y_spacing: float | None = None

    self.ref_surface: str = "xy"

    # --------------- read-only cached fields --------------
    _xp = field(init=False, repr=False, default=None)
    _yp = field(init=False, repr=False, default=None)
    _coords = field(init=False, repr=False, default=None)
    _points = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self._xp = Axis1D(self.x1, self.x2, self.x_spacing, self.num_x, self.offset)
        self._yp = Axis1D(self.y1, self.y2, self.y_spacing, self.num_y, self.offset)

    @property
    def coords(self) -> np.ndarray:
        if self._coords is None:
            self._update()
        return self._coords

    @property
    def points(self) -> list[np.ndarray]:
        if self._points is None:
            self._update()
        return self._points

    @property
    def num_points(self) -> np.ndarray:
        xp, yp = self.points  # 1-D grids
        return np.array([len(xp), len(yp)])

    @property
    def dimensions(self) -> tuple:
        return ((self.x1, self.y1), (self.x2, self.y2))

    def set_dimensions(self, x1=None, x2=None, y1=None, y2=None):
        """set the desired dimensions"""
        self._xp = self._xp.update(
            lo=self.x1 if x1 is None else x1,
            hi=self.x2 if x2 is None else x2,
            n_pts=None,
        )
        self._yp = self._yp.update(
            lo=self.y1 if y1 is None else y1,
            hi=self.y2 if y2 is None else y2,
            n_pts=None,
        )
        self._update()

    def set_spacing(self, x_spacing=None, y_spacing=None):
        """
        set the spacing desired in the dimension
        """
        self._xp = self._xp.update(
            spacing=self.x_spacing if x_spacing is None else x_spacing,
            n_pts=None,
        )
        self._yp = self._yp.update(
            spacing=self.y_spacing if y_spacing is None else y_spacing,
            n_pts=None,
        )
        self._update()

    def set_num_points(self, num_x=None, num_y=None):
        """
        set the number of points desired in a dimension, instead of setting the spacing
        """
        self._xp = self._xp.update(
            spacing=None,
            n_pts=self.num_x if num_x is None else self.num_x,
        )
        self._yp = self._yp.update(
            spacing=None,
            n_pts=self.num_y if num_y is None else self.num_y,
        )
        self._update()

    def _update(self) -> None:
        """Populate grid, basis, coords, num_points, points."""

        self.x1, self.x2 = self._xp.lo, self._xp.hi
        self.y1, self.y2 = self._yp.lo, self._yp.hi
        self.num_x, self.x_spacing = self._xp.n_pts, self._xp.spacing
        self.num_y, self.y_spacing = self._yp.n_pts, self._yp.spacing

        # set points
        self._points = [self._xp.points, self._yp.points]

        X, Y = [grid.reshape(-1) for grid in np.meshgrid(*self._points, indexing="ij")]
        if self.ref_surface in ["xy"]:
            Z = np.full(X.shape, self.height)
        elif self.ref_surface in ["xz"]:
            Z = Y
            Y = np.full(Y.shape, self.height)
        elif self.ref_surface in ["yz"]:
            Z = Y
            Y = X
            X = np.full(X.shape, self.height)
        self._coords = np.stack([X, Y, Z], axis=-1)


@dataclass(slots=True, frozen=True)
class Axis1D:
    lo: float = 0  # lower bound
    hi: float = 5  # upper bound
    spacing: float | None = None  # spacing - will override n_pts
    n_pts: int | None = None  # …or point-count (≥1)
    offset: bool = True  # centre the grid inside [lo, hi]

    def __post_init__(self):

        span = abs(self.hi - self.lo)

        # default spacing
        n_pts = min(int(span * 10), 50)
        sigfigs = -int(np.floor(np.log10(span / n_pts)))
        default_spacing = round(span / n_pts, sigfigs)

        # default num_points
        default_n_pts = int(round(span / default_spacing))

        # priority is always to spacing
        if self.spacing is None:
            object.__setattr__(self, "n_pts", self.n_pts or default_n_pts)
            object.__setattr__(self, "spacing", self._set_spacing())
        else:
            object.__setattr__(self, "n_pts", int(round(span / self.spacing)))

        if self.lo == self.hi:
            object.__setattr__(self, "n_pts", 1)

    def update(self, **changes):
        return replace(self, **changes)

    @property
    def points(self):

        span = abs(self.hi - self.lo)

        # generate default spacing based on an approx number of points
        n_pts_init = 1 if np.isclose(span, 0) else min(int(span * 10), 50)
        sigfigs = -int(np.floor(np.log10(span / n_pts_init)))
        default_spacing = round(span / n_pts_init, sigfigs)
        # default num_points based on default spacing
        default_n_pts = int(round(span / default_spacing))

        if self.spacing is None:
            n_pts = self.n_pts or default_n_pts
            spacing = self._set_spacing(n_pts)
        else:
            n_pts = int(round(span / self.spacing))
            spacing = self.spacing

        offset = min(self.hi, self.lo)
        pts = np.array([i * spacing + offset for i in range(n_pts)])
        if self.offset:
            pts += (span - abs(pts[-1] - pts[0])) / 2
        return pts

    def _set_spacing(self, n_pts):
        """set the spacing value conservatively from a num_points value"""
        span = abs(self.hi - self.lo)
        if int(span / self.spacing) == int(n_pts):
            val = self.spacing  # no changes needed
        else:
            testval = span / round(n_pts)
            i = 1
            while i < 6:
                val = round(testval, i)
                if val != 0 and int(span / round(testval, i) + 1) == n_pts:
                    break
                i += 1
                val = testval  # if no rounded value works use the original value
        return val
