import numpy as np
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Axis1D:
    """1D axis with evenly spaced points over a span.

    Spacing is authoritative: span and spacing fully determine the grid.
    num_points_init is a convenience setter — when provided without spacing,
    the spacing is derived from span / num_points.  When both are provided,
    spacing wins and num_points is derived from span / spacing.
    """

    span: float = 1.0
    spacing_init: float | None = None  # spacing - will override num_points
    num_points_init: int | None = None  # alternatively, specify point count
    offset: bool = True  # centre the grid inside [lo, hi]

    def __post_init__(self):
        self._check_spacing(self.spacing_init)
        self._check_num_points(self.num_points_init)

    @property
    def default_spacing(self):
        if self.span == 0.0:
            return 0.0
        n = max(1, min(int(self.span * 10), 30))
        sigfigs = -int(np.floor(np.log10(self.span / n)))
        return round(self.span / n, sigfigs)

    @property
    def default_num_points(self):
        if self.span == 0:
            return 1
        return int(round(self.span / self.default_spacing))

    @property
    def spacing(self):
        """Spacing between grid points.

        Priority: spacing_init > derived from num_points_init > default.
        """
        if self.span == 0:
            return 0.0
        if self.spacing_init is not None and self.spacing_init != 0:
            return self.spacing_init
        # derive from num_points (or default num_points)
        num_points = self.num_points_init or self.default_num_points
        return self._set_spacing(num_points, self.default_spacing)

    @property
    def num_points(self):
        """Number of grid points — always derived from span / spacing.

        Uses floor division so the grid never exceeds the span.
        """
        if self.span == 0:
            return 1
        return max(1, int(self.span / self.spacing))

    @property
    def points(self):
        # Ensure float dtype to allow in-place addition of float offset
        pts = np.array([i * self.spacing for i in range(self.num_points)], dtype=float)
        if self.offset:
            pts += (self.span - abs(pts[-1] - pts[0])) / 2
        return pts

    def _set_spacing(self, num, spacing):
        """set the spacing value conservatively from a num_points value"""

        if spacing > 0 and int(self.span / spacing) == int(num):
            return spacing  # no changes needed
        else:
            testval = self.span / num
            i = 1
            while i < 6:
                val = round(testval, i)
                if val != 0 and int(self.span / val) == int(num):
                    break
                i += 1
                val = testval  # if no rounded value works use the original value
            return val

    def _check_spacing(self, spacing):
        if spacing is not None and self.span > 0:
            if spacing > self.span:
                raise ValueError("Spacing value cannot be larger than dimension")

    def _check_num_points(self, num_points):
        if num_points is not None:
            if num_points < 1:
                raise ValueError("num_points_init must be positive integer")
