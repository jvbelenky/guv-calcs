import numpy as np
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Axis1D:
    """1D axis with evenly spaced points over a span.

    Resolution is specified by either *spacing_init* (desired spacing) or
    *num_points_init* (desired point count).  These are mutually exclusive
    and represent the user's intent — they are stored verbatim and never
    modified by the class.

    The *spacing* property returns the **effective** spacing: the actual
    distance between adjacent grid points.  When spacing_init exceeds the
    span, it is clamped so the grid always contains at least one point.
    When the span later grows back, the original spacing_init takes effect
    automatically.
    """

    span: float = 1.0
    spacing_init: float | None = None  # desired spacing (user intent)
    num_points_init: int | None = None  # desired point count (user intent)
    offset: bool = True  # centre the grid inside [lo, hi]

    def __post_init__(self):
        if self.spacing_init is not None and self.spacing_init < 0:
            raise ValueError("spacing_init must be non-negative")
        if self.num_points_init is not None and self.num_points_init < 1:
            raise ValueError("num_points_init must be a positive integer")

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
        """Effective spacing between grid points.

        Priority: spacing_init > derived from num_points_init > default.
        Clamped to span so the grid always contains at least one point.
        """
        if self.span == 0:
            return 0.0
        if self.spacing_init is not None and self.spacing_init != 0:
            return min(self.spacing_init, self.span)
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

