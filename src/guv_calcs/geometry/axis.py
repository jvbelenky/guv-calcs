import numpy as np
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Axis1D:
    """
    span: float, default= 5
    spacing_init: float, default = None
        desired spacing between points. will override num_points_init. If None,
        first num_points_init will be used, then defaults generated.
    num_points_init: desired number of points
    offset: if true, the points will be centered between lo and hi. If false,

    """

    span: float = 1.0
    spacing_init: float | None = None  # spacing - will override num_points
    num_points_init: int | None = None  # alternatively, specify point count
    offset: bool = True  # centre the grid inside [lo, hi]

    def __post_init__(self):
        self._check_spacing(self.spacing_init)
        self._check_num_points(self.num_points_init)
        # print(self.spacing_init,self.num_points_init)
        # print(self.spacing, self.num_points)

    @property
    def default_spacing(self):
        # define a
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
        """
        either initial value passsed, derived from initial num_points value,
        or from default num_points value
        """
        if self.span == 0:
            return 0.0
        if self.spacing_init is None:
            # if not provided, derive from num_points (or default num_points)
            num_points = self.num_points_init or self.default_num_points
            return self._set_spacing(num_points, self.default_spacing)
        return self.spacing_init

    @property
    def num_points(self):
        if self.span == 0:
            return 1
        if self.spacing_init is None:
            return self.num_points_init or self.default_num_points
        else:
            return int(round(self.span / self.spacing_init))

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
