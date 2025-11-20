import numpy as np
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class Axis1D:
    lo: float = 0  # lower bound
    hi: float = 5  # upper bound
    spacing_init: float | None = None  # spacing - will override n_pts
    n_pts_init: int | None = None  # alternatively, specify point count
    offset: bool = True  # centre the grid inside [lo, hi]
    
    def __post_init__(self):
        self._check_spacing(self.spacing_init)
        self._check_n_pts(self.n_pts_init)
            
    @property
    def span(self):
        return abs(self.hi - self.lo)

    @property
    def default_spacing(self):
        if self.span==0.0:
            return 0.0
        n = max(1, min(int(self.span * 10), 20))
        sigfigs = -int(np.floor(np.log10(self.span / n)))
        return round(self.span / n, sigfigs)

    @property
    def default_n_pts(self):
        if self.span == 0:
            return 1
        return int(round(self.span / self.default_spacing))

    @property
    def offset_val(self):
        return min(self.lo, self.hi)
        
    @property
    def spacing(self):
        if self.spacing_init is None:
            n_pts = self.n_pts_init or self.default_n_pts
            return self._set_spacing(n_pts, self.default_spacing)
        return self.spacing_init
            
    @property
    def n_pts(self):
        if self.spacing_init is None:
            return self.n_pts_init or self.default_n_pts
        else:
            return int(round(self.span / self.spacing_init))
    
    @property
    def points(self):    
        pts = np.array([i * self.spacing + self.offset_val for i in range(self.n_pts)])
        if self.offset:
            pts += (self.span - abs(pts[-1] - pts[0])) / 2       
        return pts

    def _set_spacing(self, num, spacing):
        """set the spacing value conservatively from a num_points value"""
        
        if spacing >0 and int(self.span / spacing) == int(num):
            return spacing  # no changes needed
        else:
            testval = self.span / round(num)
            i = 1
            while i < 6:
                val = round(testval, i)
                if val != 0 and int(self.span / round(testval, i) + 1) == num:
                    breakpoin
                i += 1
                val = testval  # if no rounded value works use the original value
            return val
            
    def _check_spacing(self, spacing):
        if spacing is not None:
            if spacing > self.span:
                raise ValueError("Spacing value cannot be larger than dimension")
            
    def _check_n_pts(self, n_pts):
        if n_pts is not None:
            if n_pts < 1:
                raise ValueError("n_pts_init must be positive integer")