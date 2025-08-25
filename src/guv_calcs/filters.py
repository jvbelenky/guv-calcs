from abc import ABC
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .calc_zone import CalcPlane


class FilterBase(ABC):
    def __init__(self, p0, pU, pV, values, filter_id=None, name=None, enabled=True):

        self.filter_id = filter_id or "Filter"
        self.name = str(self.filter_id) if name is None else str(name)
        self.enabled = enabled

        self.p0 = np.asarray(p0, dtype=float)
        self.pU = np.asarray(pU, dtype=float)
        self.pV = np.asarray(pV, dtype=float)

        self.u = self.pU - self.p0
        self.v = self.pV - self.p0
        # pre-compute inverse 2×2 for (u,v)-decomposition
        A = np.column_stack([self.u, self.v])  # shape (3,2)
        self.inv = np.linalg.pinv(A.T @ A) @ A.T  # maps 3-vec → (a,b)
        # plane normal - should point into the volume
        self.n = np.cross(self.u, self.v)
        self.n /= np.linalg.norm(self.n)

        self.values = values
        self.alpha_nodes = np.linspace(0, np.linalg.norm(self.u), values.shape[0])
        self.beta_nodes = np.linspace(0, np.linalg.norm(self.v), values.shape[1])

    def apply(self, position, values, coords):
        """
        Multiply `values[i]` by the interpolated measured factor
        if the ray lamp.position → coords[i] crosses the rectangle exactly once.
        """
        if self.lookup is None:
            raise ValueError(f"Value grid for filter {self.name} missing")

        rel = coords - position  # shape (N,3)
        denom = rel @ self.n  # (N,)
        # ignore rays parallel to plane - tbh this bit isnt really necessary
        good = np.abs(denom) > 1e-12
        if not good.any():
            return values

        # signed distance of lamp and voxel from plane
        d0 = (position - self.p0) @ self.n
        d1 = (coords[good] - self.p0) @ self.n
        mask = (d0 > 0) & (d1 < 0)  # crossed in +→– sense
        if not mask.any():
            return values

        # parametric t at intersection, then hit-point world coords
        t = d0 / (d0 - d1[mask])
        hit = position + rel[good][mask] * t[:, None]  # (M,3)

        # convert hit→(a,b) in rectangle basis, check if inside [0,1]^2
        ab = (self.inv @ (hit - self.p0).T).T  # (M,2) → (a, b)
        inside = (ab[:, 0] >= 0) & (ab[:, 0] <= 1) & (ab[:, 1] >= 0) & (ab[:, 1] <= 1)
        if not inside.any():
            return values
            
        # convert to physical distances along each axis
        coords_lookup = np.column_stack(
            (
                ab[inside][:, 1] * np.linalg.norm(self.v),  # b·|v|
                ab[inside][:, 0] * np.linalg.norm(self.u),
            )
        )  # a·|u|

        factors = self.lookup(coords_lookup)  # (M_in,)
        values[np.flatnonzero(good)[mask][inside]] *= factors
        return values

    def get_calc_state(self):
        return [self.p0, self.u, self.v, self.values]


class MultFilter(FilterBase):
    """
    p0 : (3,) array
        World-space coordinates of the rectangle’s origin (corner 0,0).
    pU : (3,) array
        Corner at (1,0).  u-axis = pU – p0
    pV : (3,) array
        Corner at (0,1).  v-axis = pV – p0
    values : (Ny, Nx) array
        Grid of measured values
    """

    def __init__(self, p0, pU, pV, values, filter_id="MultFilter", **kwargs):

        super().__init__(
            p0=p0, pU=pU, pV=pV, values=values, filter_id=filter_id, **kwargs
        )

        self.lookup = RegularGridInterpolator(
            (self.beta_nodes, self.alpha_nodes),
            self.values.astype("float32"),
            bounds_error=False,
            fill_value=1.0,
        )

    def calculate_values(self, lamps, ref_manager=None, hard=False):
        pass


class MeasuredCorrection(FilterBase):
    """
    p0 : (3,) array
        World-space coordinates of the rectangle’s origin (corner 0,0).
    pU : (3,) array
        Corner at (1,0).  u-axis = pU – p0
    pV : (3,) array
        Corner at (0,1).  v-axis = pV – p0
    values : (Ny, Nx) array
        Grid of measured values
    """

    def __init__(self, p0, pU, pV, values, filter_id="CorrectionFilter", **kwargs):

        super().__init__(
            p0=p0, pU=pU, pV=pV, values=values, filter_id=filter_id, **kwargs
        )

        self.plane = CalcPlane.from_vectors(
            p0=p0,
            pU=pU,
            pV=pV,
            zone_id=self.filter_id,
            name=self.name,
            horiz=True,
            num_x=self.values.shape[0],
            num_y=self.values.shape[1],
        )

        self.lookup = None

    def calculate_values(self, lamps, ref_manager=None, hard=False):
        """
        calculate the modeled incidence on the plane and build a correction grid
        """
        self.plane.calculate_values(lamps=lamps, ref_manager=ref_manager, hard=hard)

        if self.plane.values.shape != self.values.shape:
            raise ValueError(
                f"Measured and modeled values in filter {self.name} must be the same shape"
            )

        correction = self.values / self.plane.values

        self.lookup = RegularGridInterpolator(
            (self.beta_nodes, self.alpha_nodes),
            correction.astype("float32"),
            bounds_error=False,
            fill_value=1.0,
        )


class BoxObstacle:
    def __init__(self, p_lo, p_hi, label=None, obs_id=None):
        self.id    = obs_id or self._next_id()
        self.label = label  or self.id
        p_lo, p_hi = np.minimum(p_lo, p_hi), np.maximum(p_lo, p_hi)
        x1,y1,z1 = p_lo; x2,y2,z2 = p_hi

        # 6 faces, normals point outward
        self.faces = [
            CorrectionGridFilter(p0=[x1,y1,z1], pU=[x2,y1,z1], pV=[x1,y2,z1],
                                 corr=np.zeros((1,1)), name=f"{self.id}_-z"),
            CorrectionGridFilter(p0=[x1,y1,z2], pU=[x2,y1,z2], pV=[x1,y2,z2],
                                 corr=np.zeros((1,1)), name=f"{self.id}_+z"),
            CorrectionGridFilter(p0=[x1,y1,z1], pU=[x2,y1,z1], pV=[x1,y1,z2],
                                 corr=np.zeros((1,1)), name=f"{self.id}_-y"),
            CorrectionGridFilter(p0=[x1,y2,z1], pU=[x2,y2,z1], pV=[x1,y2,z2],
                                 corr=np.zeros((1,1)), name=f"{self.id}_+y"),
            CorrectionGridFilter(p0=[x1,y1,z1], pU=[x1,y2,z1], pV=[x1,y1,z2],
                                 corr=np.zeros((1,1)), name=f"{self.id}_-x"),
            CorrectionGridFilter(p0=[x2,y1,z1], pU=[x2,y2,z1], pV=[x2,y1,z2],
                                 corr=np.zeros((1,1)), name=f"{self.id}_+x"),
        ]