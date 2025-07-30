import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .calc_zone import CalcPlane


class MeasuredCorrection:
    """
    A position-dependent entrance-plane measured.


    Parameters
    ----------
    p0 : (3,) array
        World-space coordinates of the rectangle’s origin (corner 0,0).
    pU : (3,) array
        Corner at (1,0).  u-axis = pU – p0
    pV : (3,) array
        Corner at (0,1).  v-axis = pV – p0
    measured : (Ny, Nx) array
        Grid of measured values
    """

    def __init__(self, p0, pU, pV, measured, filter_id=None, name=None):

        self.filter_id = str(filter_id)
        self.name = filter_id if name is None else name

        self.measured = measured

        self.p0 = np.asarray(p0, dtype=float)
        self.u = np.asarray(pU, dtype=float) - self.p0
        self.v = np.asarray(pV, dtype=float) - self.p0
        # pre-compute inverse 2×2 for (u,v)-decomposition
        A = np.column_stack([self.u, self.v])  # shape (3,2)
        self.inv = np.linalg.pinv(A.T @ A) @ A.T  # maps 3-vec → (α,β)
        # plane normal (points *into* the volume if u×v chosen that way)
        self.n = np.cross(self.u, self.v)
        self.n /= np.linalg.norm(self.n)
        # 2-D interpolator → scalar

        self.x_nodes = np.linspace(0, self.u[0], measured.shape[0])
        self.y_nodes = np.linspace(0, self.v[1], measured.shape[1])

        self.plane = CalcPlane.from_vectors(
            p0=p0,
            pU=pU,
            pV=pV,
            zone_id=self.filter_id,
            name=self.name,
            horiz=True,
            num_x=measured.shape[0],
            num_y=measured.shape[1],
        )

        self.lookup = None

    def calculate_values(self, lamps, ref_manager=None, hard=False):
        """
        calculate the modeled incidence on the plane and build a correction grid
        """

        self.plane.calculate_values(lamps=lamps, ref_manager=ref_manager, hard=hard)

        if self.plane.values.shape != self.measured.shape:
            raise ValueError(
                f"Measured and modeled values in filter {self.name} must be the same shape"
            )

        correction = self.measured / self.plane.values

        self.lookup = RegularGridInterpolator(
            (self.x_nodes, self.y_nodes),
            correction.astype("float32"),
            bounds_error=False,
            fill_value=1.0,
        )

    def apply(self, lamp, values, coords):
        """
        Multiply `values[i]` by the interpolated measured factor
        if the ray lamp.position → coords[i] crosses the rectangle exactly once.
        """
        if self.lookup is None:
            raise ValueError(f"Correction filter {self.name} has not been calculated")

        rel = coords - lamp.position  # shape (N,3)
        denom = rel @ self.n  # (N,)
        # ignore rays parallel to plane - tbh this bit isnt really necessary
        good = np.abs(denom) > 1e-12
        if not good.any():
            return values

        # signed distance of lamp and voxel from plane
        d0 = (lamp.position - self.p0) @ self.n
        d1 = (coords[good] - self.p0) @ self.n
        mask = (d0 > 0) & (d1 < 0)  # crossed in +→– sense
        if not mask.any():
            return values

        # parametric t at intersection, then hit-point world coords
        t = d0 / (d0 - d1[mask])
        hit = lamp.position + rel[good][mask] * t[:, None]  # (M,3)

        # convert hit→(α,β) in rectangle basis, check if inside [0,1]²
        ab = (self.inv @ (hit - self.p0).T).T  # (M,2)
        inside = (ab[:, 0] >= 0) & (ab[:, 0] <= 1) & (ab[:, 1] >= 0) & (ab[:, 1] <= 1)
        if not inside.any():
            return values

        # interpolate measured on the measured grid (swap order to (x,y))
        factors = self.lookup(ab[inside][:, ::-1])  # (M_in,)
        # broadcast back to full values array
        idx = np.flatnonzero(good)[mask][inside]
        values[idx] *= factors
        return values

    def get_calc_state(self):
        return [self.p0, self.u, self.v, self.measured]
