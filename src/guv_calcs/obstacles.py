import numpy as np
from .filters import ConstFilter


class BoxObstacle:
    def __init__(
        self,
        lo,
        hi,
        transmittance=0.0,
        mu=0.96,
        obs_id=None,
        name=None,
        enabled=True,
        eps=1e-12,
    ):
        self.obs_id = obs_id or "Obstacle"
        self.name = str(self.obs_id) if name is None else str(name)
        self.enabled = enabled

        self.eps = eps

        self.lo, self.hi = np.minimum(lo, hi), np.maximum(lo, hi)

        self.transmittance = transmittance
        self.mu = mu

    def apply(self, values, position, coords):
        p = position.astype(float)  # (3,)
        c = coords.astype(float)  # (N,3)
        d = c - p  # (N,3)
        # Reject segments parallel to an axis but starting outside that slab
        par = np.abs(d) < self.eps  # (N,3)
        parok = (~par) | ((self.lo <= p) & (p <= self.hi))  # broadcast (3,)
        parok = np.all(parok, axis=1)  # (N,)

        inv_d = np.where(np.abs(d) > self.eps, 1.0 / d, np.inf)  # (N,3)
        t0 = (self.lo - p) * inv_d  # (N,3)
        t1 = (self.hi - p) * inv_d  # (N,3)
        tmin = np.maximum.reduce(np.minimum(t0, t1), axis=1)  # (N,)
        tmax = np.minimum.reduce(np.maximum(t0, t1), axis=1)  # (N,)

        # Segment intersects box if the interval overlaps [0,1]
        hit = (
            parok
            & (tmax >= -self.eps)
            & (tmin <= 1 + self.eps)
            & (tmax >= tmin - self.eps)
        )
        if not hit.any():
            return values

        # Length inside the box
        t_enter = np.maximum(tmin, 0.0)
        t_exit = np.minimum(tmax, 1.0)
        Lfrac = np.clip(t_exit - t_enter, 0.0, None)  # (N,)
        idx = np.flatnonzero(hit & (Lfrac > 0))
        if idx.size == 0:
            return values

        Lin = Lfrac[idx] * np.linalg.norm(d[idx], axis=1)  # physical length
        # Count boundary crossings for surface loss (0/1/2)
        inside_p = np.all((p >= self.lo - self.eps) & (p <= self.hi + self.eps))
        inside_c = np.all(
            (c[idx] >= self.lo - self.eps) & (c[idx] <= self.hi + self.eps), axis=1
        )
        k = np.where(inside_p ^ inside_c, 1, 2)  # enter/exit or through
        # (If you want to penalize tangential touches: add (Lfrac==0)→k=0 branch.)

        factors = (self.transmittance ** k) * np.exp(-self.mu * Lin)
        values[idx] *= factors.astype(values.dtype, copy=False)
        return values


class BoxObstacleOld2:
    def __init__(
        self, lo, hi, opacity=1.0, obs_id=None, name=None, enabled=True, eps=1e-12
    ):
        self.obs_id = obs_id or "Obstacle"
        self.name = str(self.obs_id) if name is None else str(name)
        self.enabled = enabled

        self.eps = eps

        self.lo, self.hi = np.minimum(lo, hi), np.maximum(lo, hi)

        self.opacity = opacity

    def apply(self, values, position, coords):
        """Zero rays whose segment [emitter_pos → coords[i]] intersects the box."""

        p = position.astype(float)  # (3,)
        c = coords.astype(float)  # (N,3)
        d = c - p  # (N,3)

        # Parallel-axis rejection: if d_j==0 and p_j is outside [lo_j, hi_j] → miss
        par = np.abs(d) < self.eps  # (N,3)
        par_ok = (~par) | ((self.lo <= p) & (p <= self.hi))  # broadcast (3,) → (N,3)
        par_ok = np.all(par_ok, axis=1)  # (N,)

        # Slab intersection (vectorized)
        inv_d = np.where(np.abs(d) > self.eps, 1.0 / d, np.inf)  # (N,3)
        t0 = (self.lo - p) * inv_d  # (N,3)
        t1 = (self.hi - p) * inv_d  # (N,3)
        tmin = np.maximum.reduce(np.minimum(t0, t1), axis=1)  # (N,)
        tmax = np.minimum.reduce(np.maximum(t0, t1), axis=1)  # (N,)

        # Segment hit if 0 ≤ tmin ≤ tmax ≤ 1 (allow tiny eps)
        hit = (
            par_ok
            & (tmax >= -self.eps)
            & (tmin <= 1 + self.eps)
            & (tmax >= tmin - self.eps)
        )
        if hit.any():
            values[hit] *= 1.0 - self.opacity
            return values


class BoxObstacleOld:
    def __init__(self, lo, hi, obs_id=None, name=None, enabled=True):
        self.obs_id = obs_id or "Obstacle"
        self.name = str(self.obs_id) if name is None else str(name)
        self.enabled = enabled

        self.lo, self.hi = np.minimum(lo, hi), np.maximum(lo, hi)
        x1, y1, z1 = self.lo
        x2, y2, z2 = self.hi

        # 6 faces, normals point outward
        self.faces = [
            ConstFilter(
                p0=[x1, y1, z1],
                pU=[x2, y1, z1],
                pV=[x1, y2, z1],
                values=0,
                name=f"{self.obs_id}_-z",
            ),
            ConstFilter(
                p0=[x1, y1, z2],
                pU=[x2, y1, z2],
                pV=[x1, y2, z2],
                values=0,
                name=f"{self.obs_id}_+z",
            ),
            ConstFilter(
                p0=[x1, y1, z1],
                pU=[x2, y1, z1],
                pV=[x1, y1, z2],
                values=0,
                name=f"{self.obs_id}_-y",
            ),
            ConstFilter(
                p0=[x1, y2, z1],
                pU=[x2, y2, z1],
                pV=[x1, y2, z2],
                values=0,
                name=f"{self.obs_id}_+y",
            ),
            ConstFilter(
                p0=[x1, y1, z1],
                pU=[x1, y2, z1],
                pV=[x1, y1, z2],
                values=0,
                name=f"{self.obs_id}_-x",
            ),
            ConstFilter(
                p0=[x2, y1, z1],
                pU=[x2, y2, z1],
                pV=[x2, y1, z2],
                values=0,
                name=f"{self.obs_id}_+x",
            ),
        ]

    def apply(self, values, position, coords):

        for face in self.faces:
            values = face.apply(values, position, coords)

        return values
