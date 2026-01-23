import numpy as np
from .filters import ConstFilter
from .calc_zone import CalcPlane
from .reflectance import Surface


def _box_face_points(mins, maxs, face: str):
    """
    Return (p0, pU, pV) corner points for a box face.

    The points define a plane where:
    - p0 is the origin corner
    - pU defines the U-axis edge (p0 → pU)
    - pV defines the V-axis edge (p0 → pV)
    - Normal = cross(U, V) points OUTWARD from the box

    Parameters
    ----------
    mins, maxs : array-like
        Box corners (x_min, y_min, z_min) and (x_max, y_max, z_max)
    face : str
        One of: 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'

    Returns
    -------
    tuple of (p0, pU, pV) as numpy arrays
    """
    x1, y1, z1 = mins
    x2, y2, z2 = maxs

    # Each face defined so that cross(pU-p0, pV-p0) points outward
    face_points = {
        # -X face: normal points -X, so we need cross product to give (-1, 0, 0)
        "xmin": ([x1, y2, z1], [x1, y1, z1], [x1, y2, z2]),
        # +X face: normal points +X
        "xmax": ([x2, y1, z1], [x2, y2, z1], [x2, y1, z2]),
        # -Y face: normal points -Y
        "ymin": ([x1, y1, z1], [x2, y1, z1], [x1, y1, z2]),
        # +Y face: normal points +Y
        "ymax": ([x2, y2, z1], [x1, y2, z1], [x2, y2, z2]),
        # -Z face: normal points -Z
        "zmin": ([x1, y2, z1], [x2, y2, z1], [x1, y1, z1]),
        # +Z face: normal points +Z
        "zmax": ([x1, y1, z2], [x2, y1, z2], [x1, y2, z2]),
    }

    if face not in face_points:
        raise ValueError(f"face must be one of {list(face_points.keys())}, got {face!r}")

    p0, pU, pV = face_points[face]
    return np.array(p0), np.array(pU), np.array(pV)


class BoxObstacle:
    """
    A box-shaped obstacle defined by two corner points.

    Creates 6 surfaces (one per face) that can participate in reflectance
    calculations and occlusion filtering.
    """

    FACE_KEYS = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")

    def __init__(
        self,
        p1,
        p2,
        obs_id: str = None,
        name: str = None,
        R: float = 0.0,
        T: float = 0.0,
        face_overrides: dict = None,
        spacing: float = None,
        num_points: int = 10,
        enabled: bool = True,
    ):
        """
        Create a box obstacle from two corner points.

        Parameters
        ----------
        p1, p2 : array-like
            Two opposite corners of the box (x, y, z)
        obs_id : str, optional
            Unique identifier for this obstacle
        name : str, optional
            Display name for this obstacle
        R : float, default=0.0
            Default reflectance for all faces [0, 1]
        T : float, default=0.0
            Default transmittance for all faces [0, 1]
        face_overrides : dict, optional
            Per-face overrides, e.g. {"xmin": {"R": 0.5, "T": 0.1}}
        spacing : float, optional
            Grid spacing for surface calculations
        num_points : int, default=10
            Number of grid points per axis for surface calculations
        enabled : bool, default=True
            Whether this obstacle is active
        """
        self._obs_id = obs_id or "Obstacle"
        self.name = name if name is not None else str(self._obs_id)
        self.enabled = enabled

        self.p1 = np.asarray(p1, float)
        self.p2 = np.asarray(p2, float)

        # Default optical properties
        self._default_R = R
        self._default_T = T
        self._face_overrides = face_overrides or {}
        self._spacing = spacing
        self._num_points = num_points

        # Create surfaces for each face
        self.surfaces = {}
        self._build_surfaces()

    @property
    def obs_id(self) -> str:
        return self._obs_id

    @property
    def id(self) -> str:
        return self._obs_id

    def _assign_id(self, value: str) -> None:
        """Used by registry to assign unique ID"""
        old_id = self._obs_id
        self._obs_id = value
        # Update surface IDs to match
        new_surfaces = {}
        for fk in self.FACE_KEYS:
            old_key = f"{old_id}:{fk}"
            new_key = f"{value}:{fk}"
            if old_key in self.surfaces:
                surface = self.surfaces[old_key]
                surface.plane._assign_id(new_key)
                new_surfaces[new_key] = surface
        self.surfaces = new_surfaces

    def _build_surfaces(self):
        """Create Surface objects for each face of the box."""
        mins = np.minimum(self.p1, self.p2)
        maxs = np.maximum(self.p1, self.p2)

        for fk in self.FACE_KEYS:
            sid = f"{self._obs_id}:{fk}"

            # Get optical properties (face override or default)
            overrides = self._face_overrides.get(fk, {})
            R = overrides.get("R", self._default_R)
            T = overrides.get("T", self._default_T)

            # Get the corner points for this face
            p0, pU, pV = _box_face_points(mins, maxs, fk)

            # Create the CalcPlane using from_points (supports arbitrary orientation)
            # num_points_init expects a tuple (num_u, num_v) for the 2D grid
            num_pts = (self._num_points, self._num_points) if self._num_points else None
            plane = CalcPlane.from_points(
                p0=p0,
                pU=pU,
                pV=pV,
                zone_id=sid,
                spacing=self._spacing,
                num_points_init=num_pts,
                # Surface-appropriate settings
                horiz=True,
                vert=False,
                use_normal=True,
            )

            self.surfaces[sid] = Surface(R=R, T=T, plane=plane)

    def set_corners(self, p1, p2):
        """Update the box corners and rebuild all surface geometry."""
        self.p1 = np.asarray(p1, float)
        self.p2 = np.asarray(p2, float)
        self._build_surfaces()

    def to_dict(self) -> dict:
        """Serialize the obstacle for storage."""
        return {
            "obs_id": self._obs_id,
            "name": self.name,
            "p1": list(self.p1),
            "p2": list(self.p2),
            "R": self._default_R,
            "T": self._default_T,
            "face_overrides": self._face_overrides,
            "spacing": self._spacing,
            "num_points": self._num_points,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BoxObstacle":
        """Deserialize an obstacle from storage."""
        return cls(
            p1=data["p1"],
            p2=data["p2"],
            obs_id=data.get("obs_id"),
            name=data.get("name"),
            R=data.get("R", 0.0),
            T=data.get("T", 0.0),
            face_overrides=data.get("face_overrides"),
            spacing=data.get("spacing"),
            num_points=data.get("num_points", 10),
            enabled=data.get("enabled", True),
        )

    def __repr__(self):
        return (
            f"BoxObstacle(id={self._obs_id!r}, "
            f"p1={list(self.p1)}, p2={list(self.p2)}, "
            f"R={self._default_R}, T={self._default_T}, "
            f"enabled={self.enabled})"
        )


class BoxObstacleSlab:
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
        self._obs_id = obs_id or "Obstacle"
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
