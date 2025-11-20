from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass, field, replace


@dataclass
class ZoneGeometry(ABC):
    """
    A *pure-geometry* object.

    - Owns *only* the spatial/sampling state (coords,  spacing, etc.).
    - Knows nothing about lamps, reflectance, dose, field-of-view, export …
    - Subclasses implement `update()` to populate `.coords`, `num_points`, etc.
    ─────────────────────────────────────────────────────────────────────────────
    Contract expected by LightingCalculator / CalcZone:
    • coords        : (N,3) float32 array          – world coordinates
    • num_points    : tuple[int,…]                 – grid shape
    • points        : list[np.ndarray]             – 1-D grids per axis
    """

    offset: bool = True  # common to all
    coords: np.ndarray = field(init=False, repr=False)
    num_points: tuple = field(init=False)
    points: list[np.ndarray] = field(init=False)

    def __eq__(self, other):
        if not isinstance(other, ZoneGeometry):
            raise NotImplementedError
        return self.to_dict() == other.to_dict()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ZoneGeometry):
            return NotImplemented
        # Cheap structural equality – avoid comparing large arrays
        return self.get_calc_state() == other.get_calc_state()

    @abstractmethod
    def update(self) -> None:
        """Compute coords, num_points."""
        raise NotImplementedError

    @abstractmethod
    def get_calc_state(self) -> Sequence:
        """Return a tuple of parameters that trigger a recalculation."""
        raise NotImplementedError

    # ---------- (de)serialisation ----------
    def to_dict(self) -> Dict[str, Any]:
    """Round-trip safe – stores only *constructor* fields, never derived data."""
    data = asdict(self)
    # Remove derived / non-init fields
    for f in fields(self):
        if f.init is False:
            data.pop(f.name, None)
            data["class"] = self.__class__.__name__
    return data

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ZoneGeometry":
        klass = (
            cls
            if d.get("class") in (None, cls.__name__)
            else _locate_subclass(d["class"])
        )
        init_args = {k: v for k, v in d.items() if k != "class"}
        obj = klass(**init_args)  # type: ignore[arg-type]
        obj.update()
        return obj

    # ---------- JSON convenience ----------
    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> "ZoneGeometry":
        return cls.from_dict(json.loads(s))


@dataclass
class PlanarGeometry(ZoneGeometry):
    """
    May have an inferred or set normal
    """

    normal: np.ndarray | None = None  # (3,)

    @property
    def normal(self):
        if self.normal is None:
        return self._get_normal()
        return self.normal

    @property
    def basis(self):
        return self.get_basis()

    def get_basis(self, normal=None):
        # Generate arbitrary vector not parallel to n

        n = self.normal if normal is None else normal
        tmp = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])

        u = np.cross(n, tmp)
        u = u / np.linalg.norm(u)

        v = np.cross(n, u)
        return np.stack([u, v, n], axis=1)

    def _get_normal(self):
        """determine the plane's normal from the coordinates"""
        pts = self._get_coplanar()
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        n = np.cross(v1, v2)
        return n / np.linalg.norm(n)

    def _get_coplanar(self, tol=1e-12):
        """
        Return the first three non‑collinear points.
        points : (N, 3) array_like
        tol    : float   – threshold for ‖v1 × v2‖ below which we call it collinear
        """
        combos = combinations(range(len(self.coords)), 3)
        for i, j, k in combos:
            v1 = self.coords[j] - self.coords[i]
            v2 = self.coords[k] - self.coords[i]
        if np.linalg.norm(np.cross(v1, v2)) > tol:
            return np.array((self.coords[i], self.coords[j], self.coords[k]))
        raise ValueError("All points collinear within tolerance")
