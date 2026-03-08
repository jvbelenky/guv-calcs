from dataclasses import dataclass, replace, field
import numbers
import numpy as np
import warnings
from .axis import Axis1D
from .polygon import Polygon2D
from .._serialization import init_from_dict, migrate_surface_grid_dict, migrate_volume_grid_dict
from ..units import convert_length, convert_length_tuple


@dataclass(frozen=True, eq=False)
class _GridBase:
    """Shared machinery for surface and volume grids."""

    polygon: Polygon2D
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    u_vec: tuple[float, float, float] = (1.0, 0.0, 0.0)
    v_vec: tuple[float, float, float] = (0.0, 1.0, 0.0)
    spacing_init: tuple[float, ...] | None = None
    num_points_init: tuple[int, ...] | None = None
    offset: bool = True
    _cache: dict = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self):
        if not isinstance(self.polygon, Polygon2D):
            poly = Polygon2D(vertices=tuple(tuple(v) for v in self.polygon))
            object.__setattr__(self, "polygon", poly)

        u = np.array(self.u_vec, float)
        v = np.array(self.v_vec, float)

        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise ValueError("u_vec and v_vec must be finite numeric vectors.")

        if np.linalg.norm(u) < 1e-12 or np.linalg.norm(v) < 1e-12:
            raise ValueError("u_vec and v_vec must be non-zero.")

        cross_norm = np.linalg.norm(np.cross(u, v))
        if cross_norm < 1e-12 * np.linalg.norm(u) * np.linalg.norm(v):
            raise ValueError("u_vec and v_vec must not be parallel.")

        if type(self.offset) is not bool:
            raise TypeError("must be either True or False")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    @property
    def _spans(self) -> tuple:
        raise NotImplementedError

    @property
    def is_rectangular(self) -> bool:
        """True when polygon is a 4-vertex axis-aligned rectangle."""
        if self._cache.get("is_rectangular") is not None:
            return self._cache["is_rectangular"]
        poly = self.polygon
        if poly.n_vertices != 4:
            self._cache["is_rectangular"] = False
            return False
        xs = [v[0] for v in poly.vertices]
        ys = [v[1] for v in poly.vertices]
        x_vals = sorted(set(xs))
        y_vals = sorted(set(ys))
        result = len(x_vals) == 2 and len(y_vals) == 2
        self._cache["is_rectangular"] = result
        return result

    @property
    def u_hat(self):
        u = np.asarray(self.u_vec, float)
        return u / np.linalg.norm(u)

    @property
    def v_hat(self):
        u = self.u_hat
        v = np.asarray(self.v_vec, float)
        v = v - np.dot(u, v) * u
        return v / np.linalg.norm(v)

    @property
    def normal(self):
        n = np.cross(self.u_hat, self.v_hat)
        return n / np.linalg.norm(n)

    @property
    def basis(self):
        if self._cache.get("basis") is not None:
            return self._cache["basis"]
        basis = np.stack([self.u_hat, self.v_hat, self.normal], axis=1)
        self._cache["basis"] = basis
        return basis

    @property
    def axes(self):
        if self._cache.get("axes") is not None:
            return self._cache["axes"]
        spacing = self.spacing_init or (None,) * len(self._spans)
        num_points = self.num_points_init or (None,) * len(self._spans)
        axes = [
            Axis1D(span=abs(s), spacing_init=sp, num_points_init=n, offset=self.offset)
            for s, sp, n in zip(self._spans, spacing, num_points)
        ]
        self._cache["axes"] = axes
        return axes

    @property
    def spacing(self) -> tuple:
        return tuple(float(axis.spacing) for axis in self.axes)

    @property
    def x_spacing(self) -> float:
        return self.spacing[0] if len(self.spacing) > 0 else None

    @property
    def y_spacing(self) -> float:
        return self.spacing[1] if len(self.spacing) > 1 else None

    @property
    def num_x(self) -> int:
        return len(self.axes[0].points) if len(self.axes) > 0 else None

    @property
    def num_y(self) -> int:
        return len(self.axes[1].points) if len(self.axes) > 1 else None

    @property
    def _xy_mask(self) -> np.ndarray:
        """Boolean mask for points inside polygon in xy plane."""
        if self._cache.get("_xy_mask") is not None:
            return self._cache["_xy_mask"]
        ss, tt = np.meshgrid(self.axes[0].points, self.axes[1].points, indexing="ij")
        points_2d = np.column_stack([ss.ravel(), tt.ravel()])
        mask = self.polygon.contains_points(points_2d)
        self._cache["_xy_mask"] = mask
        return mask

    @property
    def total_points(self) -> int:
        """Total number of grid points (always a single int)."""
        return len(self.coords)

    @property
    def update_state(self) -> tuple:
        """Extension point for subclasses. Always empty for grids."""
        return ()

    def update(self, **changes):
        new = replace(self, **changes)
        object.__setattr__(new, "_cache", {})
        return new

    def to_dict(self):
        data = {
            "polygon": self.polygon.to_dict(),
            "origin": tuple(float(x) for x in self.origin),
            "u_vec": tuple(float(x) for x in self.u_vec),
            "v_vec": tuple(float(x) for x in self.v_vec),
            "spacing": tuple(self.spacing),
            "offset": self.offset,
        }
        data.update(self._extra_dict())
        return data

    def _extra_dict(self):
        return {}

    @property
    def calc_state(self) -> tuple:
        return (
            self.polygon.vertices,
            tuple(self.origin),
            tuple(self.u_vec),
            tuple(self.v_vec),
            self.spacing,
            self.offset,
        ) + self._extra_calc_state()

    def _extra_calc_state(self):
        return ()

    def _convert_units(self, old_units, new_units):
        """Convert all spatial coordinates from old_units to new_units."""
        factor = convert_length(old_units, new_units, 1.0)
        new_verts = tuple(
            tuple(c * factor for c in v) for v in self.polygon.vertices
        )
        new_poly = Polygon2D(vertices=new_verts)
        new_origin = convert_length_tuple(old_units, new_units, *self.origin)
        new_spacing = convert_length_tuple(old_units, new_units, *self.spacing)
        new_u_vec = tuple(c * factor for c in self.u_vec)
        new_v_vec = tuple(c * factor for c in self.v_vec)
        updates = dict(polygon=new_poly, origin=new_origin,
                       u_vec=new_u_vec, v_vec=new_v_vec,
                       spacing_init=new_spacing, num_points_init=self.num_points_init)
        updates.update(self._extra_convert_fields(old_units, new_units))
        return self.update(**updates)

    def _extra_convert_fields(self, old_units, new_units):
        return {}


@dataclass(frozen=True, eq=False)
class SurfaceGrid(_GridBase):
    """A 2D surface grid in 3D space. Replaces PlaneGrid and PolygonGrid."""

    @property
    def _spans(self) -> tuple:
        x_min, y_min, x_max, y_max = self.polygon.bounding_box
        return (x_max - x_min, y_max - y_min)

    def __repr__(self):
        if self.is_rectangular:
            return (
                f"SurfaceGrid(dimensions={self.dimensions}, "
                f"spacing={self.spacing}, "
                f"num_points={self.num_points}, "
                f"offset={self.offset}, "
                f"normal={self.normal})"
            )
        return (
            f"SurfaceGrid(polygon={self.polygon.n_vertices} vertices, "
            f"spacing={self.spacing}, "
            f"num_points={self.num_points}, "
            f"offset={self.offset})"
        )

    @property
    def coords(self) -> np.ndarray:
        if self._cache.get("coords") is not None:
            return self._cache["coords"]
        mesh = np.meshgrid(self.axes[0].points, self.axes[1].points, indexing="ij")
        s_vals, t_vals = [grid.reshape(-1) for grid in mesh]
        if not self.is_rectangular:
            mask = self._xy_mask
            s_vals, t_vals = s_vals[mask], t_vals[mask]
        origin = np.asarray(self.origin, float)
        coords = origin + s_vals[:, None] * self.u_hat + t_vals[:, None] * self.v_hat
        self._cache["coords"] = coords
        return coords

    @property
    def num_points(self) -> tuple:
        if self.is_rectangular:
            return (self.num_x, self.num_y)
        return (len(self.coords),)

    # ---- legacy dimensional properties for axis-aligned planes ----

    @property
    def _in_plane_indices(self):
        u = np.asarray(self.u_vec, float)
        v = np.asarray(self.v_vec, float)
        return [i for i in range(3) if u[i] != 0 or v[i] != 0]

    @property
    def mins(self):
        if not self.is_rectangular:
            x_min, y_min, _, _ = self.polygon.bounding_box
            return (x_min, y_min)
        idx = self._in_plane_indices
        ext = self.extent
        return tuple(min(self.origin[i], ext[i]) for i in idx)

    @property
    def maxs(self):
        if not self.is_rectangular:
            _, _, x_max, y_max = self.polygon.bounding_box
            return (x_max, y_max)
        idx = self._in_plane_indices
        ext = self.extent
        return tuple(max(self.origin[i], ext[i]) for i in idx)

    @property
    def dimensions(self):
        return tuple((a, b) for a, b in zip(self.mins, self.maxs))

    @property
    def x1(self) -> float:
        return self.mins[0] if len(self.mins) > 0 else None

    @property
    def x2(self) -> float:
        return self.maxs[0] if len(self.mins) > 0 else None

    @property
    def y1(self) -> float:
        return self.mins[1] if len(self.mins) > 1 else None

    @property
    def y2(self) -> float:
        return self.maxs[1] if len(self.mins) > 1 else None

    @property
    def z1(self) -> float | None:
        """Always None for surface grids."""
        return None

    @property
    def z2(self) -> float | None:
        """Always None for surface grids."""
        return None

    @property
    def z_spacing(self) -> float | None:
        """Always None for surface grids."""
        return None

    @property
    def num_z(self) -> int | None:
        """Always None for surface grids."""
        return None

    @property
    def extent(self):
        return np.array(
            np.asarray(self.origin, float)
            + self.u_hat * self._spans[0]
            + self.v_hat * self._spans[1]
        )

    # ---- legacy properties (height, ref_surface, direction) ----

    @property
    def ref_surface(self) -> str | None:
        n = self.normal
        labels = ["yz", "xz", "xy"]
        idx = int(np.argmax(np.abs(n)))
        if not np.isclose(abs(n[idx]), 1.0, atol=1e-6):
            return None
        return labels[idx]

    @property
    def height(self) -> float | None:
        rs = self.ref_surface
        if rs is None:
            return None
        if rs == "xy":
            return float(self.origin[2])
        elif rs == "xz":
            return float(self.origin[1])
        elif rs == "yz":
            return float(self.origin[0])

    @property
    def direction(self) -> int | None:
        rs = self.ref_surface
        if rs is None:
            return None
        n = self.normal
        if rs == "xy":
            return 1 if n[2] > 0 else -1
        elif rs == "xz":
            return 1 if n[1] > 0 else -1
        elif rs == "yz":
            return 1 if n[0] > 0 else -1

    # ---- factory methods ----

    @classmethod
    def from_legacy(
        cls,
        *,
        mins: tuple,
        maxs: tuple,
        height: float = 0,
        ref_surface: str = "xy",
        direction: int = 1,
        **kwargs,
    ):
        if not isinstance(ref_surface, str):
            raise TypeError("ref_surface must be a string in [`xy`,`xz`,`yz`]")
        if ref_surface.lower() not in ["xy", "xz", "yz"]:
            raise ValueError("ref_surface must be a string in [`xy`,`xz`,`yz`]")
        if not isinstance(height, numbers.Number):
            raise TypeError("Height must be numeric")
        if direction is not None and direction not in [1, 0, -1]:
            raise ValueError("Direction must be in [1, 0, -1]")

        span_u = maxs[0] - mins[0]
        span_v = maxs[1] - mins[1]
        polygon = Polygon2D.rectangle(span_u, span_v)

        if ref_surface == "xy":
            if direction == 1:
                origin = (mins[0], mins[1], height)
                u_vec = (1, 0, 0)
                v_vec = (0, 1, 0)
            else:
                origin = (mins[0], maxs[1], height)
                u_vec = (1, 0, 0)
                v_vec = (0, -1, 0)
        elif ref_surface == "xz":
            if direction == 1:
                origin = (mins[0], height, maxs[1])
                u_vec = (1, 0, 0)
                v_vec = (0, 0, -1)
            else:
                origin = (mins[0], height, mins[1])
                u_vec = (1, 0, 0)
                v_vec = (0, 0, 1)
        elif ref_surface == "yz":
            if direction == 1:
                origin = (height, mins[0], mins[1])
                u_vec = (0, 1, 0)
                v_vec = (0, 0, 1)
            else:
                origin = (height, mins[0], maxs[1])
                u_vec = (0, 1, 0)
                v_vec = (0, 0, -1)

        return cls(
            polygon=polygon,
            origin=np.asarray(origin, float),
            u_vec=np.asarray(u_vec, float),
            v_vec=np.asarray(v_vec, float),
            **kwargs,
        )

    @classmethod
    def from_wall(cls, p1, p2, z_height, **kwargs):
        """Create a vertical wall plane from two 2D vertices."""
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        edge_length = np.sqrt(dx * dx + dy * dy)
        origin = (p1[0], p1[1], 0.0)
        u_vec = (dx, dy, 0.0)
        v_vec = (0.0, 0.0, 1.0)
        polygon = Polygon2D.rectangle(edge_length, z_height)
        return cls(
            polygon=polygon,
            origin=origin,
            u_vec=u_vec,
            v_vec=v_vec,
            **kwargs,
        )

    @classmethod
    def from_points(cls, *, p0, pU, pV, **kwargs):
        origin = np.asarray(p0, float)
        u = np.asarray(pU, float) - p0
        v = np.asarray(pV, float) - p0
        if not np.isclose(np.dot(u, v), 0, atol=1e-10):
            msg = f"point {pU} is not orthogonal to point {pV}"
            warnings.warn(msg)
        u_hat = u / np.linalg.norm(u)
        v_perp = v - np.dot(v, u_hat) * u_hat
        v_norm = np.linalg.norm(v_perp)
        if v_norm < 1e-12:
            raise ValueError("from_points requires non-collinear points")
        v_hat = v_perp / v_norm
        s1, t1 = np.dot(u, u_hat), np.dot(u, v_hat)
        s2, t2 = np.dot(v, u_hat), np.dot(v, v_hat)
        span_u, span_v = max(s1, s2), max(t1, t2)
        polygon = Polygon2D.rectangle(span_u, span_v)
        return cls(polygon=polygon, origin=origin, u_vec=u, v_vec=v, **kwargs)

    @classmethod
    def from_polygon(cls, polygon, height=0.0, direction=1, **kwargs):
        """Create a horizontal surface grid from a polygon at a given height."""
        if not isinstance(polygon, Polygon2D):
            polygon = Polygon2D(vertices=tuple(tuple(v) for v in polygon))
        x_min, y_min, _, y_max = polygon.bounding_box
        if direction == 1:
            origin = (x_min, y_min, height)
            u_vec = (1, 0, 0)
            v_vec = (0, 1, 0)
        else:
            origin = (x_min, y_max, height)
            u_vec = (1, 0, 0)
            v_vec = (0, -1, 0)
        # Translate polygon so its bounding box starts at (0, 0)
        shifted = polygon.translate(-x_min, -y_min)
        return cls(
            polygon=shifted,
            origin=origin,
            u_vec=u_vec,
            v_vec=v_vec,
            **kwargs,
        )

    # ---- mutation helpers ----

    def update_legacy(self, height=None, ref_surface=None, direction=None):
        """Keep current dimensions, update height/orientation/reference surface."""
        if self.ref_surface is None:
            raise ValueError(
                "update_legacy is only defined for axis-aligned planes"
            )
        return SurfaceGrid.from_legacy(
            mins=self.mins,
            maxs=self.maxs,
            spacing_init=tuple(self.spacing),
            height=self.height if height is None else height,
            ref_surface=ref_surface or self.ref_surface,
            direction=self.direction if direction is None else direction,
        )

    def update_dimensions(self, mins=None, maxs=None, polygon=None,
                          height=None, preserve_spacing=True):
        """Update grid dimensions."""
        if not self.is_rectangular:
            # Polygon surface: update polygon and/or height
            new_poly = polygon if polygon is not None else self.polygon
            new_height = height if height is not None else self.height
            # Rebuild via from_polygon to handle origin/polygon shifting
            if preserve_spacing:
                return SurfaceGrid.from_polygon(
                    polygon=new_poly,
                    height=new_height if new_height is not None else 0.0,
                    direction=self.direction if self.direction is not None else 1,
                    spacing_init=self.spacing,
                    num_points_init=None,
                    offset=self.offset,
                )
            else:
                return SurfaceGrid.from_polygon(
                    polygon=new_poly,
                    height=new_height if new_height is not None else 0.0,
                    direction=self.direction if self.direction is not None else 1,
                    num_points_init=self.num_points_init,
                    spacing_init=None,
                    offset=self.offset,
                )

        # Rectangular: use mins/maxs
        if mins is None or maxs is None:
            raise ValueError("mins and maxs required for rectangular update")
        u1, u2, v1, v2 = mins[0], maxs[0], mins[1], maxs[1]
        span_u, span_v = (u2 - u1), (v2 - v1)
        new_origin = u1 * self.u_hat + v1 * self.v_hat
        normal = self.normal
        new_origin = new_origin + np.dot(np.asarray(self.origin), normal) * normal
        new_polygon = Polygon2D.rectangle(span_u, span_v)
        if preserve_spacing:
            return self.update(
                polygon=new_polygon,
                origin=new_origin,
                spacing_init=self.spacing,
                num_points_init=None,
            )
        else:
            return self.update(
                polygon=new_polygon,
                origin=new_origin,
                num_points_init=self.num_points,
                spacing_init=None,
            )

    @classmethod
    def from_dict(cls, data):
        data = migrate_surface_grid_dict(data)
        return init_from_dict(cls, data)


@dataclass(frozen=True, eq=False)
class VolumeGrid(_GridBase):
    """A 3D volume grid. Replaces VolGrid and PolygonVolGrid."""

    depth: float = 1.0  # extrusion distance along normal

    @property
    def _spans(self) -> tuple:
        x_min, y_min, x_max, y_max = self.polygon.bounding_box
        return (x_max - x_min, y_max - y_min, self.depth)

    def __repr__(self):
        if self.is_rectangular:
            return (
                f"VolumeGrid(dimensions={self.dimensions}, "
                f"spacing={self.spacing}, "
                f"num_points={self.num_points}, "
                f"offset={self.offset})"
            )
        return (
            f"VolumeGrid(polygon={self.polygon.n_vertices} vertices, "
            f"depth={self.depth}, "
            f"spacing={self.spacing}, "
            f"num_points={self.num_points}, "
            f"offset={self.offset})"
        )

    @property
    def z_spacing(self) -> float:
        return self.spacing[2]

    @property
    def num_z(self) -> int:
        return len(self.axes[2].points)

    @property
    def coords(self) -> np.ndarray:
        if self._cache.get("coords") is not None:
            return self._cache["coords"]
        s_pts = self.axes[0].points
        t_pts = self.axes[1].points
        z_pts = self.axes[2].points
        origin = np.asarray(self.origin, float)
        u_hat, v_hat, normal = self.u_hat, self.v_hat, self.normal
        if self.is_rectangular:
            mesh = np.meshgrid(s_pts, t_pts, z_pts, indexing="ij")
            s, t, z = [grid.reshape(-1) for grid in mesh]
            coords = origin + s[:, None] * u_hat + t[:, None] * v_hat + z[:, None] * normal
        else:
            ss, tt = np.meshgrid(s_pts, t_pts, indexing="ij")
            pts_2d = np.column_stack([ss.ravel(), tt.ravel()])
            mask = self._xy_mask
            s_in, t_in = pts_2d[mask, 0], pts_2d[mask, 1]
            s_rep = np.repeat(s_in, len(z_pts))
            t_rep = np.repeat(t_in, len(z_pts))
            z_tiled = np.tile(z_pts, len(s_in))
            coords = origin + s_rep[:, None] * u_hat + t_rep[:, None] * v_hat + z_tiled[:, None] * normal
        self._cache["coords"] = coords
        return coords

    @property
    def num_points(self) -> tuple:
        if self.is_rectangular:
            return (self.num_x, self.num_y, self.num_z)
        return (len(self.coords),)

    # ---- dimensional properties ----

    @property
    def _corners(self):
        """The 8 corners of the volume in 3D space."""
        origin = np.asarray(self.origin, float)
        span_u, span_v, depth = self._spans
        u_hat, v_hat, normal = self.u_hat, self.v_hat, self.normal
        offsets = [
            su * span_u * u_hat + sv * span_v * v_hat + sn * depth * normal
            for su in (0, 1) for sv in (0, 1) for sn in (0, 1)
        ]
        return np.array([origin + o for o in offsets])

    @property
    def mins(self):
        corners = self._corners
        return tuple(float(corners[:, i].min()) for i in range(3))

    @property
    def maxs(self):
        corners = self._corners
        return tuple(float(corners[:, i].max()) for i in range(3))

    @property
    def dimensions(self):
        return tuple((a, b) for a, b in zip(self.mins, self.maxs))

    @property
    def x1(self) -> float:
        return self.mins[0]

    @property
    def x2(self) -> float:
        return self.maxs[0]

    @property
    def y1(self) -> float:
        return self.mins[1]

    @property
    def y2(self) -> float:
        return self.maxs[1]

    @property
    def z1(self) -> float:
        return self.mins[2]

    @property
    def z2(self) -> float:
        return self.maxs[2]

    def _extra_calc_state(self):
        return (self.depth,)

    # ---- polygon-specific volume properties ----

    @property
    def coords_full(self) -> np.ndarray:
        """Full bounding box coordinates (including points outside polygon)."""
        if self._cache.get("coords_full") is not None:
            return self._cache["coords_full"]
        origin = np.asarray(self.origin, float)
        u_hat, v_hat, normal = self.u_hat, self.v_hat, self.normal
        mesh = np.meshgrid(self.axes[0].points, self.axes[1].points,
                           self.axes[2].points, indexing="ij")
        s, t, z = [m.ravel() for m in mesh]
        coords_full = origin + s[:, None] * u_hat + t[:, None] * v_hat + z[:, None] * normal
        self._cache["coords_full"] = coords_full
        return coords_full

    @property
    def _mask_full(self) -> np.ndarray:
        """Boolean mask for full 3D grid (True = inside polygon)."""
        if self._cache.get("_mask_full") is not None:
            return self._cache["_mask_full"]
        mask_2d = self._xy_mask
        mask_full = np.repeat(mask_2d, len(self.axes[2].points))
        self._cache["_mask_full"] = mask_full
        return mask_full

    def values_to_full_grid(self, values: np.ndarray) -> np.ndarray:
        """Map filtered values back to full grid, with -inf outside polygon."""
        full_values = np.full(len(self.coords_full), -np.inf)
        full_values[self._mask_full] = values.flatten()
        return full_values

    # ---- factory methods ----

    @classmethod
    def from_legacy(cls, *, mins, maxs, **kwargs):
        origin = np.asarray(mins, float)
        spans = np.asarray(maxs) - origin
        polygon = Polygon2D.rectangle(float(spans[0]), float(spans[1]))
        return cls(
            polygon=polygon,
            origin=tuple(origin),
            depth=float(spans[2]),
            **kwargs,
        )

    @classmethod
    def from_polygon(cls, polygon, z_height, **kwargs):
        """Create a volume grid from a polygon footprint extruded to z_height."""
        if not isinstance(polygon, Polygon2D):
            polygon = Polygon2D(vertices=tuple(tuple(v) for v in polygon))
        x_min, y_min, _, _ = polygon.bounding_box
        shifted = polygon.translate(-x_min, -y_min)
        return cls(
            polygon=shifted,
            origin=(x_min, y_min, 0.0),
            depth=z_height,
            **kwargs,
        )

    # ---- mutation helpers ----

    def update_dimensions(self, mins=None, maxs=None, preserve_spacing=True):
        if not self.is_rectangular:
            new_z = maxs[2] if maxs is not None else self.depth
            if preserve_spacing:
                return self.update(depth=new_z, spacing_init=self.spacing,
                                   num_points_init=None)
            else:
                return self.update(depth=new_z, num_points_init=self.num_points_init,
                                   spacing_init=None)

        origin = np.asarray(mins, float)
        spans = np.asarray(maxs, float) - origin
        new_polygon = Polygon2D.rectangle(float(spans[0]), float(spans[1]))
        if preserve_spacing:
            return self.update(polygon=new_polygon, origin=origin,
                               depth=float(spans[2]),
                               spacing_init=self.spacing, num_points_init=None)
        else:
            return self.update(polygon=new_polygon, origin=origin,
                               depth=float(spans[2]),
                               num_points_init=self.num_points, spacing_init=None)

    def _extra_dict(self):
        return {"depth": self.depth}

    def _extra_convert_fields(self, old_units, new_units):
        return {"depth": convert_length(old_units, new_units, self.depth)}

    @classmethod
    def from_dict(cls, data):
        data = migrate_volume_grid_dict(data)
        return init_from_dict(cls, data)
