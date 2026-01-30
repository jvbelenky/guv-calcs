from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from ..units import convert_units, LengthUnits
from .lamp_surface_plotter import LampSurfacePlotter
from .intensity_map import IntensityMap

if TYPE_CHECKING:
    from .lamp_geometry import LampGeometry


class LampSurface:
    """Represents the emissive surface of a lamp; manages source discretization."""

    def __init__(
        self,
        width: float | None = None,
        length: float | None = None,
        height: float | None = None,
        units: "LengthUnits" = LengthUnits.METERS,
        source_density: int = 1,
        intensity_map=None,
    ):
        # Track user-provided values (None means user didn't specify)
        self._user_width = width
        self._user_length = length
        self._user_height = height
        self._user_units = units

        # Working values (0.0 if user didn't specify and no IES loaded yet)
        self.width = width if width is not None else 0.0
        self.length = length if length is not None else 0.0
        self.height = height if height is not None else 0.0
        self.units = LengthUnits.from_any(units)

        # Back-reference to geometry container (set via set_geometry)
        self._geometry: "LampGeometry | None" = None

        self._source_density = source_density
        self._intensity_map = IntensityMap(intensity_map)

        # Dirty flags
        self._position_dirty = True
        self._grid_dirty = True

        # Cached values
        self._position_cache = None
        self._surface_points_cache = None
        self._num_points_cache = None
        self._intensity_map_cache = None

        self.plotter = LampSurfacePlotter(self)

    # ---- Geometry back-reference ----

    def set_geometry(self, geometry: "LampGeometry"):
        """Set back-reference to parent LampGeometry. Called by LampGeometry.__init__."""
        self._geometry = geometry
        self._invalidate_grid()

    @property
    def _pose(self):
        """Access pose via geometry back-reference."""
        if self._geometry is None:
            raise RuntimeError("LampSurface requires geometry back-reference. Use LampGeometry.")
        return self._geometry.pose

    # ---- Public properties with lazy computation ----

    @property
    def position(self):
        """Surface center position in world coordinates."""
        if self._geometry is None:
            return np.array([0.0, 0.0, 0.0])
        if self._position_dirty:
            self._position_cache = self._geometry.surface_position
            self._position_dirty = False
        return self._position_cache

    @property
    def surface_points(self):
        """Grid points on the lamp surface in world coordinates."""
        if self._grid_dirty:
            self._recompute_grid()
        return self._surface_points_cache

    @property
    def num_points_width(self):
        """Number of grid points along width dimension."""
        if self._grid_dirty:
            self._recompute_grid()
        return self._num_points_cache[1]

    @property
    def num_points_length(self):
        """Number of grid points along length dimension."""
        if self._grid_dirty:
            self._recompute_grid()
        return self._num_points_cache[0]

    @property
    def intensity_map(self):
        """Current resampled intensity map."""
        if self._grid_dirty:
            self._recompute_grid()
        return self._intensity_map_cache

    @property
    def intensity_map_orig(self):
        """Original intensity map (for backward compatibility)."""
        return self._intensity_map.original

    @property
    def photometric_distance(self):
        """Photometric distance for far-field calculations."""
        if self.width and self.length:
            return max(self.width, self.length) * 10
        return None

    @property
    def source_density(self):
        """Source discretization density."""
        return self._source_density

    @source_density.setter
    def source_density(self, value):
        self._source_density = value
        self._invalidate_grid()

    # ---- Setters ----

    def set_source_density(self, source_density):
        """Change source discretization."""
        self.source_density = source_density

    def set_width(self, width):
        """Change x-axis extent of lamp emissive surface."""
        if width is not None and width < 0:
            raise ValueError(f"width must be non-negative, got {width}")
        self._user_width = width
        self.width = width if width is not None else 0.0
        self._invalidate_grid()

    def set_length(self, length):
        """Change y-axis extent of lamp emissive surface."""
        if length is not None and length < 0:
            raise ValueError(f"length must be non-negative, got {length}")
        self._user_length = length
        self.length = length if length is not None else 0.0
        self._invalidate_grid()

    def set_height(self, height):
        """Set z-axis extent of luminous opening (for 3D sources like cylinders)."""
        if height is not None and height < 0:
            raise ValueError(f"height must be non-negative, got {height}")
        self._user_height = height
        self.height = height if height is not None else 0.0
        self._invalidate_grid()

    def set_units(self, units):
        """Set units and convert all values."""
        units = LengthUnits.from_any(units)
        if units != self.units:
            self.width, self.length = convert_units(
                self.units, units, self.width, self.length
            )
            self.units = units
            self._invalidate_grid()

    def set_ies(self, ies):
        """
        Populate length/width/height/units values from an IESFile object.
        Only overwrites values if user didn't explicitly provide them.
        """
        if ies is not None:
            units_dict = {1: LengthUnits.FEET, 2: LengthUnits.METERS}
            self.units = units_dict[ies.units]

            if self._user_width is None:
                self.width = abs(ies.width)
            if self._user_length is None:
                self.length = abs(ies.length)
            if self._user_height is None:
                self.height = abs(ies.height)

            self._invalidate_grid()

    def load_intensity_map(self, intensity_map):
        """Load a new intensity map after instantiation."""
        self._intensity_map = IntensityMap(intensity_map)
        self._invalidate_grid()

    # ---- Serialization ----

    def to_dict(self):
        """Serialize surface state for persistence."""
        orig = self._intensity_map.original
        return {
            "width": self.width,
            "length": self.length,
            "height": self.height,
            "units": self.units.value if hasattr(self.units, "value") else str(self.units),
            "source_density": self._source_density,
            "intensity_map": orig.tolist() if orig is not None else None,
            "_user_width": self._user_width,
            "_user_length": self._user_length,
            "_user_height": self._user_height,
            "_user_units": self._user_units,
        }

    # ---- Plotting delegates ----

    def plot_surface_points(self, fig=None, ax=None, title=""):
        """Plot the discretization of the emissive surface."""
        return self.plotter.plot_surface_points(fig=fig, ax=ax, title=title)

    def plot_intensity_map(self, fig=None, ax=None, title="", show_cbar=True):
        """Plot the relative intensity map of the emissive surface."""
        return self.plotter.plot_intensity_map(fig=fig, ax=ax, title=title, show_cbar=show_cbar)

    def plot_surface(self, fig_width=10):
        """Combined grid points and intensity map plot."""
        return self.plotter.plot_surface(fig_width=fig_width)

    # ---- Invalidation helpers ----

    def _invalidate_position(self):
        """Mark position cache as stale."""
        self._position_dirty = True

    def _invalidate_grid(self):
        """Mark grid-related caches as stale."""
        self._grid_dirty = True
        self._invalidate_position()

    # ---- Lazy computation ----

    def _recompute_grid(self):
        """Lazily compute grid-related values."""
        num_u, num_v = self._get_num_points()
        self._num_points_cache = (num_u, num_v)

        if self._source_density and self._geometry is not None:
            u_points, v_points = self._generate_raw_points(num_u, num_v)
            vv, uu = np.meshgrid(v_points, u_points)

            x_local = vv.ravel()
            y_local = uu.ravel()
            z_local = np.full_like(x_local, 0)

            local = np.vstack([x_local, y_local, z_local]).T
            surface_points = self._pose.transform_to_world(local).T
            self._surface_points_cache = surface_points[::-1]
        else:
            self._surface_points_cache = self.position

        self._intensity_map_cache = self._intensity_map.resample(
            num_u, num_v, self._generate_raw_points
        )
        self._grid_dirty = False

    def _make_points_1d(self, extent, num_points):
        """Generate evenly-spaced points centered at origin."""
        if not extent or num_points == 1:
            return np.array([0.0])
        spacing = extent / num_points
        start = -extent / 2 + spacing / 2
        stop = extent / 2 - spacing / 2
        return np.linspace(start, stop, num_points)

    def _generate_raw_points(self, num_points_u, num_points_v):
        """Generate points on lamp surface prior to world transform."""
        u_points = self._make_points_1d(self.length, num_points_u)
        v_points = self._make_points_1d(self.width, num_points_v)
        return u_points, v_points

    def _get_num_points(self):
        """Calculate the number of u and v points."""
        if self._source_density:
            num_points = 2 * self._source_density - 1
        else:
            num_points = 1

        if self.width and self.length:
            num_points_v = max(
                num_points, num_points * int(round(self.width / self.length))
            )
            num_points_u = max(
                num_points, num_points * int(round(self.length / self.width))
            )
            if num_points_u % 2 == 0:
                num_points_u += 1
            if num_points_v % 2 == 0:
                num_points_v += 1
        else:
            num_points_u, num_points_v = 1, 1

        return num_points_u, num_points_v
