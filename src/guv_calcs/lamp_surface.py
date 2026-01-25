import numpy as np
from .units import convert_units, LengthUnits
from .lamp_orientation import LampOrientation
from .lamp_surface_plotter import LampSurfacePlotter
from .intensity_map import IntensityMap


class LampSurface:
    """Represents the emissive surface of a lamp; manages source discretization."""

    def __init__(
        self,
        pose: "LampOrientation",
        width: float | None = None,
        length: float | None = None,
        depth: float = 0.0,
        units: "LengthUnits" = LengthUnits.METERS,
        source_density: int = 1,
        intensity_map=None,
    ):
        # Track user-provided values (None means user didn't specify)
        self._user_width = width
        self._user_length = length

        # Working values (0.0 if user didn't specify and no IES loaded yet)
        self.width = width if width is not None else 0.0
        self.length = length if length is not None else 0.0
        self.depth = depth
        self.units = LengthUnits.from_any(units)

        self._pose = pose
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

    # ---- Public properties with lazy computation ----

    @property
    def position(self):
        """Surface center position in world coordinates."""
        if self._position_dirty:
            self._position_cache = self._calculate_surface_position()
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

    def set_depth(self, depth):
        """Change the z axis offset of the surface."""
        self.depth = depth
        self._invalidate_position()

    def set_units(self, units):
        """Set units and convert all values."""
        units = LengthUnits.from_any(units)
        if units != self.units:
            self.width, self.length, self.depth = convert_units(
                self.units, units, self.width, self.length, self.depth
            )
            self.units = units
            self._invalidate_grid()

    def set_pose(self, pose):
        """Update the lamp pose/orientation."""
        self._pose = pose
        self._invalidate_position()
        self._invalidate_grid()

    def set_ies(self, ies):
        """
        Populate length/width/units values from an IESFile object.
        Only overwrites width/length if user didn't explicitly provide them.
        """
        if ies is not None:
            units_dict = {1: LengthUnits.FEET, 2: LengthUnits.METERS}
            self.units = units_dict[ies.units]
            if self._user_width is None:
                self.width = ies.width
            if self._user_length is None:
                self.length = ies.length
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
            "depth": self.depth,
            "units": self.units.value if hasattr(self.units, "value") else str(self.units),
            "source_density": self._source_density,
            "intensity_map": orig.tolist() if orig is not None else None,
            "_user_width": self._user_width,
            "_user_length": self._user_length,
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

        if self._source_density:
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

    def _calculate_surface_position(self):
        """Compute the surface center based on the lamp's depth and aim direction."""
        direction = self._pose.aim_point - self._pose.position
        normal = direction / np.linalg.norm(direction)
        return self._pose.position + normal * self.depth

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
