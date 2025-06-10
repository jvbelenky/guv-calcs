from pathlib import Path
import pathlib
import io
import warnings
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import RegularGridInterpolator
from .units import convert_units


class LampSurface:
    def __init__(
        self, width, length, depth, units, source_density, intensity_map, pose
    ):
        """
        Represents the emissive surface of a lamp; manages functions
        related to source discretization.
        """
        self.units = units if units is not None else "meters"
        self.width = width if width is not None else 0
        self.length = length if length is not None else 0
        self.depth = depth if depth is not None else 0

        self._pose = pose
        self.position = self._calculate_surface_position()

        self.source_density = 1 if source_density is None else source_density
        # store original for future operations
        self.intensity_map_orig = self._load_intensity_map(intensity_map)
        # this is the working copy
        self.intensity_map = self.intensity_map_orig

        self.surface_points = None
        self.num_points_width = None
        self.num_points_length = None
        self.photometric_distance = None

        self._update()

    def set_source_density(self, source_density):
        """change source discretization"""
        self.source_density = source_density
        self._update()

    def set_width(self, width):
        """change x-axis extent of lamp emissive surface"""
        self.width = width
        self._update()

    def set_length(self, length):
        """change y-axis extent of lamp emissive surface"""
        self.length = length
        self._update()

    def set_depth(self, depth):
        """change the z axis offset of the surface"""
        self.depth = depth
        self._update()

    def set_units(self, units):
        """set units and convert all values"""
        if units != self.units:
            self.width, self.length, self.depth = convert_units(
                units, self.units, self.width, self.length, self.depth
            )
            self.units = units
            self._update()

    def set_pose(self, pose):
        self._pose = pose
        self._update()

    def set_ies(self, ies):
        """
        populate length/width/depth units values from an IESFile object
        """
        if ies is not None:
            units_dict = {1: "feet", 2: "meters"}
            self.units = units_dict[ies.units]
            self.width = ies.width
            self.length = ies.length
            self.depth = ies.height
            self._update()

    def load_intensity_map(self, intensity_map):
        """external method for loading relative intensity map after lamp object has been instantiated"""
        self.intensity_map_orig = self._load_intensity_map(intensity_map)
        self.intensity_map = self.intensity_map_orig
        self._update()

    def plot_surface_points(self, fig=None, ax=None, title="", figsize=(6, 4)):
        """plot the discretization of the emissive surface"""

        if fig is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = plt.gcf()
        else:
            if ax is None:
                ax = fig.axes[0]

        u_points, v_points = self._generate_raw_points(
            self.num_points_length, self.num_points_width
        )
        vv, uu = np.meshgrid(v_points, u_points)
        points = np.array([vv.flatten(), uu.flatten()[::-1]])
        ax.scatter(*points)
        if self.width:
            ax.set_xlim(-self.width / 2, self.width / 2)
        if self.length:
            ax.set_ylim(-self.length / 2, self.length / 2)
        if title is None:
            title = "Source density = " + str(self.source_density)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        return fig, ax

    def plot_intensity_map(
        self, fig=None, ax=None, title="", figsize=(6, 4), show_cbar=True
    ):
        """plot the relative intensity map of the emissive surface"""
        if fig is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = plt.gcf()
        else:
            if ax is None:
                ax = fig.axes[0]

        if self.width and self.length:
            extent = [
                -self.width / 2,
                self.width / 2,
                -self.length / 2,
                self.length / 2,
            ]
            img = ax.imshow(self.intensity_map, extent=extent)
        else:
            img = ax.imshow(self.intensity_map)
        if show_cbar:
            cbar = fig.colorbar(img, pad=0.03)
            cbar.set_label("Surface relative intensity", loc="center")
        ax.set_title(title)
        return fig, ax

    def plot_surface(self, fig_width=10):
        """
        convenience pltoting function that incorporates both the grid points
        and relative map plotting functions
        """
        width = self.width if self.width else 1
        length = self.length if self.length else 1

        fig_length = fig_width / (width / length * 2)
        fig, ax = plt.subplots(1, 2, figsize=(fig_width, min(max(fig_length, 2), 50)))

        self.plot_surface_points(fig=fig, ax=ax[0], title="Surface grid points")
        self.plot_intensity_map(
            fig=fig, ax=ax[1], show_cbar=False, title="Relative intensity map"
        )

        axins = inset_axes(
            ax[1],
            width="5%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(1.1, 0.0, 1, 1),
            bbox_transform=ax[1].transAxes,
            borderpad=0,
        )

        fig.colorbar(ax[1].get_images()[0], cax=axins)
        return fig

    # ------------------------- Internals ---------------------

    def _update(self):
        """
        _update all emissive surface parameters--surface grid points,
        relative intensity map, and photometric distance
        """
        self.position = self._calculate_surface_position()
        self.surface_points = self._generate_surface_points()
        self.intensity_map = self._generate_intensity_map()
        if all([self.width, self.length, self.units]):
            self.photometric_distance = max(self.width, self.length) * 10
        else:
            self.photometric_distance = None

    def _calculate_surface_position(self):
        """Compute the surface center based on the lamp's depth and aim direction."""
        # direction = self.aim_point - self.mounting_position
        direction = self._pose.aim_point - self._pose.position
        normal = direction / np.linalg.norm(direction)
        return self._pose.position + normal * self.depth

    def _generate_surface_points(self):
        """
        generate the points with which the calculations should be performed.
        If the source is approximately square and source_density is 1, only
        one point is generated. If source is more than twice as long as wide,
        (or vice versa), 2 or more points will be generated even if density is 1.
        Total number of points will increase quadratically with density.
        """

        if self.source_density:
            num_points_u, num_points_v = self._get_num_points()
            u_points, v_points = self._generate_raw_points(num_points_u, num_points_v)
            vv, uu = np.meshgrid(v_points, u_points)

            x_local = vv.ravel()
            y_local = uu.ravel()
            z_local = np.full_like(x_local, 0)

            local = np.vstack([x_local, y_local, z_local]).T  # shape (3, N)
            # rotate/translate into world
            surface_points = self._pose.transform_to_world(local).T
            surface_points = surface_points[::-1]
            self.num_points_width = num_points_v
            self.num_points_length = num_points_u

        else:
            surface_points = self.position
            self.num_points_length = 1
            self.num_points_width = 1

        return surface_points

    def _generate_raw_points(self, num_points_u, num_points_v):
        """generate the points on the surface of the lamp, prior to transforming them"""
        if self.width:
            spacing_v = self.width / num_points_v
            # If there's only one point, place it at the center
            if num_points_v == 1:
                v_points = np.array([0])  # Single point at the center of the width
            else:
                startv = -self.width / 2 + spacing_v / 2
                stopv = self.width / 2 - spacing_v / 2
                v_points = np.linspace(startv, stopv, num_points_v)
        else:
            v_points = np.array([0])
        if self.length:
            spacing_u = self.length / num_points_u
            if num_points_u == 1:
                u_points = np.array([0])  # Single point at the center of the length
            else:
                startu = -self.length / 2 + spacing_u / 2
                stopu = self.length / 2 - spacing_u / 2
                u_points = np.linspace(startu, stopu, num_points_u)
        else:
            u_points = np.array([0])
        return u_points, v_points

    def _get_num_points(self):
        """calculate the number of u and v points"""
        if self.source_density:
            num_points = self.source_density + self.source_density - 1
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

    def _load_intensity_map(self, arg):
        """check filetype and return correct intensity_map as array"""

        if arg is None:
            intensity_map = None
        elif isinstance(arg, (str, pathlib.Path)):
            # check if this is a file
            if Path(arg).is_file():
                intensity_map = np.genfromtxt(Path(arg), delimiter=",")
            else:
                msg = f"File {arg} not found. intensity_map will not be used."
                warnings.warn(msg, stacklevel=3)
                intensity_map = None
        elif isinstance(arg, bytes):
            try:
                data = arg.decode("utf-8-sig")
                intensity_map = np.genfromtxt(io.StringIO(data), delimiter=",")
            except UnicodeDecodeError:
                msg = (
                    "Could not read intensity map file. Intensity map will not be used."
                )
                warnings.warn(msg, stacklevel=3)
                intensity_map = None
        elif isinstance(arg, (list, np.ndarray)):
            intensity_map = np.array(arg)
        else:
            msg = f"Argument type {type(arg)} for argument intensity_map is invalid. intensity_map will not be used."
            warnings.warn(msg, stacklevel=3)
            intensity_map = None

        # clean nans
        if intensity_map is not None:
            if np.isnan(intensity_map).any():
                msg = "File contains invalid values. Intensity map will not be used."
                warnings.warn(msg, stacklevel=3)
                intensity_map = None
        return intensity_map

    def _generate_intensity_map(self):
        """if the relative map is None or ones, generate"""

        if self.intensity_map is None:
            # if no relative intensity map is provided
            intensity_map = np.ones((self.num_points_length, self.num_points_width))
        elif self.intensity_map.shape == (
            self.num_points_length,
            self.num_points_width,
        ):
            # intensity map does not need updating
            intensity_map = self.intensity_map
        else:
            if self.intensity_map_orig is None:
                intensity_map = np.ones((self.num_points_length, self.num_points_width))
            else:
                # reshape the provided relative map to the current coordinates
                # make interpolator based on original intensity map
                num_points_u, num_points_v = self.intensity_map_orig.shape
                x_orig, y_orig = self._generate_raw_points(num_points_u, num_points_v)
                interpolator = RegularGridInterpolator(
                    (x_orig, y_orig),
                    self.intensity_map_orig,
                    bounds_error=False,
                    fill_value=None,
                )

                x_new, y_new = self._generate_raw_points(
                    self.num_points_length, self.num_points_width
                )
                # x_new, y_new = np.unique(self.surface_points.T[0]), np.unique(self.surface_points.T[1])
                x_new_grid, y_new_grid = np.meshgrid(x_new, y_new)
                # Create points for interpolation and extrapolation
                points_new = np.array([x_new_grid.ravel(), y_new_grid.ravel()]).T
                intensity_map = (
                    interpolator(points_new).reshape(len(x_new), len(y_new)).T
                )

                # normalize
                intensity_map = intensity_map / intensity_map.mean()

        return intensity_map
