"""Plotting methods for LampSurface."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class LampSurfacePlotter:
    """Handles plotting for LampSurface objects."""

    def __init__(self, surface):
        self.surface = surface

    def _get_fig_ax(self, fig, ax):
        """Get or create figure and axes."""
        if fig is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = plt.gcf()
        else:
            if ax is None:
                ax = fig.axes[0]
        return fig, ax

    def plot_surface_points(self, fig=None, ax=None, title=""):
        """Plot the discretization of the emissive surface."""
        fig, ax = self._get_fig_ax(fig, ax)

        u_points, v_points = self.surface._generate_raw_points(
            self.surface.num_points_length, self.surface.num_points_width
        )
        vv, uu = np.meshgrid(v_points, u_points)
        points = np.array([vv.flatten(), uu.flatten()[::-1]])
        ax.scatter(*points)
        if self.surface.width:
            ax.set_xlim(-self.surface.width / 2, self.surface.width / 2)
        if self.surface.length:
            ax.set_ylim(-self.surface.length / 2, self.surface.length / 2)
        if title is None:
            title = "Source density = " + str(self.surface.source_density)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        return fig, ax

    def plot_intensity_map(self, fig=None, ax=None, title="", show_cbar=True):
        """Plot the relative intensity map of the emissive surface."""
        fig, ax = self._get_fig_ax(fig, ax)

        if self.surface.width and self.surface.length:
            extent = [
                -self.surface.width / 2,
                self.surface.width / 2,
                -self.surface.length / 2,
                self.surface.length / 2,
            ]
            img = ax.imshow(self.surface.intensity_map, extent=extent)
        else:
            img = ax.imshow(self.surface.intensity_map)
        if show_cbar:
            cbar = fig.colorbar(img, pad=0.03)
            cbar.set_label("Surface relative intensity", loc="center")
        ax.set_title(title)
        return fig, ax

    def plot_surface(self, fig_width=10):
        """Combined grid points and intensity map plot."""
        width = self.surface.width if self.surface.width else 1
        length = self.surface.length if self.surface.length else 1

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
