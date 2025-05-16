import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from .trigonometry import to_polar


class LampPlotter:
    """
    Class for plotting
    """

    def __init__(self, lamp):
        self.lamp = lamp
        self.spectra = self.lamp.spectra

    def plot_ies(self, title=""):
        """standard polar plot of an ies file"""
        if self.lamp.ies is not None:
            fig, ax = self.lamp.ies.plot(title=title)
        else:
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        return fig, ax

    def plot_web(
        self,
        title="",
        elev=30,
        azim=-60,
        figsize=(6, 4),
        color="#cc61ff",
        alpha=0.4,
        xlim=None,
        ylim=None,
        zlim=None,
    ):
        """plot photometric web, where distance r is set by the irradiance value"""

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        if self.lamp.ies is not None:
            scale = self.lamp.values.max()
            x, y, z = self.lamp.transform_to_world(
                self.lamp.photometric_coords, scale=scale
            )
            Theta, Phi, R = to_polar(*self.lamp.photometric_coords.T)
            tri = Delaunay(np.column_stack((Theta.flatten(), Phi.flatten())))

            ax.plot_trisurf(x, y, z, triangles=tri.simplices, color=color, alpha=alpha)
            if self.lamp.aim_point is not None:
                ax.plot(
                    *np.array((self.lamp.aim_point, self.lamp.position)).T,
                    linestyle="--",
                    color="black",
                    alpha=0.7,
                )
        ax.set_title(title)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if zlim is not None:
            ax.set_zlim(zlim)
        ax.view_init(azim=azim, elev=elev)
        return fig, ax

    def plot_3d(
        self,
        title="",
        elev=45,
        azim=-45,
        figsize=(6, 4),
        alpha=0.7,
        cmap="rainbow",
        fig=None,
        ax=None,
    ):
        """
        plot in cartesian 3d space of the true positions of the irradiance values
        mostly a convenience visualization function. Generally irradiance values
        should use a polar plot.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        if self.lamp.ies is not None:
            x, y, z = self.lamp.transform(self.lamp.coords).T
            intensity = self.lamp.values.flatten()

            ax.scatter(x, y, z, c=intensity, cmap="rainbow", alpha=alpha)
            if self.lamp.aim_point is not None:
                ax.plot(
                    *np.array((self.lamp.aim_point, self.lamp.position)).T,
                    linestyle="--",
                    color="black",
                    alpha=0.7,
                )
        ax.set_title(title)
        ax.view_init(azim=azim, elev=elev)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        return fig, ax

    def plot_spectra(
        self,
        title="",
        fig=None,
        ax=None,
        figsize=(6.4, 4.8),
        yscale="linear",
        label=None,
        weights=False,
    ):

        """
        see Spectrum.plot for details
        Some code is duplicated to maintain api compatibility
        """
        if fig is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = plt.gcf()
        else:
            if ax is None:
                ax = fig.axes[0]

        if self.spectra is not None:
            fig, ax = self.spectra.plot(
                title=title,
                fig=fig,
                ax=ax,
                figsize=figsize,
                yscale=yscale,
                label=label,
                weights=weights,
            )

        ax.grid(True, which="both", ls="--", c="gray", alpha=0.3)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Relative intensity [%]")
        ax.set_yscale(yscale)
        ax.set_title(title)
        return fig, ax
