import numpy as np
from collections.itertools import combinations
import matplotlib.pyplot as plt
from .calc_manager import LightingCalculator


class Zone:
    """Flags + shared behaviour common to ALL zones (planes, volumes, …)."""

    def __init__(
        self,
        zone_id=None,
        name=None,
        offset=None,
        dose=None,
        hours=None,
        enabled=None,
        show_values=None,
        **kwargs,
    ):

        self.zone_id = str(zone_id)
        self.name = zone_id if name is None else name
        self.offset = True if offset is None else offset
        self.dose = False if dose is None else dose
        if self.dose:
            self.units = "mJ/cm²"
        else:
            self.units = "uW/cm²"
        self.hours = 8.0 if hours is None else abs(hours)  # only used if dose is true
        self.enabled = True if enabled is None else enabled
        self.show_values = True if show_values is None else show_values

        # may need a different lighting calculator defined for Planes and Volumes...
        self.calculator = LightingCalculator(self)

    def to_dict(self):

        data = {}
        data["zone_id"] = self.zone_id
        data["name"] = self.name
        data["offset"] = self.offset
        data["dose"] = self.dose
        data["hours"] = self.hours
        data["enabled"] = self.enabled
        data["show_values"] = self.show_values

        data.update(self._extra_dict())

    def _extra_dict(self):
        return {}


class Plane(Zone):
    def __init__(
        self,
        *,  # keyword only to avoid arg clashes
        vert: bool = False,
        horiz: bool = False,
        fov_vert: float = 180.0,
        fov_horiz: float = 360.0,
        height: float = 0.0,
        normal: np.ndarray | None = None,
        **zone_flags,  # anything Zone cares about
    ):
        # set generic flags first
        super().__init__(**zone_flags)

    def plot(self):
        """"""
        pass  # tmp

    def export(self, fname):
        """"""
        pass  # tmp

    def _extra_dict(self):

        zone_data = super()._extra_dict()

        data = {
            "fov_vert": self.fov_vert,
            "fov_horiz": self.fov_horiz,
            "vert": self.vert,
            "horiz": self.horiz,
            "normal": self.normal,
        }
        zone_data.update(data)
        return zone_data


class CalcRectangle(Plane):
    def __init__(
        self,
        x1=0,
        x2=6,
        y1=0,
        y2=4,
    ):

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2


class Volume(Zone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all flag kwargs

    def plot(self):
        """"""

    def export(self, fname):
        """"""


class CalcPlaneBase:
    def __init__(self, coords, normal, vert, horiz, fov_vert, fov_horiz):

        self.coords = coords
        points = [np.unique(val) for val in self.coords.T]
        self.num_points = np.array([len(val) for val in points if len(val) > 1])

        self.normal = normal
        self.basis = self._get_basis()

        self.vert = False if vert is None else vert
        self.horiz = False if horiz is None else horiz
        self.fov_vert = 180 if fov_vert is None else fov_vert
        self.fov_horiz = 360 if fov_horiz is None else abs(fov_horiz)

        # everything to do with calculation is here
        self.calculator = LightingCalculator(self)
        self.base_values = None
        self.reflected_values = np.zeros(self.num_points)
        self.values = np.zeros(self.num_points)

        self.lamp_values = {}
        self.lamp_values_base = {}

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

    def _get_normal(self):
        """determine the plane's normal from the coordinates"""
        pts = self._get_coplanar()
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        n = np.cross(v1, v2)
        return n / np.linalg.norm(n)

    def _get_basis(self, normal=None):
        # Generate arbitrary vector not parallel to n

        n = self._get_normal() if normal is None else normal
        tmp = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])

        u = np.cross(n, tmp)
        u = u / np.linalg.norm(u)

        v = np.cross(n, u)
        return np.stack([u, v, n], axis=1)

    def _reset_params(self):
        self.normal = self._get_normal()
        self.vert = False
        self.horiz = False
        self.fov_vert = 180
        self.fov_horiz = 360

    def set_calc_type(self, calc_type, point=None, fov_vert=None, fov_horiz=None):
        """
        TODO:
        the eye calc type should be handled pretty differently in the calc manager eventually
        the directional
        """

        calc_types = [
            "planar_norm",
            "planar_max",
            "fluence",
            "eye",
            "directional",
            "target",
        ]
        calc_type = calc_type.lower()

        if calc_type not in calc_types:
            msg = f"{calc_type} is not a valid calculation type; valid calculation types are {calc_types}"
            raise KeyError(msg)

        if calc_type == "planar_norm":
            self._reset_params()
            self.horiz = True  # may not be necessary later
        elif calc_type == "planar_max":
            self._reset_params()
        elif calc_type == "fluence":
            self._reset_params()
            self.normal = None
        elif calc_type == "eye":
            self._reset_params()
            self.normal = None
            self.vert = True
            self.fov_vert = 80 if fov_vert is None else fov_vert
            self.fov_horiz = 180 if fov_horiz is None else fov_horiz
        else:
            if point is None:
                msg = "For the directional and target calculation types, a normal must be provided"
                raise ValueError(msg)
            elif len(point) != 3:
                msg = "Target point must be a cartesian point of len(3)"
                raise ValueError(msg)

            if calc_type == "directional":
                norm = point - self.coords.mean(axis=0)
                self.normal = norm / np.linalg.norm(norm)
                self.basis = self._get_basis(self.normal)
            elif calc_type == "target":
                norms = point - self.coords.astype("float32")
                norms = (norms.T / np.linalg.norm(norms, axis=1)).T
                self.normal = norms

    def _write_rows(self):
        """
        export solution to csv
        """

        num_x, num_y = self.num_points

        values = self.get_values()
        if self.base_values is None:
            vals = [[-1] * num_y] * num_x
        elif values.shape != (num_x, num_y):
            vals = [[-1] * num_y] * num_x
        else:
            vals = values
        zvals = self.coords.T[2].reshape(num_x, num_y).T[::-1]

        xpoints = self.coords.T[0].reshape(num_x, num_y).T[0].tolist()
        ypoints = self.coords.T[1].reshape(num_x, num_y)[0].tolist()

        if len(np.unique(xpoints)) == 1 and len(np.unique(ypoints)) == 1:
            xpoints = self.coords.T[0].reshape(num_x, num_y)[0].tolist()
            ypoints = self.coords.T[1].reshape(num_x, num_y).T[0].tolist()
            vals = vals.T
            zvals = zvals.T

        rows = [[""] + xpoints]
        rows += np.concatenate(([np.flip(ypoints)], vals)).T.tolist()
        rows += [""]
        rows += [[""] + list(line) for line in zvals]
        return rows

        def plot_plane(self, fig=None, ax=None, vmin=None, vmax=None, title=None):
            """Plot the image of the radiation pattern"""
            if fig is None:
                if ax is None:
                    fig, ax = plt.subplots()
                else:
                    fig = plt.gcf()
            else:
                if ax is None:
                    ax = fig.axes[0]

            title = "" if title is None else title
            values = self.get_values()
            xp, yp = [
                np.unique(val) for val in self.coords.T if len(np.unique(val)) > 1
            ]
            if values is not None:
                vmin = values.min() if vmin is None else vmin
                vmax = values.max() if vmax is None else vmax
                extent = [xp[0], xp[-1], yp[0], yp[-1]]
                values = values.T[::-1]
                img = ax.imshow(
                    values, extent=extent, vmin=vmin, vmax=vmax, cmap="plasma"
                )
                cbar = fig.colorbar(img, pad=0.03)
                ax.set_title(title)
                cbar.set_label(self.units, loc="center")
            return fig, ax
