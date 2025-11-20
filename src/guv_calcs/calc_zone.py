import inspect
import warnings
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numbers
from .calc_manager import LightingCalculator
from .io import rows_to_bytes
from .rect_grid import VolGrid, PlaneGrid


class CalcZone(object):
    """
    TODO: dummy this out?

    Base class representing a calculation zone.

    This class provides a template for setting up zones within which various
    calculations related to lighting conditions are performed. Subclasses should
    provide specific implementations of the coordinate setting method.

    NOTE: I changed this from an abstract base class to an object superclass
    to make it more convenient to work with the website, but this class doesn't really
    work on its own

    Parameters:
    --------
    zone_id: str
        identification tag for internal tracking
    name: str, default=None
        externally visible name for zone
    offset: bool, default=True
    dose: bool, default=False
        whether to calculate a dose over N hours or just fluence
    hours: float, default = 8.0
        number of hours to calculate dose over. Only relevant if dose is True.
    enabled: bool, default = True
        whether or not the calc zone is enabled for calculations
    """

    def __init__(
        self,
        zone_id=None,
        name=None,
        dose=None,
        hours=None,
        enabled=None,
        show_values=None,
        colormap=None,
    ):
        self.zone_id = zone_id
        self.name = str(zone_id) if name is None else str(name)
        self.dose = False if dose is None else dose
        if self.dose:
            self.units = "mJ/cm²"
        else:
            self.units = "uW/cm²"
        self.hours = 8.0 if hours is None else abs(hours)  # only used if dose is true
        self.enabled = True if enabled is None else enabled
        self.show_values = True if show_values is None else show_values
        self.colormap = colormap

        self.calculator = LightingCalculator(self)

        # these will all be calculated after spacing is set, which is set in the subclass
        self.calctype = "Zone"
        self.geometry = None
        self.values = None
        self.reflected_values = None
        self.lamp_values = {}
        self.lamp_values_base = {}
        self.calc_state = None

    def __eq__(self, other):
        if not isinstance(other, CalcZone):
            return NotImplemented

        return self.to_dict() == other.to_dict()

    def __repr__(self):
        return (
            f"Calc{self.calctype}(id={self.zone_id!r}, name={self.name!r}, "
            f"enabled={self.enabled}, "
            f"dose={self.dose}, "
            f"dose_hours={self.hours}, "
        )

    def to_dict(self):

        data = {}
        data["zone_id"] = self.zone_id
        data["name"] = self.name
        data["dose"] = self.dose
        data["hours"] = self.hours
        data["enabled"] = self.enabled
        data["show_values"] = self.show_values
        data["colormap"] = self.colormap

        data.update(self._extra_dict())

    def _extra_dict(self):
        return {}

    def save(self, filename):
        """save zone data to json file"""
        data = self.to_dict()
        with open(filename, "w") as json_file:
            json_file.write(json.dumps(data))

    @classmethod
    def from_dict(cls, data):
        keys = list(inspect.signature(cls.__init__).parameters.keys())[1:]
        return cls(**{k: v for k, v in data.items() if k in keys})

    def set_dimensions(self, dimensions):
        raise NotImplementedError

    def set_spacing(self, spacing):
        raise NotImplementedError

    def set_num_points(self, spacing):
        raise NotImplementedError

    def _write_rows(self):
        raise NotImplementedError

    def set_value_type(self, dose):
        """
        if true get_values() will return in dose (mJ/cm2) over hours
        otherwise get_values() will return in irradiance or fluence (uW/cm2)
        """
        if type(dose) is not bool:
            raise TypeError("Dose must be either True or False")

        self.dose = dose
        if self.dose:
            self.units = "mJ/cm²"
        else:
            self.units = "uW/cm²"

    def set_dose_time(self, hours):
        """
        Set the time over which the dose will be calculate in hours
        """
        if type(hours) not in [float, int]:
            raise TypeError("Hours must be numeric")
        self.hours = hours

    def calculate_values(self, lamps, ref_manager=None, hard=False):
        """
        Calculate all the values for all the lamps
        """

        new_calc_state = self.get_calc_state()

        # updates self.lamp_values_base and self.lamp_values
        self.base_values = self.calculator.compute(
            lamps=lamps, filters=filters, obstacles=obstacles, hard=hard
        )
        if ref_manager is not None:
            # calculate reflectance -- warning, may be expensive!
            ref_manager.calculate_reflectance(self, hard=hard)
            # add in reflected values, if applicable
            self.reflected_values = ref_manager.get_total_reflectance(self)
        else:
            self.reflected_values = np.zeros(self.geometry.num_points).astype("float32")

        # sum
        self.values = self.base_values + self.reflected_values
        self.calc_state = new_calc_state

        return self.get_values()

    def get_values(self):
        """
        return
        """
        if self.values is None:
            values = None
        else:
            if self.dose:
                values = self.values * 3.6 * self.hours
            else:
                values = self.values
        return values

    def export(self, fname=None):
        """
        export the calculation zone's results to a .csv file
        if the spacing has been updated but the values not recalculated,
        exported values will be blank.
        """
        try:
            rows = self._write_rows()  # implemented in subclass
            csv_bytes = rows_to_bytes(rows)

            if fname is not None:
                with open(fname, "wb") as csvfile:
                    csvfile.write(csv_bytes)
            else:
                return csv_bytes
        except NotImplementedError:
            pass

    def copy(self, zone_id):
        """
        return a copy of this CalcZone with the same attributes and a new zone_id
        """
        zone = copy.deepcopy(self)
        zone.zone_id = zone_id
        # clear calculated values
        zone.values = None
        zone.reflected_values = None
        zone.lamp_values = {}
        zone.lamp_values_base = {}
        return zone


class CalcVol(CalcZone):
    """
    Represents a volumetric calculation zone.
    A subclass of CalcZone designed for three-dimensional volumetric calculations.
    """

    def __init__(
        self,
        zone_id=None,
        name=None,
        x1=None,
        x2=None,
        y1=None,
        y2=None,
        z1=None,
        z2=None,
        num_x=None,
        num_y=None,
        num_z=None,
        x_spacing=None,
        y_spacing=None,
        z_spacing=None,
        offset=None,
        dose=None,
        hours=None,
        enabled=None,
        show_values=None,
    ):

        super().__init__(
            zone_id=zone_id or "CalcVol",
            name=name,
            dose=dose,
            hours=hours,
            enabled=enabled,
            show_values=show_values,
        )
        self.calctype = "Volume"
        self.geometry = VolGrid(
            mins=(x1 or 0, y1 or 0, z1 or 0),
            maxs=(x2 or 6, y2 or 4, z2 or 2.7),
            n_pts=(num_x, num_y, num_z),
            spacings=(x_spacing, y_spacing, z_spacing),
            offset=offset or True,
        )

        self.values = np.zeros(self.geometry.num_points).astype("float32")
        self.reflected_values = np.zeros(self.geometry.num_points).astype("float32")

    # attribute passthrough to geometry -------------
    def __getattr__(self, name):
        # called only if normal attribute lookup fails
        geometry = self.__dict__.get("geometry", None)
        if geometry is not None and hasattr(geometry, name):
            return getattr(geometry, name)
        raise AttributeError(name)

    def _extra_dict(self):

        zone_data = super()._extra_dict()

        data = {
            "x1": self.geometry.x1,
            "x2": self.geometry.x2,
            "x_spacing": self.geometry.y_spacing,
            "y1": self.geometry.x1,
            "y2": self.geometry.x2,
            "y_spacing": self.geometry.y_spacing,
            "z1": self.geometry.z1,
            "z2": self.geometry.z2,
            "z_spacing": self.geometry.z_spacing,
            "calctype": "Volume",
        }
        zone_data.update(data)
        return zone_data

    def __repr__(self):
        return super().__repr__() + (
            f"dimensions=(x=({self.x1},{self.x2}), y=({self.y1},{self.y2}), z=({self.z1},{self.z2})), "
            f"grid={self.num_x}x{self.num_y}x{self.num_z}, "
            f"offset={self.offset}, "
            f"enabled={self.enabled})"
        )

    def get_calc_state(self):
        """
        return a set of paramters that, if changed, indicate that
        this calc zone must be recalculated
        """
        return [
            self.geometry.offset,
            self.geometry.x1,
            self.geometry.x2,
            self.geometry.y1,
            self.geometry.y2,
            self.geometry.z1,
            self.geometry.z2,
            self.geometry.x_spacing,
            self.geometry.y_spacing,
            self.geometry.z_spacing,
        ]

    def get_update_state(self):
        """
        return a set of parameters that, if changed, indicate that the
        calc zone need not be be recalculated, but may need updating
        Currently there are no relevant update parameters for a calc volume
        """
        return []

    def set_dimensions(
        self,
        x1=None,
        x2=None,
        y1=None,
        y2=None,
        z1=None,
        z2=None,
        preserve_spacing=True,
    ):
        mins = (x1 or self.geometry.x1, y1 or self.geometry.y1, z1 or self.geometry.z1)
        maxs = (x2 or self.geometry.x2, y2 or self.geometry.y2, z2 or self.geometry.z2)

        if preserve_spacing:
            spacings = (
                self.geometry.x_spacing,
                self.geometry.y_spacing,
                self.geometry.z_spacing,
            )
            self.geometry = self.geometry.update(
                mins=mins, maxs=maxs, spacings=spacings
            )
        else:
            n_pts = (self.geometry.num_x, self.geometry.num_y, self.geometry.num_z)
            self.geometry = self.geometry.update(mins=mins, maxs=maxs, n_pts=n_pts)
        return self

    def set_spacing(self, x_spacing=None, y_spacing=None, z_spacing=None):
        """
        set the spacing desired in the dimension
        """
        spacings = (
            x_spacing or self.geometry.x_spacing,
            y_spacing or self.geometry.y_spacing,
            z_spacing or self.geometry.z_spacing,
        )
        self.geometry = self.geometry.update(spacings=spacings)
        return self

    def set_num_points(self, num_x=None, num_y=None, num_z=None):
        """
        set the number of points desired in a dimension, instead of setting the spacing
        """
        n_pts = (
            num_x or self.geometry.num_x,
            num_y or self.geometry.num_y,
            num_z or self.geometry.num_z,
        )
        self.geometry = self.geometry.update(n_pts=n_pts)
        return self
    
    def set_offset(self, offset):
        self.geometry = self.geometry.update(offset=offset)

    def _write_rows(self):
        """
        export solution to csv file
        designed to be in the same format as the Acuity Visual export
        """

        header = """Data format notes:
        
        Data consists of numZ horizontal grids of fluence rate values; each grid contains numX by numY points.

        numX; numY; numZ are given on the first line of data.
        The next line contains numX values; indicating the X-coordinate of each grid column.
        The next line contains numY values; indicating the Y-coordinate of each grid row.
        The next line contains numZ values; indicating the Z-coordinate of each horizontal grid.
        A blank line separates the position data from the first horizontal grid of fluence rate values.
        A blank line separates each subsequent horizontal grid of fluence rate values.

        fluence rate values are given in µW/cm²
        """

        lines = header.split("\n")
        rows = [[line] for line in lines]
        rows += [self.geometry.num_points]
        rows += self.geometry.points
        values = self.get_values()
        for i in range(self.geometry.num_z):
            rows += [""]
            if values is None:
                rows += [[""] * self.geometry.num_x] * self.geometry.num_y
            elif values.shape != (
                self.geometry.num_x,
                self.geometry.num_y,
                self.geometry.num_z,
            ):
                rows += [[""] * self.geometry.num_x] * self.geometry.num_y
            else:
                rows += values.T[i].tolist()
        return rows

    def plot_volume(
        self,
        title=None,
    ):
        """
        Plot the fluence values as an isosurface using Plotly.
        """

        if self.values is None:
            raise ValueError("No values calculated for this volume.")

        X, Y, Z = np.meshgrid(*self.geometry.points, indexing="ij")
        x, y, z = X.flatten(), Y.flatten(), Z.flatten()
        values = self.values.flatten()
        isomin = self.values.mean() / 2
        fig = go.Figure()
        fig.add_trace(
            go.Isosurface(
                x=x,
                y=y,
                z=z,
                value=values,
                isomin=isomin,
                surface_count=3,
                opacity=0.25,
                showscale=False,
                colorbar=None,
                colorscale=self.colormap,
                caps=dict(x_show=False, y_show=False, z_show=False),
                name=self.name + " Values",
            )
        )
        fig.update_layout(
            title=dict(
                text=self.name if title is None else title,
                x=0.5,  # center horizontally
                y=0.85,  # lower this value to move the title down (default is 0.95)
                xanchor="center",
                yanchor="top",
                font=dict(size=18),
            ),
            scene=dict(
                xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"
            ),
            height=450,
        )
        fig.update_scenes(camera_projection_type="orthographic")
        return fig


class CalcPlane(CalcZone):
    """
    Represents a planar calculation zone.
    A subclass of CalcZone designed for two-dimensional planar calculations at a specific height.
    """

    def __init__(
        self,
        zone_id=None,
        name=None,
        x1=None,
        x2=None,
        y1=None,
        y2=None,
        height=None,
        ref_surface="xy",
        direction=None,
        num_x=None,
        num_y=None,
        x_spacing=None,
        y_spacing=None,
        offset=None,
        fov_vert=None,
        fov_horiz=None,
        vert=None,
        horiz=None,
        dose=None,
        hours=None,
        enabled=None,
        show_values=None,
    ):

        super().__init__(
            zone_id=zone_id or "CalcPlane",
            name=name,
            dose=dose,
            hours=hours,
            enabled=enabled,
            show_values=show_values,
        )
        self.calctype = "Plane"

        self.geometry = PlaneGrid(
            mins=(x1 or 0, y1 or 0),
            maxs=(x2 or 6, y2 or 4),
            n_pts=(num_x, num_y),
            spacings=(x_spacing, y_spacing),
            offset=offset or True,
            height=height or 0,
            ref_surface=ref_surface or "xy",
            direction=direction or 1,
        )

        self.fov_vert = 180 if fov_vert is None else fov_vert
        self.fov_horiz = 360 if fov_horiz is None else abs(fov_horiz)
        self.vert = False if vert is None else vert
        self.horiz = False if horiz is None else horiz

        self.values = np.zeros(self.geometry.num_points)
        self.reflected_values = np.zeros(self.geometry.num_points)

    # attribute passthrough to geometry -------------
    def __getattr__(self, name):
        # called only if normal attribute lookup fails
        geometry = self.__dict__.get("geometry", None)
        if geometry is not None and hasattr(geometry, name):
            return getattr(geometry, name)
        raise AttributeError(name)

    def _extra_dict(self):

        zone_data = super()._extra_dict()

        data = {
            "x1": self.geometry.x1,
            "x2": self.geometry.x2,
            "x_spacing": self.geometry.y_spacing,
            "y1": self.geometry.x1,
            "y2": self.geometry.x2,
            "y_spacing": self.geometry.y_spacing,
            "fov_vert": self.fov_vert,
            "fov_horiz": self.fov_horiz,
            "vert": self.vert,
            "horiz": self.horiz,
            "ref_surface": self.geometry.ref_surface,
            "direction": self.geometry.direction,
            "calctype": "Plane",
        }
        zone_data.update(data)
        return zone_data

    def __repr__(self):
        a = self.ref_surface[0]
        b = self.ref_surface[1]
        return super().__repr__() + (
            f"dimensions=({a}=({self.x1},{self.x2}), {b}=({self.y1},{self.y2})), "
            f"height={self.height}, "
            f"grid={self.num_x}x{self.num_y}, "
            f"offset={self.offset}, "
            f"field_of_view=({self.fov_horiz}° horiz, {self.fov_vert}° vert), "
            f"flags=(vert={self.vert}, horiz={self.horiz}, dir={self.direction}), "
        )

    @classmethod
    def from_vectors(cls, p0, pU, pV, **kwargs):
        p0 = np.asarray(p0, float)
        u = np.asarray(pU, float) - p0
        v = np.asarray(pV, float) - p0

        # Decide which Cartesian plane we’re in (largest component of n̂)
        n = np.cross(u, v)
        axis = np.argmax(np.abs(n))

        if axis == 2:
            ref_surface = "xy"
            x1, x2 = np.unique(sorted([p0[0], pU[0], pV[0]]))
            y1, y2 = np.unique(sorted([p0[1], pU[1], pV[1]]))
            height = p0[2]
        elif axis == 1:
            ref_surface = "xz"
            x1, x2 = np.unique(sorted([p0[0], pU[0], pV[0]]))
            y1, y2 = np.unique(sorted([p0[2], pU[2], pV[2]]))
            height = p0[1]
        else:
            ref_surface = "yz"
            x1, x2 = np.unique(sorted([p0[1], pU[1], pV[1]]))
            y1, y2 = np.unique(sorted([p0[2], pU[2], pV[2]]))
            height = p0[0]

        direction = int(np.sign(n[axis])) if n[axis] != 0 else 1

        return cls(
            x1=x1,
            x2=x2,
            y1=y1,
            y2=y2,
            height=height,
            ref_surface=ref_surface,
            direction=direction,
            **kwargs,
        )

    def set_height(self, height):
        """set height of calculation plane. currently we only support vertical planes"""
        self.geometry = self.geometry.update(height=height)
        return self

    def set_ref_surface(self, ref_surface):
        """
        set the reference surface of the calc plane--must be a string in [`xy`,`xz`,`yz`]
        """
        self.geometry = self.geometry.update(ref_surface=ref_surface)
        return self

    def set_direction(self, direction):
        """
        set the direction of the plane normal
        Valid values are currently 1, -1 and 0
        """
        self.geometry = self.geometry.update(direction=direction)
        return self

    def set_dimensions(self, x1=None, x2=None, y1=None, y2=None, preserve_spacing=True):
        """set the dimensions and update the coordinate points"""
        mins = (x1 or self.geometry.x1, y1 or self.geometry.y1)
        maxs = (x2 or self.geometry.x2, y2 or self.geometry.y2)
        if preserve_spacing:
            spacings = (self.geometry.x_spacing, self.geometry.y_spacing)
            self.geometry = self.geometry.update(
                mins=mins, maxs=maxs, spacings=spacings
            )
        else:  # preserve num_points instead
            n_pts = (self.geometry.num_x, self.geometry.num_y)
            self.geometry = self.geometry.update(mins=mins, maxs=maxs, n_pts=n_pts)
        return self

    def set_spacing(self, x_spacing=None, y_spacing=None):
        """set the fineness of the grid spacing and update the coordinate points"""
        spacings = (
            x_spacing or self.geometry.x_spacing,
            y_spacing or self.geometry.y_spacing,
        )
        self.geometry = self.geometry.update(spacings=spacings)
        return self

    def set_num_points(self, num_x=None, num_y=None):
        """
        set the number of points desired in a dimension, instead of setting the spacing
        """
        n_pts = (num_x or self.geometry.num_x, num_y or self.geometry.num_y)
        self.geometry = self.geometry.update(n_pts=n_pts)
        return self
        
    def set_offset(self, offset):
        self.geometry = self.geometry.update(offset=offset)

    def get_calc_state(self):
        """
        return a set of paramters that, if changed, indicate that
        this calc zone must be recalculated
        """
        return [
            self.geometry.offset,
            self.geometry.x1,
            self.geometry.x2,
            self.geometry.x_spacing,
            self.geometry.y1,
            self.geometry.y2,
            self.geometry.y_spacing,
            self.geometry.height,
            self.geometry.ref_surface,
            self.geometry.direction,  # only for reflectance...possibly can be optimized
        ]

    def get_update_state(self):
        """
        return a set of parameters that, if changed, indicate that the
        calc zone need not be be recalculated, but may need updating
        """
        return [self.fov_vert, self.fov_horiz, self.vert, self.horiz]

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
        if values is not None:
            vmin = values.min() if vmin is None else vmin
            vmax = values.max() if vmax is None else vmax
            extent = [
                self.geometry.x1,
                self.geometry.x2,
                self.geometry.y1,
                self.geometry.y2,
            ]

            values = values.T[::-1]
            img = ax.imshow(
                values, extent=extent, vmin=vmin, vmax=vmax, cmap=self.colormap
            )
            cbar = fig.colorbar(img, pad=0.03)
            ax.set_title(title)
            cbar.set_label(self.units, loc="center")
        return fig, ax

    def _write_rows(self):
        """
        export solution to csv
        """

        # if self.ref_surface == "xy":
        # xpoints = self.points[0].tolist()
        # ypoints = self.points[1].tolist()
        # elif self.ref_surface == "xz":
        # xpoints = self.points[0].tolist()
        # ypoints = [self.height] * self.num_y
        # elif self.ref_surface == "yz":
        # xpoints = [self.height] * self.num_x
        # ypoints = self.points[1].tolist()

        # num_x, num_y, *rest = [len(np.unique(val)) for val in self.coords.T if len(np.unique(val))>1]

        # points = [np.unique(val) for val in self.coords.T]
        # num_x, num_y, *rest = [len(val) for val in points if len(val) > 1]
        num_x, num_y = self.geometry.num_x, self.geometry.num_y  # tmp

        values = self.get_values()
        if values is None:
            vals = [[-1] * num_y] * num_x
        elif values.shape != (num_x, num_y):
            vals = [[-1] * num_y] * num_x
        else:
            vals = values
        zvals = self.geometry.coords.T[2].reshape(num_x, num_y).T[::-1]

        xpoints = self.geometry.coords.T[0].reshape(num_x, num_y).T[0].tolist()
        ypoints = self.geometry.coords.T[1].reshape(num_x, num_y)[0].tolist()

        if len(np.unique(xpoints)) == 1 and len(np.unique(ypoints)) == 1:
            xpoints = self.geometry.coords.T[0].reshape(num_x, num_y)[0].tolist()
            ypoints = self.geometry.coords.T[1].reshape(num_x, num_y).T[0].tolist()
            vals = np.array(vals).T.tolist()
            zvals = zvals.T.tolist()

        rows = [[""] + xpoints]

        rows += np.concatenate(([np.flip(ypoints)], vals)).T.tolist()
        rows += [""]
        # zvals

        rows += [[""] + list(line) for line in zvals]
        return rows
