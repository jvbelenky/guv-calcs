import inspect
import json
import copy
import numpy as np
from .calc_manager import LightingCalculator
from .rect_grid import VolGrid, PlaneGrid
from .calc_zone_io import export_plane, export_volume
from .calc_zone_plot import plot_plane, plot_volume


class CalcZone(object):
    """
    Base class representing a calculation zone.

    This class provides a template for setting up zones within which various
    calculations related to lighting conditions are performed. Subclasses should
    provide specific implementations of the coordinate setting method.

    NOTE: I changed this from an abstract base class to an object superclass
    to make it more convenient to work with the website, but this class doesn't really
    work on its own--it should really be an abstract base class

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
        data["calctype"] = "Zone"

        data.update(self._extra_dict())
        return data

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

    def export(self, fname=None):
        pass

    def plot(self):
        pass

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
            return None
        if self.dose:
            return self.values * 3.6 * self.hours
        return self.values

    def get_calc_state(self):
        """if different, these parameters indicate zone must be recalculated"""
        return [self.geometry]

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

        self.values = np.zeros(self.geometry.num_points, dtype="float32")
        self.reflected_values = np.zeros(self.geometry.num_points, dtype="float32")

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
            "calctype": self.calctype,
        }
        zone_data.update(data)
        return zone_data

<<<<<<< HEAD
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

=======
>>>>>>> f07924c (move plotting and io out)
    def get_update_state(self):
        """if changes, calc zone needs updating but not recalculating"""
        return []

    def export(self, fname=None):
        """export values to csv"""
        return export_volume(self, fname=fname)

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
            self.geometry = self.geometry.update(
                mins=mins, maxs=maxs, spacings=self.geometry.spacings
            )
        else:
            self.geometry = self.geometry.update(
                mins=mins, maxs=maxs, n_pts=self.geometry.num_points
            )
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

    def plot(self, **kwargs):
        """plot fluence values as isosurface"""
        return plot_volume(self, **kwargs)

    def plot_volume(self, **kwargs):
        """alias for plot() -- kept in for compatibility"""
        return self.plot(**kwargs)


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

        self.values = np.zeros(self.geometry.num_points, dtype="float32")
        self.reflected_values = np.zeros(self.geometry.num_points, dtype="float32")

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
            "height": self.height,
            "ref_surface": self.geometry.ref_surface,
            "direction": self.geometry.direction,
            "fov_vert": self.fov_vert,
            "fov_horiz": self.fov_horiz,
            "vert": self.vert,
            "horiz": self.horiz,
            "calctype": self.calctype,
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

    def get_update_state(self):
        """
        return a set of parameters that, if changed, indicate that the
        calc zone need not be be recalculated, but may need updating
        """
        return [self.fov_vert, self.fov_horiz, self.vert, self.horiz]

    def export(self, fname=None):
        """export values to csv"""
        return export_plane(self, fname=fname)

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

    def plot(self, **kwargs):
        """Plot the image of the radiation pattern"""
        return plot_plane(self, **kwargs)

    def plot_plane(self, **kwargs):
        """alias for plot() -- kept in for compatibility"""
        return self.plot(**kwargs)
