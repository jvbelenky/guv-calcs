import inspect
import warnings
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from .calc_manager import LightingCalculator
from .io import rows_to_bytes


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
        offset=None,
        dose=None,
        hours=None,
        enabled=None,
        show_values=None,
        colormap=None,
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
        self.colormap = colormap

        self.calculator = LightingCalculator(self)

        # these will all be calculated after spacing is set, which is set in the subclass
        self.calctype = "Zone"
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None
        self.z1 = None
        self.z2 = None
        self.height = None
        self.num_x = None
        self.num_y = None
        self.num_z = None
        self.x_spacing = None
        self.y_spacing = None
        self.z_spacing = None
        self.num_points = None
        self.xp = None
        self.yp = None
        self.zp = None
        self.coords = None
        self.ref_surface = None
        self.direction = None
        self.horiz = None
        self.vert = None
        self.fov_vert = None
        self.fov_horiz = None
        self.basis = None
        self.values = None
        self.reflected_values = None
        self.lamp_values = {}
        self.lamp_values_base = {}
        self.calc_state = None

    def to_dict(self):

        data = {}
        data["zone_id"] = self.zone_id
        data["name"] = self.name
        data["offset"] = self.offset
        data["fov_vert"] = self.fov_vert
        data["fov_horiz"] = self.fov_horiz
        data["vert"] = self.vert
        data["horiz"] = self.horiz
        data["dose"] = self.dose
        data["hours"] = self.hours
        data["enabled"] = self.enabled
        data["show_values"] = self.show_values
        data["x1"] = self.x1
        data["x2"] = self.x2
        data["x_spacing"] = self.x_spacing
        # data["num_x"] = self.num_x
        data["y1"] = self.y1
        data["y2"] = self.y2
        data["y_spacing"] = self.y_spacing
        # data["num_y"] = self.num_y
        if isinstance(self, CalcPlane):
            data["height"] = self.height
            data["calctype"] = "Plane"
        elif isinstance(self, CalcVol):
            data["z1"] = self.z1
            data["z2"] = self.z2
            data["z_spacing"] = self.z_spacing
            # data["num_z"] = self.num_z
            data["calctype"] = "Volume"
        return data

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

    def _write_rows(self):
        raise NotImplementedError

    def set_offset(self, offset):
        if type(offset) is not bool:
            raise TypeError("must be either True or False")
        self.offset = offset
        self._update()

    def set_value_type(self, dose):
        """
        if true get_values() will return in dose (mJ/cm2) over hours
        otherwise get_values() will return in irradiance or fluence (uW/cm2)
        """
        if type(dose) is not bool:
            raise TypeError("Dose must be either True or False")

        # # convert values if they need converting
        # if self.values is not None:
        # if dose and not self.dose:
        # self.values = self.values * 3.6 * self.hours
        # elif self.dose and not dose:
        # self.values = self.values / (3.6 * self.hours)

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

        # if self.values is not None and self.dose:
        # self.values = (self.values * hours) / self.hours

    def _set_spacing(self, pt1, pt2, num, spacing):
        """set the spacing value conservatively from a num_points value"""
        rnge = abs(pt2 - pt1)
        if int(rnge / spacing) == int(num):
            val = spacing  # no changes needed
        else:
            testval = rnge / round(num)
            i = 1
            while i < 6:

                val = round(testval, i)
                if val != 0 and int(rnge / round(testval, i) + 1) == num:
                    break
                i += 1
                val = testval  # if no rounded value works use the original value
        return val

    def calculate_values(self, lamps, ref_manager=None, hard=False):
        """
        Calculate all the values for all the lamps
        """

        new_calc_state = self.get_calc_state()

        # updates self.lamp_values_base and self.lamp_values
        self.base_values = self.calculator.compute(lamps=lamps, hard=hard)

        if ref_manager is not None:
            # calculate reflectance -- warning, may be expensive!
            ref_manager.calculate_reflectance(self, hard=hard)
            # add in reflected values, if applicable
            self.reflected_values = ref_manager.get_total_reflectance(self)
        else:
            self.reflected_values = np.zeros(self.num_points).astype("float32")

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
            offset=offset,
            dose=dose,
            hours=hours,
            enabled=enabled,
            show_values=show_values,
        )
        self.calctype = "Volume"
        self.x1 = 0 if x1 is None else x1
        self.x2 = 6 if x2 is None else x2
        self.y1 = 0 if y1 is None else y1
        self.y2 = 4 if y2 is None else y2
        self.z1 = 0 if z1 is None else z1
        self.z2 = 2.7 if z2 is None else z2
        self.dimensions = ((self.x1, self.y1, self.z1), (self.x2, self.y2, self.z2))

        # spacing and num points
        xr = abs(self.x2 - self.x1)
        yr = abs(self.y2 - self.y1)
        zr = abs(self.z2 - self.z1)
        nx = min(int(xr * 10), 20)
        ny = min(int(yr * 10), 20)
        nz = min(int(zr * 10), 20)
        default_xs = round(xr / nx, -int(np.floor(np.log10(xr / nx))))
        default_ys = round(yr / ny, -int(np.floor(np.log10(yr / ny))))
        default_zs = round(zr / nz, -int(np.floor(np.log10(zr / nz))))
        default_num_x = int(round(xr / default_xs))
        default_num_y = int(round(yr / default_ys))
        default_num_z = int(round(zr / default_zs))
        if x_spacing is None:  # set from num_points if spacing is not provided
            self.num_x = default_num_x if num_x is None else int(num_x)
            self.x_spacing = self._set_spacing(self.x1, self.x2, self.num_x, default_xs)
        else:
            self.x_spacing = x_spacing
            self.num_x = int(round(xr / self.x_spacing))
        if y_spacing is None:
            self.num_y = default_num_y if num_y is None else int(num_y)
            self.y_spacing = self._set_spacing(self.y1, self.y2, self.num_y, default_ys)
        else:
            self.y_spacing = y_spacing
            self.num_y = int(round(yr / self.y_spacing))
        if z_spacing is None:
            self.num_z = default_num_z if num_z is None else int(num_z)
            self.z_spacing = self._set_spacing(self.z1, self.z2, self.num_z, default_zs)
        else:
            self.z_spacing = z_spacing
            self.num_z = int(round(zr / self.z_spacing))

        self._update()
        self.values = np.zeros(self.num_points).astype("float32")
        self.reflected_values = np.zeros(self.num_points).astype("float32")

    def get_calc_state(self):
        """
        return a set of paramters that, if changed, indicate that
        this calc zone must be recalculated
        """
        return [
            self.offset,
            self.x1,
            self.x2,
            self.x_spacing,
            self.y1,
            self.y2,
            self.y_spacing,
            self.z1,
            self.z2,
            self.z_spacing,
        ]

    def get_update_state(self):
        """
        return a set of parameters that, if changed, indicate that the
        calc zone need not be be recalculated, but may need updating
        Currently there are no relevant update parameters for a calc volume
        """
        return []

    def set_dimensions(self, x1=None, x2=None, y1=None, y2=None, z1=None, z2=None):
        self.x1 = self.x1 if x1 is None else x1
        self.x2 = self.x2 if x2 is None else x2
        self.y1 = self.y1 if y1 is None else y1
        self.y2 = self.y2 if y2 is None else y2
        self.z1 = self.z1 if z1 is None else z1
        self.z2 = self.z2 if z2 is None else z2
        self.dimensions = ((self.x1, self.y1, self.z1), (self.x2, self.y2, self.z2))

        # update number of points, keeping spacing
        xr = abs(self.x2 - self.x1)
        yr = abs(self.y2 - self.y1)
        zr = abs(self.z2 - self.z1)
        self.num_x = int(round(xr / self.x_spacing))
        self.num_y = int(round(yr / self.y_spacing))
        self.num_z = int(round(zr / self.z_spacing))
        self._update()

    def set_spacing(self, x_spacing=None, y_spacing=None, z_spacing=None):
        """
        set the spacing desired in the dimension
        """
        self.x_spacing = self.x_spacing if x_spacing is None else abs(x_spacing)
        self.y_spacing = self.y_spacing if y_spacing is None else abs(y_spacing)
        self.z_spacing = self.z_spacing if z_spacing is None else abs(z_spacing)
        numx = int(round(abs(self.x2 - self.x1) / self.x_spacing))
        numy = int(round(abs(self.y2 - self.y1) / self.y_spacing))
        numz = int(round(abs(self.z2 - self.z1) / self.z_spacing))
        self.num_x = numx
        self.num_y = numy
        self.num_z = numz
        self._update()

    def set_num_points(self, num_x=None, num_y=None, num_z=None):
        """
        set the number of points desired in a dimension, instead of setting the spacing
        """
        self.num_x = self.num_x if num_x is None else abs(int(num_x))
        self.num_y = self.num_y if num_y is None else abs(int(num_y))
        self.num_z = self.num_z if num_z is None else abs(int(num_z))
        if self.num_x == 0:
            warnings.warn("Number of x points must be at least 1")
            self.num_x += 1
        if self.num_y == 0:
            warnings.warn("Number of y points must be at least 1")
            self.num_y += 1
        if self.num_z == 0:
            warnings.warn("Number of z points must be at least 1")
            self.num_z += 1

        # update spacing if required
        self.x_spacing = self._set_spacing(self.x1, self.x2, self.num_x, self.x_spacing)
        self.y_spacing = self._set_spacing(self.y1, self.y2, self.num_y, self.y_spacing)
        self.z_spacing = self._set_spacing(self.z1, self.z2, self.num_z, self.z_spacing)

        self._update()
        return self

    def _update(self):
        """
        Update the number of points based on the spacing, and then the points
        """

        if self.x1 == self.x2:
            self.num_x = 1
        if self.y1 == self.y2:
            self.num_y = 1
        if self.z1 == self.z2:
            self.num_z = 1

        x_offset = min(self.x1, self.x2)
        y_offset = min(self.y1, self.y2)
        z_offset = min(self.z1, self.z2)
        xp = np.array([i * self.x_spacing + x_offset for i in range(self.num_x)])
        yp = np.array([i * self.y_spacing + y_offset for i in range(self.num_y)])
        zp = np.array([i * self.z_spacing + z_offset for i in range(self.num_z)])

        if self.offset:
            xp += (abs(self.x2 - self.x1) - abs(xp[-1] - xp[0])) / 2
            yp += (abs(self.y2 - self.y1) - abs(yp[-1] - yp[0])) / 2
            zp += (abs(self.z2 - self.z1) - abs(zp[-1] - zp[0])) / 2

        self.xp, self.yp, self.zp = xp, yp, zp
        self.points = [self.xp, self.yp, self.zp]

        X, Y, Z = [
            grid.reshape(-1) for grid in np.meshgrid(*self.points, indexing="ij")
        ]
        self.coords = np.array((X, Y, Z)).T
        self.coords = np.unique(self.coords, axis=0)

        self.num_points = np.array([len(self.xp), len(self.yp), len(self.zp)])

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
        rows += [self.num_points]
        rows += self.points
        values = self.get_values()
        for i in range(self.num_z):
            rows += [""]
            if values is None:
                rows += [[""] * self.num_x] * self.num_y
            elif values.shape != (self.num_x, self.num_y, self.num_z):
                rows += [[""] * self.num_x] * self.num_y
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

        X, Y, Z = np.meshgrid(*self.points, indexing="ij")
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
            offset=offset,
            dose=dose,
            hours=hours,
            enabled=enabled,
            show_values=show_values,
        )
        self.calctype = "Plane"
        self.height = 1.9 if height is None else height

        self.x1 = 0 if x1 is None else x1
        self.x2 = 6 if x2 is None else x2
        self.y1 = 0 if y1 is None else y1
        self.y2 = 4 if y2 is None else y2
        self.dimensions = ((self.x1, self.y1), (self.x2, self.y2))

        # TODO: this should really be much cleaner
        # perhaps just have users define cartesian points + a normal?
        # that may just need to be a slightly different class
        if not isinstance(ref_surface, str):
            raise TypeError("ref_surface must be a string in [`xy`,`xz`,`yz`]")
        if ref_surface.lower() not in ["xy", "xz", "yz"]:
            raise ValueError("ref_surface must be a string in [`xy`,`xz`,`yz`]")

        self.ref_surface = "xy" if ref_surface is None else ref_surface.lower()
        if direction is not None and direction not in [1, 0, -1]:
            raise ValueError("Direction must be in [1, 0, -1]")
        self.direction = 1 if direction is None else int(direction)  # eg planar norm
        self.basis = self._get_basis()

        self.fov_vert = 180 if fov_vert is None else fov_vert
        self.fov_horiz = 360 if fov_horiz is None else abs(fov_horiz)
        self.vert = False if vert is None else vert
        self.horiz = False if horiz is None else horiz

        # spacing and num points
        xr = abs(self.x2 - self.x1)
        yr = abs(self.y2 - self.y1)
        nx = min(int(xr * 10), 50)
        ny = min(int(yr * 10), 50)
        # default spacing
        default_xs = round(xr / nx, -int(np.floor(np.log10(xr / nx))))
        default_ys = round(yr / ny, -int(np.floor(np.log10(yr / ny))))
        # default number of points derived from default spacing
        default_num_x = int(round(xr / default_xs))
        default_num_y = int(round(yr / default_ys))

        if x_spacing is None:  # set from num_points if spacing is not provided
            self.num_x = default_num_x if num_x is None else int(num_x)
            self.x_spacing = self._set_spacing(self.x1, self.x2, self.num_x, default_xs)
        else:
            self.x_spacing = x_spacing
            self.num_x = int(round(xr / self.x_spacing))
        if y_spacing is None:
            self.num_y = default_num_y if num_y is None else int(num_y)
            self.y_spacing = self._set_spacing(self.y1, self.y2, self.num_y, default_ys)
        else:
            self.y_spacing = y_spacing
            self.num_y = int(round(yr / self.y_spacing))

        self._update()
        self.values = np.zeros(self.num_points)
        self.reflected_values = np.zeros(self.num_points)

    def _get_basis(self):
        """
        Return an orthonormal basis (u, v, n) for the surface:
        - n is the normal vector (points outward from surface)
        - u, v span the surface plane
        """

        if self.ref_surface == "xy":
            n = np.array([0, 0, 1])
        elif self.ref_surface == "xz":
            n = np.array([0, 1, 0])
        elif self.ref_surface == "yz":
            n = np.array([1, 0, 0])
        if self.direction != 0:
            n *= self.direction

        # Generate arbitrary vector not parallel to n
        tmp = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])

        u = np.cross(n, tmp)
        u = u / np.linalg.norm(u)

        v = np.cross(n, u)
        basis = np.stack([u, v, n], axis=1)

        return basis

    def set_height(self, height):
        """set height of calculation plane. currently we only support vertical planes"""
        if type(height) not in [float, int]:
            raise TypeError("Height must be numeric")
        self.height = height
        self._update()
        return self

    def set_ref_surface(self, ref_surface):
        """
        set the reference surface of the calc plane--must be a string in [`xy`,`xz`,`yz`]
        """
        if not isinstance(ref_surface, str):
            raise TypeError("ref_surface must be a string in [`xy`,`xz`,`yz`]")
        if ref_surface.lower() not in ["xy", "xz", "yz"]:
            raise ValueError("ref_surface must be a string in [`xy`,`xz`,`yz`]")
        self.ref_surface = ref_surface
        self.basis = self._get_basis()
        self._update()
        return self

    def set_direction(self, direction):
        """
        set the direction of the plane normal
        Valid values are currently 1, -1 and 0
        """
        if direction not in [1, 0, -1]:
            raise ValueError("Direction must be in [1, 0, -1]")
        self.direction = int(direction)
        self.basis = self._get_basis()
        return self

    def set_dimensions(self, x1=None, x2=None, y1=None, y2=None):
        """set the dimensions and update the coordinate points"""
        self.x1 = self.x1 if x1 is None else x1
        self.x2 = self.x2 if x2 is None else x2
        self.y1 = self.y1 if y1 is None else y1
        self.y2 = self.y2 if y2 is None else y2
        self.dimensions = ((self.x1, self.y1), (self.x2, self.y2))
        # update number of points, keeping spacing
        xr = abs(self.x2 - self.x1)
        yr = abs(self.y2 - self.y1)
        self.num_x = int(round(xr / self.x_spacing))
        self.num_y = int(round(yr / self.y_spacing))
        self._update()
        return self

    def set_spacing(self, x_spacing=None, y_spacing=None):
        """set the fineness of the grid spacing and update the coordinate points"""
        self.x_spacing = self.x_spacing if x_spacing is None else abs(x_spacing)
        self.y_spacing = self.y_spacing if y_spacing is None else abs(y_spacing)
        numx = int(round(abs(self.x2 - self.x1) / self.x_spacing))  # + 1
        numy = int(round(abs(self.y2 - self.y1) / self.y_spacing))  # + 1
        self.num_x = numx
        self.num_y = numy
        self._update()
        return self

    def set_num_points(self, num_x=None, num_y=None):
        """
        set the number of points desired in a dimension, instead of setting the spacing
        """
        self.num_x = self.num_x if num_x is None else abs(int(num_x))
        self.num_y = self.num_y if num_y is None else abs(int(num_y))
        if self.num_x == 0:
            warnings.warn("Number of x points must be at least 1")
            self.num_x += 1
        if self.num_y == 0:
            warnings.warn("Number of y points must be at least 1")
            self.num_y += 1

        # update spacing if required
        self.x_spacing = self._set_spacing(self.x1, self.x2, self.num_x, self.x_spacing)
        self.y_spacing = self._set_spacing(self.y1, self.y2, self.num_y, self.y_spacing)

        self._update()
        return self

    def get_calc_state(self):
        """
        return a set of paramters that, if changed, indicate that
        this calc zone must be recalculated
        """
        return [
            self.offset,
            self.x1,
            self.x2,
            self.x_spacing,
            self.y1,
            self.y2,
            self.y_spacing,
            self.height,
            self.ref_surface,
            self.direction,  # only for reflectance...possibly can be optimized
        ]

    def get_update_state(self):
        """
        return a set of parameters that, if changed, indicate that the
        calc zone need not be be recalculated, but may need updating
        """
        return [self.fov_vert, self.fov_horiz, self.vert, self.horiz]

    def _update(self):
        """
        Update the normal and number of points based on the spacing, and then the points
        """
        if self.x1 == self.x2:
            self.num_x = 1
        if self.y1 == self.y2:
            self.num_y = 1

        x_offset = min(self.x1, self.x2)
        y_offset = min(self.y1, self.y2)
        xp = np.array([i * self.x_spacing + x_offset for i in range(self.num_x)])
        yp = np.array([i * self.y_spacing + y_offset for i in range(self.num_y)])
        if self.offset:
            xp += (abs(self.x2 - self.x1) - abs(xp[-1] - xp[0])) / 2
            yp += (abs(self.y2 - self.y1) - abs(yp[-1] - yp[0])) / 2

        self.xp, self.yp = xp, yp
        self.points = [self.xp, self.yp]

        X, Y = [grid.reshape(-1) for grid in np.meshgrid(*self.points, indexing="ij")]

        if self.ref_surface in ["xy"]:
            Z = np.full(X.shape, self.height)
        elif self.ref_surface in ["xz"]:
            Z = Y
            Y = np.full(Y.shape, self.height)
        elif self.ref_surface in ["yz"]:
            Z = Y
            Y = X
            X = np.full(X.shape, self.height)

        self.coords = np.stack([X, Y, Z], axis=-1)
        self.num_points = np.array([len(self.xp), len(self.yp)])

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
            extent = [self.x1, self.x2, self.y1, self.y2]

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

        points = [np.unique(val) for val in self.coords.T]
        num_x, num_y, *rest = [len(val) for val in points if len(val) > 1]

        values = self.get_values()
        if values is None:
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
        # zvals

        rows += [[""] + list(line) for line in zvals]
        return rows
