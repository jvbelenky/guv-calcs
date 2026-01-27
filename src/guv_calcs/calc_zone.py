import inspect
import json
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from .calc_manager import LightingCalculator
from .rect_grid import VolGrid, PlaneGrid, PolygonGrid, WallGrid
from .calc_zone_io import export_plane, export_volume
from .calc_zone_plot import plot_plane, plot_volume
from .room_dims import RoomDimensions, PolygonRoomDimensions
from .polygon import Polygon2D


@dataclass(frozen=True)
class ZoneView:
    zone_id: str
    coords: np.ndarray
    num_points: np.ndarray
    calc_state: tuple
    update_state: tuple
    calctype: str
    fov_vert: int | None = None
    fov_horiz: int | None = None
    vert: bool | None = None
    horiz: bool | None = None
    use_normal: bool | None = None
    basis: np.ndarray | None = None

    def is_plane(self):
        if self.calctype.lower() == "plane":
            return True
        return False

    def is_volume(self):
        if self.calctype.lower() == "volume":
            return True
        return False


@dataclass
class ZoneResult:
    base_values: np.ndarray | None = None
    reflected_values: np.ndarray | None = None

    def init(self, num_points=None):
        self.base_values = None
        self.reflected_values = None
        return self

    @property
    def values(self):
        if self.base_values is None:
            return None
        if self.reflected_values is None:
            return self.base_values
        return self.base_values + self.reflected_values


class CalcZone(ABC):
    """
    Abstract base class representing a calculation zone.

    Subclasses must implement calctype, to_view(), export(), and plot().

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
    show_values: bool, default = True
    colormap: str, default = None
    """

    def __init__(
        self,
        zone_id=None,
        name=None,
        dose=False,
        hours=8.0,
        enabled=True,
        show_values=True,
        colormap=None,
    ):
        self._zone_id = zone_id
        self.name = str(zone_id) if name is None else str(name)
        self.dose = False if dose is None else dose
        self.hours = hours  # only used if dose is true
        self.enabled = enabled
        self.show_values = show_values
        self.colormap = colormap

        # these will all be calculated after spacing is set, which is set in the subclass
        self._geometry = None
        self.calculator = LightingCalculator()
        self.result = ZoneResult()

    def __getattr__(self, name):
        """
        attribute passthrough to geometry - called if normal lookup fails
        """
        geometry = self.__dict__.get("_geometry", None)
        if geometry is not None and hasattr(geometry, name):
            return getattr(geometry, name)
        raise AttributeError(name)

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

    @property
    def id(self) -> str:
        return self._zone_id

    @property
    def zone_id(self) -> str:
        return self._zone_id

    def _assign_id(self, value: str) -> None:
        """should only be used by Scene"""
        self._zone_id = value

    @property
    @abstractmethod
    def calctype(self) -> str:
        """Return zone type identifier ('Plane' or 'Volume')."""
        ...

    @property
    def units(self):
        # todo: probably should be called unit_label or something instead
        if self.dose:
            return "mJ/cm²"
        return "uW/cm²"

    @property
    def calc_state(self):
        """if changes, these parameters indicate zone must be recalculated"""
        if self.geometry is None:
            return ()
        return self.geometry.calc_state

    @property
    def update_state(self):
        """if changes, calc zone needs updating but not recalculating"""
        if self.geometry is None:
            return ()
        return self.geometry.update_state

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, new_geom):
        if self._geometry != new_geom:  # only clear if geometry is different
            if hasattr(self, "result") and self.result is not None:
                self.result.init(new_geom.num_points)
        self._geometry = new_geom

    @classmethod
    def from_dict(cls, data):
        keys = list(inspect.signature(cls.__init__).parameters.keys())[1:]
        return cls(**{k: v for k, v in data.items() if k in keys})

    def to_dict(self):
        data = {}
        data["zone_id"] = self.zone_id
        data["name"] = self.name
        data["dose"] = self.dose
        data["hours"] = self.hours
        data["enabled"] = self.enabled
        data["show_values"] = self.show_values
        data["colormap"] = self.colormap
        if self.geometry is not None:
            data["geometry"] = self.geometry.to_dict()

        data.update(self._extra_dict())
        return data

    def _extra_dict(self):
        return {}

    def save(self, filename):
        """save zone data to json file"""
        data = self.to_dict()
        with open(filename, "w") as json_file:
            json_file.write(json.dumps(data))

    @abstractmethod
    def to_view(self) -> ZoneView:
        """Return a snapshot of the zone's state for calculation."""
        ...

    @abstractmethod
    def export(self, fname=None):
        """Export zone values to file."""
        ...

    @abstractmethod
    def plot(self, **kwargs):
        """Plot the zone values."""
        ...

    def copy(self, **kwargs):
        """
        create a fresh copy of this calc zone
        """
        dct = self.to_dict()
        # replace dict with keyword args if any
        for key, val in dct.items():
            new_val = kwargs.get(key, None)
            if new_val is not None:
                dct[key] = new_val
        return type(self).from_dict(dct)

    def set_value_type(self, dose):
        """
        if true get_values() will return in dose (mJ/cm2) over hours
        otherwise get_values() will return in irradiance or fluence (uW/cm2)
        """
        if type(dose) is not bool:
            raise TypeError("Dose must be either True or False")
        self.dose = dose

    def set_dose_time(self, hours):
        """
        Set the time over which the dose will be calculate in hours
        """
        if type(hours) not in [float, int]:
            raise TypeError("Hours must be numeric")
        self.hours = hours

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
        if self.geometry is not None:
            mins = self._strip_tpl(
                x1 or self.geometry.x1, y1 or self.geometry.y1, z1 or self.geometry.z1
            )
            maxs = self._strip_tpl(
                x2 or self.geometry.x2, y2 or self.geometry.y2, z2 or self.geometry.z2
            )
            self.geometry = self.geometry.update_dimensions(
                mins=mins, maxs=maxs, preserve_spacing=preserve_spacing
            )
        return self

    def set_spacing(self, x_spacing=None, y_spacing=None, z_spacing=None):
        """
        set the spacing desired in the dimension
        """
        if self.geometry is not None:
            spacing = self._strip_tpl(
                x_spacing or self.geometry.x_spacing,
                y_spacing or self.geometry.y_spacing,
                z_spacing or self.geometry.z_spacing,
            )
            self.geometry = self.geometry.update(spacing_init=spacing)
        return self

    def set_num_points(self, num_x=None, num_y=None, num_z=None):
        """
        set the number of points desired in a dimension, instead of setting the spacing
        """
        if self.geometry is not None:
            num_points_init = self._strip_tpl(
                num_x or self.geometry.num_x,
                num_y or self.geometry.num_y,
                num_z or self.geometry.num_z,
            )
            self.geometry = self.geometry.update(num_points_init=num_points_init)
        return self

    def set_offset(self, offset):
        if self.geometry is not None:
            self.geometry = self.geometry.update(offset=offset)
        return self

    @staticmethod
    def _strip_tpl(*args):
        return tuple(val for val in args if val is not None)

    def calculate_values(self, lamps, ref_manager=None, hard=False):
        """Calculate all the values for all the lamps"""

        if self.enabled:

            self.result.base_values = self.calculator.compute(
                lamps=lamps, zv=self.to_view(), hard=hard
            )
            if ref_manager is not None:
                # calculate reflectance -- warning, may be expensive!
                self.result.reflected_values = ref_manager.calculate_reflectance(
                    self.to_view(), hard=hard
                )

        return self.get_values()

    def get_values(self):
        """return dose-adjusted values"""
        if self.result.values is None:
            return None
        if self.dose:
            return self.result.values * 3.6 * self.hours
        return self.result.values

    @property
    def lamp_cache(self):
        """lamp_values are stored here"""
        return self.calculator.cache.lamp_cache

    @property
    def base_values(self):
        return self.result.base_values

    @property
    def reflected_values(self):
        return self.result.reflected_values

    @property
    def values(self):
        return self.result.values


class CalcVol(CalcZone):
    """
    Represents a volumetric calculation zone.
    A subclass of CalcZone designed for three-dimensional volumetric calculations.
    """

    def __init__(
        self,
        zone_id: str | None = None,
        name: str | None = None,
        geometry: VolGrid | None = None,
        dose: bool = False,
        hours: int = 8,
        enabled: bool = True,
        show_values: bool = True,
        colormap: str | None = None,
        # legacy -- ignored if geometry is not None
        x1: float | None = None,
        x2: float | None = None,
        y1: float | None = None,
        y2: float | None = None,
        z1: float | None = None,
        z2: float | None = None,
        num_x: int | None = None,
        num_y: int | None = None,
        num_z: int | None = None,
        x_spacing: float | None = None,
        y_spacing: float | None = None,
        z_spacing: float | None = None,
        offset: bool | None = None,
    ):

        super().__init__(
            zone_id=zone_id or "CalcVol",
            name=name,
            dose=dose,
            hours=hours,
            enabled=enabled,
            show_values=show_values,
            colormap=colormap,
        )
        if geometry is None:
            self.geometry = VolGrid.from_legacy(
                mins=(x1 or 0.0, y1 or 0.0, z1 or 0.0),
                maxs=(x2 or 6.0, y2 or 4.0, z2 or 2.7),
                num_points_init=(num_x, num_y, num_z),
                spacing_init=(x_spacing, y_spacing, z_spacing),
                offset=True if offset is None else bool(offset),
            )
        else:
            self.geometry = geometry

    def _extra_dict(self) -> dict:
        zone_data = super()._extra_dict()
        data = {"calctype": self.calctype}
        zone_data.update(data)
        return zone_data

    @property
    def calctype(self) -> str:
        return "Volume"

    def __repr__(self):
        return super().__repr__() + f"geometry: {self.geometry.__repr__()})"

    @classmethod
    def from_dict(cls, data):
        keys = list(inspect.signature(cls.__init__).parameters.keys())[1:]
        if data.get("geometry") is not None:
            geometry = VolGrid.from_dict(data.pop("geometry"))
            data["geometry"] = geometry
        return cls(**{k: v for k, v in data.items() if k in keys})

    @classmethod
    def from_dims(
        cls,
        dims: "RoomDimensions | PolygonRoomDimensions",
        spacing: float | None = None,
        num_points: int | None = None,
        offset: bool = True,
        **kwargs,
    ):
        """Create a CalcVol from room dimensions (uses bounding box for polygon rooms)."""
        geometry = VolGrid.from_legacy(
            mins=(0, 0, 0),
            maxs=(dims.x, dims.y, dims.z),
            spacing_init=spacing,
            num_points_init=num_points,
            offset=offset,
        )
        return cls(geometry=geometry, **kwargs)

    def to_view(self):
        """take a snapshot of the zone's state"""
        return ZoneView(
            zone_id=self.zone_id,
            coords=self.geometry.coords,
            num_points=self.geometry.num_points,
            calc_state=self.calc_state,
            update_state=self.update_state,
            calctype=self.calctype,
        )

    def export(self, fname=None):
        """export values to csv"""
        return export_volume(self, fname=fname)

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
        zone_id: str | None = None,
        name: str | None = None,
        geometry: "PlaneGrid | PolygonGrid | WallGrid | None" = None,
        dose: bool = False,
        hours: int = 8,
        enabled: bool = True,
        show_values: bool = True,
        colormap: str | None = None,
        # soon to be legacy-ish
        fov_vert: int | float = 180,
        fov_horiz: int | float = 360,
        vert: bool = False,
        horiz: bool = False,
        use_normal: bool = True,
        # legacy only! ignored if geometry is not None
        x1: float | None = None,
        x2: float | None = None,
        y1: float | None = None,
        y2: float | None = None,
        num_x: int | None = None,
        num_y: int | None = None,
        x_spacing: float | None = None,
        y_spacing: float | None = None,
        offset: bool | None = None,
        height: float | None = None,
        ref_surface: str | None = None,
        direction: int | None = None,
    ):

        super().__init__(
            zone_id=zone_id or "CalcPlane",
            name=name,
            dose=dose,
            hours=hours,
            enabled=enabled,
            show_values=show_values,
            colormap=colormap,
        )

        if geometry is None:
            # legacy initialization strategy
            self.geometry = PlaneGrid.from_legacy(
                mins=(x1 or 0.0, y1 or 0.0),
                maxs=(x2 or 6.0, y2 or 4.0),
                spacing_init=(x_spacing, y_spacing),
                num_points_init=(num_x, num_y),
                offset=True if offset is None else bool(offset),
                height=height or 0,
                ref_surface=ref_surface or "xy",
                direction=direction or 1,
            )
        else:
            self.geometry = geometry

        self.fov_vert = fov_vert
        self.fov_horiz = fov_horiz
        # flags to be killed and replaced by PlaneType enum or something
        self.use_normal = use_normal
        self.vert = vert
        self.horiz = horiz

    @property
    def calctype(self):
        return "Plane"

    def _extra_dict(self):

        zone_data = super()._extra_dict()
        data = {
            "fov_vert": self.fov_vert,
            "fov_horiz": self.fov_horiz,
            "use_normal": self.use_normal,
            "vert": self.vert,
            "horiz": self.horiz,
            "calctype": self.calctype,
        }
        zone_data.update(data)
        return zone_data

    def __repr__(self):
        return super().__repr__() + (
            f"geometry={self.geometry.__repr__()}, "
            f"field_of_view=({self.fov_horiz}° horiz, {self.fov_vert}° vert), "
            f"flags=(vert={self.vert}, horiz={self.horiz}, use_normal={self.use_normal}), "
        )

    @classmethod
    def from_dict(cls, data):
        keys = list(inspect.signature(cls.__init__).parameters.keys())[1:]
        if data.get("geometry") is not None:
            geom_data = data.pop("geometry")
            # Detect geometry type from data
            if "polygon" in geom_data:
                if "p1" in geom_data:
                    geometry = WallGrid.from_dict(geom_data)
                else:
                    geometry = PolygonGrid.from_dict(geom_data)
            else:
                geometry = PlaneGrid.from_dict(geom_data)
            data["geometry"] = geometry
        return cls(**{k: v for k, v in data.items() if k in keys})

    @classmethod
    def from_points(
        cls,
        p0,
        pU,
        pV,
        num_points_init=None,
        spacing=None,
        offset=True,
        **kwargs,
    ):
        """define a calc plane from an arbitrary origin, U point, and V point"""
        geometry = PlaneGrid.from_points(
            p0=p0,
            pU=pU,
            pV=pV,
            spacing_init=spacing,
            num_points_init=num_points_init,
            offset=offset,
        )
        return cls(geometry=geometry, **kwargs)

    @classmethod
    def from_face(
        cls,
        wall: str,
        dims: "RoomDimensions | PolygonRoomDimensions",
        normal_offset: float = 0.0,
        spacing: float | None = None,
        num_points: int | None = None,
        offset: bool = True,
        **kwargs,
    ):
        if wall.lower() not in dims.faces.keys():
            raise KeyError(
                f"{wall} is not a valid wall ID, must be in {dims.faces.keys()}"
            )

        face_data = dims.faces[wall]

        # Handle polygon room dimensions
        if dims.is_polygon:
            if wall in ("floor", "ceiling"):
                # face_data: (x_min, x_max, y_min, y_max, height, ref_surface, direction, polygon)
                base_height = face_data[4]
                direction = face_data[6]
                polygon = face_data[7]
                height = base_height + normal_offset * direction

                geometry = PolygonGrid(
                    polygon=polygon,
                    height=height,
                    spacing_init=spacing,
                    num_points_init=num_points,
                    offset=offset,
                    direction=direction,
                )
            else:
                # Wall: (x1, y1, x2, y2, edge_length, z_height, normal_2d)
                x1, y1, x2, y2, edge_length, z_height, normal_2d = face_data

                geometry = WallGrid(
                    p1=(x1, y1),
                    p2=(x2, y2),
                    z_height=z_height,
                    normal_2d=normal_2d,
                    spacing_init=spacing,
                    num_points_init=num_points,
                    offset=offset,
                )
        else:
            # Rectangular room - original behavior
            x1, x2, y1, y2, base_height, ref_surface, direction = face_data
            height = base_height + normal_offset * direction

            geometry = PlaneGrid.from_legacy(
                mins=(x1, y1),
                maxs=(x2, y2),
                height=height,
                ref_surface=ref_surface,
                spacing_init=spacing,
                num_points_init=num_points,
                offset=offset,
            )

        return cls(geometry=geometry, **kwargs)

    @classmethod
    def from_polygon(
        cls,
        polygon: "Polygon2D | list[tuple[float, float]]",
        height: float = 0.0,
        direction: int = 1,
        spacing: float | None = None,
        num_points: int | None = None,
        offset: bool = True,
        **kwargs,
    ):
        """Create a CalcPlane from a polygon at a specified height."""
        if not isinstance(polygon, Polygon2D):
            polygon = Polygon2D(vertices=tuple(tuple(v) for v in polygon))

        geometry = PolygonGrid(
            polygon=polygon,
            height=height,
            spacing_init=spacing,
            num_points_init=num_points,
            offset=offset,
            direction=direction,
        )
        return cls(geometry=geometry, **kwargs)

    @property
    def update_state(self):
        """if changes, calc_zone needs updating but not recalculating"""
        return super().update_state + (
            self.fov_vert,
            self.fov_horiz,
            self.vert,
            self.horiz,
            self.use_normal,
        )

    def to_view(self):
        """take a snapshot of the zone's state"""
        return ZoneView(
            zone_id=self.zone_id,
            coords=self.geometry.coords,
            num_points=self.geometry.num_points,
            calc_state=self.calc_state,
            update_state=self.update_state,
            calctype=self.calctype,
            fov_vert=self.fov_vert,
            fov_horiz=self.fov_horiz,
            vert=self.vert,
            horiz=self.horiz,
            use_normal=self.use_normal,
            basis=self.basis,
        )

    def set_height(self, height):
        self.geometry = self.geometry.update_legacy(height=height)
        return self

    def set_ref_surface(self, ref_surface):
        self.geometry = self.geometry.update_legacy(ref_surface=ref_surface)
        return self

    def set_direction(self, direction):
        self.geometry = self.geometry.update_legacy(direction=direction)
        return self

    def export(self, fname=None):
        """export values to csv"""
        return export_plane(self, fname=fname)

    def plot(self, **kwargs):
        """Plot the image of the radiation pattern"""
        return plot_plane(self, **kwargs)

    def plot_plane(self, **kwargs):
        """alias for plot() -- kept in for compatibility"""
        return self.plot(**kwargs)
