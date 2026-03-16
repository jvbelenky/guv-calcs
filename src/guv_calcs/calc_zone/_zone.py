import json
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import timedelta
from ..calc_manager import LightingCalculator
from ..geometry import SurfaceGrid, VolumeGrid, GridPoint
from ._io import export_plane, export_volume
from ._plot import plot_plane, plot_volume
from ..geometry import RoomDimensions
from ..geometry import Polygon2D
from .._serialization import init_from_dict, deserialize_geometry, migrate_zone_dict, migrate_legacy_zone_geometry
from ..plane_calc_mode import PlaneCalcMode


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
    view_direction: tuple | None = None
    view_target: tuple | None = None

    def is_plane(self):
        if self.calctype.lower() == "plane":
            return True
        return False

    def is_volume(self):
        if self.calctype.lower() == "volume":
            return True
        return False

    def has_view_mode(self):
        return self.view_direction is not None or self.view_target is not None

    def compute_view_normals(self):
        """Return (N, 3) normalized view direction vectors for each grid point.

        For directional mode: broadcasts the single direction to all points.
        For target mode: computes per-point direction toward the target.
        Returns None if no view mode is active.
        """
        if self.view_direction is not None:
            d = np.asarray(self.view_direction, dtype="float64")
            d = d / np.linalg.norm(d)
            return np.broadcast_to(d, self.coords.shape).copy()
        if self.view_target is not None:
            target = np.asarray(self.view_target, dtype="float64")
            diff = target - self.coords
            norms = np.linalg.norm(diff, axis=1, keepdims=True)
            # Guard against coincident points (target == grid point)
            norms = np.where(norms < 1e-12, 1.0, norms)
            return diff / norms
        return None


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
    """

    _grid_cls = None  # subclasses set to VolumeGrid or SurfaceGrid

    def __init__(
        self,
        zone_id=None,
        name=None,
        dose=False,
        hours=None,
        minutes=None,
        seconds=None,
        enabled=True,
        display_mode="heatmap",
    ):
        self._zone_id = zone_id
        self.name = str(zone_id) if name is None else str(name)
        self.dose = False if dose is None else dose
        self._exposure_time = self._parse_exposure_time(hours, minutes, seconds)
        self.enabled = enabled
        self.display_mode = display_mode

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
    def value_units(self):
        # todo: probably should be called unit_label or something instead
        if self.dose:
            return "mJ/cm²"
        return "uW/cm²"

    @property
    def calc_state(self):
        """if changes, these parameters indicate zone must be recalculated"""
        if self.geometry is None:
            return ()
        return self.geometry.calc_state + (self.result.base_values is not None,)

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
        data = migrate_zone_dict(data)
        if cls._grid_cls is not None:
            data = migrate_legacy_zone_geometry(data, cls._grid_cls)
            data = deserialize_geometry(data, cls._grid_cls)
        return init_from_dict(cls, data)

    def to_dict(self):
        data = {}
        data["zone_id"] = self.zone_id
        data["name"] = self.name
        data["calctype"] = self.calctype
        data["dose"] = self.dose
        data["exposure_time"] = self._exposure_time.total_seconds()
        data["enabled"] = self.enabled
        data["display_mode"] = self.display_mode
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

    def set_dose_time(self, hours=None, minutes=None, seconds=None):
        """Set the exposure time for dose calculation."""
        self._exposure_time = self._parse_exposure_time(hours, minutes, seconds, default=False)
        
    @property
    def exposure_time(self) -> timedelta:
        return self._exposure_time

    @exposure_time.setter
    def exposure_time(self, value: timedelta):
        if not isinstance(value, timedelta):
            raise TypeError("exposure_time must be a timedelta")
        self._exposure_time = value

    @property
    def hours(self) -> float:
        return self._exposure_time.total_seconds() / 3600

    @property
    def minutes(self) -> float:
        return self._exposure_time.total_seconds() / 60

    @property
    def seconds(self) -> float:
        return self._exposure_time.total_seconds()


    def convert_units(self, old_units, new_units):
        """Convert geometry coordinates without invalidating calculated values."""
        if self._geometry is not None:
            self._geometry = self._geometry._convert_units(old_units, new_units)
            # Update cached calc_state to match new geometry, preventing unnecessary recalc
            if self.calculator.cache.calc_state is not None:
                self.calculator.cache = replace(
                    self.calculator.cache,
                    calc_state=self.calc_state,
                    update_state=self.update_state,
                )

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
                self._coalesce(x1, self.geometry.x1),
                self._coalesce(y1, self.geometry.y1),
                self._coalesce(z1, self.geometry.z1),
            )
            maxs = self._strip_tpl(
                self._coalesce(x2, self.geometry.x2),
                self._coalesce(y2, self.geometry.y2),
                self._coalesce(z2, self.geometry.z2),
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
                self._coalesce(x_spacing, self.geometry.x_spacing),
                self._coalesce(y_spacing, self.geometry.y_spacing),
                self._coalesce(z_spacing, self.geometry.z_spacing),
            )
            self.geometry = self.geometry.update(spacing_init=spacing)
        return self

    def set_num_points(self, num_x=None, num_y=None, num_z=None):
        """
        set the number of points desired in a dimension, instead of setting the spacing
        """
        if self.geometry is not None:
            num_points_init = self._strip_tpl(
                self._coalesce(num_x, self.geometry.num_x),
                self._coalesce(num_y, self.geometry.num_y),
                self._coalesce(num_z, self.geometry.num_z),
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

    @staticmethod
    def _coalesce(value, default):
        return default if value is None else value

    def calculate_values(self, lamps, surfaces=None, enable_occlusion=True,
                         enable_reflectance=True, hard=False):
        """Calculate all the values for all the lamps."""

        if self.enabled:
            zv = self.to_view()

            self.result.base_values = self.calculator.compute(
                lamps=lamps, zv=zv, surfaces=surfaces,
                enable_occlusion=enable_occlusion, hard=hard
            )

            if surfaces and enable_reflectance:
                self.result.reflected_values = self.calculator.compute_reflectance(
                    surfaces=surfaces, zv=zv,
                    enable_occlusion=enable_occlusion, hard=hard
                )
            else:
                self.result.reflected_values = None

        return self.get_values()

    def get_values(self):
        """return dose-adjusted values"""
        if self.result.values is None:
            return None
        if self.dose:
            return self.result.values * self._exposure_time.total_seconds() / 1e3
        return self.result.values

    def get_statistics(self) -> dict | None:
        """Return NaN-aware statistics for calculated values.

        Returns:
            Dict with min, max, mean, std, max_min, avg_min keys,
            or None if uncalculated.
        """
        values = self.get_values()
        if values is None:
            return None
        mn, mx, avg = float(np.min(values)), float(np.max(values)), float(np.mean(values))
        return {
            "min": mn, "max": mx, "mean": avg,
            "std": float(np.std(values)),
            "max_min": mx / mn if mn != 0 else float("inf"),
            "avg_min": avg / mn if mn != 0 else float("inf"),
        }

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
        
    @staticmethod
    def _parse_exposure_time(hours=None, minutes=None, seconds=None, default=True):
        """Parse time params into a timedelta. Values are summed."""
        h = hours or 0
        m = minutes or 0
        s = seconds or 0
        for name, val in [("hours", h), ("minutes", m), ("seconds", s)]:
            if not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if val < 0:
                raise ValueError(f"{name} must be non-negative")
        if h == 0 and m == 0 and s == 0:
            if default:
                return timedelta(hours=8)
            raise ValueError("Specify at least one of hours, minutes, seconds")
        return timedelta(hours=h, minutes=m, seconds=s)


class CalcVol(CalcZone):
    """Volumetric calculation zone for three-dimensional calculations."""

    _grid_cls = VolumeGrid

    def __init__(
        self,
        zone_id: str | None = None,
        name: str | None = None,
        geometry: VolumeGrid | None = None,
        dose: bool = False,
        hours: int | float | None = None,
        minutes: int | float | None = None,
        seconds: int | float | None = None,
        enabled: bool = True,
        display_mode: str = "heatmap",
    ):

        super().__init__(
            zone_id=zone_id or "CalcVol",
            name=name,
            dose=dose,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            enabled=enabled,
            display_mode=display_mode,
        )
        if geometry is None:
            geometry = VolumeGrid.from_polygon(
                Polygon2D.rectangle(6.0, 4.0), z_height=2.7, offset=True,
            )
        self.geometry = geometry

    @property
    def calctype(self) -> str:
        return "Volume"

    def __repr__(self):
        return super().__repr__() + f"geometry: {self.geometry.__repr__()})"

    @classmethod
    def from_dims(
        cls,
        dims: "RoomDimensions",
        spacing: float | None = None,
        num_points: int | None = None,
        offset: bool = True,
        **kwargs,
    ):
        """Create a CalcVol from room dimensions."""
        geometry = VolumeGrid.from_polygon(
            polygon=dims.polygon,
            z_height=dims.z,
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
    """Planar calculation zone for two-dimensional calculations at a specific height."""

    _grid_cls = SurfaceGrid

    # Backward-compatible defaults when no calc_mode is provided
    _DEFAULT_HORIZ = False
    _DEFAULT_VERT = False
    _DEFAULT_USE_NORMAL = True
    _DEFAULT_FOV_VERT = 180
    _DEFAULT_FOV_HORIZ = 360

    def __init__(
        self,
        zone_id: str | None = None,
        name: str | None = None,
        geometry: "SurfaceGrid | None" = None,
        dose: bool = False,
        hours: int | float | None = None,
        minutes: int | float | None = None,
        seconds: int | float | None = None,
        enabled: bool = True,
        display_mode: str = "heatmap",
        calc_mode: str | None = None,
        fov_vert: int | float | None = None,
        fov_horiz: int | float | None = None,
        vert: bool | None = None,
        horiz: bool | None = None,
        use_normal: bool | None = None,
        view_direction: tuple | list | None = None,
        view_target: tuple | list | None = None,
    ):

        super().__init__(
            zone_id=zone_id or "CalcPlane",
            name=name,
            dose=dose,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            enabled=enabled,
            display_mode=display_mode,
        )

        if geometry is None:
            geometry = SurfaceGrid.from_polygon(
                Polygon2D.rectangle(6.0, 4.0), height=0, direction=1, offset=True,
            )
        self.geometry = geometry

        # Validate mutual exclusivity of explicit view params
        if view_direction is not None and view_target is not None:
            raise ValueError(
                "view_direction and view_target are mutually exclusive"
            )

        self.view_direction = view_direction
        self.view_target = view_target

        # Resolve flags: calc_mode spec → explicit overrides → defaults
        if calc_mode is not None:
            ct = PlaneCalcMode.from_token(calc_mode)
            spec = ct.spec
            self.horiz = horiz if horiz is not None else spec.horiz
            self.vert = vert if vert is not None else spec.vert
            self.use_normal = use_normal if use_normal is not None else spec.use_normal
            self.fov_vert = fov_vert if fov_vert is not None else spec.fov_vert
            self.fov_horiz = fov_horiz if fov_horiz is not None else spec.fov_horiz
            # Apply view params from spec if not explicitly provided
            if view_direction is None and view_target is None:
                self.view_direction = spec.view_direction
                self.view_target = spec.view_target
        else:
            self.horiz = horiz if horiz is not None else self._DEFAULT_HORIZ
            self.vert = vert if vert is not None else self._DEFAULT_VERT
            self.use_normal = use_normal if use_normal is not None else self._DEFAULT_USE_NORMAL
            self.fov_vert = fov_vert if fov_vert is not None else self._DEFAULT_FOV_VERT
            self.fov_horiz = fov_horiz if fov_horiz is not None else self._DEFAULT_FOV_HORIZ

    @property
    def calctype(self):
        return "Plane"

    @property
    def view_mode(self) -> str | None:
        """Return the active view mode, or None for normal (surface-normal) mode."""
        if self.view_direction is not None:
            return "directional"
        if self.view_target is not None:
            return "target"
        return None

    @property
    def calc_mode(self) -> str:
        """Derive the calculation mode from current flags.

        Returns the matching PlaneCalcMode value if flags match a known mode,
        otherwise returns "custom".
        """
        return PlaneCalcMode.from_flags(
            horiz=self.horiz,
            vert=self.vert,
            use_normal=self.use_normal,
            fov_vert=float(self.fov_vert),
            fov_horiz=float(self.fov_horiz),
            view_direction=self.view_direction,
            view_target=self.view_target,
        ).value

    def set_calc_mode(self, value: str):
        """Set all calculation flags from a named calc mode.

        After calling this, individual flags can be overridden, which may
        cause the derived calc_mode to become "custom".
        """
        ct = PlaneCalcMode.from_token(value)
        if ct is PlaneCalcMode.CUSTOM:
            raise ValueError(
                "Cannot set calc_mode to 'custom' — set individual flags instead"
            )
        spec = ct.spec
        self.horiz = spec.horiz
        self.vert = spec.vert
        self.use_normal = spec.use_normal
        self.fov_vert = spec.fov_vert
        self.fov_horiz = spec.fov_horiz
        self.view_direction = spec.view_direction
        self.view_target = spec.view_target
        return self

    def _extra_dict(self):
        return {
            "calc_mode": self.calc_mode,
            "fov_vert": self.fov_vert,
            "fov_horiz": self.fov_horiz,
            "use_normal": self.use_normal,
            "vert": self.vert,
            "horiz": self.horiz,
            "view_direction": self.view_direction,
            "view_target": self.view_target,
        }

    def __repr__(self):
        return super().__repr__() + (
            f"geometry={self.geometry.__repr__()}, "
            f"field_of_view=({self.fov_horiz}° horiz, {self.fov_vert}° vert), "
            f"flags=(vert={self.vert}, horiz={self.horiz}, use_normal={self.use_normal}), "
        )

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
        geometry = SurfaceGrid.from_points(
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
        dims: "RoomDimensions",
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

        face = dims.faces[wall]
        geometry = face.to_grid(
            normal_offset=normal_offset,
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

        geometry = SurfaceGrid.from_polygon(
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
            self.view_direction,
            self.view_target,
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
            view_direction=self.view_direction,
            view_target=self.view_target,
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


class CalcPoint(CalcPlane):
    """Single-point calculation zone with full CalcPlane view/FOV support."""

    _grid_cls = GridPoint

    def __init__(
        self,
        position=(0.0, 0.0, 0.0),
        normal_direction=None,
        zone_id=None,
        name=None,
        geometry=None,
        dose=False,
        hours=None,
        minutes=None,
        seconds=None,
        enabled=True,
        calc_mode=None,
        fov_vert=None,
        fov_horiz=None,
        vert=None,
        horiz=None,
        use_normal=None,
        view_direction=None,
        view_target=None,
    ):
        if geometry is None:
            n = normal_direction or (0.0, 0.0, 1.0)
            geometry = GridPoint(position=position, normal_direction=n)

        super().__init__(
            zone_id=zone_id or "CalcPoint",
            name=name,
            geometry=geometry,
            dose=dose,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            enabled=enabled,
            display_mode="heatmap",
            calc_mode=calc_mode,
            fov_vert=fov_vert,
            fov_horiz=fov_horiz,
            vert=vert,
            horiz=horiz,
            use_normal=use_normal,
            view_direction=view_direction,
            view_target=view_target,
        )

    @property
    def calctype(self):
        return "Point"

    @property
    def position(self):
        return self.geometry.position

    def _extra_dict(self):
        d = super()._extra_dict()
        d["position"] = list(self.geometry.position)
        d["normal_direction"] = list(self.geometry.normal_direction)
        return d

    def export(self, fname=None):
        raise NotImplementedError("CalcPoint does not support CSV export")

    def plot(self, **kwargs):
        raise NotImplementedError("CalcPoint does not support plotting")


