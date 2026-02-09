import numpy as np
import copy
import hashlib
from dataclasses import dataclass
from .calc_zone import CalcPlane
from .calc_manager import apply_plane_filters
from .room_dims import RoomDimensions
from .rect_grid import PlaneGrid
from .poly_grid import PolygonGrid
from ._serialization import init_from_dict


class ReflectanceManager:
    """
    Class for managing reflective surfaces and their interactions

    Attributes:
    surfaces: dict, default = {}
    max_num_passes: int, default=100
        When calculating interreflections, the maximum number of passes before
        the calculation concludes.
    threshold: float in [0,1], default=0.02
        When calculating interreflections, the threshold below which additional
        reflection contributions are no longer calculated. Interreflection
        calculation will step when the number of loops reaches max_num_passes
        or when the contributions fall below the threshold times the initial value,
        whichever happens first.
    """

    def __init__(
        self,
        surfaces=None,
        max_num_passes: int = 100,
        threshold: float = 0.02,
        enabled: bool = True,
    ):

        self.surfaces = {} if surfaces is None else surfaces
        self.max_num_passes = int(max_num_passes)
        self.threshold = float(threshold)
        self.enabled = bool(enabled)
        if self.threshold < 0 or self.threshold > 1:
            raise ValueError("threshold must be a float between 0 and 1")

        self.zone_dict = {}  # will contain all values from all contributions

    def __eq__(self, other):
        if not isinstance(other, ReflectanceManager):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self):
        surface_rep = "(\n"
        for key, surf in self.surfaces.items():
            surface_rep += surf.__repr__() + "\n"
        return (
            f"ReflectanceManager("
            f"max_num_passes={self.max_num_passes}, "
            f"threshold={self.threshold}, "
            f"surfaces={surface_rep})"
        )

    @classmethod
    def from_dict(cls, data):
        return init_from_dict(cls, data)

    def to_dict(self):
        """Normalized configuration-only dict for equality."""
        return {
            "max_num_passes": self.max_num_passes,
            "threshold": self.threshold,
            "enabled": self.enabled,
        }

    @property
    def calc_state(self):
        tpl = tuple((key,) + val.calc_state for key, val in self.surfaces.items())
        return (self.max_num_passes, self.threshold, self.enabled) + tpl

    @property
    def keys(self):
        return self.surfaces.keys()

    @property
    def reflectances(self):
        return {key: val.R for key, val in self.surfaces.items()}

    @property
    def x_spacings(self):
        return {key: val.plane.x_spacing for key, val in self.surfaces.items()}

    @property
    def y_spacings(self):
        return {key: val.plane.y_spacing for key, val in self.surfaces.items()}

    def calculate_incidence(self, lamps, hard=False):
        """
        calculate the incident irradiances on all reflective walls
        """
        if self.enabled:
            # first pass
            for wall, surface in self.surfaces.items():
                surface.calculate_incidence(lamps, hard=hard)
            # subsequent passes - runs once, and whenever reflectance changes
            self._interreflectance(lamps, hard=hard)

    def _interreflectance(self, lamps, hard=False):
        """
        calculate additional interreflectance
        """
        # create dict of ref managers for each wall
        managers = self._create_managers()

        i = 0  # increases to self.max_num_passes
        percent = 1  # falls to self.threshold
        while percent > self.threshold and i < self.max_num_passes:
            pc = []
            for wall, surface in self.surfaces.items():
                init = surface.plane.values.mean()
                surface.calculate_incidence(
                    lamps, ref_manager=managers[wall], hard=hard
                )
                final = surface.plane.values.mean()
                if final > 0:
                    pc.append((abs(final - init) / final))
                else:
                    pc.append(0)
            percent = np.mean(pc)
            managers = self._update_managers(managers)
            i = i + 1

    def _update_managers(self, managers: dict) -> dict:
        """Update all interreflection managers with newly calculated surface incidences"""
        for wall, manager in managers.items():
            subwalls = list(manager.surfaces.keys())
            for subwall in subwalls:
                # update reflected values
                old_values = manager.surfaces[subwall].plane.result.reflected_values
                new_values = self.surfaces[subwall].plane.result.reflected_values
                if old_values is not None:
                    np.copyto(old_values, new_values)
        return managers  # Updated in place

    def _create_managers(self):
        """
        create a dict of reflection managers for each wall
        """
        managers = {}
        for wall, surface in self.surfaces.items():
            ref_manager = copy.deepcopy(self)
            # assign planes
            for subwall, surface in ref_manager.surfaces.items():
                new_plane = copy.deepcopy(self.surfaces[subwall].plane)
                ref_manager.surfaces[subwall].plane = new_plane
            # remove the surface being reflected upon
            del ref_manager.surfaces[wall]
            managers[wall] = ref_manager
        return managers

    def calculate_reflectance(self, zv, hard=False):
        """
        calculate the reflectance contribution to a calc zone from each surface
        """
        if self.enabled:
            threshold = 0  # zone.base_values.mean() * 0.01  # 1% of total value

            total_values = {}
            for wall, surface in self.surfaces.items():
                # first make sure incident irradiance is calculated
                if surface.plane.values is None:
                    raise ValueError("Incidence must be calculated before reflectance")
                if surface.R * surface.plane.values.mean() > threshold:
                    surface.calculate_reflectance(zv, hard=hard)
                    total_values[wall] = surface.zone_dict[zv.zone_id].values
                else:
                    total_values[wall] = np.zeros(zv.num_points)
            self.zone_dict[zv.zone_id] = total_values
            return self._get_total_reflectance(zv)
        else:
            return np.zeros(zv.num_points).astype("float32")

    def _get_total_reflectance(self, zv):
        """sum over all surfaces to get the total reflected values for that calc zone"""
        dct = self.zone_dict[zv.zone_id]
        values = np.zeros(zv.num_points).astype("float32")
        for wall, surface_vals in dct.items():
            if surface_vals is not None:
                values += surface_vals * self.reflectances[wall]
        return values


class Surface:
    """
    Class that represents a single surface defined by a calculation plane
    and optical properties R (reflectance) and T (transmittance).

    Constraints:
        - R and T must each be in [0, 1]
        - R + T must be <= 1 (the remainder is absorbed)
    """

    def __init__(self, R: float, plane: CalcPlane, T: float = 0.0):

        if not isinstance(R, (float, int)):
            raise TypeError("R must be a float in range [0, 1]")
        if R > 1 or R < 0:
            raise ValueError("R must be a float in range [0, 1]")

        if not isinstance(T, (float, int)):
            raise TypeError("T must be a float in range [0, 1]")
        if T > 1 or T < 0:
            raise ValueError("T must be a float in range [0, 1]")

        if R + T > 1:
            raise ValueError(
                "R + T must be <= 1 (cannot reflect and transmit more than 100%)"
            )

        if not isinstance(plane, CalcPlane):
            raise TypeError("plane must be a CalcPlane object")

        self.R = R
        self.T = T
        self.plane = plane
        self.zone_dict: dict[str, SurfaceZoneCache] = {}

    def __getattr__(self, name):
        """passthrough to plane attributes"""
        plane = self.__dict__.get("plane", None)
        if plane is not None and hasattr(plane, name):
            return getattr(plane, name)
        raise AttributeError

    def __eq__(self, other):
        if not isinstance(other, Surface):
            return NotImplemented
        return (
            self.R == other.R
            and self.T == other.T
            and self.plane.to_dict() == other.plane.to_dict()
        )

    def __repr__(self):
        return (
            f"Surface(id={self.plane.zone_id}, "
            f"R={self.R:.3g}, T={self.T:.3g}, "
            f"geometry={self.plane.geometry.__repr__()})"
        )

    @property
    def id(self) -> str:
        return self.plane.zone_id

    def _assign_id(self, value: str) -> None:
        self.plane._assign_id(value)

    def to_dict(self):
        return {"R": self.R, "T": self.T, "plane": self.plane.to_dict()}

    @classmethod
    def from_dict(cls, data):
        plane_data = data.get("plane", {})
        plane = CalcPlane.from_dict(plane_data)
        R = data.get("R", 0.0)
        T = data.get("T", 0.0)
        return cls(R=R, T=T, plane=plane)

    @property
    def calc_state(self):
        """check if the surface needs to be recalculated"""
        return self.plane.calc_state

    @property
    def update_state(self):
        """check if surface needs updating"""
        arr = self.plane.values
        return self.plane.update_state + (hashlib.sha1(arr.tobytes()).digest(),)

    def set_reflectance(self, R: float):
        if not (0 <= R <= 1):
            raise ValueError("R must be in [0, 1]")
        if R + self.T > 1:
            raise ValueError("R + T must be <= 1")
        self.R = float(R)

    def set_transmittance(self, T: float):
        if not (0 <= T <= 1):
            raise ValueError("T must be in [0, 1]")
        if self.R + T > 1:
            raise ValueError("R + T must be <= 1")
        self.T = float(T)

    def set_spacing(self, x_spacing=None, y_spacing=None):
        self.plane.set_spacing(x_spacing=x_spacing, y_spacing=y_spacing)

    def set_num_points(self, num_x=None, num_y=None):
        self.plane.set_num_points(num_x=num_x, num_y=num_y)

    def calculate_incidence(self, lamps, ref_manager=None, hard=False):
        """calculate incoming radiation onto all surfaces"""
        return self.plane.calculate_values(
            lamps=lamps, ref_manager=ref_manager, hard=hard
        )

    def calculate_reflectance(self, zv, hard=False):
        """
        calculate the reflective contribution of this reflective surface
        to a provided calculation zone

        Arguments:
            zone: a view of the calculation zone onto which reflectance is calculated
            lamp: optional. if provided, and incidence not yet calculated, uses this
            lamp to calculate incidence. mostly this is just for
        """

        if self.zone_dict.get(zv.zone_id) is None:
            self.zone_dict[zv.zone_id] = SurfaceZoneCache()

        cache = self.zone_dict[zv.zone_id]

        RECALCULATE = cache.needs_recalc(zv.calc_state, self.calc_state) or hard
        UPDATE = cache.needs_update(zv.update_state, self.update_state) or RECALCULATE

        if RECALCULATE:
            form_factors, theta_zone = self._calculate_coordinates(zv)
        else:
            form_factors = cache.form_factors
            theta_zone = cache.theta_zone
        if UPDATE:
            values = self._calculate_values(form_factors, theta_zone, zv)
        else:
            values = cache.values

        # update the state
        self.zone_dict[zv.zone_id] = SurfaceZoneCache(
            zone_calc_state=zv.calc_state,
            zone_update_state=zv.update_state,
            surface_calc_state=self.calc_state,
            surface_update_state=self.update_state,
            form_factors=form_factors,
            theta_zone=theta_zone,
            values=values,
        )

        return (values * self.R).astype("float32")

    def _calculate_values(self, form_factors, theta_zone, zv):

        I_r = self.plane.values[:, :, np.newaxis, np.newaxis, np.newaxis]
        element_size = self.plane.x_spacing * self.plane.y_spacing

        values = (I_r * element_size * form_factors).astype("float32")

        # clean nans
        if (~np.isfinite(values)).any():
            values = np.ma.masked_invalid(values)

        if zv.is_plane():
            values = apply_plane_filters(values, theta_zone, zv)

        # Sum over all self.plane points to get total values at each volume point
        values = np.sum(values, axis=(0, 1))  # Collapse the dimensions
        return values.reshape(*zv.num_points)

    def _calculate_coordinates(self, zv):
        """
        return the angles and distances between the points of the reflective
        surface and the calculation zone

        this is the expensive step!
        """
        surface_points = self.plane.coords.reshape(*self.plane.num_points, 3)
        zone_points = zv.coords.reshape(*zv.num_points, 3)

        differences = (
            surface_points[:, :, np.newaxis, np.newaxis, np.newaxis, :] - zone_points
        )
        x, y, z = differences.reshape(-1, 3).T
        distances = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        distances = distances.reshape(differences.shape[0:-1])
        # distances = np.linalg.norm(differences, axis=-1) # notably slower!

        # angles relative to reflective surface -- always between 0 and 90 unless the calc zone has been misspecified
        rel_surface = differences @ self.plane.basis
        cos_theta_surface = -rel_surface[..., 2] / distances
        cos_theta_surface[cos_theta_surface < 0] = 0
        # theta_surface = np.arccos(cos_theta_surface)
        form_factors = cos_theta_surface / (np.pi * distances ** 2)
        form_factors = form_factors.astype("float32")

        #  angles relative to calculation zone. only relevant for planes
        if zv.is_plane():
            rel_zone = differences @ zv.basis
            cos_theta_zone = rel_zone[..., 2] / distances
            theta_zone = np.arccos(cos_theta_zone).astype("float32")
        else:
            theta_zone = None

        # # ? absolute? angles
        # cos_theta = -differences[..., 2] / distances
        # theta = np.arccos(cos_theta)

        return form_factors, theta_zone


@dataclass
class SurfaceZoneCache:
    zone_calc_state: tuple | None = None  # (zone_calc_state, surface_calc_state)
    zone_update_state: tuple | None = None  # (zone_update_state, surface_update_state)
    surface_calc_state: tuple | None = None  # (zone_calc_state, surface_calc_state)
    surface_update_state: tuple | None = (
        None  # (zone_update_state, surface_update_state)
    )
    form_factors: np.ndarray | None = None
    theta_zone: np.ndarray | None = None
    values: np.ndarray | None = None

    def needs_recalc(self, zone_calc_state, surface_calc_state):
        if self.zone_calc_state != zone_calc_state:
            return True
        if self.surface_calc_state != surface_calc_state:
            return True
        if self.new_zone:
            return True
        return False

    def needs_update(self, zone_update_state, surface_update_state):
        if self.zone_update_state != zone_update_state:
            return True
        if self.surface_update_state != surface_update_state:
            return True
        return False

    @property
    def new_zone(self):
        if self.values is None:
            return True
        return False


def init_room_surfaces(
    dims: "RoomDimensions",
    reflectances: dict = None,
    x_spacings: dict = None,
    y_spacings: dict = None,
    num_x: dict = None,
    num_y: dict = None,
):
    """Initialize room surfaces for reflectance calculations."""
    keys = dims.faces.keys()
    # build defaults
    default_reflectances = {surface: 0.0 for surface in keys}
    default_spacings = {surface: None for surface in keys}
    default_nums = {surface: 10 for surface in keys}
    # build what gets used
    reflectances = {**default_reflectances, **(reflectances or {})}
    x_spacings = {**default_spacings, **(x_spacings or {})}
    y_spacings = {**default_spacings, **(y_spacings or {})}
    num_x = {**default_nums, **(num_x or {})}
    num_y = {**default_nums, **(num_y or {})}

    surfaces = {}

    if dims.is_polygon:
        # Polygon room: floor/ceiling use PolygonGrid, walls use PlaneGrid.from_wall()
        for face_id in ["floor", "ceiling"]:
            face = dims.faces[face_id]

            geometry = PolygonGrid(
                polygon=face.polygon,
                height=face.height,
                spacing_init=(x_spacings[face_id], y_spacings[face_id]),
                num_points_init=(num_x[face_id], num_y[face_id]),
                direction=face.direction,
            )
            plane = CalcPlane(zone_id=face_id, geometry=geometry, horiz=True)
            surfaces[face_id] = Surface(R=reflectances[face_id], plane=plane)

        for wall_id in dims.wall_ids:
            wall = dims.faces[wall_id]

            geometry = PlaneGrid.from_wall(
                p1=(wall.x1, wall.y1),
                p2=(wall.x2, wall.y2),
                z_height=wall.z_height,
                normal_2d=wall.normal_2d,
                spacing_init=(x_spacings[wall_id], y_spacings[wall_id]),
                num_points_init=(num_x[wall_id], num_y[wall_id]),
            )
            plane = CalcPlane(zone_id=wall_id, geometry=geometry, horiz=True)
            surfaces[wall_id] = Surface(R=reflectances[wall_id], plane=plane)
    else:
        # Rectangular room: all faces use CalcPlane.from_face()
        for wall, reflectance in reflectances.items():
            plane = CalcPlane.from_face(
                zone_id=wall,
                wall=wall,
                dims=dims,
                spacing=(x_spacings[wall], y_spacings[wall]),
                num_points=(num_x[wall], num_y[wall]),
                horiz=True,
            )
            surfaces[wall] = Surface(R=reflectance, plane=plane)

    return surfaces
