import numpy as np
import copy
import inspect
from dataclasses import dataclass
from .calc_zone import CalcPlane
from .room_dims import RoomDimensions


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
        self, surfaces=None, max_num_passes: int = 100, threshold: float = 0.02
    ):

        self.surfaces = {} if surfaces is None else surfaces
        self.max_num_passes = int(max_num_passes)
        self.threshold = float(threshold)
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
        keys = list(inspect.signature(cls.__init__).parameters.keys())[1:]
        surface_data = data.get("surfaces", {})
        surfaces = {}
        for key, val in surface_data.items():
            surfaces[key] = ReflectiveSurface.from_dict(val)
        data["surfaces"] = surfaces
        return cls(**{k: v for k, v in data.items() if k in keys})

    def to_dict(self):
        """Normalized configuration-only dict for equality."""
        surf_dict = {key: val.to_dict() for key, val in self.surfaces.items()}
        return {
            "surfaces": surf_dict,
            "max_num_passes": self.max_num_passes,
            "threshold": self.threshold,
        }

    def copy(self, **kwargs):
        """return fresh copy of this class"""
        dct = self.to_dict()
        # replace dict with keyword args if any
        for key, val in dct.items():
            new_val = kwargs.get(key, None)
            if new_val is not None:
                dct[key] = new_val
        return ReflectanceManager.from_dict(dct)

    @property
    def calc_state(self):
        tpl = tuple((key,) + val.calc_state for key, val in self.surfaces.items())
        return (self.max_num_passes, self.threshold) + tpl

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
                np.copyto(old_values, new_values)
        return managers  # Updated in place

    def _create_managers(self):
        """
        create a dict of reflection managers for each wall
        """
        managers = {}
        for wall, surface in self.surfaces.items():
            ref_manager = self.copy()
            # assign planes
            for subwall, surface in ref_manager.surfaces.items():
                new_plane = copy.deepcopy(self.surfaces[subwall].plane)
                ref_manager.surfaces[subwall].plane = new_plane
            # remove the surface being reflected upon
            del ref_manager.surfaces[wall]
            managers[wall] = ref_manager
        return managers

    def calculate_reflectance(self, zone, hard=False):
        """
        calculate the reflectance contribution to a calc zone from each surface
        """

        threshold = zone.base_values.mean() * 0.01  # 1% of total value

        total_values = {}
        for wall, surface in self.surfaces.items():
            if surface.R * surface.plane.values.mean() > threshold:
                surface.calculate_reflectance(zone, hard=hard)
                total_values[wall] = surface.zone_dict[zone.zone_id].values
            else:
                total_values[wall] = np.zeros(zone.num_points)
        self.zone_dict[zone.zone_id] = total_values

        return self.get_total_reflectance(zone)

    def get_total_reflectance(self, zone):
        """sum over all surfaces to get the total reflected values for that calc zone"""
        dct = self.zone_dict[zone.zone_id]
        values = np.zeros(zone.num_points).astype("float32")
        for wall, surface_vals in dct.items():
            if surface_vals is not None:
                values += surface_vals * self.reflectances[wall]
        return values


class ReflectiveSurface:
    """
    Class that represents a single reflective surface defined by a calculation
    zone and a float value R between 0 and 1.
    """

    def __init__(self, R, plane):

        if not isinstance(R, (float, int)):
            raise TypeError("R must be a float in range [0, 1]")
        if R > 1 or R < 0:
            raise ValueError("R must be a float in range [0, 1]")

        if not isinstance(plane, CalcPlane):
            raise TypeError("plane must be a CalcPlane object")

        self.R = R
        self.plane = plane
        self.zone_dict: dict[str, SurfaceZoneCache] = {}

    def __eq__(self, other):
        if not isinstance(other, ReflectiveSurface):
            return NotImplemented
        return self.R == other.R and self.plane.to_dict() == other.plane.to_dict()

    def __repr__(self):
        p = self.plane
        a = p.ref_surface[0]
        b = p.ref_surface[1]
        return (
            f"ReflectiveSurface(id={p.zone_id}, "
            f"R={self.R:.3g}, "
            f"dimensions=({a}=({p.x1},{p.x2}), {b}=({p.y1},{p.y2})), "
            f"height={p.height}, "
            f"grid={p.num_x}x{p.num_y})"
        )

    def to_dict(self):
        return {"R": self.R, "plane": self.plane.to_dict()}

    @classmethod
    def from_dict(cls, data):
        plane_data = data.get("plane", {})
        plane = CalcPlane.from_dict(plane_data)
        R = data.get("R", 0.0)
        return cls(R=R, plane=plane)

    @property
    def calc_state(self):
        """check if the surface needs to be recalculated"""
        return self.plane.calc_state

    @property
    def update_state(self):
        """check if surface needs updating"""
        return self.plane.update_state + (self.plane.values.sum(),)

    def set_reflectance(self, R: float):
        if not (0 <= R <= 1):
            raise ValueError("R must be in [0, 1]")
        self.R = float(R)

    def set_spacing(self, x_spacing=None, y_spacing=None):
        self.plane.set_spacing(x_spacing=x_spacing, y_spacing=y_spacing)

    def set_num_points(self, num_x=None, num_y=None):
        self.plane.set_num_points(num_x=num_x, num_y=num_y)

    def calculate_incidence(self, lamps, ref_manager=None, hard=False):
        """calculate incoming radiation onto all surfaces"""
        return self.plane.calculate_values(
            lamps=lamps, ref_manager=ref_manager, hard=hard
        )

    def calculate_reflectance(self, zone, hard=False):
        """
        calculate the reflective contribution of this reflective surface
        to a provided calculation zone

        Arguments:
            zone: a calculation zone onto which reflectance is calculated
            lamp: optional. if provided, and incidence not yet calculated, uses this
            lamp to calculate incidence. mostly this is just for
        """
        # first make sure incident irradiance is calculated
        if self.plane.values is None:
            raise ValueError("Incidence must be calculated before reflectance")

        if self.zone_dict.get(zone.zone_id) is None:
            self.zone_dict[zone.zone_id] = SurfaceZoneCache()

        cache = self.zone_dict[zone.zone_id]

        RECALCULATE = cache.needs_recalc(zone.calc_state, self.calc_state) or hard
        UPDATE = cache.needs_update(zone.update_state, self.update_state) or RECALCULATE

        if RECALCULATE:
            form_factors, theta_zone = self.calculate_coordinates(zone)
        else:
            form_factors = cache.form_factors
            theta_zone = cache.theta_zone
        if UPDATE:
            I_r = self.plane.values[:, :, np.newaxis, np.newaxis, np.newaxis]
            element_size = self.plane.x_spacing * self.plane.y_spacing

            values = (I_r * element_size * form_factors).astype("float32")

            values = self.apply_filters(values, theta_zone, zone)

            # Sum over all self.plane points to get total values at each volume point
            values = np.sum(values, axis=(0, 1))  # Collapse the dimensions
            values = values.reshape(*zone.num_points)
        else:
            values = cache.values

        # update the state
        self.zone_dict[zone.zone_id] = SurfaceZoneCache(
            zone_calc_state=zone.calc_state,
            zone_update_state=zone.update_state,
            surface_calc_state=self.calc_state,
            surface_update_state=self.update_state,
            form_factors=form_factors,
            theta_zone=theta_zone,
            values=values,
        )

        return (values * self.R).astype("float32")

    def apply_filters(self, values, theta_zone, zone):
        """apply field-of-view based calculations"""

        # clean nans

        if (~np.isfinite(values)).any():
            values = np.ma.masked_invalid(values)

        if zone.calctype == "Plane":

            # apply normals
            if zone.direction != 0:
                values[theta_zone > np.pi / 2] = 0

            # apply vertical field of view
            values[theta_zone < (np.pi / 2 - np.radians(zone.fov_vert / 2))] = 0
            values[theta_zone > (np.pi / 2 + np.radians(zone.fov_vert / 2))] = 0

            if zone.vert:
                values *= np.sin(theta_zone)
            if zone.horiz:
                values *= abs(np.cos(theta_zone))

        return values

    def calculate_coordinates(self, zone):
        """
        return the angles and distances between the points of the reflective
        surface and the calculation zone

        this is the expensive step!
        """
        surface_points = self.plane.coords.reshape(*self.plane.num_points, 3)
        zone_points = zone.coords.reshape(*zone.num_points, 3)

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
        if zone.calctype == "Plane":
            rel_zone = differences @ zone.basis
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

    # check that the keys match
    if reflectances.keys() != keys:
        raise KeyError(f"Reflectance keys must be {keys}, not {reflectances.keys()}")
    if x_spacings.keys() != keys:
        raise KeyError(f"x_spacing keys must be {keys}, not {x_spacings.keys()}")
    if y_spacings.keys() != keys:
        raise KeyError(f"y_spacing keys must be {keys}, not {y_spacings.keys()}")
    if num_x.keys() != keys:
        raise KeyError(f"num_x keys must be {keys}, not {num_x.keys()}")
    if num_y.keys() != keys:
        raise KeyError(f"num_y keys must be {keys}, not {num_y.keys()}")

    face_dict = dims.faces

    surfaces = {}
    for wall, reflectance in reflectances.items():
        x1, x2, y1, y2, height, rs, d = face_dict[wall]
        plane = CalcPlane(
            zone_id=wall,
            x1=x1,
            x2=x2,
            y1=y1,
            y2=y2,
            height=height,
            ref_surface=rs,
            direction=d,
            horiz=True,
            x_spacing=x_spacings[wall],
            y_spacing=y_spacings[wall],
            num_x=num_x[wall],
            num_y=num_y[wall],
        )
        surfaces[wall] = ReflectiveSurface(R=reflectance, plane=plane)
    return surfaces
