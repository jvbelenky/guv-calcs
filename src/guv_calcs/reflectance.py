import numpy as np
import copy
from .calc_zone import CalcPlane


class ReflectanceManager:
    """
    Class for managing reflective surfaces and their interactions

    Attributes:
    x,y,z: floats
        cartesian dimensions--should be the same as the Room object
    reflectances: dict { str: float}
        dict with keys [`floor`, `ceiling`, `south`, `north`, `east`, `west`]
        and float values between 0 and 1. All values default 0.0
    x_spacings: dict { str: float}
        dict with same keys as `reflectances` and float values greater than 0.
        All values default 0.5. Determines spacing in the relative x direction
    y_spacings: dict { str: float}
        dict with same keys as `reflectances` and float values greater than 0.
        All values default 0.5. Determines spacing in the relative y direction
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
        x,
        y,
        z,
        reflectances=None,
        x_spacings=None,
        y_spacings=None,
        max_num_passes=None,
        threshold=None,
    ):

        self.x = x
        self.y = y
        self.z = z

        keys = ["floor", "ceiling", "south", "north", "east", "west"]
        default_reflectances = {surface: 0.0 for surface in keys}
        default_spacings = {surface: 0.5 for surface in keys}

        self.reflectances = {**default_reflectances, **(reflectances or {})}
        self.x_spacings = {**default_spacings, **(x_spacings or {})}
        self.y_spacings = {**default_spacings, **(y_spacings or {})}
        self.max_num_passes = 100 if max_num_passes is None else int(max_num_passes)
        self.threshold = 0.02 if threshold is None else threshold
        if not isinstance(self.threshold, float):
            raise TypeError("threshold must be a float between 0 and 1")
        if self.threshold < 0 or self.threshold > 1:
            raise ValueError("threshold must be a float between 0 and 1")

        self.zone_dict = {}  # will contain all values from all contributions
        self.surfaces = {}
        self.managers = {}
        self._initialize_surfaces()

    def set_reflectance(self, R, wall_id=None):
        """set reflectance by wall_id or, if wall_if is None, to all walls"""
        keys = self.reflectances.keys()
        if wall_id is None:
            # set this value for all walls
            for wall in keys:
                # self.reflectances[wall] = R
                self.surfaces[wall].R = R
        else:
            if wall_id not in keys:
                raise KeyError(f"wall_id must be in {keys}")
            else:
                # self.reflectances[wall_id] = R
                self.surfaces[wall_id].R = R
        # update the dict
        for key, val in self.surfaces.items():
            self.reflectances[key] = val.R

    def set_spacing(self, x_spacing=None, y_spacing=None, wall_id=None):
        """set x and y spacing by wall_id or, if wall_if is None, to all walls"""
        keys = self.x_spacings.keys()
        if wall_id is None:
            # set this value for all walls
            for wall in keys:
                self.surfaces[wall].plane.set_spacing(
                    x_spacing=x_spacing, y_spacing=y_spacing
                )
        else:
            if wall_id not in keys:
                raise KeyError(f"wall_id must be in {keys}")
            else:
                self.surfaces[wall_id].plane.set_spacing(
                    x_spacing=x_spacing, y_spacing=y_spacing
                )
        for key, val in self.surfaces.items():
            self.x_spacings[key] = val.plane.x_spacing
            self.y_spacings[key] = val.plane.y_spacing

    def _initialize_surfaces(self):

        for wall, reflectance in self.reflectances.items():
            (
                x1,
                x2,
                y1,
                y2,
                height,
                ref_surface,
                direction,
            ) = self._get_surface_dimensions(wall)
            plane = CalcPlane(
                zone_id=wall,
                x1=x1,
                x2=x2,
                y1=y1,
                y2=y2,
                height=height,
                ref_surface=ref_surface,
                direction=direction,
                horiz=True,
                x_spacing=self.x_spacings[wall],
                y_spacing=self.y_spacings[wall],
            )
            self.surfaces[wall] = ReflectiveSurface(R=reflectance, plane=plane)

        # possibly this should go here? idk
        self.managers = {}
        for wall, surface in self.surfaces.items():
            ref_manager = copy.deepcopy(self)
            del ref_manager.surfaces[wall]
            self.managers[wall] = ref_manager

    def _get_surface_dimensions(self, wall):
        """retrieve the dimensions of a particular wall based on its id"""

        if wall == "floor":
            x1, x2, y1, y2 = 0, self.x, 0, self.y
            height = 0
            ref_surface = "xy"
            direction = 1
        elif wall == "ceiling":
            x1, x2, y1, y2 = 0, self.x, 0, self.y
            height = self.z
            ref_surface = "xy"
            direction = -1
        elif wall == "south":
            x1, x2, y1, y2 = 0, self.x, 0, self.z
            height = 0
            ref_surface = "xz"
            direction = 1
        elif wall == "north":
            x1, x2, y1, y2 = 0, self.x, 0, self.z
            height = self.y
            ref_surface = "xz"
            direction = -1
        elif wall == "west":
            x1, x2, y1, y2 = 0, self.y, 0, self.z
            height = 0
            ref_surface = "yz"
            direction = 1
        elif wall == "east":
            x1, x2, y1, y2 = 0, self.y, 0, self.z
            height = self.x
            ref_surface = "yz"
            direction = -1
        return x1, x2, y1, y2, height, ref_surface, direction

    def update_dimensions(self, x=None, y=None, z=None):
        """update the wall dimensions based on changes to the Room parent class"""
        self.x = self.x if x is None else x
        self.y = self.y if y is None else y
        self.z = self.z if z is None else z
        for wall, surface in self.surfaces.items():
            x1, x2, y1, y2, height, _, _ = self._get_surface_dimensions(wall)
            surface.plane.height = height
            surface.plane.set_dimensions(x1, x2, y1, y2)

    def calculate_incidence(self, lamps, hard=False):
        """
        calculate the incident irradiances on all reflective walls
        """

        # first pass
        for wall, surface in self.surfaces.items():
            surface.calculate_incidence(lamps, hard=hard)
        # subsequent passes
        self._interreflectance(lamps, hard=hard)

    def _interreflectance(self, lamps, hard=False):
        """
        calculate additional interreflectance
        """
        # create dict of ref managers for each wall
        managers = self._create_managers()

        # for storing the progressively increasing reflectance values
        dct = {}
        for wall, surface in self.surfaces.items():
            dct[wall] = surface.plane.values

        i = 0  # increases to self.max_num_passes
        percent = 1  # falls to self.threshold
        while percent > self.threshold and i < self.max_num_passes:
            pc = []
            for wall, surface in self.surfaces.items():

                init = surface.plane.values.mean()

                surface.calculate_incidence(
                    lamps, ref_manager=managers[wall], hard=hard
                )

                dct[wall] += surface.plane.reflected_values

                final = surface.plane.values.mean()
                if final > 0:
                    pc.append((abs(final - init) / final))
                else:
                    pc.append(0)
            percent = np.mean(pc)
            managers = self._update_managers(managers)
            i = i + 1
        for wall, surface in self.surfaces.items():
            surface.plane.values = dct[wall]

    def _update_managers(self, managers: dict) -> dict:
        """Update all interreflection managers with newly calculated surface incidences"""
        for wall, manager in managers.items():
            subwalls = list(manager.surfaces.keys())  # Create a static copy of keys
            for subwall in subwalls:
                # Update the values without modifying the dictionary structure
                np.copyto(
                    manager.surfaces[subwall].plane.values,
                    self.surfaces[subwall].plane.values,
                )
                # create new plane
                new_plane = copy.deepcopy(self.surfaces[subwall].plane)
                # replace the total values with only the reflected values
                new_plane.values = new_plane.reflected_values
                # Replace the object instead of deleting in-place
                manager.surfaces[subwall] = ReflectiveSurface(
                    R=self.surfaces[subwall].R, plane=new_plane
                )

        return managers  # Updated in place

    def _create_managers(self):
        """
        create a dict of reflection managers for each wall
        """
        managers = {}
        for wall, surface in self.surfaces.items():
            ref_manager = ReflectanceManager(
                x=self.x,
                y=self.y,
                z=self.z,
                reflectances=self.reflectances,
                x_spacings=self.x_spacings,
                y_spacings=self.y_spacings,
            )
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
                values = surface.zone_dict[zone.zone_id]["values"]
                total_values[wall] = values
            else:
                total_values[wall] = np.zeros(zone.num_points)
        self.zone_dict[zone.zone_id] = total_values

        return total_values

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
        self.num_passes = 0  # init
        self.zone_dict = {}

    def calculate_incidence(self, lamps, ref_manager=None, hard=False):
        """calculate incoming radiation onto all surfaces"""
        self.plane.calculate_values(lamps=lamps, ref_manager=ref_manager, hard=hard)

    def get_calc_state(self):
        """check if the surface needs to be recalculated"""
        return self.plane.get_calc_state()

    def get_update_state(self):
        """"""
        return self.plane.get_update_state() + [self.plane.values.sum()]

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
            self.zone_dict[zone.zone_id] = {}

        calc_state = zone.get_calc_state()
        update_state = zone.get_update_state()
        surface_calc_state = self.get_calc_state()
        surface_update_state = self.get_update_state()
        NEW_ZONE = self.zone_dict[zone.zone_id].get("values") is None
        ZONE_RECALC = calc_state != self.zone_dict[zone.zone_id].get("calc_state")
        ZONE_UPDATE = update_state != self.zone_dict[zone.zone_id].get("update_state")
        SURF_RECALC = surface_calc_state != self.zone_dict[zone.zone_id].get(
            "surface_calc_state"
        )
        SURF_UPDATE = surface_update_state != self.zone_dict[zone.zone_id].get(
            "surface_update_state"
        )

        RECALCULATE = NEW_ZONE or ZONE_RECALC or SURF_RECALC or hard
        UPDATE = RECALCULATE or NEW_ZONE or ZONE_UPDATE or SURF_UPDATE or hard

        if RECALCULATE:
            form_factors, theta_zone = self._calculate_coordinates(zone)
        else:
            form_factors = self.zone_dict[zone.zone_id]["form_factors"]
            theta_zone = self.zone_dict[zone.zone_id]["theta_zone"]
        if UPDATE:
            I_r = self.plane.values[:, :, np.newaxis, np.newaxis, np.newaxis]
            element_size = self.plane.x_spacing * self.plane.y_spacing

            values = (I_r * element_size * form_factors).astype("float32")

            values = self._apply_filters(values, theta_zone, zone)

            # Sum over all self.plane points to get total values at each volume point
            values = np.sum(values, axis=(0, 1))  # Collapse the dimensions
            values = values.reshape(*zone.num_points)
        else:
            values = self.zone_dict[zone.zone_id]["values"]

        # update the state
        self.zone_dict[zone.zone_id] = {
            "form_factors": form_factors,
            "theta_zone": theta_zone,
            "values": values,
            "calc_state": calc_state,
            "update_state": update_state,
            "surface_calc_state": surface_calc_state,
            "surface_update_state": surface_update_state,
        }

        return (values * self.R).astype("float32")

    def _apply_filters(self, values, theta_zone, zone):
        """apply field-of-view based calculations"""

        # clean nans
        if np.isnan(values).any():
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

    def _calculate_coordinates(self, zone):
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
