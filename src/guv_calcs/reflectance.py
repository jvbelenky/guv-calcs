import numpy as np
import copy
import hashlib
from dataclasses import dataclass
from .calc_zone import CalcPlane
from .calc_manager import apply_plane_filters
from .geometry import RoomDimensions
from ._serialization import init_from_dict


class ReflectanceManager:
    """
    Interreflection solver: computes incident irradiance on room surfaces
    and iterates surface-to-surface bounces until convergence.

    Zone-level reflected contributions are computed by LightingCalculator.
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
        """Calculate incident irradiance on all surfaces, then run interreflection."""
        if self.enabled:
            # first pass: direct lamp → surface
            for wall, surface in self.surfaces.items():
                surface.calculate_incidence(lamps, hard=hard)
            # subsequent passes: surface ↔ surface bounces
            self._interreflectance(lamps, hard=hard)

    def _interreflectance(self, lamps, hard=False):
        """Iterate surface-to-surface reflections until convergence."""
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
        """Create a per-wall reflection manager (self minus that wall)."""
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


class Surface:
    """
    A surface defined by a calculation plane and optical properties
    R (reflectance) and T (transmittance). R + T <= 1; the remainder is absorbed.

    The surface provides two roles:
    - Reflector: contributes reflected light to calc zones (via R and incidence values)
    - Occluder: blocks direct/reflected light paths (via T and boundary geometry)
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
        data = dict(data)
        data["plane"] = CalcPlane.from_dict(data.get("plane", {}))
        data.setdefault("R", 0.0)
        data.setdefault("T", 0.0)
        return init_from_dict(cls, data)

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
        surfaces = ref_manager.surfaces if ref_manager else None
        return self.plane.calculate_values(
            lamps=lamps, surfaces=surfaces, hard=hard
        )

    # -- stateless computation, called by LightingCalculator --

    def _calculate_values(self, form_factors, theta_zone, zv):

        I_r = self.plane.values[:, :, np.newaxis, np.newaxis, np.newaxis]

        values = (I_r * form_factors).astype("float32")

        # clean nans
        if (~np.isfinite(values)).any():
            values = np.ma.masked_invalid(values)

        if zv.is_plane():
            values = apply_plane_filters(values, theta_zone, zv)

        # Sum over all self.plane points to get total values at each volume point
        values = np.sum(values, axis=(0, 1))  # Collapse the dimensions
        return values.reshape(*zv.num_points)

    def _calculate_coordinates(self, zv):
        """Compute analytical patch-to-point form factors and angles.

        Uses the closed-form integral of cos(theta)/(pi*d^2) over each
        rectangular surface element, eliminating grid-resolution artifacts
        that appear when zone points are close to coarse surface elements.
        """
        surface_points = self.plane.coords.reshape(*self.plane.num_points, 3)
        zone_points = zv.coords.reshape(*zv.num_points, 3)

        differences = (
            surface_points[:, :, np.newaxis, np.newaxis, np.newaxis, :] - zone_points
        )

        # project into surface local frame: [u, v, normal]
        rel_surface = differences @ self.plane.basis
        # h = perpendicular distance from zone point to surface (positive = correct side)
        h = -rel_surface[..., 2]
        safe_h = np.where(h > 0, h, 1.0)

        # analytical form factor: integrate cos(theta)/(pi*d^2) over each element
        # each element spans [-a/2, a/2] x [-b/2, b/2] centered at the element point
        a = self.plane.x_spacing
        b = self.plane.y_spacing

        form_factors = np.zeros(h.shape, dtype="float64")
        for du, dv, sign in [
            (+a / 2, +b / 2, +1),
            (-a / 2, +b / 2, -1),
            (+a / 2, -b / 2, -1),
            (-a / 2, -b / 2, +1),
        ]:
            xi = rel_surface[..., 0] + du
            eta = rel_surface[..., 1] + dv
            r = np.sqrt(xi ** 2 + eta ** 2 + h ** 2)
            form_factors += sign * np.arctan2(xi * eta, safe_h * r)

        form_factors /= np.pi
        form_factors[h <= 0] = 0
        form_factors = form_factors.astype("float32")

        # angles relative to calculation zone (only relevant for planes)
        if zv.is_plane():
            x, y, z = differences.reshape(-1, 3).T
            distances = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            distances = distances.reshape(differences.shape[0:-1])
            zero_mask = distances == 0
            safe_distances = np.where(zero_mask, 1.0, distances)

            rel_zone = differences @ zv.basis
            cos_theta_zone = rel_zone[..., 2] / safe_distances
            cos_theta_zone[zero_mask] = 0
            theta_zone = np.arccos(np.clip(cos_theta_zone, -1, 1)).astype("float32")
        else:
            theta_zone = None

        return form_factors, theta_zone


def init_room_surfaces(
    dims: "RoomDimensions",
    reflectances: dict = None,
    transmittances: dict = None,
    x_spacings: dict = None,
    y_spacings: dict = None,
    num_x: dict = None,
    num_y: dict = None,
):
    """Initialize room surfaces for reflectance calculations."""
    keys = dims.faces.keys()
    # build defaults
    default_reflectances = {surface: 0.0 for surface in keys}
    default_transmittances = {surface: 0.0 for surface in keys}
    default_spacings = {surface: None for surface in keys}
    default_nums = {surface: 10 for surface in keys}
    # build what gets used
    reflectances = {**default_reflectances, **(reflectances or {})}
    transmittances = {**default_transmittances, **(transmittances or {})}
    x_spacings = {**default_spacings, **(x_spacings or {})}
    y_spacings = {**default_spacings, **(y_spacings or {})}
    num_x = {**default_nums, **(num_x or {})}
    num_y = {**default_nums, **(num_y or {})}

    surfaces = {}

    for face_id, face in dims.faces.items():
        geometry = face.to_grid(
            spacing_init=(x_spacings[face_id], y_spacings[face_id]),
            num_points_init=(num_x[face_id], num_y[face_id]),
        )
        plane = CalcPlane(zone_id=face_id, geometry=geometry, horiz=True)
        surfaces[face_id] = Surface(
            R=reflectances[face_id], T=transmittances[face_id], plane=plane,
        )

    return surfaces
