import numpy as np
from dataclasses import dataclass, field
import warnings
from .units import convert_units, LengthUnits
from .geometry.occlusion import compute_transmission

np.seterr(divide="ignore", invalid="ignore")


@dataclass(frozen=True)
class LampCacheEntry:
    base_values: np.ndarray | None
    values: np.ndarray | None
    calc_state: tuple | None
    update_state: tuple | None


@dataclass(frozen=True)
class SurfaceCacheEntry:
    form_factors: np.ndarray | None
    theta_zone: np.ndarray | None
    values: np.ndarray | None
    surface_calc_state: tuple | None
    surface_update_state: tuple | None


@dataclass(frozen=True)
class ZoneCache:
    lamp_cache: dict = field(default_factory=dict)
    surface_cache: dict = field(default_factory=dict)
    calc_state: tuple | None = None
    update_state: tuple | None = None
    scene_geometry_state: tuple | None = None

    def needs_recalc(self, calc_state, lamp_id, lamp_calc_state):
        if calc_state != self.calc_state:
            return True  # zone has changed
        if self.lamp_cache.get(lamp_id) is None:
            return True  # new lamp
        if self.lamp_cache.get(lamp_id).calc_state != lamp_calc_state:
            return True  # lamp has changed
        return False

    def needs_update(self, update_state, lamp_id, lamp_update_state):
        if update_state != self.update_state:
            return True  # zone has changed
        if self.lamp_cache.get(lamp_id) is None:
            return True  # new lamp
        if self.lamp_cache.get(lamp_id).update_state != lamp_update_state:
            return True
        return False

    def needs_surface_recalc(self, surface_id, surface_calc_state, zone_calc_state,
                             scene_geometry_state):
        """Check if form factors need recomputing for a surface→zone pair."""
        if zone_calc_state != self.calc_state:
            return True
        if scene_geometry_state != self.scene_geometry_state:
            return True  # some surface moved → occlusion changed
        entry = self.surface_cache.get(surface_id)
        if entry is None:
            return True
        if entry.surface_calc_state != surface_calc_state:
            return True
        return False

    def needs_surface_update(self, surface_id, surface_update_state):
        """Check if reflected values need recomputing (surface incidence changed)."""
        entry = self.surface_cache.get(surface_id)
        if entry is None:
            return True
        if entry.surface_update_state != surface_update_state:
            return True
        return False

    def base_values(self, lamp_id):
        """convenience attribute for fetching lamp base values"""
        entry = self.lamp_cache.get(lamp_id)
        if entry is not None:
            return entry.base_values
        return None

    def values(self, lamp_id):
        """convenience attribute for fetching lamp values"""
        entry = self.lamp_cache.get(lamp_id)
        if entry is not None:
            return entry.values
        return None

    def by_lamp_type(self, lamps):
        """
        TODO: under construction! for replacing much of the stuff in disinfection_calculator
        return a dict of lamp contributions to the zone by wavelength/GUV type
        """
        # gather wavelengths
        wavelengths = []
        for key, lamp in lamps.items():
            if key in self.lamp_cache.keys():
                if lamp.wavelength is not None:
                    wavelengths.append(lamp.wavelength)
                else:
                    msg = f"{lamp.name} ({key}) has an undefined wavelength. Its fluence contribution will not be counted."
                    warnings.warn(msg, stacklevel=3)
        wavelengths = np.unique(wavelengths)


class LightingCalculator:
    """
    Performs all computations for a calculation zone: direct irradiance,
    occlusion, and reflected contributions.
    """

    def __init__(self):
        self.cache = ZoneCache()

    def compute(self, lamps, zv, surfaces=None, hard=False):
        """
        Calculate and return direct irradiance values at all coordinate points.
        """

        if len(lamps) == 0:
            self.cache = ZoneCache()
            return np.zeros(zv.num_points, dtype="float32")

        lamp_cache = {}
        for lamp_id, lamp in lamps.items():
            # potentially expensive
            base_values = self.calculate_lamp(lamp, zv, surfaces=surfaces, hard=hard)
            # always cheap
            values = self.apply_filters(lamp, base_values.copy(), zv)
            lamp_cache[lamp_id] = LampCacheEntry(
                base_values=base_values,
                values=values,
                calc_state=lamp.calc_state,
                update_state=lamp.update_state,
            )

        # build scene geometry state for occlusion cache invalidation
        scene_geo = None
        if surfaces:
            scene_geo = tuple(s.calc_state for s in surfaces.values())

        # update cache
        self.cache = ZoneCache(
            lamp_cache=lamp_cache,
            surface_cache=self.cache.surface_cache,  # preserve across compute calls
            calc_state=zv.calc_state,
            update_state=zv.update_state,
            scene_geometry_state=scene_geo,
        )

        # sum across lamp values
        return self.aggregate(lamps, zv)

    def compute_reflectance(self, surfaces, zv, hard=False):
        """Compute total reflected contribution from all surfaces to a zone."""
        if not surfaces:
            return np.zeros(zv.num_points, dtype="float32")

        scene_geo = self.cache.scene_geometry_state

        surface_cache = {}
        total = np.zeros(zv.num_points, dtype="float32")

        for surface_id, surface in surfaces.items():
            if surface.R == 0 or surface.plane.values is None:
                continue

            RECALC = self.cache.needs_surface_recalc(
                surface_id, surface.calc_state, zv.calc_state, scene_geo
            ) or hard
            UPDATE = self.cache.needs_surface_update(
                surface_id, surface.update_state
            ) or RECALC

            if RECALC:
                form_factors, theta_zone = surface._calculate_coordinates(zv)
                # apply occlusion: exclude self
                transmission = compute_transmission(
                    surface.plane.coords, zv.coords, surfaces, exclude=surface_id
                )
                form_factors = form_factors * transmission.reshape(form_factors.shape)
            else:
                entry = self.cache.surface_cache[surface_id]
                form_factors = entry.form_factors
                theta_zone = entry.theta_zone

            if UPDATE:
                values = surface._calculate_values(form_factors, theta_zone, zv)
            else:
                values = self.cache.surface_cache[surface_id].values

            surface_cache[surface_id] = SurfaceCacheEntry(
                form_factors=form_factors,
                theta_zone=theta_zone,
                values=values,
                surface_calc_state=surface.calc_state,
                surface_update_state=surface.update_state,
            )

            total += (values * surface.R).reshape(*zv.num_points).astype("float32")

        # update cache with new surface entries
        self.cache = ZoneCache(
            lamp_cache=self.cache.lamp_cache,
            surface_cache=surface_cache,
            calc_state=self.cache.calc_state,
            update_state=self.cache.update_state,
            scene_geometry_state=scene_geo,
        )

        return total.astype("float32")

    def _to_meters(self, R, lamp):
        """Convert distance array to meters for inverse-square calculation."""
        if lamp.surface.units != LengthUnits.METERS:
            return np.array(convert_units(lamp.surface.units, "meters", *R))
        return R

    def calculate_lamp(self, lamp, zv, surfaces=None, hard=False):
        """Calculate the zone values for a single lamp."""

        RECALC = self.cache.needs_recalc(
            calc_state=zv.calc_state,
            lamp_id=lamp.lamp_id,
            lamp_calc_state=lamp.calc_state,
        )
        if hard or RECALC:
            # get coords
            rel_coords = zv.coords - lamp.surface.position
            Theta, Phi, R = lamp.transform_to_lamp(rel_coords, which="polar")
            R = self._to_meters(R, lamp)
            # fetch intensity values from photometric data
            phot = lamp.ies.photometry.interpolated()
            values = phot.get_intensity(Theta, Phi) / R ** 2

            # near field only if necessary
            if lamp.surface.source_density > 0 and lamp.surface.photometric_distance:
                values = self.calculate_nearfield(lamp, R, values, zv)

            # apply occlusion from room surfaces
            if surfaces:
                transmission = compute_transmission(
                    lamp.surface.position, zv.coords, surfaces
                )
                values *= transmission.reshape(values.shape)

            if any(~np.isfinite(values)):  # mask any nans and infs near light source
                values = np.ma.masked_invalid(values)

            return values.astype("float32")
        # if no recalculation required use cached version
        return self.cache.base_values(lamp.lamp_id)

    def apply_filters(self, lamp, values, zv):
        """
        update the values of a single lamp based on the calc zone properties,
        but which don't require a full recalculation
        """

        if zv.is_plane():
            rel_coords = zv.coords - lamp.surface.position
            x, y, z = (rel_coords @ zv.basis).T
            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            theta = np.arccos(-z / r)
            values = apply_plane_filters(values, theta, zv)

        # TODO: This should maybe actually be in Lamp/Photometry
        values = values * lamp.intensity_units.factor

        return values.reshape(*zv.num_points)

    def calculate_nearfield(self, lamp, R, values, zv):
        """
        calculate the values within the photometric distance
        over a discretized source
        """
        near_idx = np.where(R < lamp.surface.photometric_distance)
        # set current values to zero
        values[near_idx] = 0
        # redo calculation in a loop
        num_points = len(lamp.surface.surface_points)
        points = lamp.surface.surface_points
        intensity_values = lamp.surface.intensity_map.reshape(-1)
        for point, val in zip(points, intensity_values):

            rel_coords = zv.coords - point
            Theta, Phi, R = lamp.transform_to_lamp(rel_coords, which="polar")
            Theta_n, Phi_n, R_n = Theta[near_idx], Phi[near_idx], R[near_idx]

            R_n = self._to_meters(R_n, lamp)

            phot = lamp.ies.photometry.interpolated()
            near_values = phot.get_intensity(Theta_n, Phi_n) / R_n ** 2
            near_values = near_values * val / num_points
            values[near_idx] += near_values

        return values

    def aggregate(self, lamps, zv):
        """sum across lamp_values"""

        lamp_values = [self.cache.values(lamp_id) for lamp_id in lamps.keys()]

        if zv.is_plane() and zv.fov_horiz < 360 and len(lamps) > 1:
            base_values = self.calculate_horizontal_fov(lamps, lamp_values, zv)
        else:
            base_values = sum(lamp_values)
        return base_values.reshape(*zv.num_points).astype("float32")

    def calculate_horizontal_fov(self, lamps, lamp_values, zv):
        """
        Vectorized function to compute the largest possible value for all lamps
        within a horizontal view field.
        """

        # Compute relative coordinates: Shape (num_points, num_lamps, 3)
        lamp_positions = np.array([lamp.surface.position for lamp in lamps.values()])
        rel_coords = zv.coords[:, None, :] - lamp_positions[None, :, :]
        rel_coords = rel_coords @ zv.basis

        # Calculate horizontal angles (in degrees)
        angles = np.degrees(np.arctan2(rel_coords[..., 1], rel_coords[..., 0]))
        angles = angles % 360  # Wrap; Shape (N, M)
        angles = angles[:, :, None]  # Expand; Shape (N, M, 1)

        # Compute pairwise angular differences for all rows
        diffs = np.abs(angles - angles.transpose(0, 2, 1))
        diffs = np.minimum(diffs, 360 - diffs)  # Wrap angular differences to [0, 180]

        # Create the adjacency mask for each pair within 180 degrees
        adjacency = diffs <= zv.fov_horiz / 2  # Shape (N, M, M)

        # current values to be transformed
        values = np.array([val.reshape(-1) for val in lamp_values]).T
        # Sum the values for all connected components (using the adjacency mask)
        value_sums = adjacency @ values[:, :, None]  # Shape (N, M, 1)
        # Remove the last singleton dimension,
        value_sums = value_sums.squeeze(-1)  # Shape (N, M)

        return np.max(value_sums, axis=1)  # Shape (N,)


# reused in reflectance.py
def apply_plane_filters(values, theta, zv):
    """apply angular view filters to a plane"""
    if zv.is_plane():
        # apply normals/directions
        if zv.use_normal:
            values[theta > np.pi / 2] = 0

        # apply vertical/horizontal irradiances
        if zv.vert:
            values *= np.sin(theta)
        if zv.horiz:
            values *= abs(np.cos(theta))

        if zv.fov_vert < 180:
            values[theta < (np.pi / 2 - np.radians(zv.fov_vert / 2))] = 0
            values[theta > (np.pi / 2 + np.radians(zv.fov_vert / 2))] = 0
        return values
    else:
        return values
