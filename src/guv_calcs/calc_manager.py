import numpy as np
from dataclasses import dataclass, field, replace

# import matplotlib.pyplot as plt
from copy import deepcopy

# from .trigonometry import attitude, to_polar
from .units import convert_units


@dataclass(frozen=True)
class ZoneCache:
    lamp_values_base: dict = field(default_factory=dict)
    lamp_values: dict = field(default_factory=dict)
    calc_state: tuple | None = None
    update_state: tuple | None = None

    def update(self, **changes):
        return replace(self, **changes)

    @classmethod
    def clear(cls):
        return cls()

    def needs_recalc(self, calc_state):
        return calc_state != self.calc_state

    def needs_update(self, update_state):
        return update_state != self.update_state

    def new_lamp(self, lamp_id):
        return self.lamp_values_base.get(lamp_id) is None


class LightingCalculator:
    """
    Performs all computations for a calculation zone
    """

    def __init__(self):
        self.cache = ZoneCache()
        
    def compute(self, lamps, zv, hard=False):
        """
        Calculate and return irradiance values at all coordinate points within the zone.
        """

        if len(lamps) == 0:
            self.cache = self.cache.clear()
            return np.zeros(zv.num_points, dtype="float32")

        # this will only recalculate if the lamp has changed
        lamp_values_base = {}
        for lamp_id, lamp in lamps.items():
            result = self.calculate_lamp(lamp, zv, hard=hard)
            lamp_values_base[lamp_id] = result

        # this step is always cheap
        lamp_values = {}
        for lamp_id, values in lamp_values_base.items():
            result = self.apply_filters(lamps[lamp_id], values.copy(), zv)
            lamp_values[lamp_id] = result

        # sum across lamp values
        base_values = self.aggregate(lamps, lamp_values, zv)

        # update cache
        self.cache = self.cache.update(
            lamp_values_base=lamp_values_base,
            lamp_values=lamp_values,
            calc_state=zv.calc_state,
            update_state=zv.update_state,
        )

        return base_values

    def calculate_lamp(self, lamp, zv, hard: bool):
        """
        Calculate the zone values for a single lamp
        """

        NEW_LAMP = self.cache.new_lamp(lamp.lamp_id)
        LAMP_RECALC = lamp.calc_state != lamp.get_calc_state()
        ZONE_RECALC = self.cache.needs_recalc(zv.calc_state)

        if hard or NEW_LAMP or LAMP_RECALC or ZONE_RECALC:
            # get coords
            rel_coords = zv.coords - lamp.surface.position
            Theta, Phi, R = lamp.transform_to_lamp(rel_coords, which="polar")
            if lamp.surface.units.lower() != "meters":
                R = np.array(convert_units(lamp.surface.units, "meters", *R))

            # fetch intensity values from photometric data
            phot = lamp.ies.photometry.interpolated()
            values = phot.get_intensity(Theta, Phi) / R ** 2

            # near field only if necessary
            if lamp.surface.source_density > 0 and lamp.surface.photometric_distance:
                values = self.calculate_nearfield(lamp, R, values, zv)

            if any(~np.isfinite(values)):  # mask any nans and infs near light source
                values = np.ma.masked_invalid(values)

            return values.astype("float32")
        # if no recalculation required use cached version
        return self.cache.lamp_values_base[lamp.lamp_id]
        
    def apply_filters(self, lamp, values, zv):
        # apply measured correction filters
        if filters is not None:
            for filt in filters.values():
                if (
                    lamp.surface.source_density > 0
                    and lamp.surface.photometric_distance
                ):
                    new_values = np.zeros(values.shape)
                    for point in lamp.surface.surface_points:
                        tmpvals = filt.apply(deepcopy(values), point, self.zone.coords)
                        new_values += tmpvals / len(lamp.surface.surface_points)
                    values = new_values
                else:
                    values = filt.apply(values, lamp.position, self.zone.coords)

        if obstacles is not None:
            for obs in obstacles.values():
                if (
                    lamp.surface.source_density > 0
                    and lamp.surface.photometric_distance
                ):
                    new_values = np.zeros(values.shape)
                    for point in lamp.surface.surface_points:
                        tmpvals = obs.apply(deepcopy(values), point, self.zone.coords)
                        new_values += tmpvals / len(lamp.surface.surface_points)
                    values = new_values
                else:
                    values = obs.apply(values, lamp.position, self.zone.coords)

        if zv.is_plane():
            rel_coords = zv.coords - lamp.surface.position
            x, y, z = (rel_coords @ zv.basis).T
            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            theta = np.arccos(-z / r)

            # apply normals/directions
            if zv.direction != 0:
                values[theta > np.pi / 2] = 0

            # apply vertical/horizontal irradiances
            if zv.vert:
                values *= np.sin(theta)
            if zv.horiz:
                values *= abs(np.cos(theta))

            if zv.fov_vert < 180:
                values[theta < (np.pi / 2 - np.radians(zv.fov_vert / 2))] = 0
                values[theta > (np.pi / 2 + np.radians(zv.fov_vert / 2))] = 0

        # TODO: This should actually be in Lamp/Photometry
        if lamp.intensity_units.lower() == "mw/sr":
            values = values / 10  # convert from mW/Sr to uW/cm2

        return values.reshape(*zv.num_points)

    def aggregate(self, lamps, lamp_values, zv):
        """sum across lamp_values"""

        if zv.is_plane() and zv.fov_horiz < 360 and len(lamps) > 1:
            base_values = self.calculate_horizontal_fov(lamps, lamp_values, zv)
        else:
            base_values = sum(lamp_values.values())
        return base_values.reshape(*zv.num_points).astype("float32")

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

            if lamp.surface.units.lower() != "meters":
                R_n = np.array(convert_units(lamp.surface.units, "meters", *R_n))

            phot = lamp.ies.photometry.interpolated()
            near_values = phot.get_intensity(Theta_n, Phi_n) / R_n ** 2
            # interpdict = lamp.lampdict["interp_vals"]
            # near_values = get_intensity(Theta_n, Phi_n, interpdict) / R_n ** 2
            near_values = near_values * val / num_points
            values[near_idx] += near_values

        return values

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
        vals = lamp_values.values()
        values = np.array([val.reshape(-1) for val in vals]).T
        # Sum the values for all connected components (using the adjacency mask)
        value_sums = adjacency @ values[:, :, None]  # Shape (N, M, 1)
        # Remove the last singleton dimension,
        value_sums = value_sums.squeeze(-1)  # Shape (N, M)

        return np.max(value_sums, axis=1)  # Shape (N,)
