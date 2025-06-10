import numpy as np

# from .trigonometry import attitude, to_polar
from .units import convert_units


class LightingCalculator:
    """
    Performs all computations for a calculation zone
    """

    def __init__(self, zone):
        self.zone = zone

    def compute(self, lamps, hard=False):
        """
        Calculate and return irradiance values at all coordinate points within the zone.
        """

        if len(lamps) > 0:
            # this will only recalculate if the lamp has changed--unless it is a 'hard' recalculation
            self.zone.lamp_values_base = {
                lamp_id: self._calculate_lamp(lamp, hard)
                for lamp_id, lamp in lamps.items()
            }

            # this step is always cheap
            self.zone.lamp_values = {
                lamp_id: self._apply_filters(lamps[lamp_id], values.copy())
                for lamp_id, values in self.zone.lamp_values_base.items()
            }

            # sum the base lamp values
            if (
                self.zone.calctype == "Plane"
                and self.zone.fov_horiz < 360
                and len(lamps) > 1
            ):
                values = self._calculate_horizontal_fov(lamps)
            else:
                values = sum(self.zone.lamp_values.values())

            # reshape
            values = values.reshape(*self.zone.num_points).astype("float32")
        else:
            self.zone.lamp_values_base = {}
            self.zone.lamp_values = {}
            values = np.zeros(self.zone.num_points).astype("float32")

        return values

    def _calculate_lamp(self, lamp, hard=False):
        """
        Calculate the zone values for a single lamp
        """

        NEW_LAMP = self.zone.lamp_values_base.get(lamp.lamp_id) is None
        LAMP_UPDATE = lamp.calc_state != lamp.get_calc_state()
        ZONE_UPDATE = self.zone.calc_state != self.zone.get_calc_state()

        if hard or NEW_LAMP or LAMP_UPDATE or ZONE_UPDATE:
            # get coords
            rel_coords = self.zone.coords - lamp.surface.position
            Theta, Phi, R = lamp.transform_to_lamp(rel_coords, which="polar")
            if lamp.surface.units.lower() != "meters":
                R = np.array(convert_units(lamp.surface.units, "meters", *R))

            # fetch intensity values from photometric data
            phot = lamp.ies.photometry.interpolated()
            values = phot.get_intensity(Theta, Phi) / R ** 2

            # near field only if necessary
            if lamp.surface.source_density > 0 and lamp.surface.photometric_distance:
                values = self._calculate_nearfield(lamp, R, values)

            if np.isnan(values.any()):  # mask any nans near light source
                values = np.ma.masked_invalid(values)

            values = values.astype("float32")

        else:
            values = self.zone.lamp_values_base[lamp.lamp_id]

        return values

    def _apply_filters(self, lamp, values):
        """
        TODO:
        possibly I should really be saving on calculation time
        by not calculating unnecessary angles to begin with

        update the values of a single lamp based on the calc zone properties,
        but which don't require a full recalculation
        """

        if self.zone.calctype == "Plane":
            rel_coords = self.zone.coords - lamp.surface.position
            x, y, z = (rel_coords @ self.zone.basis).T
            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            theta = np.arccos(-z / r)

            # apply normals/directions
            if self.zone.direction != 0:
                values[theta > np.pi / 2] = 0

            # apply vertical/horizontal irradiances
            if self.zone.vert:
                values *= np.sin(theta)
            if self.zone.horiz:
                values *= abs(np.cos(theta))

            if self.zone.fov_vert < 180:
                values[theta < (np.pi / 2 - np.radians(self.zone.fov_vert / 2))] = 0
                values[theta > (np.pi / 2 + np.radians(self.zone.fov_vert / 2))] = 0

        # TODO: This should actually be in Lamp/Photometry
        if lamp.intensity_units.lower() == "mw/sr":
            values = values / 10  # convert from mW/Sr to uW/cm2

        # reshape
        values = values.reshape(*self.zone.num_points)

        return values

    def _calculate_nearfield(self, lamp, R, values):
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

            rel_coords = self.zone.coords - point
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

    def _calculate_horizontal_fov(self, lamps):
        """
        Vectorized function to compute the largest possible value for all lamps
        within a horizontal view field.
        """

        # Compute relative coordinates: Shape (num_points, num_lamps, 3)
        lamp_positions = np.array(
            [lamp.surface.position for lamp in lamps.values() if lamp.enabled]
        )
        rel_coords = self.zone.coords[:, None, :] - lamp_positions[None, :, :]
        rel_coords = rel_coords @ self.zone.basis

        # Calculate horizontal angles (in degrees)
        angles = np.degrees(np.arctan2(rel_coords[..., 1], rel_coords[..., 0]))
        angles = angles % 360  # Wrap; Shape (N, M)
        angles = angles[:, :, None]  # Expand; Shape (N, M, 1)

        # Compute pairwise angular differences for all rows
        diffs = np.abs(angles - angles.transpose(0, 2, 1))
        diffs = np.minimum(diffs, 360 - diffs)  # Wrap angular differences to [0, 180]

        # Create the adjacency mask for each pair within 180 degrees
        adjacency = diffs <= self.zone.fov_horiz / 2  # Shape (N, M, M)

        # current values to be transformed
        vals = self.zone.lamp_values.values()
        values = np.array([val.reshape(-1) for val in vals]).T
        # Sum the values for all connected components (using the adjacency mask)
        value_sums = adjacency @ values[:, :, None]  # Shape (N, M, 1)
        # Remove the last singleton dimension,
        value_sums = value_sums.squeeze(-1)  # Shape (N, M)

        return np.max(value_sums, axis=1)  # Shape (N,)
