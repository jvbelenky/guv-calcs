from dataclasses import dataclass, replace
import numpy as np
from .trigonometry import to_polar


@dataclass(frozen=True, slots=True)
class LampOrientation:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    angle: float = 0.0
    aimx: float = 0.0
    aimy: float = 0.0
    aimz: float = -1.0

    def __post_init__(self):
        if (self.aimx, self.aimy, self.aimz) == (self.x, self.y, self.z):
            raise ValueError("Aim point cannot equal position")

    def with_(self, **changes):
        return replace(self, **changes)

    @property
    def position(self):
        return np.array([self.x, self.y, self.z])

    @property
    def aim_point(self):
        return np.array([self.aimx, self.aimy, self.aimz])

    @property
    def heading(self):
        # xr, yr = self.aimx - self.x, self.aimy - self.y
        # return np.degrees(np.arctan2(yr, xr)) % 360
        dx, dy = self.aimx - self.x, self.aimy - self.y
        if np.isclose([dx, dy], 0.0).all():
            return 0.0
        return np.degrees(np.arctan2(dy, dx)) % 360

    @property
    def bank(self):
        dx, dy, dz = self.aimx - self.x, self.aimy - self.y, self.aimz - self.z
        norm = np.linalg.norm((dx, dy, dz))
        if norm == 0:
            return 0.0
        # cos(tilt) = -dz / |v| so tilt=0 → down, 180→up
        return np.degrees(np.arccos(np.clip(-dz / norm, -1.0, 1.0)))

    @property
    def rotation_matrix(self):
        # yaw1 = –heading, pitch = bank, yaw2 = –angle
        R1 = self._R_z(-self.heading)
        R2 = self._R_y(self.bank)
        R3 = self._R_z(-self.angle)
        return R3 @ R2 @ R1

    @property
    def inverse_rotation_matrix(self):
        return self.rotation_matrix.T

    def move(self, x=None, y=None, z=None):
        """Designate lamp position in cartesian space"""
        # determine new position   selected_lamp.
        x = self.x if x is None else x
        y = self.y if y is None else y
        z = self.z if z is None else z
        position = np.array([x, y, z])
        # update aim point based on new position
        diff = position - self.position
        aimx, aimy, aimz = self.aim_point + diff

        return self.with_(x=x, y=y, z=z, aimx=aimx, aimy=aimy, aimz=aimz)

    def rotate(self, angle):
        return self.with_(angle=angle)

    def aim(self, x=None, y=None, z=None):
        """aim lamp at a point in cartesian space"""
        aimx = self.aimx if x is None else x
        aimy = self.aimy if y is None else y
        aimz = self.aimz if z is None else z
        return self.with_(aimx=aimx, aimy=aimy, aimz=aimz)

    def tilt(self, tilt, dimensions=None, distance=None):
        """set the tilt, or bank"""
        return self.recalculate_aim_point(
            bank=tilt, dimensions=dimensions, distance=distance
        )

    def orient(self, orientation, dimensions=None, distance=None):
        """set the orientation, or heading"""
        return self.recalculate_aim_point(
            heading=orientation, dimensions=dimensions, distance=distance
        )

    def recalculate_aim_point(
        self, heading=None, bank=None, dimensions=None, distance=None
    ):
        """recalculate the aim point based on the heading and bank"""
        heading = self.heading if heading is None else heading
        bank = self.bank if bank is None else bank

        heading_rad = np.radians(heading)
        # Correcting bank angle for the pi shift
        bank_rad = np.radians(bank)  # - 180)
        # bank_rad = np.radians(np.clip(bank, 0, 180))

        # Convert from spherical to Cartesian coordinates
        dx = np.sin(bank_rad) * np.cos(heading_rad)
        dy = np.sin(bank_rad) * np.sin(heading_rad)
        dz = -np.cos(bank_rad)
        if dimensions is not None:
            distances = []
            dimx, dimy, dimz = dimensions
            if dx != 0:
                distances.append((dimx - self.x) / dx if dx > 0 else self.x / -dx)
            if dy != 0:
                distances.append((dimy - self.y) / dy if dy > 0 else self.y / -dy)
            if dz != 0:
                distances.append((dimz - self.z) / dz if dz > 0 else self.z / -dz)
            distance = min([d for d in distances])
        else:
            distance = 1 if distance is None else distance

        aimx, aimy, aimz = self.position + np.array([dx, dy, dz]) * distance
        return self.with_(aimx=aimx, aimy=aimy, aimz=aimz)

    def transform_to_world(self, coords, scale=1, which="cartesian"):
        """
        transform coordinates from the lamp frame of reference to the world
        Scale parameter should generally only be used for photometric_coords
        """
        coords = self.inverse_rotation_matrix @ coords.T
        coords = (coords / scale).T + self.position
        if which == "polar":
            return to_polar(*coords.T)
        elif which == "cartesian":
            return coords.T
        raise ValueError(f"`which` must be polar or cartesian, not {which}")

    def transform_to_lamp(self, coords, which="cartesian"):
        """
        transform coordinates to align with the lamp's coordinates
        """
        # coords = coords - self.position
        coords = self.rotation_matrix @ coords.T
        if which == "polar":
            return to_polar(*coords)
        elif which == "cartesian":
            return coords
        raise ValueError(f"`which` must be polar or cartesian, not {which}")

    # --------------- Internals --------------------------

    @staticmethod
    def _R_z(angle_deg):
        a = np.radians(angle_deg)
        return np.array(
            [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
        )

    @staticmethod
    def _R_y(angle_deg):
        a = np.radians(angle_deg)
        return np.array(
            [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
        )
