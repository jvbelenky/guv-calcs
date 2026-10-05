"""Where a fixture's beam points in its IES file's own frame."""

import numpy as np
from ..units import ParseableEnum


def _rot_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


class PhotometricAxis(ParseableEnum):
    """
    Direction, in the IES frame, that the fixture's beam goes.

    The IES frame puts theta=0 on -z, theta=90/phi=0 on +x and the zenith on +z.
    `matrix` rotates IES-frame vectors into the aim frame (beam on -z); for the
    horizontal axes the IES zenith lands on local +x, which the lamp pose sends
    to world-up when the lamp is banked to 90 degrees.
    """

    DOWN = "down"
    UP = "up"
    HORIZONTAL_0 = "horizontal_0"
    HORIZONTAL_90 = "horizontal_90"
    HORIZONTAL_180 = "horizontal_180"
    HORIZONTAL_270 = "horizontal_270"

    @classmethod
    def _default(cls):
        return cls.DOWN

    @classmethod
    def from_token(cls, token):
        token = str(token).strip().lower().replace("-", "_").replace(" ", "_")
        try:
            return cls(token)
        except ValueError:
            raise ValueError(f"Unknown PhotometricAxis: {token}")

    @property
    def is_horizontal(self):
        return self.value.startswith("horizontal")

    @property
    def phi(self):
        """beam azimuth in the ies frame, horizontal axes only"""
        return float(self.value.split("_")[1]) if self.is_horizontal else None

    @property
    def direction(self):
        """unit vector of the beam in the ies frame"""
        if self is PhotometricAxis.DOWN:
            return np.array([0.0, 0.0, -1.0])
        if self is PhotometricAxis.UP:
            return np.array([0.0, 0.0, 1.0])
        p = np.radians(self.phi)
        return np.array([np.cos(p), np.sin(p), 0.0])

    @property
    def matrix(self):
        """rotation taking ies-frame vectors into the aim frame"""
        if self is PhotometricAxis.DOWN:
            return np.eye(3)
        if self is PhotometricAxis.UP:
            m = _rot_y(180.0)
        else:
            m = _rot_y(90.0) @ _rot_z(-self.phi)
        return np.round(m, 12) + 0.0  # clean -0.0 and 1e-17 noise

    def permute_extents(self, length, width, height):
        """map ies (x, y, z) extents onto aim-frame (length, width, height)"""
        ext = np.abs(self.matrix) @ np.array([length, width, height], dtype=float)
        return (float(ext[0]), float(ext[1]), float(ext[2]))
