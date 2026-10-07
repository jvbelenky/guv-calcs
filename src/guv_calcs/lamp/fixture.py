"""Fixture (luminaire housing) definitions for lamp geometry."""

import itertools
from dataclasses import dataclass, replace
import numpy as np
from ..units import ParseableEnum
from .photometric_axis import _rot_y, _rot_z


class FixtureShape(ParseableEnum):
    """Physical shape of fixture housing."""

    RECTANGULAR = "rectangular"  # Box shape (default, MVP)
    CYLINDRICAL = "cylindrical"  # Cylinder (future)
    SPHERICAL = "spherical"  # Sphere (future)

    @classmethod
    def _default(cls):
        return cls.RECTANGULAR

    @classmethod
    def from_token(cls, token: str) -> "FixtureShape":
        """Parse string token to FixtureShape."""
        token = str(token).strip().lower()
        aliases = {
            "rectangular": cls.RECTANGULAR,
            "rect": cls.RECTANGULAR,
            "box": cls.RECTANGULAR,
            "cylindrical": cls.CYLINDRICAL,
            "cylinder": cls.CYLINDRICAL,
            "round": cls.CYLINDRICAL,
            "spherical": cls.SPHERICAL,
            "sphere": cls.SPHERICAL,
        }
        try:
            return aliases[token]
        except KeyError:
            raise ValueError(f"Unknown FixtureShape: {token}")


def _rot_x(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _clean(m):
    return np.round(m, 12) + 0.0


class HousingFace(ParseableEnum):
    """A face of the housing box (housing frame: x = length, y = width, z = height)."""

    BOTTOM = "bottom"  # -z
    TOP = "top"        # +z
    FRONT = "front"    # +x
    BACK = "back"      # -x
    LEFT = "left"      # +y
    RIGHT = "right"    # -y

    @classmethod
    def _default(cls):
        return cls.TOP

    @classmethod
    def from_token(cls, token):
        token = str(token).strip().lower()
        aliases = {"-z": "bottom", "+z": "top", "+x": "front", "-x": "back", "+y": "left", "-y": "right"}
        try:
            return cls(aliases.get(token, token))
        except ValueError:
            raise ValueError(f"Unknown HousingFace: {token}")

    @property
    def normal(self):
        """outward unit normal in the housing frame"""
        return np.array(_FACE_NORMALS[self], dtype=float)

    @property
    def opposite(self):
        return _OPPOSITE[self]

    @property
    def rotation(self):
        """rotation taking this face's outward normal onto the aim frame's +z (behind the beam)"""
        return _MOUNT_ROTATIONS[self]


_FACE_NORMALS = {
    HousingFace.BOTTOM: (0, 0, -1), HousingFace.TOP: (0, 0, 1),
    HousingFace.FRONT: (1, 0, 0), HousingFace.BACK: (-1, 0, 0),
    HousingFace.LEFT: (0, 1, 0), HousingFace.RIGHT: (0, -1, 0),
}
_OPPOSITE = {
    HousingFace.BOTTOM: HousingFace.TOP, HousingFace.TOP: HousingFace.BOTTOM,
    HousingFace.FRONT: HousingFace.BACK, HousingFace.BACK: HousingFace.FRONT,
    HousingFace.LEFT: HousingFace.RIGHT, HousingFace.RIGHT: HousingFace.LEFT,
}
_MOUNT_ROTATIONS = {
    HousingFace.TOP: np.eye(3), HousingFace.BOTTOM: _clean(_rot_x(180)),
    HousingFace.FRONT: _clean(_rot_y(-90)), HousingFace.BACK: _clean(_rot_y(90)),
    HousingFace.LEFT: _clean(_rot_x(90)), HousingFace.RIGHT: _clean(_rot_x(-90)),
}
_ROLLS = (0, 90, 180, 270)


@dataclass(frozen=True, slots=True, repr=False, init=False)
class Fixture:
    """
    Physical housing of a luminaire.

    Lamp position = photometric center = emitter. The housing is a box in its
    own frame (x = length, y = width, z = height). `mount_face` is the face
    that sits against the wall or ceiling; the beam points directly away from
    it. `roll` turns the housing about that direction in 90 degree steps.
    `emitter_offset` is the emitter's position in housing axes measured from
    the mounting point (the center of the mount face); None puts the emitter
    so the luminous opening sits flush with the opposite (front) face (the
    classic layout); this stays correct when the housing is resized. The
    luminous opening (LM-63 width/length/height, aim frame) is not the
    housing, but the housing box always contains it. `photometric_depth` (legacy) is the
    distance from the front face back to the emitter.
    """

    housing_width: float
    housing_length: float
    housing_height: float
    mount_face: HousingFace
    roll: int
    emitter_offset: tuple | None
    shape: FixtureShape

    def __init__(self, housing_width=0.0, housing_length=0.0, housing_height=0.0,
                 mount_face=None, roll=0, emitter_offset=None,
                 shape=FixtureShape.RECTANGULAR, photometric_depth=None):
        face = HousingFace.from_any(mount_face)
        roll = int(round(float(roll or 0))) % 360
        if roll not in _ROLLS:
            raise ValueError(f"roll must be one of {_ROLLS}, got {roll}")
        set_ = object.__setattr__
        set_(self, "housing_width", float(housing_width or 0.0))
        set_(self, "housing_length", float(housing_length or 0.0))
        set_(self, "housing_height", float(housing_height or 0.0))
        set_(self, "mount_face", face)
        set_(self, "roll", roll)
        set_(self, "shape", FixtureShape.from_any(shape))
        offset = None
        if emitter_offset is not None:
            arr = np.asarray(emitter_offset, dtype=float).ravel()
            if arr.shape != (3,):
                raise ValueError(f"emitter_offset needs 3 components, got {emitter_offset!r}")
            offset = tuple(float(v) + 0.0 for v in arr)
        set_(self, "emitter_offset", offset)
        if offset is None and photometric_depth:
            set_(self, "emitter_offset", self._offset_for_depth(float(photometric_depth), np.zeros(3)))

    def __repr__(self):
        return (
            f"Fixture({self.housing_width}×{self.housing_length}×{self.housing_height}, "
            f"mount_face={self.mount_face.value!r}, roll={self.roll}, "
            f"emitter_offset={self.emitter_offset}, shape={self.shape.value!r})"
        )

    # ---- frames ----

    def _half(self):
        return np.array([self.housing_length, self.housing_width, self.housing_height]) / 2

    def _mount_center(self):
        return self.mount_face.normal * self._half()

    @property
    def housing_matrix(self):
        """rotation taking housing-frame vectors into the aim frame"""
        return _clean(_rot_z(self.roll) @ self.mount_face.rotation)

    def emitter_position(self, luminous_height=0.0):
        """emitter position in the housing frame, relative to the box center.

        By default the luminous opening (of depth luminous_height along the
        mount normal) sits flush with the front face.
        """
        if self.emitter_offset is None:
            n = self.mount_face.normal
            return -self._mount_center() + n * (abs(luminous_height) / 2)
        return self._mount_center() + np.array(self.emitter_offset)

    # ---- legacy depth ----

    @property
    def photometric_depth(self):
        """distance from the front face back to the emitter (negative = protruding)"""
        n = self.mount_face.normal
        return float((self.emitter_position() + self._mount_center()) @ n) + 0.0

    def _offset_for_depth(self, depth, in_plane):
        n = self.mount_face.normal
        emitter = -self._mount_center() + n * depth + in_plane
        return tuple(float(v) + 0.0 for v in emitter - self._mount_center())

    def with_photometric_depth(self, depth):
        """copy with only the depth along the mount normal replaced"""
        n = self.mount_face.normal
        e = self.emitter_position()
        in_plane = e - n * (e @ n)
        return replace(self, emitter_offset=self._offset_for_depth(float(depth), in_plane))

    # ---- box ----

    def aim_bounds(self, luminous=(0.0, 0.0, 0.0)):
        """(lo, hi) of the housing box in the aim frame, emitter at the origin.

        `luminous` is the luminous opening's (length, width, height) in the aim
        frame, centered on the emitter; the box is grown to contain it.
        """
        lum = np.abs(np.asarray(luminous, dtype=float)) / 2
        half = self._half()
        signs = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)
        corners = signs * half - self.emitter_position(luminous_height=2 * lum[2])
        aim = corners @ self.housing_matrix.T
        lo, hi = aim.min(axis=0), aim.max(axis=0)
        if lum.any():
            lo, hi = np.minimum(lo, -lum), np.maximum(hi, lum)
        return lo + 0.0, hi + 0.0

    def aim_corners(self, luminous=(0.0, 0.0, 0.0)):
        """8 box corners in the aim frame: 0-3 at z-min, 4-7 at z-max"""
        (x0, y0, z0), (x1, y1, z1) = self.aim_bounds(luminous)
        return np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ])

    def aim_to_housing_extents(self, length, width, height):
        """map aim-frame (length, width, height) extents onto housing (L, W, H)"""
        ext = np.abs(self.housing_matrix.T) @ np.array([length, width, height], dtype=float)
        return (float(ext[0]), float(ext[1]), float(ext[2]))

    # ---- serialization ----

    def to_dict(self) -> dict:
        """Serialize fixture state for persistence."""
        return {
            "housing_width": self.housing_width,
            "housing_length": self.housing_length,
            "housing_height": self.housing_height,
            "mount_face": self.mount_face.value,
            "roll": self.roll,
            "emitter_offset": list(self.emitter_offset) if self.emitter_offset is not None else None,
            "photometric_depth": self.photometric_depth,
            "shape": self.shape.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Fixture":
        """Deserialize fixture; 0.8.x dicts carry only photometric_depth."""
        keys = ("housing_width", "housing_length", "housing_height", "mount_face",
                "roll", "emitter_offset", "shape", "photometric_depth")
        kwargs = {k: data[k] for k in keys if data.get(k) is not None}
        if "emitter_offset" in kwargs or "mount_face" in data:
            # current format: the offset (or its absence) is authoritative
            kwargs.pop("photometric_depth", None)
        return cls(**kwargs)

    @property
    def has_dimensions(self) -> bool:
        """True if fixture has non-zero housing dimensions."""
        return self.housing_width > 0 or self.housing_length > 0 or self.housing_height > 0
