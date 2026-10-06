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
    """Face of the housing the beam leaves through (housing frame)."""

    BOTTOM = "bottom"  # -z
    TOP = "top"        # +z
    FRONT = "front"    # +x
    BACK = "back"      # -x
    LEFT = "left"      # +y
    RIGHT = "right"    # -y

    @classmethod
    def _default(cls):
        return cls.BOTTOM

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
    def rotation(self):
        """rotation taking this face's outward normal onto the aim direction (-z)"""
        return _FACE_ROTATIONS[self]


_FACE_NORMALS = {
    HousingFace.BOTTOM: (0, 0, -1), HousingFace.TOP: (0, 0, 1),
    HousingFace.FRONT: (1, 0, 0), HousingFace.BACK: (-1, 0, 0),
    HousingFace.LEFT: (0, 1, 0), HousingFace.RIGHT: (0, -1, 0),
}
_FACE_ROTATIONS = {
    HousingFace.BOTTOM: np.eye(3), HousingFace.TOP: _clean(_rot_x(180)),
    HousingFace.FRONT: _clean(_rot_y(90)), HousingFace.BACK: _clean(_rot_y(-90)),
    HousingFace.LEFT: _clean(_rot_x(-90)), HousingFace.RIGHT: _clean(_rot_x(90)),
}
_ROLLS = (0, 90, 180, 270)


@dataclass(frozen=True, slots=True, repr=False, init=False)
class Fixture:
    """
    Physical housing of a luminaire.

    Lamp position = photometric center = emitter. The housing is a box in its
    own frame (x = length, y = width, z = height); `beam_face` and `roll` say
    how that frame sits relative to the beam, and `emitter_offset` is the
    emitter's position in housing axes measured from the center of the beam
    face. Defaults reproduce the classic layout (box behind the emitting face).
    `photometric_depth` (legacy) sets the offset along the face's inward normal.
    """

    housing_width: float
    housing_length: float
    housing_height: float
    beam_face: HousingFace
    roll: int
    emitter_offset: tuple
    shape: FixtureShape

    def __init__(self, housing_width=0.0, housing_length=0.0, housing_height=0.0,
                 beam_face=None, roll=0, emitter_offset=None,
                 shape=FixtureShape.RECTANGULAR, photometric_depth=None):
        face = HousingFace.from_any(beam_face)
        roll = int(round(float(roll or 0))) % 360
        if roll not in _ROLLS:
            raise ValueError(f"roll must be one of {_ROLLS}, got {roll}")
        if emitter_offset is None:
            offset = -face.normal * float(photometric_depth or 0.0)
        else:
            offset = np.asarray(emitter_offset, dtype=float).ravel()
            if offset.shape != (3,):
                raise ValueError(f"emitter_offset needs 3 components, got {emitter_offset!r}")
        set_ = object.__setattr__
        set_(self, "housing_width", float(housing_width or 0.0))
        set_(self, "housing_length", float(housing_length or 0.0))
        set_(self, "housing_height", float(housing_height or 0.0))
        set_(self, "beam_face", face)
        set_(self, "roll", roll)
        set_(self, "emitter_offset", tuple(float(v) + 0.0 for v in offset))
        set_(self, "shape", FixtureShape.from_any(shape))

    def __repr__(self):
        return (
            f"Fixture({self.housing_width}×{self.housing_length}×{self.housing_height}, "
            f"beam_face={self.beam_face.value!r}, roll={self.roll}, "
            f"emitter_offset={self.emitter_offset}, shape={self.shape.value!r})"
        )

    @property
    def housing_matrix(self):
        """rotation taking housing-frame vectors into the aim frame"""
        return _clean(_rot_z(self.roll) @ self.beam_face.rotation)

    @property
    def photometric_depth(self):
        """offset along the beam face's inward normal (negative = protruding)"""
        return float(-self.beam_face.normal @ np.array(self.emitter_offset)) + 0.0

    def with_photometric_depth(self, depth):
        """copy with only the inward-normal offset component replaced"""
        n = self.beam_face.normal
        off = np.array(self.emitter_offset)
        off = off - n * (off @ n) - n * float(depth)
        return replace(self, emitter_offset=tuple(off))

    def emitter_position(self):
        """emitter position in the housing frame, relative to the box center"""
        half = np.array([self.housing_length, self.housing_width, self.housing_height]) / 2
        return self.beam_face.normal * half + np.array(self.emitter_offset)

    def aim_bounds(self, surface_height=0.0):
        """(lo, hi) of the housing box in the aim frame, emitter at the origin"""
        half = np.array([self.housing_length, self.housing_width, self.housing_height]) / 2
        signs = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)
        corners = signs * half - self.emitter_position()
        aim = corners @ self.housing_matrix.T
        lo, hi = aim.min(axis=0), aim.max(axis=0)
        if surface_height and surface_height > 0:
            lo[2] = min(lo[2], -surface_height / 2)
            hi[2] = max(hi[2], surface_height / 2)
        return lo + 0.0, hi + 0.0

    def aim_corners(self, surface_height=0.0):
        """8 box corners in the aim frame: 0-3 at z-min, 4-7 at z-max"""
        (x0, y0, z0), (x1, y1, z1) = self.aim_bounds(surface_height)
        return np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ])

    def aim_to_housing_extents(self, length, width, height):
        """map aim-frame (length, width, height) extents onto housing (L, W, H)"""
        ext = np.abs(self.housing_matrix.T) @ np.array([length, width, height], dtype=float)
        return (float(ext[0]), float(ext[1]), float(ext[2]))

    def to_dict(self) -> dict:
        """Serialize fixture state for persistence."""
        return {
            "housing_width": self.housing_width,
            "housing_length": self.housing_length,
            "housing_height": self.housing_height,
            "beam_face": self.beam_face.value,
            "roll": self.roll,
            "emitter_offset": list(self.emitter_offset),
            "photometric_depth": self.photometric_depth,
            "shape": self.shape.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Fixture":
        """Deserialize fixture; legacy dicts carry only photometric_depth."""
        keys = ("housing_width", "housing_length", "housing_height", "beam_face",
                "roll", "emitter_offset", "shape", "photometric_depth")
        kwargs = {k: data[k] for k in keys if data.get(k) is not None}
        if "emitter_offset" in kwargs:
            kwargs.pop("photometric_depth", None)
        return cls(**kwargs)

    @property
    def has_dimensions(self) -> bool:
        """True if fixture has non-zero housing dimensions."""
        return self.housing_width > 0 or self.housing_length > 0 or self.housing_height > 0
