"""Fixture (luminaire housing) definitions for lamp geometry."""

from enum import StrEnum
from dataclasses import dataclass


class FixtureShape(StrEnum):
    """Physical shape of fixture housing."""

    RECTANGULAR = "rectangular"  # Box shape (default, MVP)
    CYLINDRICAL = "cylindrical"  # Cylinder (future)
    SPHERICAL = "spherical"  # Sphere (future)

    @classmethod
    def from_any(cls, arg) -> "FixtureShape":
        """Convert various inputs to FixtureShape."""
        if arg is None:
            return cls.RECTANGULAR
        if isinstance(arg, cls):
            return arg
        return cls.from_token(arg)

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


@dataclass(frozen=True, slots=True)
class Fixture:
    """
    Physical housing of a luminaire.

    The fixture contains the luminous surface and defines the physical
    boundaries that must fit within the room geometry.

    Lamp position = photometric center = luminous surface center.
    Housing extends "behind" the surface (in -Z direction, opposite to aim).

    Attributes:
        housing_width: X-axis extent of physical housing
        housing_length: Y-axis extent of physical housing
        housing_height: How far fixture extends behind the luminous surface
        shape: Physical shape of the housing
    """

    housing_width: float = 0.0
    housing_length: float = 0.0
    housing_height: float = 0.0
    shape: FixtureShape = FixtureShape.RECTANGULAR

    def to_dict(self) -> dict:
        """Serialize fixture state for persistence."""
        return {
            "housing_width": self.housing_width,
            "housing_length": self.housing_length,
            "housing_height": self.housing_height,
            "shape": self.shape.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Fixture":
        """Deserialize fixture from dict. Handles backward compatibility."""
        # Ignore legacy fields: 'height', 'mount_type'
        return cls(
            housing_width=data.get("housing_width", 0.0),
            housing_length=data.get("housing_length", 0.0),
            housing_height=data.get("housing_height", 0.0),
            shape=FixtureShape.from_any(data.get("shape")),
        )

    @property
    def has_dimensions(self) -> bool:
        """True if fixture has non-zero housing dimensions."""
        return self.housing_width > 0 or self.housing_length > 0 or self.housing_height > 0
