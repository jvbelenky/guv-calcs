"""Container for lamp spatial/geometric properties."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from .fixture import Fixture

if TYPE_CHECKING:
    from .lamp_orientation import LampOrientation
    from .lamp_surface import LampSurface


class LampGeometry:
    """
    Container for all spatial/geometric lamp properties.

    Owns the pose (orientation), surface (luminous area), and fixture (housing).
    This eliminates pose duplication between Lamp and LampSurface by providing
    a single source of truth for orientation.
    """

    def __init__(
        self,
        pose: "LampOrientation",
        surface: "LampSurface",
        fixture: Fixture,
    ):
        self._pose = pose
        self._surface = surface
        self._fixture = fixture
        # Set back-reference so surface can access pose
        self._surface.set_geometry(self)

    # --- Properties ---

    @property
    def pose(self) -> "LampOrientation":
        """Current lamp orientation/position."""
        return self._pose

    @property
    def surface(self) -> "LampSurface":
        """Luminous surface definition."""
        return self._surface

    @property
    def fixture(self) -> Fixture:
        """Physical fixture housing."""
        return self._fixture

    # --- Position/orientation (delegates to pose, updates surface) ---

    def move(self, x=None, y=None, z=None) -> "LampGeometry":
        """Move lamp to new position, maintaining aim direction."""
        self._pose = self._pose.move(x=x, y=y, z=z)
        self._surface._invalidate_grid()
        return self

    def rotate(self, angle) -> "LampGeometry":
        """Rotate lamp around its axis."""
        self._pose = self._pose.rotate(angle)
        self._surface._invalidate_grid()
        return self

    def aim(self, x=None, y=None, z=None) -> "LampGeometry":
        """Aim lamp at a point in cartesian space."""
        self._pose = self._pose.aim(x=x, y=y, z=z)
        self._surface._invalidate_grid()
        return self

    def recalculate_aim_point(self, heading=None, bank=None, dimensions=None, distance=None):
        """Recalculate aim point from heading/bank angles."""
        self._pose = self._pose.recalculate_aim_point(
            heading=heading, bank=bank, dimensions=dimensions, distance=distance
        )
        self._surface._invalidate_grid()
        return self

    # --- Computed positions ---

    @property
    def surface_position(self) -> np.ndarray:
        """Position of luminous surface center (same as lamp position)."""
        return self._pose.position

    def get_bounding_box_corners(self) -> np.ndarray:
        """
        Calculate 8 corners of fixture housing in world coordinates.

        The bounding box is computed from:
        - Surface center (= lamp position, = origin in local frame)
        - housing_width/length for XY extent (symmetric around surface)
        - housing_height extending "behind" surface (in -Z direction, away from aim)
        - LampSurface.height (luminous z-extent) extends both directions from surface

        Returns:
            (8, 3) array of corner coordinates
        """
        hw = self._fixture.housing_width / 2
        hl = self._fixture.housing_length / 2
        hh = self._fixture.housing_height

        # Surface z-extent (for 3D luminous openings like cylinders)
        surface_z = self._surface.height / 2

        # Local coordinate system: +Z points OPPOSITE to aim direction
        # (away from where light goes, i.e., behind the lamp)
        # Local bounding box:
        # - XY: symmetric around origin (surface center)
        # - Z: from -surface_z (front, toward aim) to +housing_height (behind, away from aim)
        z_min = -surface_z
        z_max = hh

        local_corners = np.array([
            [-hw, -hl, z_min],
            [hw, -hl, z_min],
            [hw, hl, z_min],
            [-hw, hl, z_min],
            [-hw, -hl, z_max],
            [hw, -hl, z_max],
            [hw, hl, z_max],
            [-hw, hl, z_max],
        ])

        # Transform to world coordinates
        world_corners = self._pose.transform_to_world(local_corners)
        return world_corners.T  # Return as (8, 3)

    # --- Serialization ---

    def to_dict(self) -> dict:
        """Serialize geometry state."""
        return {
            "pose": {
                "x": float(self._pose.x),
                "y": float(self._pose.y),
                "z": float(self._pose.z),
                "angle": float(self._pose.angle),
                "aimx": float(self._pose.aimx),
                "aimy": float(self._pose.aimy),
                "aimz": float(self._pose.aimz),
            },
            "surface": self._surface.to_dict(),
            "fixture": self._fixture.to_dict(),
        }
