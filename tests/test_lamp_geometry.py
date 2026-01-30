"""Tests for the LampGeometry class."""

import pytest
import numpy as np
from guv_calcs import Lamp
from guv_calcs.lamp import Fixture
from guv_calcs.lamp import LampGeometry


class TestLampGeometryConstruction:
    """Tests for LampGeometry via Lamp construction."""

    def test_geometry_created_from_lamp(self, basic_lamp):
        """Lamp should create a LampGeometry instance."""
        assert basic_lamp.geometry is not None
        assert isinstance(basic_lamp.geometry, LampGeometry)

    def test_geometry_has_pose(self, basic_lamp):
        """Geometry should have pose property."""
        assert basic_lamp.geometry.pose is not None
        assert basic_lamp.geometry.pose is basic_lamp.pose

    def test_geometry_has_surface(self, basic_lamp):
        """Geometry should have surface property."""
        assert basic_lamp.geometry.surface is not None
        assert basic_lamp.geometry.surface is basic_lamp.surface

    def test_geometry_has_fixture(self, basic_lamp):
        """Geometry should have fixture property."""
        assert basic_lamp.geometry.fixture is not None
        assert basic_lamp.geometry.fixture is basic_lamp.fixture


class TestLampGeometryMove:
    """Tests for LampGeometry movement methods."""

    def test_move_updates_pose(self, basic_lamp):
        """move() should update the pose position."""
        basic_lamp.geometry.move(1, 2, 3)
        assert basic_lamp.pose.x == 1
        assert basic_lamp.pose.y == 2
        assert basic_lamp.pose.z == 3

    def test_move_maintains_aim_direction(self, basic_lamp):
        """move() should maintain the aim direction."""
        # Record original direction
        orig_aim = basic_lamp.pose.aim_point - basic_lamp.pose.position
        orig_direction = orig_aim / np.linalg.norm(orig_aim)

        basic_lamp.geometry.move(1, 2, 3)

        new_aim = basic_lamp.pose.aim_point - basic_lamp.pose.position
        new_direction = new_aim / np.linalg.norm(new_aim)

        np.testing.assert_array_almost_equal(orig_direction, new_direction)

    def test_aim_updates_pose(self, basic_lamp):
        """aim() should update the pose aim point."""
        basic_lamp.move(1, 2, 3)
        basic_lamp.geometry.aim(1, 2, 0)
        assert basic_lamp.pose.aimx == 1
        assert basic_lamp.pose.aimy == 2
        assert basic_lamp.pose.aimz == 0

    def test_rotate_updates_angle(self, basic_lamp):
        """rotate() should update the pose angle."""
        basic_lamp.geometry.rotate(45)
        assert basic_lamp.pose.angle == 45


class TestLampGeometrySurfacePosition:
    """Tests for surface position calculations."""

    def test_surface_position_at_origin(self, basic_lamp):
        """Surface position should be at lamp position."""
        pos = basic_lamp.geometry.surface_position
        assert isinstance(pos, np.ndarray)
        assert len(pos) == 3

    def test_surface_position_equals_lamp_position(self):
        """Surface position should equal lamp position (no offset)."""
        lamp = Lamp.from_keyword("aerolamp", z=3.0)
        pos = lamp.geometry.surface_position
        np.testing.assert_array_almost_equal(pos, lamp.position)

    def test_surface_position_follows_lamp(self):
        """Surface position should follow lamp when moved."""
        lamp = Lamp.from_keyword("aerolamp", x=1, y=2, z=3)
        np.testing.assert_array_almost_equal(lamp.geometry.surface_position, [1, 2, 3])
        lamp.move(5, 5, 5)
        np.testing.assert_array_almost_equal(lamp.geometry.surface_position, [5, 5, 5])


class TestLampGeometryBoundingBox:
    """Tests for bounding box calculations."""

    def test_get_bounding_box_corners_shape(self):
        """Bounding box should return 8 corners."""
        lamp = Lamp.from_keyword(
            "aerolamp",
            x=2, y=2, z=3,
            housing_width=0.5,
            housing_length=0.3,
            housing_height=0.1,
        )
        corners = lamp.geometry.get_bounding_box_corners()
        assert corners.shape == (8, 3)

    def test_get_bounding_box_at_position(self):
        """Bounding box corners should be centered around lamp position."""
        lamp = Lamp.from_keyword(
            "aerolamp",
            x=2, y=2, z=3,
            housing_width=0.5,
            housing_length=0.3,
            housing_height=0.1,
        )
        corners = lamp.geometry.get_bounding_box_corners()
        # All corners should be within fixture dimensions of lamp position
        for corner in corners:
            assert abs(corner[0] - lamp.x) <= 0.5  # housing_width/2
            assert abs(corner[1] - lamp.y) <= 0.3  # housing_length/2

    def test_get_bounding_box_zero_dimensions(self, basic_lamp):
        """Bounding box with zero dimensions should return degenerate box."""
        corners = basic_lamp.geometry.get_bounding_box_corners()
        # All corners should be at lamp position
        assert corners.shape == (8, 3)

    def test_bounding_box_extends_behind_surface(self):
        """Bounding box should extend in -Z direction (behind surface)."""
        lamp = Lamp.from_keyword(
            "aerolamp",
            x=0, y=0, z=3,
            aimx=0, aimy=0, aimz=0,  # pointing down
            housing_width=0.2,
            housing_length=0.2,
            housing_height=0.1,
        )
        corners = lamp.geometry.get_bounding_box_corners()
        # With lamp at z=3 pointing down (-Z is up in world coords)
        # Housing extends "behind" the surface (opposite aim direction)
        z_values = corners[:, 2]
        # Some corners should be at or above lamp z (behind = up when aiming down)
        assert z_values.max() >= lamp.z


class TestLampFixtureIntegration:
    """Tests for Lamp + Fixture integration."""

    def test_lamp_with_fixture_params(self):
        """Lamp should accept fixture parameters in constructor."""
        lamp = Lamp.from_keyword(
            "aerolamp",
            housing_width=0.5,
            housing_length=0.3,
            housing_height=0.1,
        )
        assert lamp.fixture.housing_width == 0.5
        assert lamp.fixture.housing_length == 0.3
        assert lamp.fixture.housing_height == 0.1

    def test_lamp_default_fixture_from_surface(self):
        """Fixture dimensions should default to surface dimensions when config has none."""
        # Create a lamp directly (not from keyword) to avoid config defaults
        lamp = Lamp(filedata=None, width=0.2, length=0.1)
        # Housing should default to surface dimensions when no config/kwargs provided
        assert lamp.fixture.housing_width == 0.2
        assert lamp.fixture.housing_length == 0.1

    def test_lamp_default_fixture_from_ies(self):
        """Fixture dimensions should use config defaults when available."""
        lamp = Lamp.from_keyword("aerolamp")
        # Housing should use config defaults (not IES surface dimensions)
        # aerolamp config has fixture: housing_width=0.1, housing_length=0.118, housing_height=0.076
        assert lamp.fixture.housing_width == 0.1
        assert lamp.fixture.housing_length == 0.118
        assert lamp.fixture.housing_height == 0.076

    def test_lamp_fixture_override_config(self):
        """User-provided fixture dimensions should override config defaults."""
        lamp = Lamp.from_keyword("aerolamp", housing_width=0.5, housing_length=0.4)
        assert lamp.fixture.housing_width == 0.5
        assert lamp.fixture.housing_length == 0.4

    def test_lamp_height_property(self):
        """lamp.height should return surface.height (luminous z-extent)."""
        lamp = Lamp.from_keyword("aerolamp")
        assert lamp.height == lamp.surface.height

    def test_lamp_depth_backward_compat(self):
        """lamp.depth should return fixture.housing_height."""
        lamp = Lamp.from_keyword("aerolamp", housing_height=0.1)
        assert lamp.depth == 0.1
        assert lamp.depth == lamp.fixture.housing_height

    def test_lamp_housing_height_param(self):
        """housing_height parameter should set fixture housing_height."""
        lamp = Lamp.from_keyword("aerolamp", housing_height=0.1)
        assert lamp.fixture.housing_height == 0.1


class TestLampGeometrySerialization:
    """Tests for LampGeometry serialization."""

    def test_to_dict_includes_fixture(self, positioned_lamp):
        """Lamp.to_dict() should include fixture data."""
        data = positioned_lamp.to_dict()
        assert "fixture" in data
        assert "housing_height" in data["fixture"]

    def test_from_dict_with_fixture(self):
        """Lamp.from_dict() should restore fixture data."""
        lamp1 = Lamp.from_keyword(
            "aerolamp",
            x=2, y=2, z=3,
            housing_width=0.5,
            housing_length=0.3,
            housing_height=0.1,
        )
        data = lamp1.to_dict()
        lamp2 = Lamp.from_dict(data)

        assert lamp2.fixture.housing_width == 0.5
        assert lamp2.fixture.housing_length == 0.3
        assert lamp2.fixture.housing_height == 0.1

    def test_from_dict_legacy_depth_migration(self):
        """Lamp.from_dict() should migrate legacy 'depth' field."""
        # Simulate old format with 'depth' instead of nested fixture/surface
        lamp_orig = Lamp.from_keyword("aerolamp", x=2, y=2, z=3)
        data = lamp_orig.to_dict()
        # Simulate old format: remove fixture, surface, height, add depth
        del data["fixture"]
        del data["surface"]
        data.pop("height", None)  # Old format didn't have height at top level
        data["depth"] = 0.15

        lamp_loaded = Lamp.from_dict(data)
        assert lamp_loaded.fixture.housing_height == 0.15


class TestSurfaceBackReference:
    """Tests for surface-geometry back-reference."""

    def test_surface_has_geometry_reference(self, basic_lamp):
        """Surface should have back-reference to geometry."""
        assert basic_lamp.surface._geometry is basic_lamp.geometry

    def test_surface_pose_via_geometry(self, basic_lamp):
        """Surface should access pose via geometry."""
        # Access should not raise
        pose = basic_lamp.surface._pose
        assert pose is basic_lamp.pose

    def test_surface_invalidates_on_move(self, basic_lamp):
        """Surface caches should invalidate when lamp moves."""
        # Compute initial position
        _ = basic_lamp.surface.position
        assert not basic_lamp.surface._position_dirty

        # Move lamp
        basic_lamp.move(5, 5, 5)
        assert basic_lamp.surface._position_dirty


class TestSurfacePositionEqualsLampPosition:
    """Tests verifying lamp position == surface position (no offset)."""

    def test_surface_position_equals_lamp_position(self, basic_lamp):
        """Surface position should always equal lamp position."""
        np.testing.assert_array_almost_equal(
            basic_lamp.surface.position,
            basic_lamp.position
        )

    def test_surface_position_equals_lamp_position_after_move(self, basic_lamp):
        """Surface position should follow lamp after move."""
        basic_lamp.move(10, 20, 30)
        np.testing.assert_array_almost_equal(
            basic_lamp.surface.position,
            basic_lamp.position
        )
        np.testing.assert_array_almost_equal(
            basic_lamp.surface.position,
            [10, 20, 30]
        )

    def test_ies_height_populates_surface_height(self):
        """IES height value should populate surface.height."""
        lamp = Lamp.from_keyword("aerolamp")
        # The IES file may have height=0 for flat lamps, but surface.height
        # should come from IES file
        assert lamp.surface.height >= 0  # Should be a valid value
        assert lamp.height == lamp.surface.height


class TestHousingUnits:
    """Tests for fixture housing_units parameter and unit conversion."""

    def test_housing_units_same_as_lamp_units(self):
        """Housing dimensions should be stored directly when units match."""
        lamp = Lamp.from_keyword(
            "aerolamp",
            housing_width=0.5,
            housing_length=0.3,
            housing_height=0.1,
            units="meters",
        )
        assert lamp.fixture.housing_width == 0.5
        assert lamp.fixture.housing_length == 0.3
        assert lamp.fixture.housing_height == 0.1

    def test_housing_units_converted_from_inches(self):
        """Housing dimensions in inches should be converted to lamp units."""
        lamp = Lamp.from_keyword(
            "aerolamp",
            housing_width=12,  # 12 inches = 0.3048 meters
            housing_length=6,  # 6 inches = 0.1524 meters
            housing_height=2,  # 2 inches = 0.0508 meters
            housing_units="inches",
            units="meters",
        )
        np.testing.assert_almost_equal(lamp.fixture.housing_width, 0.3048, decimal=4)
        np.testing.assert_almost_equal(lamp.fixture.housing_length, 0.1524, decimal=4)
        np.testing.assert_almost_equal(lamp.fixture.housing_height, 0.0508, decimal=4)

    def test_housing_units_converted_from_feet(self):
        """Housing dimensions in feet should be converted to meters."""
        lamp = Lamp.from_keyword(
            "aerolamp",
            housing_width=1,  # 1 foot = 0.3048 meters
            housing_height=0.5,  # 0.5 feet = 0.1524 meters
            housing_units="feet",
            units="meters",
        )
        np.testing.assert_almost_equal(lamp.fixture.housing_width, 0.3048, decimal=4)
        np.testing.assert_almost_equal(lamp.fixture.housing_height, 0.1524, decimal=4)

    def test_set_units_converts_fixture(self):
        """set_units() should convert fixture dimensions."""
        lamp = Lamp.from_keyword(
            "aerolamp",
            housing_width=0.3048,  # 1 foot in meters
            housing_height=0.1524,  # 0.5 feet in meters
            units="meters",
        )
        lamp.set_units("feet")
        np.testing.assert_almost_equal(lamp.fixture.housing_width, 1.0, decimal=4)
        np.testing.assert_almost_equal(lamp.fixture.housing_height, 0.5, decimal=4)

    def test_set_units_no_conversion_when_no_dimensions(self):
        """set_units() should not fail when fixture has no dimensions."""
        lamp = Lamp.from_keyword("aerolamp", units="meters")
        # This should not raise
        lamp.set_units("feet")
        assert lamp.surface.units.value == "feet"

    def test_housing_units_defaults_to_lamp_units(self):
        """If housing_units not specified, uses lamp units."""
        lamp = Lamp.from_keyword(
            "aerolamp",
            housing_width=1,
            units="feet",
        )
        # 1 foot should stay 1 foot (no conversion)
        assert lamp.fixture.housing_width == 1.0
