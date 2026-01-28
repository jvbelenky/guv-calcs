"""Tests for the Room class."""

import pytest
import numpy as np
from guv_calcs import Room, Lamp, CalcPlane, CalcVol, Polygon2D


class TestRoomInitialization:
    """Tests for Room initialization and default values."""

    def test_default_dimensions_meters(self):
        """Room should have default dimensions of 6x4x2.7 meters."""
        room = Room()
        assert room.x == 6
        assert room.y == 4
        assert room.z == 2.7
        assert room.units == "meters"

    def test_default_dimensions_feet(self):
        """Room should have default dimensions of 20x13x9 feet when units='feet'."""
        room = Room(units="feet")
        assert room.x == 20
        assert room.y == 13
        assert room.z == 9
        assert room.units == "feet"

    def test_custom_dimensions(self):
        """Room should accept custom dimensions."""
        room = Room(x=10, y=8, z=3.0, units="meters")
        assert room.x == 10
        assert room.y == 8
        assert room.z == 3.0

    def test_volume_calculation(self):
        """Room volume should be calculated correctly."""
        room = Room(x=6, y=4, z=2.7, units="meters")
        expected_volume = 6 * 4 * 2.7
        assert np.isclose(room.volume, expected_volume)

    def test_dimensions_property(self):
        """Room dimensions property should return tuple of (x, y, z)."""
        room = Room(x=6, y=4, z=2.7)
        assert room.dimensions == (6, 4, 2.7)

    def test_reflectance_enabled_by_default(self):
        """Reflectance should be enabled by default."""
        room = Room()
        assert room.ref_manager.enabled is True

    def test_reflectance_disabled(self):
        """Reflectance can be disabled at initialization."""
        room = Room(enable_reflectance=False)
        assert room.ref_manager.enabled is False


class TestRoomUnits:
    """Tests for unit handling in Room."""

    def test_set_units_changes_unit_label(self, basic_room):
        """set_units should change the units label."""
        basic_room.set_units("feet")
        assert basic_room.units == "feet"

    def test_initial_units_meters(self, basic_room):
        """Room initialized with meters should have meters as units."""
        assert basic_room.units == "meters"

    def test_initial_units_feet(self, room_feet):
        """Room initialized with feet should have feet as units."""
        assert room_feet.units == "feet"

    def test_set_dimensions(self, basic_room):
        """Room dimensions can be changed after initialization."""
        basic_room.set_dimensions(x=10, y=8, z=3.5)
        assert basic_room.x == 10
        assert basic_room.y == 8
        assert basic_room.z == 3.5


class TestRoomLampManagement:
    """Tests for lamp management in Room."""

    def test_add_lamp(self, basic_room, basic_lamp):
        """Lamp should be added to room.lamps dict."""
        basic_room.add_lamp(basic_lamp)
        assert len(basic_room.lamps) == 1
        assert basic_lamp.lamp_id in basic_room.lamps

    def test_add_multiple_lamps(self, basic_room):
        """Multiple lamps can be added to a room."""
        lamp1 = Lamp.from_keyword("aerolamp")
        lamp2 = Lamp.from_keyword("ushio_b1")
        basic_room.add_lamp(lamp1).add_lamp(lamp2)
        assert len(basic_room.lamps) == 2

    def test_place_lamp_by_keyword(self, basic_room):
        """place_lamp should work with a keyword string."""
        basic_room.place_lamp("aerolamp")
        assert len(basic_room.lamps) == 1

    def test_remove_lamp(self, room_with_lamp):
        """Lamp should be removed from room."""
        lamp_id = list(room_with_lamp.lamps.keys())[0]
        room_with_lamp.remove_lamp(lamp_id)
        assert len(room_with_lamp.lamps) == 0


class TestLampPlacementModes:
    """Tests for lamp placement modes (downlight, tilted, corner, edge)."""

    def test_place_lamp_downlight_mode(self, basic_room):
        """Downlight mode places lamp in center pointing down."""
        basic_room.place_lamp("aerolamp", mode="downlight")
        lamp = list(basic_room.lamps.values())[0]
        # Should be roughly centered
        assert 2.5 < lamp.position[0] < 3.5
        assert 1.5 < lamp.position[1] < 2.5
        # Aiming straight down (aim x,y equals position x,y)
        assert lamp.aimx == pytest.approx(lamp.position[0], abs=0.01)
        assert lamp.aimy == pytest.approx(lamp.position[1], abs=0.01)

    def test_place_lamp_tilted_mode_deprecated(self, basic_room):
        """Tilted mode is deprecated but still works."""
        with pytest.warns(DeprecationWarning, match="mode='tilted' is deprecated"):
            basic_room.place_lamp("aerolamp", mode="tilted")
        lamp = list(basic_room.lamps.values())[0]
        # Should be near edge/corner
        x, y = lamp.position[:2]
        near_edge = (x < 0.5 or x > 5.5 or y < 0.5 or y > 3.5)
        assert near_edge
        # Aiming inward (different from position)
        assert lamp.aimx != pytest.approx(lamp.position[0], abs=0.1) or \
               lamp.aimy != pytest.approx(lamp.position[1], abs=0.1)

    def test_place_lamp_corner_mode(self, basic_room):
        """Corner mode places lamp at corners."""
        basic_room.place_lamp("aerolamp", mode="corner")
        lamp = list(basic_room.lamps.values())[0]
        x, y = lamp.position[:2]
        # Should be near a corner
        near_corner = (x < 0.5 or x > 5.5) and (y < 0.5 or y > 3.5)
        assert near_corner

    def test_place_lamp_edge_mode(self, basic_room):
        """Edge mode places lamp at edge centers."""
        basic_room.place_lamp("aerolamp", mode="edge")
        lamp = list(basic_room.lamps.values())[0]
        x, y = lamp.position[:2]
        # Should be near an edge center (mid x or mid y, but on edge)
        near_x_mid = 2.5 < x < 3.5
        near_y_mid = 1.5 < y < 2.5
        on_edge = (y < 0.5 or y > 3.5) or (x < 0.5 or x > 5.5)
        assert (near_x_mid or near_y_mid) and on_edge

    def test_place_lamp_corner_mode_fills_all_corners(self, basic_room):
        """Corner mode fills all 4 corners before switching to edges."""
        for _ in range(4):
            basic_room.place_lamp("aerolamp", mode="corner")
        positions = [(l.position[0], l.position[1]) for l in basic_room.lamps.values()]
        # All 4 should be at different corners
        corners_hit = set()
        for x, y in positions:
            if x < 0.5 and y < 0.5:
                corners_hit.add("bl")
            elif x > 5.5 and y < 0.5:
                corners_hit.add("br")
            elif x < 0.5 and y > 3.5:
                corners_hit.add("tl")
            elif x > 5.5 and y > 3.5:
                corners_hit.add("tr")
        assert len(corners_hit) == 4

    def test_place_lamp_with_tilt(self, basic_room):
        """Tilt parameter forces exact tilt angle."""
        basic_room.place_lamp("aerolamp", mode="corner", tilt=30.0)
        lamp = list(basic_room.lamps.values())[0]
        # Calculate actual tilt
        from guv_calcs.lamp_helpers import calculate_tilt
        actual_tilt = calculate_tilt(
            lamp.position[2],
            (lamp.position[0], lamp.position[1]),
            (lamp.aimx, lamp.aimy)
        )
        assert actual_tilt == pytest.approx(30.0, abs=2.0)

    def test_place_lamp_with_max_tilt(self, basic_room):
        """Max_tilt parameter limits tilt angle."""
        basic_room.place_lamp("aerolamp", mode="corner", max_tilt=20.0)
        lamp = list(basic_room.lamps.values())[0]
        from guv_calcs.lamp_helpers import calculate_tilt
        actual_tilt = calculate_tilt(
            lamp.position[2],
            (lamp.position[0], lamp.position[1]),
            (lamp.aimx, lamp.aimy)
        )
        assert actual_tilt <= 20.5  # Small tolerance

    def test_place_lamp_invalid_mode(self, basic_room):
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="invalid lamp placement mode"):
            basic_room.place_lamp("aerolamp", mode="invalid")

    def test_place_lamp_horizontal_mode(self, basic_room):
        """Horizontal mode places lamp at edge aimed straight horizontally."""
        basic_room.place_lamp("aerolamp", mode="horizontal")
        lamp = list(basic_room.lamps.values())[0]
        # Lamp Z should equal aim Z (horizontal)
        assert lamp.position[2] == pytest.approx(lamp.aimz, abs=0.001)


class TestPolygonRoomPlacement:
    """Tests for lamp placement in polygon rooms."""

    def test_polygon_room_corner_mode(self):
        """Corner mode works with polygon rooms."""
        room = Room(polygon=Polygon2D.rectangle(6.0, 4.0), z=2.7)
        room.place_lamp("aerolamp", mode="corner")
        lamp = list(room.lamps.values())[0]
        x, y = lamp.position[:2]
        # Should be near a corner
        near_corner = (x < 0.5 or x > 5.5) and (y < 0.5 or y > 3.5)
        assert near_corner

    def test_polygon_room_edge_mode(self):
        """Edge mode works with polygon rooms."""
        room = Room(polygon=Polygon2D.rectangle(6.0, 4.0), z=2.7)
        room.place_lamp("aerolamp", mode="edge")
        lamp = list(room.lamps.values())[0]
        # Should be inside room
        assert room.scene.dim.polygon.contains_point(lamp.position[0], lamp.position[1])

    def test_l_shaped_room_corner_mode(self):
        """Corner mode works with L-shaped polygon rooms."""
        l_shape = Polygon2D(vertices=(
            (0.0, 0.0), (6.0, 0.0), (6.0, 2.0),
            (3.0, 2.0), (3.0, 4.0), (0.0, 4.0)
        ))
        room = Room(polygon=l_shape, z=2.7)
        # L-shape has 6 vertices but only 5 convex corners (excludes reflex vertex at (3,2))
        for _ in range(5):
            room.place_lamp("aerolamp", mode="corner")
        assert len(room.lamps) == 5
        # All lamps should be inside room
        for lamp in room.lamps.values():
            x, y = lamp.position[:2]
            inside = l_shape.contains_point(x, y)
            # Or very close to boundary (offset)
            from guv_calcs.lamp_helpers import _distance_to_polygon_boundary
            import numpy as np
            near_boundary = _distance_to_polygon_boundary(np.array([x, y]), l_shape) < 0.2
            assert inside or near_boundary


class TestRoomZoneManagement:
    """Tests for calc zone management in Room."""

    def test_add_standard_zones(self, room_with_lamp):
        """add_standard_zones should create SkinLimits, EyeLimits, WholeRoomFluence."""
        room_with_lamp.add_standard_zones()
        assert "SkinLimits" in room_with_lamp.calc_zones
        assert "EyeLimits" in room_with_lamp.calc_zones
        assert "WholeRoomFluence" in room_with_lamp.calc_zones

    def test_add_calc_zone(self, basic_room, calc_plane):
        """Custom calc zone can be added to room."""
        basic_room.add_calc_zone(calc_plane)
        assert "TestPlane" in basic_room.calc_zones

    def test_remove_calc_zone(self, room_with_zones):
        """Calc zone can be removed from room."""
        room_with_zones.remove_calc_zone("SkinLimits")
        assert "SkinLimits" not in room_with_zones.calc_zones


class TestRoomCalculation:
    """Tests for Room.calculate() functionality."""

    def test_calculate_populates_values(self, room_with_zones):
        """calculate() should populate zone values."""
        room_with_zones.calculate()
        zone = room_with_zones.calc_zones["WholeRoomFluence"]
        assert zone.values is not None
        assert zone.values.size > 0

    def test_calculate_values_positive(self, calculated_room):
        """Calculated fluence values should be positive."""
        zone = calculated_room.calc_zones["WholeRoomFluence"]
        assert np.all(zone.values >= 0)

    def test_calculate_returns_self(self, room_with_zones):
        """calculate() should return self for method chaining."""
        result = room_with_zones.calculate()
        assert result is room_with_zones


class TestRoomSerialization:
    """Tests for Room save/load functionality."""

    def test_to_dict(self, basic_room):
        """Room should serialize to dict."""
        data = basic_room.to_dict()
        assert data["x"] == 6
        assert data["y"] == 4
        assert data["z"] == 2.7
        assert data["units"] == "meters"

    def test_from_dict(self, basic_room):
        """Room should deserialize from dict."""
        data = basic_room.to_dict()
        new_room = Room.from_dict(data)
        assert new_room.x == basic_room.x
        assert new_room.y == basic_room.y
        assert new_room.z == basic_room.z

    def test_save_load_round_trip(self, room_with_zones, temp_file):
        """Room should survive save/load round trip."""
        room_with_zones.save(temp_file)
        loaded_room = Room.load(temp_file)
        assert loaded_room.x == room_with_zones.x
        assert loaded_room.y == room_with_zones.y
        assert loaded_room.z == room_with_zones.z
        assert "WholeRoomFluence" in loaded_room.calc_zones

    def test_to_dict_from_dict_basic(self, basic_room):
        """Room should survive to_dict/from_dict round trip."""
        data = basic_room.to_dict()
        loaded_room = Room.from_dict(data)
        assert loaded_room.x == basic_room.x
        assert loaded_room.y == basic_room.y
        assert loaded_room.z == basic_room.z

    def test_copy(self, basic_room):
        """Room copy should be independent of original."""
        copy = basic_room.copy()
        copy.set_dimensions(x=10)
        assert basic_room.x == 6
        assert copy.x == 10


class TestRoomReflectance:
    """Tests for reflectance settings in Room."""

    def test_enable_reflectance(self, room_no_reflectance):
        """Reflectance can be enabled after initialization."""
        room_no_reflectance.enable_reflectance(True)
        assert room_no_reflectance.ref_manager.enabled is True

    def test_set_reflectance(self, basic_room):
        """Reflectance value can be set."""
        basic_room.set_reflectance(0.5)
        # Verify all surfaces have the reflectance set
        for surface in basic_room.surfaces.values():
            assert surface.R == 0.5

    def test_set_max_num_passes(self, basic_room):
        """Max number of reflectance passes can be set."""
        basic_room.set_max_num_passes(50)
        assert basic_room.ref_manager.max_num_passes == 50


class TestPolygonRoom:
    """Tests for polygon-based room shapes."""

    def test_polygon_room_creation_from_list(self):
        """Polygon room can be created from list of vertices."""
        vertices = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
        room = Room(polygon=vertices, z=2.7)
        assert room.is_polygon is True
        assert room.polygon is not None
        assert room.polygon.n_vertices == 6

    def test_polygon_room_creation_from_polygon2d(self):
        """Polygon room can be created from Polygon2D object."""
        poly = Polygon2D(vertices=[(0, 0), (6, 0), (6, 4), (0, 4)])
        room = Room(polygon=poly, z=2.7)
        assert room.is_polygon is True
        assert room.polygon.n_vertices == 4

    def test_polygon_room_bounding_box_dimensions(self):
        """Polygon room x/y properties return bounding box dimensions."""
        vertices = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
        room = Room(polygon=vertices, z=2.7)
        assert room.x == 4.0
        assert room.y == 4.0
        assert room.z == 2.7

    def test_polygon_room_volume(self):
        """Polygon room volume is based on polygon area times height."""
        # L-shaped room with area = 4*2 + 2*2 = 12 sq units
        vertices = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
        room = Room(polygon=vertices, z=2.7)
        expected_volume = 12.0 * 2.7
        assert np.isclose(room.volume, expected_volume)

    def test_polygon_room_surfaces(self):
        """Polygon room has numbered wall surfaces."""
        vertices = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
        room = Room(polygon=vertices, z=2.7)
        assert "floor" in room.surfaces
        assert "ceiling" in room.surfaces
        assert "wall_0" in room.surfaces
        assert "wall_5" in room.surfaces
        assert len(room.surfaces) == 8  # floor + ceiling + 6 walls

    def test_polygon_room_set_wall_reflectance(self):
        """Wall reflectance can be set by wall ID for polygon rooms."""
        vertices = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
        room = Room(polygon=vertices, z=2.7)
        room.set_reflectance(0.5, wall_id="wall_0")
        assert room.surfaces["wall_0"].R == 0.5
        assert room.surfaces["wall_1"].R == 0.0  # Other walls unchanged

    def test_polygon_room_serialization(self):
        """Polygon room survives to_dict/from_dict round trip."""
        vertices = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
        room = Room(polygon=vertices, z=2.7, units="meters")
        data = room.to_dict()
        loaded_room = Room.from_dict(data)
        assert loaded_room.is_polygon is True
        assert loaded_room.polygon.n_vertices == 6
        assert loaded_room.z == 2.7
        assert np.isclose(loaded_room.volume, room.volume)

    def test_rectangular_room_is_not_polygon(self):
        """Rectangular room created with x/y/z is not polygon."""
        room = Room(x=6, y=4, z=2.7)
        assert room.is_polygon is False
        assert room.polygon is None

    def test_polygon_room_repr(self):
        """Polygon room has informative repr."""
        vertices = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
        room = Room(polygon=vertices, z=2.7)
        repr_str = repr(room)
        assert "polygon=" in repr_str
        assert "6 vertices" in repr_str
