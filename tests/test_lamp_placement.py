"""Tests for lamp placement algorithms in lamp_helpers.py"""

import pytest
import numpy as np
from guv_calcs.polygon import Polygon2D
from guv_calcs.lamp_placement import (
    # Geometry utilities
    _point_to_segment_distance,
    _ray_polygon_intersection,
    _distance_to_polygon_boundary,
    # Corner placement
    get_corners,
    _rank_corners_by_visibility,
    new_lamp_position_corner,
    # Edge placement
    get_edge_centers,
    new_lamp_position_edge,
    _calculate_edge_perpendicular_aim,
    # Visibility/aiming
    farthest_visible_point,
    _calculate_cone_coverage,
    # Tilt utilities
    calculate_tilt,
    clamp_aim_to_max_tilt,
    apply_tilt,
    # Existing downlight functions
    new_lamp_position,
    new_lamp_position_polygon,
)


class TestGeometryUtilities:
    def test_point_to_segment_distance_on_segment(self):
        """Point on segment should have distance 0"""
        dist = _point_to_segment_distance(0.5, 0.0, 0.0, 0.0, 1.0, 0.0)
        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_point_to_segment_distance_perpendicular(self):
        """Point perpendicular to segment midpoint"""
        dist = _point_to_segment_distance(0.5, 1.0, 0.0, 0.0, 1.0, 0.0)
        assert dist == pytest.approx(1.0)

    def test_point_to_segment_distance_past_endpoint(self):
        """Point past segment endpoint"""
        dist = _point_to_segment_distance(2.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        assert dist == pytest.approx(1.0)

    def test_ray_polygon_intersection_simple(self):
        """Ray from center to edge"""
        rect = Polygon2D.rectangle(4.0, 4.0)
        # Ray from center pointing right
        intersection = _ray_polygon_intersection((2.0, 2.0), (1.0, 0.0), rect)
        assert intersection is not None
        assert intersection[0] == pytest.approx(4.0)
        assert intersection[1] == pytest.approx(2.0)

    def test_ray_polygon_intersection_diagonal(self):
        """Ray at 45 degrees"""
        rect = Polygon2D.rectangle(4.0, 4.0)
        intersection = _ray_polygon_intersection((2.0, 2.0), (1.0, 1.0), rect)
        assert intersection is not None
        # Should hit corner or near it
        assert intersection[0] == pytest.approx(4.0)
        assert intersection[1] == pytest.approx(4.0)

    def test_distance_to_polygon_boundary(self):
        """Point inside rectangle"""
        rect = Polygon2D.rectangle(4.0, 4.0)
        dist = _distance_to_polygon_boundary(np.array([2.0, 2.0]), rect)
        assert dist == pytest.approx(2.0)  # Equidistant from all sides


class TestCornerPlacement:
    def test_get_corners_rectangle(self):
        """Rectangle has 4 corners"""
        rect = Polygon2D.rectangle(4.0, 3.0)
        corners = get_corners(rect)
        assert len(corners) == 4
        assert (0.0, 0.0) in corners
        assert (4.0, 0.0) in corners
        assert (4.0, 3.0) in corners
        assert (0.0, 3.0) in corners

    def test_get_corners_triangle(self):
        """Triangle has 3 corners"""
        tri = Polygon2D(vertices=((0.0, 0.0), (4.0, 0.0), (2.0, 3.0)))
        corners = get_corners(tri)
        assert len(corners) == 3

    def test_get_corners_l_shape(self):
        """L-shaped polygon has 5 convex corners (excludes 1 reflex vertex)"""
        l_shape = Polygon2D(vertices=(
            (0.0, 0.0), (4.0, 0.0), (4.0, 2.0),
            (2.0, 2.0), (2.0, 4.0), (0.0, 4.0)
        ))
        corners = get_corners(l_shape)
        # L-shape has 6 vertices but 1 is reflex at (2,2) - the inner corner
        assert len(corners) == 5
        # Reflex vertex at (2,2) should be excluded
        assert (2.0, 2.0) not in corners

    def test_rank_corners_by_visibility_returns_all(self):
        """Should return ranking for all corners"""
        rect = Polygon2D.rectangle(4.0, 4.0)
        ranked = _rank_corners_by_visibility(rect)
        assert len(ranked) == 4
        assert set(ranked) == {0, 1, 2, 3}

    def test_new_lamp_position_corner_first_lamp(self):
        """First lamp goes to best visibility corner"""
        rect = Polygon2D.rectangle(4.0, 4.0)
        pos, aim = new_lamp_position_corner(1, rect)
        # Position should be near a corner
        x, y = pos
        assert (x < 0.5 or x > 3.5) or (y < 0.5 or y > 3.5)
        # Aim should be inside polygon
        assert rect.contains_point(*aim) or _is_near_boundary(aim, rect)

    def test_new_lamp_position_corner_fills_corners_first(self):
        """Subsequent lamps should fill other corners"""
        rect = Polygon2D.rectangle(4.0, 4.0)
        positions = []

        for i in range(4):
            pos, aim = new_lamp_position_corner(i + 1, rect, positions)
            positions.append(pos)

        # All 4 lamps should be at different corners (roughly)
        # Compare each pair only once
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                p1, p2 = positions[i], positions[j]
                dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
                assert dist > 1.0  # Corners are at least 3.0 apart, minus offset

    def test_new_lamp_position_corner_delegates_to_edge(self):
        """After all corners occupied, delegates to edge placement"""
        rect = Polygon2D.rectangle(4.0, 4.0)
        positions = []

        # Fill all 4 corners
        for i in range(4):
            pos, aim = new_lamp_position_corner(i + 1, rect, positions)
            positions.append(pos)

        # 5th lamp should be on edge (around midpoint)
        pos5, aim5 = new_lamp_position_corner(5, rect, positions)
        x, y = pos5
        # Should be near an edge center, not a corner
        # Edge centers for a 4x4 rect are at (2, 0), (4, 2), (2, 4), (0, 2)
        # With offset, position should be slightly inside
        is_near_bottom_edge = (1.5 < x < 2.5) and (y < 0.5)
        is_near_top_edge = (1.5 < x < 2.5) and (y > 3.5)
        is_near_left_edge = (x < 0.5) and (1.5 < y < 2.5)
        is_near_right_edge = (x > 3.5) and (1.5 < y < 2.5)
        assert is_near_bottom_edge or is_near_top_edge or is_near_left_edge or is_near_right_edge


class TestEdgePlacement:
    def test_get_edge_centers_rectangle(self):
        """Rectangle has 4 edge centers"""
        rect = Polygon2D.rectangle(4.0, 4.0)
        centers = get_edge_centers(rect)
        assert len(centers) == 4
        # Each center should have (x, y, edge_index)
        for x, y, idx in centers:
            assert 0 <= idx < 4

    def test_get_edge_centers_positions(self):
        """Edge centers are at midpoints"""
        rect = Polygon2D.rectangle(4.0, 4.0)
        centers = get_edge_centers(rect)
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        # Should include midpoints
        assert any(abs(x - 2.0) < 0.01 for x in xs)  # Top or bottom edge
        assert any(abs(y - 2.0) < 0.01 for y in ys)  # Left or right edge

    def test_new_lamp_position_edge_first_lamp(self):
        """First lamp goes to edge center farthest from corners"""
        rect = Polygon2D.rectangle(4.0, 4.0)
        pos, aim = new_lamp_position_edge(1, rect)
        x, y = pos
        # Should be near an edge midpoint
        near_x_mid = 1.8 < x < 2.2
        near_y_mid = 1.8 < y < 2.2
        # Either x or y should be near midpoint
        assert near_x_mid or near_y_mid

    def test_new_lamp_position_edge_perpendicular_aim(self):
        """Edge lamps aim perpendicular to wall"""
        rect = Polygon2D.rectangle(4.0, 4.0)
        pos, aim = new_lamp_position_edge(1, rect)
        x, y = pos

        # If near bottom edge (y near 0), aim should be in y direction
        if y < 0.5:
            assert aim[1] > y  # Aiming up
        # If near top edge (y near 4), aim should be in -y direction
        elif y > 3.5:
            assert aim[1] < y  # Aiming down
        # If near left edge (x near 0), aim should be in x direction
        elif x < 0.5:
            assert aim[0] > x  # Aiming right
        # If near right edge (x near 4), aim should be in -x direction
        elif x > 3.5:
            assert aim[0] < x  # Aiming left

    def test_calculate_edge_perpendicular_aim(self):
        """Perpendicular aim hits opposite wall"""
        rect = Polygon2D.rectangle(4.0, 4.0)
        # Lamp at bottom edge center
        lamp_pos = (2.0, 0.1)
        aim = _calculate_edge_perpendicular_aim(lamp_pos, 0, rect)
        # Should aim toward top edge
        assert aim[1] > 2.0
        # x should stay roughly the same (perpendicular)
        assert abs(aim[0] - 2.0) < 0.5

    def test_edge_placement_prioritizes_long_sightlines(self):
        """Edge mode places lamps on edges with longest sightlines first"""
        # 6x4 room: short edges (left/right) have 6m sightlines, long edges have 4m
        rect = Polygon2D.rectangle(6.0, 4.0)
        positions = []

        # First two lamps should go on short edges (left/right) with 6m sightlines
        pos1, _ = new_lamp_position_edge(1, rect, positions)
        positions.append(pos1)
        pos2, _ = new_lamp_position_edge(2, rect, positions)
        positions.append(pos2)

        # Both should be on left or right edge (x near 0 or 6)
        for pos in [pos1, pos2]:
            on_short_edge = (pos[0] < 0.5 or pos[0] > 5.5)
            assert on_short_edge, f"Expected lamp on short edge, got pos={pos}"


class TestVisibilityAndAiming:
    def test_farthest_visible_point_rectangle(self):
        """Farthest visible from corner is opposite corner"""
        rect = Polygon2D.rectangle(4.0, 4.0)
        # From near origin
        farthest = farthest_visible_point((0.1, 0.1), rect)
        # Should be toward opposite corner
        assert farthest[0] > 2.0
        assert farthest[1] > 2.0

    def test_farthest_visible_point_l_shape(self):
        """L-shape has limited visibility from inside corner"""
        l_shape = Polygon2D(vertices=(
            (0.0, 0.0), (4.0, 0.0), (4.0, 2.0),
            (2.0, 2.0), (2.0, 4.0), (0.0, 4.0)
        ))
        # From near the inner corner, can't see far
        farthest = farthest_visible_point((1.9, 2.1), l_shape)
        # Point should be inside polygon or on boundary
        assert l_shape.contains_point(*farthest) or _is_near_boundary(farthest, l_shape)

    def test_calculate_cone_coverage_unobstructed(self):
        """Full coverage when aiming at open space"""
        rect = Polygon2D.rectangle(10.0, 10.0)
        coverage = _calculate_cone_coverage((5.0, 0.5), (5.0, 5.0), rect, 15.0)
        assert coverage > 0.8  # Most of cone should be unobstructed

    def test_calculate_cone_coverage_near_wall(self):
        """Reduced coverage when aiming near wall"""
        rect = Polygon2D.rectangle(4.0, 4.0)
        # From center aiming at wall corner
        coverage = _calculate_cone_coverage((2.0, 2.0), (0.0, 0.0), rect, 30.0)
        # Some rays will hit walls before reaching corner
        assert 0.0 < coverage < 1.0


class TestTiltUtilities:
    def test_calculate_tilt_straight_down(self):
        """Zero tilt when aiming directly below"""
        tilt = calculate_tilt(3.0, (2.0, 2.0), (2.0, 2.0))
        assert tilt == pytest.approx(0.0)

    def test_calculate_tilt_45_degrees(self):
        """45 degree tilt when h_dist == lamp_z"""
        tilt = calculate_tilt(3.0, (0.0, 0.0), (3.0, 0.0))
        assert tilt == pytest.approx(45.0)

    def test_calculate_tilt_horizontal(self):
        """90 degrees when lamp_z is 0"""
        tilt = calculate_tilt(0.0, (0.0, 0.0), (1.0, 0.0))
        assert tilt == pytest.approx(90.0)

    def test_clamp_aim_to_max_tilt_no_change(self):
        """No change when within max_tilt"""
        rect = Polygon2D.rectangle(4.0, 4.0)
        lamp_pos = (2.0, 0.5, 3.0)
        aim = (2.0, 2.0)  # About 27 degree tilt
        result = clamp_aim_to_max_tilt(lamp_pos, aim, 45.0, rect)
        assert result[0] == pytest.approx(aim[0])
        assert result[1] == pytest.approx(aim[1])

    def test_clamp_aim_to_max_tilt_clamped(self):
        """Aim is adjusted when exceeding max_tilt"""
        rect = Polygon2D.rectangle(10.0, 10.0)
        lamp_pos = (5.0, 0.5, 3.0)
        aim = (5.0, 9.0)  # Large tilt
        result = clamp_aim_to_max_tilt(lamp_pos, aim, 30.0, rect)
        # New aim should be closer
        new_dist = np.hypot(result[0] - 5.0, result[1] - 0.5)
        old_dist = np.hypot(aim[0] - 5.0, aim[1] - 0.5)
        assert new_dist < old_dist

    def test_apply_tilt_forces_angle(self):
        """apply_tilt forces exact tilt angle"""
        rect = Polygon2D.rectangle(10.0, 10.0)
        lamp_pos = (5.0, 0.5, 3.0)
        aim = (5.0, 5.0)  # Some initial aim

        result = apply_tilt(lamp_pos, aim, rect, 45.0)
        actual_tilt = calculate_tilt(3.0, (5.0, 0.5), result)
        assert actual_tilt == pytest.approx(45.0, abs=1.0)

    def test_apply_tilt_maintains_direction(self):
        """apply_tilt keeps same direction"""
        rect = Polygon2D.rectangle(10.0, 10.0)
        lamp_pos = (5.0, 0.5, 3.0)
        aim = (7.0, 3.0)  # Diagonal aim

        result = apply_tilt(lamp_pos, aim, rect, 30.0)
        # Direction should be same
        orig_angle = np.arctan2(aim[1] - 0.5, aim[0] - 5.0)
        new_angle = np.arctan2(result[1] - 0.5, result[0] - 5.0)
        assert orig_angle == pytest.approx(new_angle, abs=0.1)


class TestConcavePolygons:
    def test_u_shape_corner_placement_inside(self):
        """Corner placement in U-shape keeps lamps inside polygon"""
        u_shape = Polygon2D(vertices=[
            (0, 0), (4, 0), (4, 3), (3, 3), (3, 1), (1, 1), (1, 3), (0, 3)
        ])
        corners = get_corners(u_shape)
        positions = []

        for i in range(len(corners)):
            pos, aim = new_lamp_position_corner(i + 1, u_shape, positions)
            positions.append(pos)
            # All positions must be inside the polygon
            assert u_shape.contains_point(*pos), f"Lamp {i+1} at {pos} is outside polygon"

    def test_u_shape_excludes_reflex_vertices(self):
        """U-shape should exclude reflex vertices from corners"""
        u_shape = Polygon2D(vertices=[
            (0, 0), (4, 0), (4, 3), (3, 3), (3, 1), (1, 1), (1, 3), (0, 3)
        ])
        corners = get_corners(u_shape)
        # U-shape has 8 vertices but 2 are reflex (at bottom of interior)
        assert len(corners) == 6


class TestDownlightPlacement:
    def test_new_lamp_position_first_in_center(self):
        """First lamp should be in center"""
        x, y = new_lamp_position(1, 4.0, 4.0)
        assert x == pytest.approx(2.0, abs=0.1)
        assert y == pytest.approx(2.0, abs=0.1)

    def test_new_lamp_position_polygon_first_in_center(self):
        """First lamp in polygon should be near centroid"""
        rect = Polygon2D.rectangle(4.0, 4.0)
        x, y = new_lamp_position_polygon(1, rect)
        cx, cy = rect.centroid
        assert x == pytest.approx(cx, abs=0.5)
        assert y == pytest.approx(cy, abs=0.5)

    def test_new_lamp_position_polygon_l_shape(self):
        """L-shape placement puts lamps in both sections"""
        l_shape = Polygon2D(vertices=(
            (0.0, 0.0), (4.0, 0.0), (4.0, 2.0),
            (2.0, 2.0), (2.0, 4.0), (0.0, 4.0)
        ))
        positions = [new_lamp_position_polygon(i + 1, l_shape) for i in range(3)]
        # Should cover different areas
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        # Should have variety in positions
        assert max(xs) - min(xs) > 1.0 or max(ys) - min(ys) > 1.0


class TestIntegration:
    def test_corner_then_edge_coverage(self):
        """Corner + edge modes together cover room well"""
        rect = Polygon2D.rectangle(6.0, 6.0)

        # Place 4 corner lamps
        corner_positions = []
        for i in range(4):
            pos, _ = new_lamp_position_corner(i + 1, rect, corner_positions)
            corner_positions.append(pos)

        # Place 4 edge lamps
        edge_positions = []
        for i in range(4):
            pos, _ = new_lamp_position_edge(i + 1, rect, edge_positions)
            edge_positions.append(pos)

        # All positions should be inside or near boundary
        for pos in corner_positions + edge_positions:
            assert rect.contains_point(*pos) or _is_near_boundary(pos, rect, tolerance=0.2)

    def test_mixed_room_shapes(self):
        """Algorithms work on different polygon shapes"""
        shapes = [
            Polygon2D.rectangle(4.0, 4.0),  # Rectangle
            Polygon2D(vertices=((0.0, 0.0), (4.0, 0.0), (2.0, 3.0))),  # Triangle
            Polygon2D(vertices=(  # L-shape
                (0.0, 0.0), (4.0, 0.0), (4.0, 2.0),
                (2.0, 2.0), (2.0, 4.0), (0.0, 4.0)
            )),
            Polygon2D(vertices=(  # Pentagon
                (2.0, 0.0), (4.0, 1.5), (3.2, 4.0), (0.8, 4.0), (0.0, 1.5)
            )),
        ]

        for shape in shapes:
            # Should not raise
            pos, aim = new_lamp_position_corner(1, shape)
            assert shape.contains_point(*pos) or _is_near_boundary(pos, shape)

            pos, aim = new_lamp_position_edge(1, shape)
            assert shape.contains_point(*pos) or _is_near_boundary(pos, shape)


def _is_near_boundary(point, polygon, tolerance=0.1):
    """Check if point is near polygon boundary."""
    from guv_calcs.lamp_placement import _distance_to_polygon_boundary
    dist = _distance_to_polygon_boundary(np.array(point), polygon)
    return dist < tolerance


class TestLampPlacerAPI:
    """Tests for the new LampPlacer factory methods and place_lamp functionality."""

    def test_for_room_with_xy(self):
        """for_room creates placer from x/y dimensions."""
        from guv_calcs.lamp_placement import LampPlacer
        placer = LampPlacer.for_room(x=4, y=3, z=2.5)
        assert placer.polygon is not None
        assert placer.z == 2.5
        assert placer.polygon.x == 4
        assert placer.polygon.y == 3

    def test_for_room_with_polygon(self):
        """for_room creates placer from polygon."""
        from guv_calcs.lamp_placement import LampPlacer
        poly = Polygon2D.rectangle(5, 4)
        placer = LampPlacer.for_room(polygon=poly, z=3.0)
        assert placer.polygon is poly
        assert placer.z == 3.0

    def test_for_room_requires_dimensions(self):
        """for_room raises error without x/y or polygon."""
        from guv_calcs.lamp_placement import LampPlacer
        with pytest.raises(ValueError, match="Must provide either"):
            LampPlacer.for_room(z=3.0)

    def test_for_dims_rectangular(self):
        """for_dims works with rectangular RoomDimensions."""
        from guv_calcs.lamp_placement import LampPlacer
        from guv_calcs.room_dims import RoomDimensions
        dims = RoomDimensions(x=6, y=4, z=2.7)
        placer = LampPlacer.for_dims(dims)
        assert placer.polygon.x == 6
        assert placer.polygon.y == 4
        assert placer.z == 2.7

    def test_for_dims_polygon(self):
        """for_dims works with PolygonRoomDimensions."""
        from guv_calcs.lamp_placement import LampPlacer
        from guv_calcs.room_dims import PolygonRoomDimensions
        poly = Polygon2D.rectangle(5, 5)
        dims = PolygonRoomDimensions(polygon=poly, z=3.0)
        placer = LampPlacer.for_dims(dims)
        assert placer.polygon is poly
        assert placer.z == 3.0

    def test_for_dims_with_existing(self):
        """for_dims tracks existing positions."""
        from guv_calcs.lamp_placement import LampPlacer
        from guv_calcs.room_dims import RoomDimensions
        dims = RoomDimensions(x=6, y=4, z=2.7)
        existing = [(1.0, 1.0), (5.0, 3.0)]
        placer = LampPlacer.for_dims(dims, existing=existing)
        assert len(placer._existing) == 2

    def test_place_returns_result(self):
        """place returns PlacementResult with position and aim."""
        from guv_calcs.lamp_placement import LampPlacer
        placer = LampPlacer.for_room(x=4, y=4, z=3)
        result = placer.place("corner", lamp_idx=1)
        assert hasattr(result, "position")
        assert hasattr(result, "aim")
        assert len(result.position) == 2
        assert len(result.aim) == 2

    def test_place_different_modes(self):
        """place works with all modes."""
        from guv_calcs.lamp_placement import LampPlacer
        placer = LampPlacer.for_room(x=4, y=4, z=3)
        for mode in ["downlight", "corner", "edge", "horizontal"]:
            result = placer.place(mode, lamp_idx=1)
            assert result.position is not None
            assert result.aim is not None
