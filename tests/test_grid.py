"""Tests for SurfaceGrid and VolumeGrid."""

import pytest
import numpy as np
from guv_calcs.geometry import SurfaceGrid, VolumeGrid, GridPoint, Polygon2D


class TestSurfaceGridCreation:
    """Tests for SurfaceGrid construction paths."""

    def test_from_legacy_xy(self):
        g = SurfaceGrid.from_legacy(mins=(0, 0), maxs=(6, 4), height=1.0)
        assert g.height == 1.0
        assert g.ref_surface == "xy"
        assert g.direction == 1
        assert g.is_rectangular

    def test_from_legacy_xz(self):
        g = SurfaceGrid.from_legacy(mins=(0, 0), maxs=(6, 2.7),
                                     height=0.0, ref_surface="xz", direction=1)
        assert g.ref_surface == "xz"
        assert g.is_rectangular

    def test_from_legacy_yz(self):
        g = SurfaceGrid.from_legacy(mins=(0, 0), maxs=(4, 2.7),
                                     height=6.0, ref_surface="yz", direction=-1)
        assert g.ref_surface == "yz"
        assert g.direction == -1

    def test_from_legacy_direction_minus1(self):
        g = SurfaceGrid.from_legacy(mins=(0, 0), maxs=(6, 4), height=2.7, direction=-1)
        assert g.direction == -1
        assert np.isclose(g.height, 2.7)

    def test_from_wall(self):
        g = SurfaceGrid.from_wall(p1=(0, 0), p2=(6, 0), z_height=2.7)
        assert g.is_rectangular
        assert g.coords.shape[1] == 3
        # All z values should be between 0 and 2.7
        assert g.coords[:, 2].min() >= 0
        assert g.coords[:, 2].max() <= 2.7

    def test_from_wall_diagonal(self):
        g = SurfaceGrid.from_wall(p1=(0, 0), p2=(3, 4), z_height=2.7)
        assert g.is_rectangular
        edge_len = np.sqrt(3**2 + 4**2)
        assert np.isclose(g._spans[0], edge_len)

    def test_from_points(self):
        g = SurfaceGrid.from_points(
            p0=(0, 0, 0), pU=(6, 0, 0), pV=(0, 4, 0)
        )
        assert g.is_rectangular
        assert g.coords.shape[1] == 3

    def test_from_polygon(self):
        poly = Polygon2D(vertices=[(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)])
        g = SurfaceGrid.from_polygon(polygon=poly, height=1.0)
        assert not g.is_rectangular
        # All points should be inside the polygon
        for pt in g.coords:
            assert poly.contains_point(pt[0], pt[1]) or poly.point_on_boundary(pt[0], pt[1])

    def test_from_polygon_direction_minus1(self):
        poly = Polygon2D.rectangle(6, 4)
        g = SurfaceGrid.from_polygon(polygon=poly, height=2.7, direction=-1)
        assert g.direction == -1
        assert np.isclose(g.height, 2.7)


class TestSurfaceGridProperties:
    """Tests for SurfaceGrid properties."""

    def test_rectangular_num_points_is_tuple_2(self):
        g = SurfaceGrid.from_legacy(mins=(0, 0), maxs=(6, 4), height=0)
        assert len(g.num_points) == 2

    def test_polygon_num_points_is_tuple_1(self):
        poly = Polygon2D(vertices=[(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)])
        g = SurfaceGrid.from_polygon(polygon=poly, height=0)
        assert len(g.num_points) == 1

    def test_spacing(self):
        g = SurfaceGrid.from_legacy(mins=(0, 0), maxs=(6, 4), height=0,
                                     spacing_init=(0.5, 0.5))
        assert g.x_spacing == 0.5
        assert g.y_spacing == 0.5

    def test_mins_maxs_rectangular(self):
        g = SurfaceGrid.from_legacy(mins=(1, 2), maxs=(5, 7), height=0)
        assert g.x1 == 1
        assert g.x2 == 5
        assert g.y1 == 2
        assert g.y2 == 7

    def test_calc_state_changes_with_height(self):
        g1 = SurfaceGrid.from_legacy(mins=(0, 0), maxs=(6, 4), height=0)
        g2 = SurfaceGrid.from_legacy(mins=(0, 0), maxs=(6, 4), height=1)
        assert g1.calc_state != g2.calc_state


class TestSurfaceGridMutation:
    """Tests for SurfaceGrid update methods."""

    def test_update_legacy_height(self):
        g = SurfaceGrid.from_legacy(mins=(0, 0), maxs=(6, 4), height=0)
        g2 = g.update_legacy(height=2.0)
        assert g2.height == 2.0

    def test_update_legacy_direction(self):
        g = SurfaceGrid.from_legacy(mins=(0, 0), maxs=(6, 4), height=0, direction=1)
        g2 = g.update_legacy(direction=-1)
        assert g2.direction == -1

    def test_update_dimensions_rectangular(self):
        g = SurfaceGrid.from_legacy(mins=(0, 0), maxs=(6, 4), height=0)
        g2 = g.update_dimensions(mins=(1, 1), maxs=(5, 3))
        assert g2.x1 == 1
        assert g2.x2 == 5

    def test_update_dimensions_polygon(self):
        poly = Polygon2D(vertices=[(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)])
        g = SurfaceGrid.from_polygon(polygon=poly, height=0)
        g2 = g.update_dimensions(height=2.0)
        assert g2.height == 2.0


class TestSurfaceGridSerialization:
    """Tests for SurfaceGrid serialization."""

    def test_round_trip_rectangular(self):
        g = SurfaceGrid.from_legacy(mins=(0, 0), maxs=(6, 4), height=1.5,
                                     spacing_init=(0.5, 0.5))
        d = g.to_dict()
        g2 = SurfaceGrid.from_dict(d)
        assert g == g2

    def test_round_trip_polygon(self):
        poly = Polygon2D(vertices=[(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)])
        g = SurfaceGrid.from_polygon(polygon=poly, height=1.0,
                                      spacing_init=(0.5, 0.5))
        d = g.to_dict()
        g2 = SurfaceGrid.from_dict(d)
        assert np.allclose(g.coords, g2.coords)

    def test_legacy_planegrid_dict(self):
        """Old PlaneGrid dict format should deserialize correctly."""
        old_dict = {
            "origin": (0, 0, 1.5),
            "spans": (6.0, 4.0),
            "spacing_init": (0.5, 0.5),
            "u_vec": (1, 0, 0),
            "v_vec": (0, 1, 0),
        }
        g = SurfaceGrid.from_dict(old_dict)
        assert g.is_rectangular
        assert np.isclose(g.height, 1.5)

class TestSurfaceGridUnitConversion:
    """Tests for SurfaceGrid unit conversion."""

    def test_convert_units(self):
        g = SurfaceGrid.from_legacy(mins=(0, 0), maxs=(6, 4), height=1.0,
                                     spacing_init=(0.5, 0.5))
        g2 = g._convert_units("meters", "feet")
        factor = 1.0 / 0.3048
        assert np.isclose(g2.height, 1.0 * factor, rtol=1e-4)
        assert np.isclose(g2.x_spacing, 0.5 * factor, rtol=1e-4)


class TestVolumeGridCreation:
    """Tests for VolumeGrid construction paths."""

    def test_from_legacy(self):
        g = VolumeGrid.from_legacy(mins=(0, 0, 0), maxs=(6, 4, 2.7))
        assert g.is_rectangular
        assert g.x1 == 0
        assert g.x2 == 6
        assert g.y1 == 0
        assert g.y2 == 4
        assert np.isclose(g.z1, 0)
        assert np.isclose(g.z2, 2.7)

    def test_from_polygon(self):
        poly = Polygon2D(vertices=[(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)])
        g = VolumeGrid.from_polygon(polygon=poly, z_height=2.7)
        assert not g.is_rectangular
        assert np.isclose(g.depth, 2.7)

    def test_from_polygon_rectangular(self):
        poly = Polygon2D.rectangle(6, 4)
        g = VolumeGrid.from_polygon(polygon=poly, z_height=2.7)
        # Rectangle polygon should still be detected as rectangular
        assert g.is_rectangular


class TestVolumeGridProperties:
    """Tests for VolumeGrid properties."""

    def test_rectangular_num_points_is_tuple_3(self):
        g = VolumeGrid.from_legacy(mins=(0, 0, 0), maxs=(6, 4, 2.7))
        assert len(g.num_points) == 3

    def test_polygon_num_points_is_tuple_1(self):
        poly = Polygon2D(vertices=[(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)])
        g = VolumeGrid.from_polygon(polygon=poly, z_height=2.7)
        assert len(g.num_points) == 1

    def test_coords_shape_rectangular(self):
        g = VolumeGrid.from_legacy(mins=(0, 0, 0), maxs=(6, 4, 2.7),
                                    spacing_init=(1, 1, 1))
        assert g.coords.shape[1] == 3
        assert g.coords.shape[0] == g.num_x * g.num_y * g.num_z

    def test_coords_shape_polygon(self):
        poly = Polygon2D(vertices=[(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)])
        g = VolumeGrid.from_polygon(polygon=poly, z_height=2.7,
                                     spacing_init=(1, 1, 1))
        assert g.coords.shape[1] == 3
        assert g.num_points == (len(g.coords),)

    def test_z_spacing(self):
        g = VolumeGrid.from_legacy(mins=(0, 0, 0), maxs=(6, 4, 2.7),
                                    spacing_init=(0.5, 0.5, 0.5))
        assert g.z_spacing == 0.5

    def test_polygon_values_to_full_grid(self):
        poly = Polygon2D(vertices=[(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)])
        g = VolumeGrid.from_polygon(polygon=poly, z_height=2.7,
                                     spacing_init=(1, 1, 1))
        values = np.ones(g.num_points[0])
        full = g.values_to_full_grid(values)
        assert len(full) == len(g.coords_full)
        assert np.isfinite(full[g._mask_full]).all()


class TestVolumeGridSerialization:
    """Tests for VolumeGrid serialization."""

    def test_round_trip_rectangular(self):
        g = VolumeGrid.from_legacy(mins=(0, 0, 0), maxs=(6, 4, 2.7),
                                    spacing_init=(0.5, 0.5, 0.5))
        d = g.to_dict()
        g2 = VolumeGrid.from_dict(d)
        assert g == g2

    def test_round_trip_polygon(self):
        poly = Polygon2D(vertices=[(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)])
        g = VolumeGrid.from_polygon(polygon=poly, z_height=2.7,
                                     spacing_init=(0.5, 0.5, 0.5))
        d = g.to_dict()
        g2 = VolumeGrid.from_dict(d)
        assert np.allclose(g.coords, g2.coords)

    def test_legacy_volgrid_dict(self):
        """Old VolGrid dict format should deserialize correctly."""
        old_dict = {
            "origin": (0, 0, 0),
            "spans": (6.0, 4.0, 2.7),
            "spacing_init": (0.5, 0.5, 0.5),
        }
        g = VolumeGrid.from_dict(old_dict)
        assert g.is_rectangular
        assert np.isclose(g.z2, 2.7)

class TestVolumeGridUnitConversion:
    """Tests for VolumeGrid unit conversion."""

    def test_convert_units(self):
        g = VolumeGrid.from_legacy(mins=(0, 0, 0), maxs=(6, 4, 2.7),
                                    spacing_init=(0.5, 0.5, 0.5))
        g2 = g._convert_units("meters", "feet")
        factor = 1.0 / 0.3048
        assert np.isclose(g2.z2, 2.7 * factor, rtol=1e-4)
        assert np.isclose(g2.z_spacing, 0.5 * factor, rtol=1e-4)


class TestRectangularFastPath:
    """Test that rectangular grids produce same results whether detected as rectangular or not."""

    def test_surface_rect_vs_polygon(self):
        """A rectangular polygon surface should give same coords as from_legacy."""
        g_legacy = SurfaceGrid.from_legacy(
            mins=(0, 0), maxs=(6, 4), height=1.0,
            spacing_init=(1.0, 1.0)
        )
        poly = Polygon2D.rectangle(6, 4)
        g_poly = SurfaceGrid.from_polygon(
            polygon=poly, height=1.0,
            spacing_init=(1.0, 1.0)
        )
        # Both should produce matching coordinate sets
        assert np.allclose(sorted(g_legacy.coords.tolist()),
                          sorted(g_poly.coords.tolist()))

    def test_volume_rect_vs_polygon(self):
        """A rectangular polygon volume should give same coords as from_legacy."""
        g_legacy = VolumeGrid.from_legacy(
            mins=(0, 0, 0), maxs=(6, 4, 2.7),
            spacing_init=(1.0, 1.0, 1.0)
        )
        poly = Polygon2D.rectangle(6, 4)
        g_poly = VolumeGrid.from_polygon(
            polygon=poly, z_height=2.7,
            spacing_init=(1.0, 1.0, 1.0)
        )
        assert np.allclose(sorted(g_legacy.coords.tolist()),
                          sorted(g_poly.coords.tolist()))


class TestTotalPoints:
    """Tests for the total_points property."""

    def test_surface_rectangular(self):
        g = SurfaceGrid.from_legacy(mins=(0, 0), maxs=(6, 4), height=0,
                                     spacing_init=(1.0, 1.0))
        assert g.total_points == g.num_x * g.num_y

    def test_surface_polygon(self):
        poly = Polygon2D(vertices=[(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)])
        g = SurfaceGrid.from_polygon(polygon=poly, height=0, spacing_init=(1.0, 1.0))
        assert g.total_points == g.num_points[0]

    def test_volume_rectangular(self):
        g = VolumeGrid.from_legacy(mins=(0, 0, 0), maxs=(6, 4, 2.7),
                                    spacing_init=(1.0, 1.0, 1.0))
        assert g.total_points == g.num_x * g.num_y * g.num_z

    def test_volume_polygon(self):
        poly = Polygon2D(vertices=[(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)])
        g = VolumeGrid.from_polygon(polygon=poly, z_height=2.7,
                                     spacing_init=(1.0, 1.0, 1.0))
        assert g.total_points == g.num_points[0]


class TestGridPoint:
    """Tests for GridPoint."""

    def test_single_point_at_position(self):
        g = GridPoint(position=(3.0, 2.0, 1.5))
        assert g.coords.shape == (1, 3)
        assert np.allclose(g.coords[0], [3.0, 2.0, 1.5], atol=1e-6)

    def test_default_normal_is_z_up(self):
        g = GridPoint(position=(0, 0, 0))
        assert np.allclose(g.normal, [0, 0, 1], atol=1e-6)

    def test_custom_normal(self):
        g = GridPoint(position=(0, 0, 0), aim_point=(0, 1, 0))
        assert np.allclose(g.normal, [0, 1, 0], atol=1e-6)

    def test_normal_z_down(self):
        g = GridPoint(position=(0, 0, 0), aim_point=(0, 0, -1))
        assert np.allclose(g.normal, [0, 0, -1], atol=1e-6)

    def test_num_points(self):
        g = GridPoint(position=(1, 2, 3))
        assert g.num_points == (1,)

    def test_serialization_round_trip(self):
        g = GridPoint(position=(3.0, 2.0, 1.5), aim_point=(3.0, 3.5, 1.5))
        d = g.to_dict()
        g2 = GridPoint.from_dict(d)
        assert np.allclose(g.coords, g2.coords, atol=1e-6)
        assert np.allclose(g.normal, g2.normal, atol=1e-6)
        assert g.aim_point == g2.aim_point

    def test_arbitrary_normal(self):
        # aim_point (1,1,1) from origin → normal = (1,1,1)/sqrt(3)
        n = np.array([1.0, 1.0, 1.0])
        n = n / np.linalg.norm(n)
        g = GridPoint(position=(0, 0, 0), aim_point=(1, 1, 1))
        assert np.allclose(g.normal, n, atol=1e-6)

    def test_aim_determines_direction(self):
        # aim_point far along Z produces same normal as close along Z
        g = GridPoint(position=(0, 0, 0), aim_point=(0, 0, 5))
        assert np.allclose(g.normal, [0, 0, 1], atol=1e-6)

    def test_basis_orthogonal(self):
        g = GridPoint(position=(0, 0, 0), aim_point=(1, 1, 0))
        basis = g.basis
        assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-10)



class TestSerializationMigration:
    """Tests for old spacing_init key migration."""

    def test_old_spacing_init_key_surface(self):
        """Old dicts with 'spacing_init' key should still load."""
        old_dict = {
            "polygon": {"vertices": [(0, 0), (6, 0), (6, 4), (0, 4)]},
            "origin": (0, 0, 0),
            "u_vec": (1, 0, 0),
            "v_vec": (0, 1, 0),
            "spacing_init": (0.5, 0.5),
            "offset": True,
        }
        g = SurfaceGrid.from_dict(old_dict)
        assert np.isclose(g.x_spacing, 0.5)

    def test_spacing_init_round_trip_surface(self):
        """spacing_init survives to_dict/from_dict round-trip."""
        g = SurfaceGrid.from_legacy(mins=(0, 0), maxs=(6, 4), height=0,
                                     spacing_init=(0.5, 0.5))
        d = g.to_dict()
        assert "spacing_init" in d
        g2 = SurfaceGrid.from_dict(d)
        assert g == g2
        assert g2.spacing_init == (0.5, 0.5)

    def test_num_points_init_round_trip_surface(self):
        """num_points_init survives to_dict/from_dict round-trip."""
        g = SurfaceGrid.from_legacy(mins=(0, 0), maxs=(6, 4), height=0,
                                     num_points_init=(10, 10))
        d = g.to_dict()
        assert "num_points_init" in d
        assert "spacing_init" not in d
        g2 = SurfaceGrid.from_dict(d)
        assert g == g2
        assert g2.num_points_init == (10, 10)

    def test_old_spacing_init_key_volume(self):
        """Old dicts with 'spacing_init' key should still load."""
        old_dict = {
            "polygon": {"vertices": [(0, 0), (6, 0), (6, 4), (0, 4)]},
            "origin": (0, 0, 0),
            "u_vec": (1, 0, 0),
            "v_vec": (0, 1, 0),
            "spacing_init": (0.5, 0.5, 0.5),
            "depth": 2.7,
            "offset": True,
        }
        g = VolumeGrid.from_dict(old_dict)
        assert np.isclose(g.z_spacing, 0.5)

    def test_spacing_init_round_trip_volume(self):
        """spacing_init survives to_dict/from_dict round-trip."""
        g = VolumeGrid.from_legacy(mins=(0, 0, 0), maxs=(6, 4, 2.7),
                                    spacing_init=(0.5, 0.5, 0.5))
        d = g.to_dict()
        assert "spacing_init" in d
        g2 = VolumeGrid.from_dict(d)
        assert g == g2
        assert g2.spacing_init == (0.5, 0.5, 0.5)

    def test_num_points_init_round_trip_volume(self):
        """num_points_init survives to_dict/from_dict round-trip."""
        g = VolumeGrid.from_legacy(mins=(0, 0, 0), maxs=(6, 4, 2.7),
                                    num_points_init=(10, 10, 10))
        d = g.to_dict()
        assert "num_points_init" in d
        assert "spacing_init" not in d
        g2 = VolumeGrid.from_dict(d)
        assert g == g2
        assert g2.num_points_init == (10, 10, 10)
