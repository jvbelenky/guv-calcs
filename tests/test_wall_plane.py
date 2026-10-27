"""Tests for wall-region calc planes and wall_frame_of."""

import pytest
import numpy as np
from guv_calcs import CalcPlane
from guv_calcs.geometry import SurfaceGrid, Polygon2D, RoomDimensions
from guv_calcs.geometry.wall_plane import wall_frame_of, WallFrame


def _sorted_rows(a):
    a = np.round(np.asarray(a, float), 9)
    return a[np.lexsort(a.T[::-1])]


class TestFromWallRegion:

    def test_whole_wall_matches_from_wall(self):
        p1, p2, z = (1.0, 2.0), (5.0, 4.0), 3.0
        ref = SurfaceGrid.from_wall(p1, p2, z, spacing_init=(0.5, 0.5))
        g = SurfaceGrid.from_wall_region(p1, p2, z, spacing_init=(0.5, 0.5))
        assert g.num_points == ref.num_points
        assert np.allclose(g.coords, ref.coords)
        assert np.allclose(g.normal, ref.normal)

    def test_normal_points_left_of_wall_direction(self):
        g = SurfaceGrid.from_wall_region((0, 0), (4, 0), 3.0)
        assert np.allclose(g.normal, (0, 1, 0))

    def test_flip_reverses_normal_keeps_position(self):
        g = SurfaceGrid.from_wall_region((0, 0), (4, 0), 3.0)
        f = SurfaceGrid.from_wall_region((0, 0), (4, 0), 3.0, flip=True)
        assert np.allclose(f.normal, (0, -1, 0))
        assert f.coords.shape == g.coords.shape
        assert np.allclose(_sorted_rows(f.coords), _sorted_rows(g.coords))

    def test_normal_offset_moves_into_room(self):
        for flip in (False, True):
            g = SurfaceGrid.from_wall_region(
                (0, 0), (4, 0), 3.0, normal_offset=0.5, flip=flip
            )
            assert np.allclose(g.coords[:, 1], 0.5)

    def test_angled_wall(self):
        p1, p2 = np.array([4.0, 0.0]), np.array([6.0, 3.0])
        g = SurfaceGrid.from_wall_region(tuple(p1), tuple(p2), 2.5)
        d = p2 - p1
        rel = g.coords[:, :2] - p1
        cross = d[0] * rel[:, 1] - d[1] * rel[:, 0]
        assert np.allclose(cross, 0, atol=1e-9)
        expected = np.array([-3.0, 2.0, 0.0]) / np.sqrt(13)
        assert np.isclose(np.dot(g.normal, expected), 1.0)

    def test_polygon_region_in_wall_coords(self):
        poly = Polygon2D(vertices=((1, 0.5), (3, 0.5), (3, 2), (1, 2)))
        g = SurfaceGrid.from_wall_region((0, 0), (4, 0), 3.0, polygon=poly)
        xs, zs = g.coords[:, 0], g.coords[:, 2]
        assert xs.min() >= 1 - 1e-9 and xs.max() <= 3 + 1e-9
        assert zs.min() >= 0.5 - 1e-9 and zs.max() <= 2 + 1e-9
        assert np.allclose(g.coords[:, 1], 0)
        bv = g.boundary_vertices
        assert np.isclose(bv[:, 0].min(), 1) and np.isclose(bv[:, 0].max(), 3)
        assert np.isclose(bv[:, 2].min(), 0.5) and np.isclose(bv[:, 2].max(), 2)

    def test_triangle_region_masks(self):
        poly = Polygon2D(vertices=((0, 0), (4, 0), (0, 3)))
        g = SurfaceGrid.from_wall_region(
            (0, 0), (4, 0), 3.0, polygon=poly, spacing_init=(0.25, 0.25)
        )
        assert not g.is_rectangular
        full = g.values_to_full_grid(np.ones(len(g.coords)), fill=np.nan)
        assert np.isnan(full).any()
        assert np.isfinite(full).sum() == len(g.coords)
        # every kept point lies under the hypotenuse along/4 + up/3 <= 1
        along, up = g.coords[:, 0], g.coords[:, 2]
        assert np.all(along / 4 + up / 3 <= 1 + 1e-9)

    def test_outline_outside_wall_warns(self):
        poly = Polygon2D(vertices=((1, 1), (3, 1), (3, 3.5), (1, 3.5)))
        with pytest.warns(UserWarning):
            g = SurfaceGrid.from_wall_region((0, 0), (4, 0), 3.0, polygon=poly)
        assert len(g.coords) > 0
        assert np.isclose(g.boundary_vertices[:, 2].max(), 3.5)

    def test_round_trip_to_dict(self):
        poly = Polygon2D(vertices=((0.5, 0), (3, 0), (2, 2.5)))
        g = SurfaceGrid.from_wall_region(
            (4, 0), (6, 3), 3.0, polygon=poly, normal_offset=0.2, flip=True
        )
        g2 = SurfaceGrid.from_dict(g.to_dict())
        assert np.allclose(g2.coords, g.coords)
        assert np.allclose(g2.normal, g.normal)


RECT = RoomDimensions(polygon=Polygon2D.rectangle(4, 6), z=3.0)
PENT = RoomDimensions(
    polygon=Polygon2D(vertices=((0, 0), (4, 0), (6, 3), (2, 5), (-1, 2))), z=2.7
)


class TestWallFrameOf:

    def test_inverts_from_wall_region(self):
        region = Polygon2D(vertices=((0.5, 0.25), (1.5, 0.25), (1.0, 1.5)))
        cases = [(RECT, "south"), (RECT, "east"), (PENT, "wall_1"), (PENT, "wall_3")]
        for dims, wall in cases:
            face = dims.faces[wall]
            for flip in (False, True):
                for offset in (0.0, 0.4):
                    for poly in (None, region):
                        g = SurfaceGrid.from_wall_region(
                            (face.x1, face.y1), (face.x2, face.y2), face.z_height,
                            polygon=poly, normal_offset=offset, flip=flip,
                        )
                        frame = wall_frame_of(g, dims)
                        assert isinstance(frame, WallFrame)
                        assert frame.wall_id == wall
                        assert np.isclose(frame.normal_offset, offset)
                        assert frame.flip == flip
                        expected = poly or Polygon2D.rectangle(
                            face.edge_length, face.z_height
                        )
                        assert np.allclose(
                            _sorted_rows(frame.polygon.vertices),
                            _sorted_rows(expected.vertices),
                        )

    def test_legacy_xz_plane_inside_room(self):
        g = SurfaceGrid.from_legacy(
            mins=(0, 0), maxs=(4, 3), height=0.5, ref_surface="xz", direction=1
        )
        frame = wall_frame_of(g, RECT)
        assert frame.wall_id == "south"
        assert np.isclose(frame.normal_offset, 0.5)
        assert frame.flip is False
        assert np.allclose(
            _sorted_rows(frame.polygon.vertices),
            _sorted_rows(Polygon2D.rectangle(4, 3).vertices),
        )

    def test_legacy_xz_plane_facing_away(self):
        g = SurfaceGrid.from_legacy(
            mins=(0, 0), maxs=(4, 3), height=0.5, ref_surface="xz", direction=-1
        )
        frame = wall_frame_of(g, RECT)
        assert frame.wall_id == "south"
        assert np.isclose(frame.normal_offset, 0.5)
        assert frame.flip is True

    def test_legacy_yz_plane_near_east_wall(self):
        g = SurfaceGrid.from_legacy(
            mins=(0, 0), maxs=(6, 3), height=3.8, ref_surface="yz", direction=-1
        )
        frame = wall_frame_of(g, RECT)
        assert frame.wall_id == "east"
        assert np.isclose(frame.normal_offset, 0.2)
        assert frame.flip is False

    def test_horizontal_plane_returns_none(self):
        g = SurfaceGrid.from_legacy(mins=(0, 0), maxs=(4, 6), height=1.0)
        assert wall_frame_of(g, RECT) is None

    def test_plane_parallel_to_no_wall_returns_none(self):
        a = np.radians(30)
        g = SurfaceGrid.from_wall_region(
            (1, 1), (1 + 2 * np.cos(a), 1 + 2 * np.sin(a)), 3.0
        )
        assert wall_frame_of(g, RECT) is None


class TestCalcPlaneFromWallRegion:

    def test_from_wall_region_uses_dims_faces(self):
        face = PENT.faces["wall_1"]
        poly = Polygon2D(vertices=((0.5, 0.2), (2.5, 0.2), (2.5, 2.0), (0.5, 2.0)))
        zone = CalcPlane.from_wall_region(
            "wall_1", PENT, polygon=poly, normal_offset=0.3, flip=True,
            spacing=(0.25, 0.25), zone_id="w1",
        )
        ref = SurfaceGrid.from_wall_region(
            (face.x1, face.y1), (face.x2, face.y2), face.z_height,
            polygon=poly, normal_offset=0.3, flip=True, spacing_init=(0.25, 0.25),
        )
        assert zone.zone_id == "w1"
        assert np.allclose(zone.coords, ref.coords)
        assert np.allclose(zone.geometry.normal, ref.normal)

    def test_unknown_wall_raises_keyerror(self):
        with pytest.raises(KeyError):
            CalcPlane.from_wall_region("wall_9", PENT)
        with pytest.raises(KeyError):
            CalcPlane.from_wall_region("south", PENT)
        with pytest.raises(KeyError):
            CalcPlane.from_wall_region("floor", RECT)
