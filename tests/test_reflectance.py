"""Tests for Surface, ReflectanceManager, and init_room_surfaces."""

import pytest
import numpy as np
from guv_calcs import CalcPlane, SurfaceGrid, Polygon2D
from guv_calcs.reflectance import Surface, ReflectanceManager, init_room_surfaces
from guv_calcs.geometry import RoomDimensions


def _make_plane(zone_id="test"):
    """Create a simple CalcPlane for Surface tests."""
    grid = SurfaceGrid.from_legacy(mins=(0, 0), maxs=(6, 4), height=0)
    return CalcPlane(zone_id=zone_id, geometry=grid)


class TestSurfaceValidation:

    def test_R_out_of_range(self):
        with pytest.raises(ValueError):
            Surface(R=1.5, plane=_make_plane())

    def test_T_out_of_range(self):
        with pytest.raises(ValueError):
            Surface(R=0, T=-0.1, plane=_make_plane())

    def test_R_plus_T_exceeds_one(self):
        with pytest.raises(ValueError):
            Surface(R=0.8, T=0.5, plane=_make_plane())

    def test_set_reflectance_validates(self):
        s = Surface(R=0.0, plane=_make_plane())
        with pytest.raises(ValueError):
            s.set_reflectance(1.5)

    def test_set_transmittance_validates(self):
        s = Surface(R=0.8, plane=_make_plane())
        with pytest.raises(ValueError):
            s.set_transmittance(0.5)

    def test_serialization_round_trip(self):
        s = Surface(R=0.3, T=0.2, plane=_make_plane("surf1"))
        data = s.to_dict()
        loaded = Surface.from_dict(data)
        assert loaded.R == 0.3
        assert loaded.T == 0.2
        assert loaded.plane.zone_id == "surf1"


class TestReflectanceManager:

    def test_threshold_out_of_range(self):
        with pytest.raises(ValueError):
            ReflectanceManager(threshold=-0.1)

    def test_disabled_skips_calculation(self):
        """Disabled manager leaves surface values as None."""
        mgr = ReflectanceManager(enabled=False)
        grid = SurfaceGrid.from_legacy(mins=(0, 0), maxs=(6, 4), height=0)
        plane = CalcPlane(zone_id="floor", geometry=grid)
        surface = Surface(R=0.5, plane=plane)
        mgr.calculate_incidence(lamps=[], surfaces={"floor": surface})
        assert surface.plane.values is None

    def test_serialization_round_trip(self):
        mgr = ReflectanceManager(max_num_passes=50, threshold=0.05, enabled=False)
        data = mgr.to_dict()
        loaded = ReflectanceManager.from_dict(data)
        assert loaded.max_num_passes == 50
        assert loaded.threshold == 0.05
        assert loaded.enabled is False


class TestInitRoomSurfaces:

    def test_creates_all_faces_rectangular(self):
        """Rectangular room -> 6 surfaces."""
        dims = RoomDimensions(polygon=Polygon2D.rectangle(6, 4), z=2.7)
        surfaces = init_room_surfaces(dims)
        assert "floor" in surfaces
        assert "ceiling" in surfaces
        assert len(surfaces) == 6

    def test_polygon_room_surfaces(self):
        """L-shaped room -> floor + ceiling + N walls."""
        poly = Polygon2D(vertices=[(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)])
        dims = RoomDimensions(polygon=poly, z=2.7)
        surfaces = init_room_surfaces(dims)
        assert "floor" in surfaces
        assert "ceiling" in surfaces
        # 6 walls for 6-vertex polygon
        assert len(surfaces) == 8  # floor + ceiling + 6 walls

    def test_custom_reflectance(self):
        """Custom reflectance applied to specific surface."""
        dims = RoomDimensions(polygon=Polygon2D.rectangle(6, 4), z=2.7)
        surfaces = init_room_surfaces(dims, reflectances={"floor": 0.5})
        assert surfaces["floor"].R == 0.5
        assert surfaces["ceiling"].R == 0.0
