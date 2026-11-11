"""Tests for Surface, ReflectanceManager, and init_room_surfaces."""

import pytest
import numpy as np
from guv_calcs import CalcPlane, SurfaceGrid, Polygon2D, Room, Lamp, Object
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


class TestNonRectangularReflectors:
    """A masked polygon grid (polygon-room floor, polygon-object top) has a
    flat point list rather than an (nx, ny) grid; it must still act as a
    reflector."""

    @staticmethod
    def _room(**kwargs):
        from guv_calcs import Room, Lamp
        room = Room(z=2.7, enable_reflectance=True, **kwargs)
        room.set_reflectance(0.078)
        lamp = Lamp.from_keyword("aerolamp")
        lamp.move(2, 3, 2.6)
        room.add_lamp(lamp)
        return room

    def test_polygon_room_calculates_with_reflections(self):
        room = self._room(polygon=[(0, 0), (4, 0), (4, 3), (2, 3), (2, 6), (0, 6)])
        assert len(room.surfaces["floor"].plane.num_points) == 1  # masked grid
        room.calculate()
        floor = room.surfaces["floor"].plane
        assert floor.values is not None and np.isfinite(floor.values).any()
        assert floor.reflected_values is not None

    def test_reflective_polygon_object_calculates(self):
        from guv_calcs import Object
        room = self._room(x=4, y=6)
        obj = Object.extrusion(
            [(0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)], 1.0,
            object_id="l", position=(2, 3, 0), R=0.5,
        )
        room.add_object(obj)
        assert len(obj.surfaces["l:top"].plane.num_points) == 1  # masked grid
        room.calculate()
        top = obj.surfaces["l:top"].plane
        assert top.values is not None and np.isfinite(top.values).any()
        # The room floor received bounced light from the object's faces too
        assert room.surfaces["floor"].plane.reflected_values is not None


class TestReflectedViewFOV:
    """Reflected light obeys a zone's view-centered FOV.

    Only the floor reflects; it is lit by a downlight and gridded finely. A
    point 0.5 m above it looks straight down.
    """

    R = 0.5

    def _room(self, fov_vert, fov_horiz):
        from guv_calcs import Room, Lamp, CalcPoint
        room = Room(x=4, y=6, z=2.7, enable_reflectance=True)
        room.set_reflectance(0.0)
        room.set_reflectance(self.R, wall_id="floor")
        room.set_reflectance_spacing(0.05, 0.05, wall_id="floor")
        room.add_lamp(Lamp.from_keyword(
            "ushio_b1", x=2, y=3, z=2.6, aimx=2, aimy=3, aimz=0))
        pt = CalcPoint.at(
            (2, 3, 0.5), aim_point=(2, 3, 0), zone_id="P",
            horiz=False, use_normal=True,
            fov_vert=fov_vert, fov_horiz=fov_horiz)
        room.add_calc_zone(pt)
        return room, pt

    def test_narrow_view_reads_floor_radiance(self):
        """A narrow view straight down reads L * solid angle, where
        L = R * E / pi is the floor's radiance under the point. The FOV edge
        falls on floor cell edges (+-0.1 m at 0.5 m), since cells count whole."""
        half = np.arctan(0.1 / 0.5)
        fov = 2 * np.degrees(half)
        room, pt = self._room(fov, fov)
        room.calculate()
        floor = room.surfaces["floor"].plane
        x, y = floor.coords[:, 0], floor.coords[:, 1]
        near = (np.abs(x - 2) < 0.1) & (np.abs(y - 3) < 0.1)
        E = floor.values.ravel()[near].mean()
        solid_angle = 4 * np.arcsin(np.sin(half) * np.sin(half))
        expected = self.R * E / np.pi * solid_angle
        assert pt.result.reflected_values.ravel()[0] == pytest.approx(expected, rel=0.05)

    def test_changing_fov_updates_reflected_values(self):
        room, pt = self._room(180, 360)
        room.calculate()
        full = pt.result.reflected_values.ravel()[0]
        pt.fov_vert, pt.fov_horiz = 20, 20
        room.calculate()
        narrow = pt.result.reflected_values.ravel()[0]
        fresh_room, fresh_pt = self._room(20, 20)
        fresh_room.calculate()
        assert narrow < full
        assert narrow == pytest.approx(fresh_pt.result.reflected_values.ravel()[0])


class TestReflectanceCacheInvalidation:
    """A soft recalculation after a scene change matches a hard one."""

    @staticmethod
    def _room():
        from guv_calcs import Room, Lamp
        room = Room(x=4, y=6, z=2.7)
        room.set_reflectance(0.5)
        room.add_calc_zone(CalcPlane(zone_id="plane", geometry=SurfaceGrid.from_legacy(
            mins=(0, 0), maxs=(4, 6), height=1.0, num_points_init=(10, 10))))
        room.add_lamp(Lamp.from_keyword(
            "ushio_b1", x=2, y=3, z=2.6, aimx=2, aimy=3, aimz=0))
        return room

    @staticmethod
    def _values(room):
        return {k: np.array(z.values, dtype=float) for k, z in room.calc_zones.items()}

    def _assert_matches_hard(self, room):
        room.calculate()
        soft = self._values(room)
        room.calculate(hard=True)
        for k, v in self._values(room).items():
            np.testing.assert_allclose(soft[k], v, rtol=1e-5, err_msg=k)

    @pytest.mark.parametrize("change", [
        lambda r: r.set_reflectance_num_points(num_x=25, wall_id="south"),
        lambda r: r.set_reflectance_spacing(0.13, 0.13, wall_id="south"),
        lambda r: r.set_reflectance_spacing(0.2, 0.2),
    ], ids=["one-wall-num-points", "one-wall-spacing", "all-walls-spacing"])
    def test_surface_resolution_change(self, change):
        room = self._room()
        room.calculate()
        change(room)
        self._assert_matches_hard(room)

    def test_obstacle_move(self):
        from guv_calcs import Object
        room = self._room()
        slab = Object.box(1.0, 1.0, 0.3, object_id="slab", position=(1.5, 2.5, 1.2))
        room.add_object(slab)
        room.calculate()
        slab.move(x=2.5, y=3.5)
        self._assert_matches_hard(room)


class TestReleaseFormFactors:
    def _room(self):
        room = Room(x=4, y=6, z=2.7, enable_reflectance=True)
        room.set_reflectance(0.1)
        room.set_reflectance_num_points(num_x=4, num_y=4)  # coarse: keeps these tests quick
        room.add_lamp(Lamp.from_keyword("ushio_b1", lamp_id="L1", x=2, y=3, z=2.7, aimx=2, aimy=3, aimz=0))
        room.add_standard_zones()
        for zone in room.calc_zones.values():
            zone.set_spacing(x_spacing=0.5, y_spacing=0.5)
        return room

    def _cached_form_factor_bytes(self, room):
        zones = list(room.calc_zones.values()) + [s.plane for s in room.all_surfaces.values()]
        return sum(
            e.form_factors.nbytes
            for z in zones
            for e in z.calculator.cache.surface_cache.values()
            if e.form_factors is not None
        )

    def test_release_keeps_results_and_frees_form_factors(self):
        room = self._room().calculate()
        before = {k: z.get_values().copy() for k, z in room.calc_zones.items()}
        assert self._cached_form_factor_bytes(room) > 0
        room.release_form_factors()
        assert self._cached_form_factor_bytes(room) == 0
        for k, z in room.calc_zones.items():
            np.testing.assert_array_equal(z.get_values(), before[k])

    def test_recalculation_after_release_matches_a_fresh_room(self):
        room = self._room().calculate().release_form_factors()
        room.lamps["L1"].move(x=1.0, y=2.0, z=2.7)
        room.calculate()
        fresh = self._room()
        fresh.lamps["L1"].move(x=1.0, y=2.0, z=2.7)
        fresh.calculate()
        for k, z in room.calc_zones.items():
            np.testing.assert_allclose(z.get_values(), fresh.calc_zones[k].get_values(), rtol=1e-5)
        assert self._cached_form_factor_bytes(room) > 0  # rebuilt by the recalculation

    def test_unchanged_room_recalculates_after_release(self):
        room = self._room().calculate()
        before = {k: z.get_values().copy() for k, z in room.calc_zones.items()}
        room.release_form_factors().calculate()
        for k, z in room.calc_zones.items():
            np.testing.assert_allclose(z.get_values(), before[k], rtol=1e-5)


class TestReflectanceMemory:
    """The surface→zone form factors and reflected values are computed in
    chunks of surface points, so the float64 temporaries stay bounded no
    matter how fine the surface grid is. Before chunking, one 25x25 surface
    against the default 25^3 volume needed ~1.1 GB of temporaries for a
    39 MB result, and six surfaces did that in turn."""

    @staticmethod
    def _room(objects=False, coarse=False):
        room = Room(x=4, y=6, z=2.7, enable_reflectance=True)
        room.set_reflectance(0.1)
        room.add_lamp(Lamp.from_keyword(
            "ushio_b1", lamp_id="L1", x=2, y=3, z=2.7, aimx=2, aimy=3, aimz=0))
        room.add_standard_zones()
        if coarse:  # keeps the chunk-equivalence test quick
            room.set_reflectance_num_points(num_x=5, num_y=5)
            for zone in room.calc_zones.values():
                zone.set_spacing(x_spacing=0.5, y_spacing=0.5)
        if objects:
            room.add_object(Object.box(0.6, 0.6, 0.8, object_id="box", position=(1, 1, 0), R=0.2))
        return room

    def test_form_factor_peak_is_bounded(self):
        import tracemalloc
        room = self._room()
        room.set_reflectance_num_points(num_x=25, num_y=25, wall_id="floor")
        floor = room.surfaces["floor"]
        zv = room.calc_zones["WholeRoomFluence"].to_view()
        tracemalloc.start()
        ff = floor._calculate_coordinates(zv)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert ff.dtype == np.float32
        assert peak < ff.nbytes + 100 * 2**20, f"peak {peak/2**20:.0f} MB for a {ff.nbytes/2**20:.0f} MB result"

    def test_tiny_chunks_give_the_same_results(self, monkeypatch):
        import guv_calcs.calc_manager as calc_manager
        from guv_calcs import CalcPoint
        def build():
            room = self._room(objects=True, coarse=True)
            room.add_calc_zone(CalcPoint.at(
                (2, 3, 1.0), aim_point=(2, 0, 1.0), zone_id="view",
                horiz=False, use_normal=True, fov_vert=60, fov_horiz=90))
            return room
        default = build().calculate()
        monkeypatch.setattr(calc_manager, "_CHUNK_BYTES", 16 * 1024)  # one or a few rows per chunk
        chunked = build().calculate()
        for k, z in default.calc_zones.items():
            np.testing.assert_allclose(
                chunked.calc_zones[k].get_values(), z.get_values(), rtol=1e-5, err_msg=k)
            np.testing.assert_allclose(
                chunked.calc_zones[k].result.reflected_values,
                z.result.reflected_values, rtol=1e-5, err_msg=k)

    def test_volume_values_match_the_elementwise_sum(self):
        """The volume path is a matrix product; it must equal the plain
        sum over surface points, with non-finite surface values (the lamp
        sits on the ceiling) dropped the way a masked sum drops them."""
        room = self._room()
        room.ref_manager.calculate_incidence(room.lamps, room.surfaces)
        ceiling = room.surfaces["ceiling"]
        base = np.array(ceiling.plane.result.base_values, dtype="float32")
        base[0, 0], base[3, 4] = np.inf, np.nan
        ceiling.plane.result.base_values = base
        assert not np.isfinite(np.asarray(ceiling.plane.values, dtype=float)).all()
        zv = room.calc_zones["WholeRoomFluence"].to_view()
        ff = ceiling._calculate_coordinates(zv)
        values = ceiling._calculate_values(ff, zv)
        I_r = np.ma.masked_invalid(np.asarray(ceiling.plane.values, dtype="float64").ravel())
        expected = np.ma.sum(I_r[:, None] * ff.reshape(I_r.size, -1), axis=0)
        np.testing.assert_allclose(values.ravel(), np.ma.filled(expected, 0.0), rtol=1e-5)
