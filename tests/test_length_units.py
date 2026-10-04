"""Interoperability of every length unit: metadata, Room.set_units across all
pairs, serialization, standard zones, reporting, efficacy and placement."""

import itertools
import warnings

import numpy as np
import pytest

from guv_calcs import (
    CalcPlane,
    CalcVol,
    Lamp,
    LengthUnits,
    Object,
    Project,
    Room,
    convert_length,
    round_length,
)
from guv_calcs.lamp.lamp_placement import LampPlacer
from guv_calcs.standard_zones import create_standard_zones
from guv_calcs.geometry import RoomDimensions, Polygon2D, SurfaceGrid
from guv_calcs.safety import PhotStandard

APP_UNITS = ["meters", "centimeters", "millimeters", "feet", "inches"]
ALL_UNITS = APP_UNITS + ["yards"]
PAIRS = [(a, b) for a, b in itertools.permutations(ALL_UNITS, 2)]


def _k(src, dst):
    """Multiplicative factor from src to dst."""
    return convert_length(src, dst, 1.0)


def _furnished_room(units="meters"):
    """A 6x4x2.7 m-equivalent room with a lamp, custom zones and objects."""
    k = _k("meters", units)
    room = Room(x=6 * k, y=4 * k, z=2.7 * k, units=units, enable_reflectance=False)
    lamp = Lamp.from_keyword("aerolamp").move(3 * k, 2 * k, 2.6 * k).aim(3 * k, 2 * k, 0)
    room.add_lamp(lamp)
    room.add_calc_zone(
        CalcPlane(
            zone_id="plane",
            geometry=SurfaceGrid.from_legacy(
                mins=(1 * k, 1 * k), maxs=(5 * k, 3 * k), height=1.0 * k,
                spacing_init=(0.5 * k, 0.5 * k),
            ),
        )
    )
    room.add_calc_zone(
        CalcVol.from_legacy(
            x1=0, x2=6 * k, y1=0, y2=4 * k, z1=0, z2=2 * k,
            num_x=6, num_y=4, num_z=4, zone_id="vol",
        )
    )
    room.add_object(Object.box(1.2 * k, 0.6 * k, 0.75 * k, object_id="desk",
                               position=(2 * k, 3 * k, 0)))
    room.add_object(Object.extrusion([(0, 0), (1 * k, 0), (1 * k, 1 * k)], 0.5 * k,
                                     object_id="wedge", position=(4 * k, 1 * k, 0)))
    return room


def _snapshot(room):
    """Geometry of a furnished room as a flat dict of floats."""
    lamp = room.lamps["aerolamp"]
    plane = room.calc_zones["plane"]
    vol = room.calc_zones["vol"]
    desk = room.objects["desk"]
    wedge = room.objects["wedge"]
    snap = {
        "x": room.x, "y": room.y, "z": room.z,
        "lamp_x": lamp.x, "lamp_y": lamp.y, "lamp_z": lamp.z,
        "lamp_aimz": lamp.aimz,
        "lamp_w": lamp.surface.width, "lamp_l": lamp.surface.length,
        "plane_x1": plane.x1, "plane_x2": plane.x2, "plane_h": plane.height,
        "plane_dx": plane.x_spacing,
        "vol_x2": vol.x2, "vol_z2": vol.z2,
        "desk_x": desk.x, "desk_w": desk.width, "desk_h": desk.height,
        "wedge_h": wedge.height,
    }
    for i, (vx, vy) in enumerate(wedge._get_polygon().vertices):
        snap[f"wedge_v{i}x"] = vx
        snap[f"wedge_v{i}y"] = vy
    return snap


class TestLengthUnitsMetadata:

    @pytest.mark.parametrize("token,abbr,metric,decimals", [
        ("meters", "m", True, 2),
        ("centimeters", "cm", True, 1),
        ("millimeters", "mm", True, 0),
        ("feet", "ft", False, 2),
        ("inches", "in", False, 1),
        ("yards", "yd", False, 2),
    ])
    def test_metadata(self, token, abbr, metric, decimals):
        u = LengthUnits.from_any(token)
        assert u.abbreviation == abbr
        assert u.is_metric is metric
        assert u.decimals == decimals

    def test_every_member_has_metadata(self):
        for u in LengthUnits:
            assert u.abbreviation
            assert isinstance(u.is_metric, bool)
            assert u.decimals >= 0

    def test_abbreviations_parse_back(self):
        for u in LengthUnits:
            assert LengthUnits.from_any(u.abbreviation) is u

    def test_round_length(self):
        assert round_length("inches", 70.86614) == 70.9
        assert round_length("millimeters", 1800.4) == 1800.0
        assert round_length("meters", 1.23456) == 1.23
        assert round_length("centimeters", 1.0, 2.26) == (1.0, 2.3)
        assert round_length("feet", None) is None

    def test_round_length_returns_python_float(self):
        assert type(round_length("meters", np.float64(1.5))) is float


class TestConversionFactors:

    @pytest.mark.parametrize("src,dst", PAIRS)
    def test_round_trip_identity(self, src, dst):
        # convert_length rounds to 12 decimal places, so tiny factors (mm -> ft)
        # carry a few parts in 1e9 of rounding; geometry values are far larger
        assert _k(src, dst) * _k(dst, src) == pytest.approx(1.0, rel=1e-8)

    @pytest.mark.parametrize("src,mid,dst", list(itertools.permutations(APP_UNITS, 3))[:40])
    def test_transitive(self, src, mid, dst):
        via = convert_length(mid, dst, convert_length(src, mid, 1.0))
        assert via == pytest.approx(_k(src, dst), rel=1e-8)

    def test_known_factors(self):
        assert _k("feet", "inches") == pytest.approx(12.0)
        assert _k("meters", "centimeters") == pytest.approx(100.0)
        assert _k("meters", "millimeters") == pytest.approx(1000.0)
        assert _k("inches", "centimeters") == pytest.approx(2.54)
        assert _k("inches", "millimeters") == pytest.approx(25.4)
        assert _k("yards", "feet") == pytest.approx(3.0)


class TestRoomConstruction:

    @pytest.mark.parametrize("units,dims", [
        ("meters", (6.0, 4.0, 2.7)),
        ("centimeters", (600, 400, 270)),
        ("millimeters", (6000, 4000, 2700)),
        ("feet", (20, 13, 9)),
        ("inches", (236, 157, 106)),
    ])
    def test_default_dimensions_per_unit(self, units, dims):
        room = Room(units=units)
        assert (room.x, room.y, room.z) == pytest.approx(dims)

    @pytest.mark.parametrize("units", ALL_UNITS)
    def test_added_lamp_adopts_room_units(self, units):
        room = Room(units=units)
        lamp = Lamp.from_keyword("aerolamp")  # meters
        w_m = lamp.surface.width
        room.add_lamp(lamp)
        assert lamp.surface.units == LengthUnits.from_any(units)
        assert lamp.surface.width == pytest.approx(w_m * _k("meters", units), rel=1e-9)

    @pytest.mark.parametrize("units", ALL_UNITS)
    def test_project_units_canonical(self, units):
        p = Project(units=units)
        assert p.units == units
        assert isinstance(p.units, LengthUnits)
        assert p.create_room(room_id="r").units == units

    def test_project_units_alias(self):
        assert Project(units="in").units == "inches"
        assert Project(units="cm").units == "centimeters"

    def test_project_units_invalid(self):
        with pytest.raises(ValueError):
            Project(units="furlongs")

    def test_room_units_invalid(self):
        with pytest.raises(ValueError):
            Room(units="furlongs")


class TestSetUnitsEveryPair:

    @pytest.mark.parametrize("src,dst", PAIRS)
    def test_all_geometry_scales(self, src, dst):
        room = _furnished_room(src)
        before = _snapshot(room)
        k = _k(src, dst)
        room.set_units(dst)
        after = _snapshot(room)
        assert room.units == dst
        for key, v in before.items():
            assert after[key] == pytest.approx(v * k, rel=1e-9), key

    @pytest.mark.parametrize("src,dst", PAIRS)
    def test_round_trip_restores_geometry(self, src, dst):
        room = _furnished_room(src)
        before = _snapshot(room)
        room.set_units(dst)
        room.set_units(src)
        after = _snapshot(room)
        for key, v in before.items():
            assert after[key] == pytest.approx(v, rel=1e-9, abs=1e-9), key

    def test_chain_through_every_unit(self):
        room = _furnished_room("meters")
        before = _snapshot(room)
        for u in ["inches", "centimeters", "feet", "millimeters", "yards", "meters"]:
            room.set_units(u)
        assert room.units == "meters"
        after = _snapshot(room)
        for key, v in before.items():
            assert after[key] == pytest.approx(v, rel=1e-9, abs=1e-9), key

    @pytest.mark.parametrize("dst", ALL_UNITS)
    def test_num_points_unchanged(self, dst):
        room = _furnished_room("meters")
        plane = room.calc_zones["plane"]
        vol = room.calc_zones["vol"]
        n_before = (plane.num_x, plane.num_y, vol.num_x, vol.num_y, vol.num_z)
        room.set_units(dst)
        assert (plane.num_x, plane.num_y, vol.num_x, vol.num_y, vol.num_z) == n_before

    @pytest.mark.parametrize("dst", ALL_UNITS)
    def test_polygon_room_vertices_scale(self, dst):
        verts = [(0, 0), (6, 0), (6, 4), (3, 6), (0, 4)]
        room = Room(polygon=verts, z=2.7)
        room.set_units(dst)
        k = _k("meters", dst)
        got = np.array(room.dim.polygon.vertices)
        assert np.allclose(got, np.array(verts, dtype=float) * k)
        assert room.dim.is_polygon

    @pytest.mark.parametrize("dst", ALL_UNITS)
    def test_surfaces_follow(self, dst):
        room = Room(x=6, y=4, z=2.7)
        room.set_units(dst)
        k = _k("meters", dst)
        # all four cardinal walls remain and span the converted extents
        assert set(room.dim.wall_ids) == {"south", "north", "west", "east"}
        assert room.surfaces["floor"].geometry.x2 == pytest.approx(6 * k, rel=1e-9)

    @pytest.mark.parametrize("dst", ALL_UNITS)
    def test_same_physical_volume(self, dst):
        room = Room(x=6, y=4, z=2.7)
        m3 = room.dim.cubic_meters
        room.set_units(dst)
        assert room.dim.cubic_meters == pytest.approx(m3, rel=1e-9)
        assert room.volume == pytest.approx(6 * 4 * 2.7 * _k("meters", dst) ** 3, rel=1e-9)

    def test_set_units_preserves_results(self):
        room = _furnished_room("meters")
        room.calculate()
        values = {z: room.calc_zones[z].values.copy() for z in ("plane", "vol")}
        for u in ["inches", "centimeters", "millimeters", "feet"]:
            room.set_units(u)
            for z, v in values.items():
                assert np.allclose(room.calc_zones[z].values, v)

    @pytest.mark.parametrize("units", ["inches", "centimeters", "millimeters"])
    def test_calculation_matches_meters(self, units):
        """Irradiance is physical: the same room in any unit gives the same values."""
        ref = _furnished_room("meters")
        ref.calculate()
        other = _furnished_room(units)
        other.calculate()
        np.testing.assert_allclose(
            other.calc_zones["plane"].values, ref.calc_zones["plane"].values, rtol=1e-3
        )


class TestSerialization:

    @pytest.mark.parametrize("units", ALL_UNITS)
    def test_dict_round_trip(self, units):
        room = _furnished_room(units)
        before = _snapshot(room)
        data = room.to_dict()
        assert data["units"] == units
        back = Room.from_dict(data)
        assert back.units == units
        after = _snapshot(back)
        for key, v in before.items():
            assert after[key] == pytest.approx(v, rel=1e-9, abs=1e-9), key

    @pytest.mark.parametrize("units", ALL_UNITS)
    def test_file_round_trip(self, units, tmp_path):
        room = _furnished_room(units)
        path = tmp_path / "room.guv"
        room.save(str(path))
        back = Room.load(str(path))
        assert back.units == units
        assert back.x == pytest.approx(room.x)
        assert back.lamps["aerolamp"].surface.units == LengthUnits.from_any(units)

    def test_lamp_housing_units_in_centimeter_room(self):
        room = Room(units="centimeters")
        lamp = Lamp.from_keyword(
            "aerolamp", housing_width=12, housing_length=6, housing_height=2,
            housing_units="inches",
        )
        room.add_lamp(lamp)
        assert lamp.fixture.housing_width == pytest.approx(30.48, rel=1e-9)
        assert lamp.fixture.housing_height == pytest.approx(5.08, rel=1e-9)


class TestStandardZonesPerUnit:

    def _dims(self, units):
        k = _k("meters", units)
        return RoomDimensions(polygon=Polygon2D.rectangle(6 * k, 4 * k), z=2.7 * k,
                              units=LengthUnits.from_any(units))

    @pytest.mark.parametrize("units,height", [
        ("inches", 70.9), ("centimeters", 180.0), ("millimeters", 1800.0),
        ("feet", 5.9), ("meters", 1.8), ("yards", 1.97),
    ])
    def test_default_height_rounded(self, units, height):
        zones = {z.zone_id: z for z in create_standard_zones(PhotStandard.ACGIH, self._dims(units))}
        assert zones["EyeLimits"].height == height
        assert zones["SkinLimits"].height == height

    @pytest.mark.parametrize("units,height", [
        ("inches", 74.8), ("centimeters", 190.0), ("millimeters", 1900.0), ("feet", 6.25),
    ])
    def test_ul8802_height_rounded(self, units, height):
        zones = {z.zone_id: z for z in create_standard_zones(PhotStandard.UL8802, self._dims(units))}
        assert zones["EyeLimits"].height == height

    @pytest.mark.parametrize("units", APP_UNITS)
    def test_room_add_standard_zones(self, units):
        room = Room(units=units)
        room.add_standard_zones()
        wrf = room.calc_zones["WholeRoomFluence"]
        assert wrf.x2 == pytest.approx(room.x)
        assert wrf.z2 == pytest.approx(room.z)
        assert 0 < room.calc_zones["EyeLimits"].height < room.z


class TestReportLabels:

    @pytest.mark.parametrize("units,abbr", [
        ("meters", "m"), ("centimeters", "cm"), ("millimeters", "mm"),
        ("feet", "ft"), ("inches", "in"), ("yards", "yd"),
    ])
    def test_area_and_volume_units(self, units, abbr):
        k = _k("meters", units)
        room = Room(x=6 * k, y=4 * k, z=2.7 * k, units=units)
        text = room.generate_report().decode("cp1252")
        area = round(room.dim.polygon.area, 3)
        assert f",Floor area,{area},{abbr} 2" in text
        assert f",Volume,{round(room.volume, 3)},{abbr} 3" in text
        assert f",{units}\r\n" in text  # dimensions row names the unit


class TestEfficacyUnits:

    @pytest.mark.parametrize("units,metric", [
        ("meters", True), ("centimeters", True), ("millimeters", True),
        ("feet", False), ("inches", False), ("yards", False),
    ])
    def test_cadr_unit_follows_metric_system(self, units, metric):
        room = Room(units=units)
        room.add_standard_zones()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = room.get_efficacy_data()
        assert data._use_metric_units is metric


class TestPlacementScalesWithUnits:

    @pytest.mark.parametrize("units", ALL_UNITS)
    def test_fixture_offsets_are_physical(self, units):
        """aerolamp has a 0.1 x 0.118 x 0.076 m housing; offsets follow it in any unit."""
        room = Room(units=units)
        lamp = Lamp.from_keyword("aerolamp")
        room.add_lamp(lamp)  # converts lamp units
        k = _k("meters", units)
        f = lamp.fixture
        assert LampPlacer.ceiling_offset(lamp) == pytest.approx((0.076 + 0.02) * k, rel=1e-6)
        diag = np.hypot(0.1, 0.118)
        assert LampPlacer.wall_clearance(lamp) == pytest.approx((diag / 2 + 0.076 / 2) * k, rel=1e-6)
        assert f.housing_height == pytest.approx(0.076 * k, rel=1e-6)

    @pytest.mark.parametrize("units", ALL_UNITS)
    def test_bare_lamp_offsets_are_physical(self, units):
        """A lamp with no housing dims gets the 10 cm drop and 5 cm clearance."""
        k = _k("meters", units)
        lamp = Lamp(lamp_id="bare", units=units)
        assert not lamp.fixture.has_dimensions
        assert LampPlacer.ceiling_offset(lamp) == pytest.approx(0.1 * k, rel=1e-9)
        assert LampPlacer.wall_clearance(lamp) == pytest.approx(0.05 * k, rel=1e-9)

    @pytest.mark.parametrize("units", ALL_UNITS)
    def test_corner_inset_follows_clearance(self, units):
        k = _k("meters", units)
        placer = LampPlacer.for_room(x=6 * k, y=4 * k, z=2.7 * k, units=units)
        lamp = Lamp(lamp_id="bare", units=units)
        res = placer.get_placement(lamp, mode="corner")
        corners = [(0, 0), (6 * k, 0), (6 * k, 4 * k), (0, 4 * k)]
        d = min(np.hypot(res.x - cx, res.y - cy) for cx, cy in corners)
        assert d == pytest.approx(0.05 * k, rel=1e-6)
        assert res.z == pytest.approx(2.7 * k - 0.1 * k, rel=1e-6)

    @pytest.mark.parametrize("units", APP_UNITS)
    def test_room_place_lamp_drop_below_ceiling(self, units):
        room = Room(units=units)
        lamp = Lamp.from_keyword("aerolamp")
        room.place_lamp(lamp, mode="downlight")
        k = _k("meters", units)
        assert room.z - lamp.z == pytest.approx(round(0.1 * k, 2), rel=1e-6)

    @pytest.mark.parametrize("units", APP_UNITS)
    def test_four_corners_fill_in_any_unit(self, units):
        """Occupancy tolerance scales too, so the 2nd..4th lamps take fresh corners."""
        room = Room(units=units)
        for i in range(4):
            room.place_lamp(Lamp.from_keyword("aerolamp", lamp_id=f"l{i}"), mode="corner")
        pts = [(l.x, l.y) for l in room.lamps.values()]
        for a, b in itertools.combinations(pts, 2):
            assert np.hypot(a[0] - b[0], a[1] - b[1]) > 1.0 * _k("meters", units)

    def test_for_dims_takes_units_from_dims(self):
        dims = RoomDimensions(polygon=Polygon2D.rectangle(600, 400), z=270,
                              units=LengthUnits.CENTIMETERS)
        placer = LampPlacer.for_dims(dims)
        assert placer.units == LengthUnits.CENTIMETERS
        assert placer.scale == pytest.approx(100.0)


class TestLampSurfaceSerialization:

    @pytest.mark.parametrize("units", ALL_UNITS)
    def test_surface_dims_survive_dict_round_trip(self, units):
        """Surface width/length are re-read from the IES (meters) on load and must
        be converted into the room's units, not relabeled."""
        room = Room(units=units)
        lamp = Lamp.from_keyword("aerolamp")
        room.add_lamp(lamp)
        w, l = lamp.surface.width, lamp.surface.length
        back = Room.from_dict(room.to_dict())
        s = back.lamps["aerolamp"].surface
        assert s.units == LengthUnits.from_any(units)
        assert s.width == pytest.approx(w, rel=1e-9)
        assert s.length == pytest.approx(l, rel=1e-9)

    def test_ies_keeps_native_units_after_set_units(self):
        lamp = Lamp.from_keyword("aerolamp")
        code = lamp.ies.units
        w_ies = lamp.ies.width
        lamp.set_units("inches")
        assert lamp.ies.units == code
        assert lamp.ies.width == w_ies
        assert lamp.surface.width == pytest.approx(w_ies * _k("meters", "inches"), rel=1e-9)

    def test_set_width_in_room_units_updates_ies_in_its_units(self):
        lamp = Lamp.from_keyword("aerolamp")
        lamp.set_units("centimeters")
        lamp.set_width(8.0)  # cm
        assert lamp.surface.width == 8.0
        assert lamp.ies.width == pytest.approx(0.08, rel=1e-9)


class TestGridPointCountStability:

    @pytest.mark.parametrize("dst", ALL_UNITS)
    def test_spacing_mode_keeps_points_across_units(self, dst):
        room = _furnished_room("meters")
        plane = room.calc_zones["plane"]
        assert (plane.num_x, plane.num_y) == (8, 4)
        room.set_units(dst)
        assert (plane.num_x, plane.num_y) == (8, 4)
        room.set_units("meters")
        assert (plane.num_x, plane.num_y) == (8, 4)
