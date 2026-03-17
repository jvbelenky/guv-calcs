"""Tests for create_standard_zones from standard_zones.py."""

import pytest
from guv_calcs import CalcPlane, CalcVol, PhotStandard, Polygon2D
from guv_calcs.geometry import RoomDimensions
from guv_calcs.units import LengthUnits
from guv_calcs.standard_zones import create_standard_zones, _standard_zone_spacing, MAX_POINTS_PER_DIM


class TestCreateStandardZones:

    @pytest.fixture
    def dims_m(self):
        return RoomDimensions(polygon=Polygon2D.rectangle(6, 4), z=2.7)

    @pytest.fixture
    def dims_ft(self):
        return RoomDimensions(
            polygon=Polygon2D.rectangle(20, 13), z=9, units=LengthUnits.FEET,
        )

    def test_creates_three_zones(self, dims_m):
        zones = create_standard_zones(PhotStandard.ACGIH, dims_m)
        assert len(zones) == 3

    def test_zone_ids(self, dims_m):
        zones = create_standard_zones(PhotStandard.ACGIH, dims_m)
        ids = [z.zone_id for z in zones]
        assert "WholeRoomFluence" in ids
        assert "EyeLimits" in ids
        assert "SkinLimits" in ids

    def test_zone_types(self, dims_m):
        zones = create_standard_zones(PhotStandard.ACGIH, dims_m)
        zone_map = {z.zone_id: z for z in zones}
        assert isinstance(zone_map["WholeRoomFluence"], CalcVol)
        assert isinstance(zone_map["EyeLimits"], CalcPlane)
        assert isinstance(zone_map["SkinLimits"], CalcPlane)

    def test_default_height_meters(self, dims_m):
        zones = create_standard_zones(PhotStandard.ACGIH, dims_m)
        zone_map = {z.zone_id: z for z in zones}
        assert zone_map["EyeLimits"].height == pytest.approx(1.8)
        assert zone_map["SkinLimits"].height == pytest.approx(1.8)

    def test_default_height_feet(self, dims_ft):
        zones = create_standard_zones(PhotStandard.ACGIH, dims_ft)
        zone_map = {z.zone_id: z for z in zones}
        assert zone_map["EyeLimits"].height == pytest.approx(5.9)

    def test_ul8802_height(self, dims_m):
        zones = create_standard_zones(PhotStandard.UL8802, dims_m)
        zone_map = {z.zone_id: z for z in zones}
        assert zone_map["EyeLimits"].height == pytest.approx(1.9)

    def test_ul8802_eye_calc_mode(self, dims_m):
        """UL8802 uses fluence_rate base with fov_horiz=180 override -> custom."""
        zones = create_standard_zones(PhotStandard.UL8802, dims_m)
        zone_map = {z.zone_id: z for z in zones}
        eye = zone_map["EyeLimits"]
        # fluence_rate base flags but fov_horiz overridden to 180
        assert eye.use_normal is False
        assert eye.fov_horiz == 180
        assert eye.fov_vert == 180

    def test_default_eye_calc_mode(self, dims_m):
        zones = create_standard_zones(PhotStandard.ACGIH, dims_m)
        zone_map = {z.zone_id: z for z in zones}
        assert zone_map["EyeLimits"].calc_mode == "eye_worst_case"

    def test_dose_mode_enabled(self, dims_m):
        zones = create_standard_zones(PhotStandard.ACGIH, dims_m)
        zone_map = {z.zone_id: z for z in zones}
        assert zone_map["EyeLimits"].dose is True
        assert zone_map["EyeLimits"].hours == 8.0
        assert zone_map["SkinLimits"].dose is True
        assert zone_map["SkinLimits"].hours == 8.0

    def test_wrf_grid_25_cubed(self, dims_m):
        zones = create_standard_zones(PhotStandard.ACGIH, dims_m)
        zone_map = {z.zone_id: z for z in zones}
        assert tuple(zone_map["WholeRoomFluence"].num_points) == (25, 25, 25)

    def test_spacing_capped_at_max_points(self):
        """Very large room -> spacing increases so points per dim <= 200."""
        dims = RoomDimensions(polygon=Polygon2D.rectangle(1000, 1000), z=100)
        zones = create_standard_zones(PhotStandard.ACGIH, dims)
        for zone in zones:
            for n in zone.num_points:
                assert n <= MAX_POINTS_PER_DIM
