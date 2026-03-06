"""Tests for CalcPlane and CalcVol classes."""

import pytest
import numpy as np
from datetime import timedelta
from guv_calcs import CalcPlane, CalcVol


class TestCalcPlaneCreation:
    """Tests for CalcPlane initialization."""

    def test_default_creation(self):
        """CalcPlane should be created with defaults."""
        plane = CalcPlane(zone_id="TestPlane")
        assert plane.zone_id == "TestPlane"
        assert plane.calctype == "Plane"

    def test_custom_dimensions(self):
        """CalcPlane should accept custom dimensions."""
        plane = CalcPlane(
            zone_id="TestPlane",
            x1=0, x2=10,
            y1=0, y2=8,
            height=1.5,
        )
        assert plane.x1 == 0
        assert plane.x2 == 10
        assert plane.y1 == 0
        assert plane.y2 == 8
        assert plane.height == 1.5

    def test_custom_spacing(self):
        """CalcPlane should accept custom spacing."""
        plane = CalcPlane(
            zone_id="TestPlane",
            x_spacing=0.25,
            y_spacing=0.25,
        )
        assert plane.x_spacing == 0.25
        assert plane.y_spacing == 0.25

    def test_constructor_allows_zero_max_bounds(self):
        """Legacy constructor should preserve explicit zero upper bounds."""
        plane = CalcPlane(x1=0, x2=0, y1=0, y2=0, height=0)
        assert plane.x2 == 0
        assert plane.y2 == 0

    def test_num_points(self, calc_plane):
        """CalcPlane should have calculated num_points."""
        assert calc_plane.num_x > 0
        assert calc_plane.num_y > 0

    def test_coords_shape(self, calc_plane):
        """CalcPlane coords should have correct shape."""
        coords = calc_plane.coords
        assert coords.shape[1] == 3  # x, y, z columns
        total_points = calc_plane.num_x * calc_plane.num_y
        assert coords.shape[0] == total_points


class TestCalcPlaneProperties:
    """Tests for CalcPlane properties."""

    def test_dose_mode_default_false(self):
        """CalcPlane dose mode should default to False."""
        plane = CalcPlane(zone_id="TestPlane")
        assert plane.dose is False

    def test_dose_mode_enabled(self, calc_plane_dose):
        """CalcPlane dose mode can be enabled."""
        assert calc_plane_dose.dose is True
        assert calc_plane_dose.hours == 8.0

    def test_units_irradiance(self, calc_plane):
        """CalcPlane units should be uW/cm² in irradiance mode."""
        assert calc_plane.value_units == "uW/cm²"

    def test_units_dose(self, calc_plane_dose):
        """CalcPlane units should be mJ/cm² in dose mode."""
        assert calc_plane_dose.value_units == "mJ/cm²"

    def test_enabled_default_true(self):
        """CalcPlane enabled should default to True."""
        plane = CalcPlane(zone_id="TestPlane")
        assert plane.enabled is True

    def test_fov_defaults(self):
        """CalcPlane should have default FOV values."""
        plane = CalcPlane(zone_id="TestPlane")
        assert plane.fov_vert == 180
        assert plane.fov_horiz == 360


class TestCalcPlaneModification:
    """Tests for CalcPlane modification methods."""

    def test_set_height(self, calc_plane):
        """set_height() should update plane height."""
        calc_plane.set_height(2.0)
        assert calc_plane.height == 2.0

    def test_set_spacing(self, calc_plane):
        """set_spacing() should update spacing."""
        calc_plane.set_spacing(x_spacing=0.1, y_spacing=0.1)
        assert calc_plane.x_spacing == 0.1
        assert calc_plane.y_spacing == 0.1

    def test_set_dimensions_allows_zero_values(self):
        """set_dimensions() should treat 0 as explicit input, not fallback."""
        plane = CalcPlane(x1=1, x2=2, y1=1, y2=2, height=1.0)
        plane.set_dimensions(x1=0, y1=0)
        assert plane.x1 == 0
        assert plane.y1 == 0

    def test_set_value_type(self, calc_plane):
        """set_value_type() should toggle dose mode."""
        calc_plane.set_value_type(True)
        assert calc_plane.dose is True
        calc_plane.set_value_type(False)
        assert calc_plane.dose is False

    def test_set_dose_time(self, calc_plane):
        """set_dose_time() should update exposure time."""
        calc_plane.set_dose_time(hours=4.0)
        assert calc_plane.hours == 4.0


class TestCalcVolCreation:
    """Tests for CalcVol initialization."""

    def test_default_creation(self):
        """CalcVol should be created with defaults."""
        vol = CalcVol(zone_id="TestVol")
        assert vol.zone_id == "TestVol"
        assert vol.calctype == "Volume"

    def test_custom_dimensions(self):
        """CalcVol should accept custom dimensions."""
        vol = CalcVol(
            zone_id="TestVol",
            x1=0, x2=10,
            y1=0, y2=8,
            z1=0, z2=3.0,
        )
        assert vol.x1 == 0
        assert vol.x2 == 10
        assert vol.y1 == 0
        assert vol.y2 == 8
        assert vol.z1 == 0
        assert vol.z2 == 3.0

    def test_constructor_allows_zero_max_bounds(self):
        """Legacy constructor should preserve explicit zero upper bounds."""
        vol = CalcVol(x1=0, x2=0, y1=0, y2=0, z1=0, z2=0)
        assert vol.x2 == 0
        assert vol.y2 == 0
        assert vol.z2 == 0

    def test_custom_spacing(self):
        """CalcVol should accept custom spacing."""
        vol = CalcVol(
            zone_id="TestVol",
            x_spacing=0.25,
            y_spacing=0.25,
            z_spacing=0.25,
        )
        assert vol.x_spacing == 0.25
        assert vol.y_spacing == 0.25
        assert vol.z_spacing == 0.25

    def test_num_points(self, calc_volume):
        """CalcVol should have calculated num_points."""
        assert calc_volume.num_x > 0
        assert calc_volume.num_y > 0
        assert calc_volume.num_z > 0

    def test_coords_shape(self, calc_volume):
        """CalcVol coords should have correct shape."""
        coords = calc_volume.coords
        assert coords.shape[1] == 3  # x, y, z columns
        total_points = calc_volume.num_x * calc_volume.num_y * calc_volume.num_z
        assert coords.shape[0] == total_points


class TestCalcVolProperties:
    """Tests for CalcVol properties."""

    def test_dose_mode_default_false(self):
        """CalcVol dose mode should default to False."""
        vol = CalcVol(zone_id="TestVol")
        assert vol.dose is False

    def test_enabled_default_true(self):
        """CalcVol enabled should default to True."""
        vol = CalcVol(zone_id="TestVol")
        assert vol.enabled is True


class TestCalcZoneSerialization:
    """Tests for CalcPlane and CalcVol serialization."""

    def test_plane_to_dict(self, calc_plane):
        """CalcPlane should serialize to dict."""
        data = calc_plane.to_dict()
        assert data["zone_id"] == "TestPlane"
        assert data["calctype"] == "Plane"
        assert "geometry" in data

    def test_plane_from_dict_round_trip(self, calc_plane):
        """CalcPlane should survive to_dict/from_dict round trip."""
        data = calc_plane.to_dict()
        loaded = CalcPlane.from_dict(data)
        assert loaded.zone_id == calc_plane.zone_id
        assert loaded.height == calc_plane.height

    def test_volume_to_dict(self, calc_volume):
        """CalcVol should serialize to dict."""
        data = calc_volume.to_dict()
        assert data["zone_id"] == "TestVolume"
        assert data["calctype"] == "Volume"
        assert "geometry" in data

    def test_volume_from_dict_round_trip(self, calc_volume):
        """CalcVol should survive to_dict/from_dict round trip."""
        data = calc_volume.to_dict()
        loaded = CalcVol.from_dict(data)
        assert loaded.zone_id == calc_volume.zone_id
        assert loaded.z2 == calc_volume.z2

    def test_plane_equality(self):
        """Two CalcPlanes with same properties should be equal."""
        plane1 = CalcPlane(zone_id="Test", x1=0, x2=6, y1=0, y2=4, height=1.8)
        plane2 = CalcPlane(zone_id="Test", x1=0, x2=6, y1=0, y2=4, height=1.8)
        assert plane1 == plane2

    def test_volume_equality(self):
        """Two CalcVols with same properties should be equal."""
        vol1 = CalcVol(zone_id="Test", x1=0, x2=6, y1=0, y2=4, z1=0, z2=2.7)
        vol2 = CalcVol(zone_id="Test", x1=0, x2=6, y1=0, y2=4, z1=0, z2=2.7)
        assert vol1 == vol2


class TestCalcZoneCopy:
    """Tests for CalcZone copy functionality."""

    def test_plane_copy(self, calc_plane):
        """CalcPlane copy should be independent of original."""
        copy = calc_plane.copy(zone_id="CopiedPlane")
        assert copy.zone_id == "CopiedPlane"
        copy.set_height(3.0)
        assert calc_plane.height != copy.height

    def test_volume_copy(self, calc_volume):
        """CalcVol copy should be independent of original."""
        copy = calc_volume.copy(zone_id="CopiedVolume")
        assert copy.zone_id == "CopiedVolume"


class TestExposureTime:
    """Tests for exposure time parameters."""

    def test_default_hours(self):
        """Default exposure time is 8 hours."""
        plane = CalcPlane(zone_id="T")
        assert plane.hours == 8.0
        assert plane.exposure_time == timedelta(hours=8)

    def test_hours_param(self):
        """hours= sets exposure time in hours."""
        plane = CalcPlane(zone_id="T", hours=4)
        assert plane.hours == 4.0
        assert plane.seconds == 14400.0

    def test_minutes_param(self):
        """minutes= sets exposure time in minutes."""
        plane = CalcPlane(zone_id="T", minutes=30)
        assert plane.hours == 0.5
        assert plane.minutes == 30.0

    def test_seconds_param(self):
        """seconds= sets exposure time in seconds."""
        plane = CalcPlane(zone_id="T", seconds=3600)
        assert plane.hours == 1.0

    def test_combined_time(self):
        """hours + minutes + seconds sum together."""
        plane = CalcPlane(zone_id="T", hours=1, minutes=30)
        assert plane.exposure_time == timedelta(hours=1, minutes=30)
        assert plane.seconds == 5400.0

    def test_combined_time_all_three(self):
        """All three time params sum together."""
        plane = CalcPlane(zone_id="T", hours=1, minutes=30, seconds=45)
        assert plane.exposure_time == timedelta(hours=1, minutes=30, seconds=45)

    def test_set_dose_time_combined(self):
        """set_dose_time with multiple params sums them."""
        plane = CalcPlane(zone_id="T")
        plane.set_dose_time(hours=2, minutes=15)
        assert plane.exposure_time == timedelta(hours=2, minutes=15)

    def test_negative_raises(self):
        """Negative exposure time raises ValueError."""
        with pytest.raises(ValueError):
            CalcPlane(zone_id="T", hours=-1)

    def test_non_numeric_raises(self):
        """Non-numeric exposure time raises TypeError."""
        with pytest.raises(TypeError):
            CalcPlane(zone_id="T", hours="eight")

    def test_exposure_time_property(self):
        """exposure_time returns a timedelta."""
        plane = CalcPlane(zone_id="T", minutes=30)
        assert plane.exposure_time == timedelta(minutes=30)

    def test_exposure_time_setter(self):
        """exposure_time can be set with a timedelta."""
        plane = CalcPlane(zone_id="T")
        plane.exposure_time = timedelta(minutes=45)
        assert plane.hours == 0.75

    def test_set_dose_time_hours(self):
        """set_dose_time accepts hours kwarg."""
        plane = CalcPlane(zone_id="T")
        plane.set_dose_time(hours=4)
        assert plane.hours == 4.0

    def test_set_dose_time_minutes(self):
        """set_dose_time accepts minutes kwarg."""
        plane = CalcPlane(zone_id="T")
        plane.set_dose_time(minutes=30)
        assert plane.hours == 0.5

    def test_serialization_round_trip(self):
        """exposure_time survives to_dict/from_dict."""
        plane = CalcPlane(zone_id="T", dose=True, minutes=30)
        data = plane.to_dict()
        loaded = CalcPlane.from_dict(data)
        assert loaded.exposure_time == timedelta(minutes=30)

    def test_old_dict_migration(self):
        """Old dicts with 'hours' key should load correctly."""
        old_data = {
            "zone_id": "T",
            "dose": True,
            "hours": 4.0,
            "x1": 0, "x2": 6, "y1": 0, "y2": 4, "height": 1.0,
        }
        plane = CalcPlane.from_dict(old_data)
        assert plane.hours == 4.0

    def test_calcvol_exposure_time(self):
        """CalcVol also supports hours/minutes/seconds."""
        vol = CalcVol(zone_id="T", minutes=15)
        assert vol.minutes == 15.0
        assert vol.hours == 0.25


class TestCalcZoneConvertUnits:
    """Tests for CalcZone.convert_units preserving calculated values."""

    def test_convert_units_preserves_values(self, calculated_room):
        """Changing units should not clear calculated values."""
        zone = calculated_room.calc_zones["WholeRoomFluence"]
        values_before = zone.values.copy()
        zone.convert_units("meters", "feet")
        assert zone.values is not None
        assert np.allclose(values_before, zone.values)

    def test_convert_units_updates_geometry(self, calculated_room):
        """Geometry coordinates should scale with unit conversion."""
        zone = calculated_room.calc_zones["WholeRoomFluence"]
        spacing_before = zone.geometry.spacing
        zone.convert_units("meters", "feet")
        factor = 1.0 / 0.3048
        for old, new in zip(spacing_before, zone.geometry.spacing):
            assert np.isclose(new, old * factor, rtol=1e-4)

    def test_convert_units_updates_cache(self, calculated_room):
        """Cache calc_state should match new geometry after conversion."""
        zone = calculated_room.calc_zones["WholeRoomFluence"]
        zone.convert_units("meters", "feet")
        assert zone.calculator.cache.calc_state == zone.calc_state

    def test_convert_units_noop_without_geometry(self):
        """Should not error when geometry is None."""
        zone = CalcPlane(zone_id="test")
        zone._geometry = None
        zone.convert_units("meters", "feet")  # should not raise
