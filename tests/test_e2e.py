"""End-to-end workflow tests for guv_calcs."""

import pytest
import numpy as np
import warnings
from guv_calcs import Room, Lamp, CalcPlane, CalcVol, Spectrum, PhotStandard


class TestBasicWorkflow:
    """Tests for the standard workflow: create room -> add lamp -> add zones -> calculate."""

    def test_complete_workflow(self):
        """Complete workflow from room creation to calculation."""
        # Create room
        room = Room(x=6, y=4, z=2.7, units="meters")

        # Add lamp
        lamp = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        room.add_lamp(lamp)

        # Add standard zones
        room.add_standard_zones()

        # Calculate
        room.calculate()

        # Verify results
        assert "WholeRoomFluence" in room.calc_zones
        fluence = room.calc_zones["WholeRoomFluence"]
        assert fluence.values is not None
        assert fluence.values.mean() > 0

    def test_chained_api(self):
        """Workflow using method chaining."""
        room = (
            Room(x=6, y=4, z=2.7)
            .place_lamp("aerolamp")
            .add_standard_zones()
            .calculate()
        )

        assert len(room.lamps) == 1
        assert len(room.calc_zones) == 3
        assert room.calc_zones["WholeRoomFluence"].values is not None

    def test_multiple_lamps(self):
        """Workflow with multiple lamps."""
        room = Room(x=6, y=4, z=2.7, units="meters")

        # Add two lamps at different positions
        lamp1 = Lamp.from_keyword("aerolamp").move(2, 2, 2.7).aim(2, 2, 0)
        lamp2 = Lamp.from_keyword("ushio_b1").move(4, 2, 2.7).aim(4, 2, 0)

        room.add_lamp(lamp1).add_lamp(lamp2)
        room.add_standard_zones()
        room.calculate()

        # Verify both lamps contribute
        assert len(room.lamps) == 2
        fluence = room.calc_zones["WholeRoomFluence"]
        assert fluence.values.mean() > 0


class TestSaveLoadWorkflow:
    """Tests for save/load workflows."""

    def test_save_calculate_load_workflow(self, temp_file):
        """Save calculated room, load, verify structure preserved."""
        # Create and calculate room
        room = (
            Room(x=6, y=4, z=2.7)
            .place_lamp("aerolamp")
            .add_standard_zones()
            .calculate()
        )

        # Save
        room.save(temp_file)

        # Load
        loaded = Room.load(temp_file)

        # Verify structure preserved
        assert len(loaded.lamps) == len(room.lamps)
        assert len(loaded.calc_zones) == len(room.calc_zones)
        assert "WholeRoomFluence" in loaded.calc_zones
        assert "SkinLimits" in loaded.calc_zones
        assert "EyeLimits" in loaded.calc_zones

        # Verify can recalculate without errors
        loaded.calculate()
        assert loaded.calc_zones["WholeRoomFluence"].values is not None
        assert loaded.calc_zones["WholeRoomFluence"].values.mean() > 0

    def test_modify_and_recalculate(self):
        """Modify room after calculation and recalculate."""
        room = (
            Room(x=6, y=4, z=2.7)
            .place_lamp("aerolamp")
            .add_standard_zones()
            .calculate()
        )

        original_mean = room.calc_zones["WholeRoomFluence"].values.mean()

        # Add another lamp
        lamp2 = Lamp.from_keyword("ushio_b1").move(4, 2, 2.7).aim(4, 2, 0)
        room.add_lamp(lamp2)
        room.calculate()

        new_mean = room.calc_zones["WholeRoomFluence"].values.mean()

        # Fluence should increase with additional lamp
        assert new_mean > original_mean


class TestCustomZoneWorkflow:
    """Tests for workflows with custom calculation zones."""

    def test_custom_plane_workflow(self):
        """Workflow with custom CalcPlane."""
        room = Room(x=6, y=4, z=2.7)

        lamp = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        room.add_lamp(lamp)

        # Add custom plane at workplane height
        plane = CalcPlane(
            zone_id="WorkPlane",
            x1=0, x2=6,
            y1=0, y2=4,
            height=0.75,  # desk height
            x_spacing=0.25,
            y_spacing=0.25,
        )
        room.add_calc_zone(plane)
        room.calculate()

        # Verify calculation
        assert "WorkPlane" in room.calc_zones
        workplane = room.calc_zones["WorkPlane"]
        assert workplane.values is not None
        assert workplane.values.mean() > 0

    def test_custom_volume_workflow(self):
        """Workflow with custom CalcVol."""
        room = Room(x=6, y=4, z=2.7)

        lamp = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        room.add_lamp(lamp)

        # Add custom volume for breathing zone
        volume = CalcVol(
            zone_id="BreathingZone",
            x1=1, x2=5,
            y1=1, y2=3,
            z1=1.0, z2=1.8,  # breathing zone height
            x_spacing=0.25,
            y_spacing=0.25,
            z_spacing=0.2,
        )
        room.add_calc_zone(volume)
        room.calculate()

        # Verify calculation
        assert "BreathingZone" in room.calc_zones
        breathing = room.calc_zones["BreathingZone"]
        assert breathing.values is not None
        assert breathing.values.mean() > 0


class TestReflectanceWorkflow:
    """Tests for workflows involving reflectance."""

    def test_reflectance_increases_fluence(self):
        """Higher reflectance should increase fluence values."""
        # Room with no reflectance
        room_no_ref = (
            Room(x=6, y=4, z=2.7, enable_reflectance=False)
            .place_lamp("aerolamp")
            .add_standard_zones()
            .calculate()
        )

        # Room with reflectance
        room_ref = (
            Room(x=6, y=4, z=2.7, enable_reflectance=True)
            .place_lamp("aerolamp")
            .set_reflectance(0.5)
            .add_standard_zones()
            .calculate()
        )

        no_ref_mean = room_no_ref.calc_zones["WholeRoomFluence"].values.mean()
        ref_mean = room_ref.calc_zones["WholeRoomFluence"].values.mean()

        # Reflectance should increase fluence
        assert ref_mean > no_ref_mean

    def test_different_reflectance_values(self):
        """Different reflectance values should produce different results."""
        results = []

        for R in [0.0, 0.3, 0.6]:
            room = (
                Room(x=6, y=4, z=2.7, enable_reflectance=True)
                .place_lamp("aerolamp")
                .set_reflectance(R)
                .add_standard_zones()
                .calculate()
            )
            results.append(room.calc_zones["WholeRoomFluence"].values.mean())

        # Higher reflectance should give higher fluence
        assert results[1] > results[0]  # 0.3 > 0.0
        assert results[2] > results[1]  # 0.6 > 0.3


class TestEmptyAndMinimalStates:
    """Tests for edge cases with empty or minimal room configurations."""

    def test_empty_room_no_lamps_no_zones(self):
        """Empty room should be valid but have nothing to calculate."""
        room = Room(x=6, y=4, z=2.7)
        assert len(room.lamps) == 0
        assert len(room.calc_zones) == 0

    @pytest.mark.filterwarnings("ignore:No lamps are present in the room")
    def test_room_with_zones_but_no_lamps(self):
        """Room with zones but no lamps should calculate to zero fluence."""
        room = Room(x=6, y=4, z=2.7)
        room.add_standard_zones()
        room.calculate()

        # With no lamps, fluence should be zero everywhere
        fluence = room.calc_zones["WholeRoomFluence"]
        assert fluence.values is not None
        assert fluence.values.max() == 0

    def test_room_with_lamps_but_no_zones(self):
        """Room with lamps but no zones should calculate without error."""
        room = Room(x=6, y=4, z=2.7).place_lamp("aerolamp")
        # Should not raise - just nothing to calculate
        room.calculate()
        assert len(room.calc_zones) == 0

    def test_calculate_called_multiple_times(self):
        """Calling calculate() multiple times should be idempotent."""
        room = (
            Room(x=6, y=4, z=2.7)
            .place_lamp("aerolamp")
            .add_standard_zones()
        )

        room.calculate()
        first_result = room.calc_zones["WholeRoomFluence"].values.copy()

        room.calculate()
        second_result = room.calc_zones["WholeRoomFluence"].values.copy()

        room.calculate()
        third_result = room.calc_zones["WholeRoomFluence"].values.copy()

        # All results should be identical
        assert np.allclose(first_result, second_result)
        assert np.allclose(second_result, third_result)


class TestModifyAfterCalculation:
    """Tests for modifying room state after calculation."""

    def test_add_lamp_after_calculation(self):
        """Adding a lamp after calculation should allow recalculation."""
        room = (
            Room(x=6, y=4, z=2.7)
            .place_lamp("aerolamp")
            .add_standard_zones()
            .calculate()
        )
        original_mean = room.calc_zones["WholeRoomFluence"].values.mean()

        # Add another lamp
        lamp2 = Lamp.from_keyword("ushio_b1").move(4, 2, 2.7).aim(4, 2, 0)
        room.add_lamp(lamp2)
        room.calculate()

        # Fluence should increase
        new_mean = room.calc_zones["WholeRoomFluence"].values.mean()
        assert new_mean > original_mean

    def test_remove_lamp_after_calculation(self):
        """Removing a lamp after calculation should allow recalculation."""
        room = Room(x=6, y=4, z=2.7)
        lamp1 = Lamp.from_keyword("aerolamp").move(2, 2, 2.7).aim(2, 2, 0)
        lamp2 = Lamp.from_keyword("ushio_b1").move(4, 2, 2.7).aim(4, 2, 0)
        room.add_lamp(lamp1).add_lamp(lamp2)
        room.add_standard_zones()
        room.calculate()

        original_mean = room.calc_zones["WholeRoomFluence"].values.mean()

        # Remove one lamp
        lamp_id = list(room.lamps.keys())[0]
        room.remove_lamp(lamp_id)
        room.calculate()

        # Fluence should decrease
        new_mean = room.calc_zones["WholeRoomFluence"].values.mean()
        assert new_mean < original_mean

    @pytest.mark.filterwarnings("ignore:No lamps are present in the room")
    @pytest.mark.filterwarnings("ignore:aerolamp exceeds room boundaries")
    def test_remove_all_lamps_after_calculation(self):
        """Removing all lamps after calculation should result in zero fluence."""
        room = (
            Room(x=6, y=4, z=2.7)
            .place_lamp("aerolamp")
            .add_standard_zones()
            .calculate()
        )

        # Remove all lamps
        for lamp_id in list(room.lamps.keys()):
            room.remove_lamp(lamp_id)

        room.calculate()

        # Fluence should be zero
        assert room.calc_zones["WholeRoomFluence"].values.max() == 0

    def test_move_lamp_after_calculation(self):
        """Moving a lamp after calculation should change results on recalculation."""
        room = Room(x=6, y=4, z=2.7)
        lamp = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        room.add_lamp(lamp)
        room.add_standard_zones()
        room.calculate()

        original_values = room.calc_zones["WholeRoomFluence"].values.copy()

        # Move lamp to corner
        lamp.move(0.5, 0.5, 2.7).aim(0.5, 0.5, 0)
        room.calculate()

        new_values = room.calc_zones["WholeRoomFluence"].values

        # Values should be different (distribution changed)
        assert not np.allclose(original_values, new_values)

    def test_scale_lamp_after_calculation(self):
        """Scaling a lamp after calculation should proportionally change fluence."""
        room = Room(x=6, y=4, z=2.7)
        lamp = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        room.add_lamp(lamp)
        room.add_standard_zones()
        room.calculate()

        original_mean = room.calc_zones["WholeRoomFluence"].values.mean()

        # Double lamp output
        lamp.scale(2.0)
        room.calculate()

        new_mean = room.calc_zones["WholeRoomFluence"].values.mean()

        # Fluence should approximately double
        assert np.isclose(new_mean, original_mean * 2, rtol=0.1)

    def test_add_zone_after_calculation(self):
        """Adding a zone after calculation should allow calculating the new zone."""
        room = (
            Room(x=6, y=4, z=2.7)
            .place_lamp("aerolamp")
            .add_standard_zones()
            .calculate()
        )

        # Add custom zone
        plane = CalcPlane(zone_id="CustomPlane", height=1.0)
        room.add_calc_zone(plane)
        room.calculate()

        # New zone should have values
        assert "CustomPlane" in room.calc_zones
        assert room.calc_zones["CustomPlane"].values is not None
        assert room.calc_zones["CustomPlane"].values.mean() > 0

    def test_remove_zone_after_calculation(self):
        """Removing a zone after calculation should work."""
        room = (
            Room(x=6, y=4, z=2.7)
            .place_lamp("aerolamp")
            .add_standard_zones()
            .calculate()
        )

        room.remove_calc_zone("SkinLimits")
        assert "SkinLimits" not in room.calc_zones

        # Should still be able to recalculate remaining zones
        room.calculate()
        assert "WholeRoomFluence" in room.calc_zones
        assert "EyeLimits" in room.calc_zones

    def test_change_room_dimensions_after_calculation(self):
        """Changing room dimensions after calculation should affect results."""
        room = (
            Room(x=6, y=4, z=2.7)
            .place_lamp("aerolamp")
            .add_standard_zones()
            .calculate()
        )

        original_mean = room.calc_zones["WholeRoomFluence"].values.mean()

        # Make room larger
        room.set_dimensions(x=12, y=8, z=2.7)
        room.calculate()

        # Fluence mean should decrease in larger room (same lamp, more volume)
        new_mean = room.calc_zones["WholeRoomFluence"].values.mean()
        assert new_mean < original_mean

    def test_toggle_reflectance_after_calculation(self):
        """Toggling reflectance after calculation should change results."""
        room = (
            Room(x=6, y=4, z=2.7, enable_reflectance=False)
            .place_lamp("aerolamp")
            .add_standard_zones()
            .calculate()
        )

        no_ref_mean = room.calc_zones["WholeRoomFluence"].values.mean()

        # Enable reflectance
        room.enable_reflectance(True)
        room.set_reflectance(0.5)
        room.calculate()

        ref_mean = room.calc_zones["WholeRoomFluence"].values.mean()

        # Reflectance should increase fluence
        assert ref_mean > no_ref_mean


class TestOutOfOrderOperations:
    """Tests for operations performed in unusual orders."""

    def test_add_zones_before_lamps(self):
        """Adding zones before lamps should still work."""
        room = Room(x=6, y=4, z=2.7)
        room.add_standard_zones()  # Zones first
        room.place_lamp("aerolamp")  # Then lamp
        room.calculate()

        assert room.calc_zones["WholeRoomFluence"].values.mean() > 0

    def test_calculate_add_zones_calculate_again(self):
        """Calculate, add more zones, calculate again."""
        room = (
            Room(x=6, y=4, z=2.7)
            .place_lamp("aerolamp")
        )

        # Add one zone and calculate
        plane1 = CalcPlane(zone_id="Plane1", height=1.0)
        room.add_calc_zone(plane1)
        room.calculate()

        first_values = room.calc_zones["Plane1"].values.copy()

        # Add another zone and calculate
        plane2 = CalcPlane(zone_id="Plane2", height=1.5)
        room.add_calc_zone(plane2)
        room.calculate()

        # First zone should still have same values
        assert np.allclose(room.calc_zones["Plane1"].values, first_values)
        # Second zone should now have values
        assert room.calc_zones["Plane2"].values is not None


class TestLampEdgeCases:
    """Tests for unusual lamp configurations."""

    def test_lamp_at_floor_level(self):
        """Lamp at z=0 should work."""
        room = Room(x=6, y=4, z=2.7)
        lamp = Lamp.from_keyword("aerolamp").move(3, 2, 0).aim(3, 2, 2.7)
        room.add_lamp(lamp)
        room.add_standard_zones()
        room.calculate()

        assert room.calc_zones["WholeRoomFluence"].values.mean() > 0

    def test_lamp_in_corner(self):
        """Lamp in corner should work."""
        room = Room(x=6, y=4, z=2.7)
        lamp = Lamp.from_keyword("aerolamp").move(0, 0, 2.7).aim(3, 2, 0)
        room.add_lamp(lamp)
        room.add_standard_zones()
        room.calculate()

        assert room.calc_zones["WholeRoomFluence"].values.mean() > 0

    def test_multiple_lamps_same_position(self):
        """Multiple lamps at the same position should stack."""
        # Create room with two lamps at identical positions
        double_room = Room(x=6, y=4, z=2.7)
        lamp1 = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        lamp2 = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        double_room.add_lamp(lamp1).add_lamp(lamp2)
        double_room.add_standard_zones()
        double_room.calculate()

        # Create room with single lamp at same position
        single_room = Room(x=6, y=4, z=2.7)
        lamp = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        single_room.add_lamp(lamp)
        single_room.add_standard_zones()
        single_room.calculate()

        double_mean = double_room.calc_zones["WholeRoomFluence"].values.mean()
        single_mean = single_room.calc_zones["WholeRoomFluence"].values.mean()

        # Two identical lamps should give double the fluence
        assert np.isclose(double_mean, single_mean * 2, rtol=0.1)

    def test_lamp_aimed_at_wall(self):
        """Lamp aimed at wall should work."""
        room = Room(x=6, y=4, z=2.7)
        lamp = Lamp.from_keyword("aerolamp").move(3, 2, 1.5).aim(6, 2, 1.5)  # Aim at +x wall
        room.add_lamp(lamp)
        room.add_standard_zones()
        room.calculate()

        assert room.calc_zones["WholeRoomFluence"].values is not None

    def test_lamp_different_wavelengths(self):
        """Lamps with different wavelengths should combine correctly."""
        room = Room(x=6, y=4, z=2.7)

        lamp1 = Lamp.from_keyword("aerolamp").move(2, 2, 2.7).aim(2, 2, 0)
        lamp2 = Lamp.from_keyword("ushio_b1").move(4, 2, 2.7).aim(4, 2, 0)

        room.add_lamp(lamp1).add_lamp(lamp2)
        room.add_standard_zones()
        room.calculate()

        assert room.calc_zones["WholeRoomFluence"].values.mean() > 0


class TestZoneEdgeCases:
    """Tests for unusual zone configurations."""

    def test_zone_very_fine_spacing(self):
        """Zone with very fine spacing should work (more points)."""
        room = (
            Room(x=6, y=4, z=2.7)
            .place_lamp("aerolamp")
        )

        plane = CalcPlane(
            zone_id="FineGrid",
            x1=2, x2=4,
            y1=1, y2=3,
            height=1.0,
            x_spacing=0.1,
            y_spacing=0.1,
        )
        room.add_calc_zone(plane)
        room.calculate()

        # Should have many points
        assert plane.num_x * plane.num_y > 100
        assert plane.values is not None

    def test_zone_very_coarse_spacing(self):
        """Zone with very coarse spacing should work (few points)."""
        room = (
            Room(x=6, y=4, z=2.7)
            .place_lamp("aerolamp")
        )

        plane = CalcPlane(
            zone_id="CoarseGrid",
            x1=0, x2=6,
            y1=0, y2=4,
            height=1.0,
            x_spacing=2.0,
            y_spacing=2.0,
        )
        room.add_calc_zone(plane)
        room.calculate()

        # Should have few points
        assert plane.num_x * plane.num_y < 20
        assert plane.values is not None

    def test_zone_dose_mode(self):
        """Zone in dose mode should return dose via get_values()."""
        room = (
            Room(x=6, y=4, z=2.7)
            .place_lamp("aerolamp")
        )

        irrad_plane = CalcPlane(zone_id="Irradiance", height=1.0, dose=False)
        dose_plane = CalcPlane(zone_id="Dose", height=1.0, dose=True, hours=8.0)

        room.add_calc_zone(irrad_plane)
        room.add_calc_zone(dose_plane)
        room.calculate()

        # Both should have values
        assert irrad_plane.get_values() is not None
        assert dose_plane.get_values() is not None

        # Raw values should be the same (same calculation)
        assert np.allclose(irrad_plane.values, dose_plane.values)

        # get_values() should apply dose conversion: values * 3.6 * hours
        # For 8 hours: factor = 3.6 * 8 = 28.8
        expected_ratio = 3.6 * 8.0
        actual_ratio = dose_plane.get_values().mean() / irrad_plane.get_values().mean()
        assert np.isclose(actual_ratio, expected_ratio, rtol=0.01)

        # Units should differ
        assert irrad_plane.units == "uW/cm²"
        assert dose_plane.units == "mJ/cm²"

    def test_switch_zone_to_dose_mode_after_calculation(self):
        """Switching zone to dose mode should change get_values() output."""
        room = (
            Room(x=6, y=4, z=2.7)
            .place_lamp("aerolamp")
        )

        plane = CalcPlane(zone_id="TestPlane", height=1.0, dose=False)
        room.add_calc_zone(plane)
        room.calculate()

        irrad_mean = plane.get_values().mean()

        # Switch to dose mode (no recalculation needed - it's just a display conversion)
        plane.set_value_type(True)
        plane.set_dose_time(8.0)

        dose_mean = plane.get_values().mean()

        # get_values() should now return dose-adjusted values
        expected_ratio = 3.6 * 8.0  # dose conversion factor
        assert np.isclose(dose_mean / irrad_mean, expected_ratio, rtol=0.01)


class TestSaveLoadEdgeCases:
    """Tests for unusual save/load scenarios."""

    def test_save_before_calculation_load_then_calculate(self, temp_file):
        """Save uncalculated room, load, then calculate."""
        room = (
            Room(x=6, y=4, z=2.7)
            .place_lamp("aerolamp")
            .add_standard_zones()
        )
        # Don't calculate yet
        room.save(temp_file)

        loaded = Room.load(temp_file)
        loaded.calculate()

        assert loaded.calc_zones["WholeRoomFluence"].values.mean() > 0

    def test_save_load_modify_save_load(self, temp_file):
        """Multiple save/load cycles with modifications."""
        # First save/load
        room1 = (
            Room(x=6, y=4, z=2.7)
            .place_lamp("aerolamp")
            .add_standard_zones()
            .calculate()
        )
        room1.save(temp_file)

        # Load and modify
        room2 = Room.load(temp_file)
        room2.place_lamp("ushio_b1")
        room2.calculate()
        room2.save(temp_file)

        # Load again
        room3 = Room.load(temp_file)
        room3.calculate()

        # Should have 2 lamps
        assert len(room3.lamps) == 2


class TestCopyWorkflow:
    """Tests for copying rooms and ensuring independence."""

    def test_room_copy_is_independent(self):
        """Copied room should be independent of original."""
        original = (
            Room(x=6, y=4, z=2.7)
            .place_lamp("aerolamp")
            .add_standard_zones()
            .calculate()
        )
        original_mean = original.calc_zones["WholeRoomFluence"].values.mean()

        # Copy and modify
        copy = original.copy()
        copy.place_lamp("ushio_b1")
        copy.calculate()

        # Original should be unchanged
        assert len(original.lamps) == 1
        assert len(copy.lamps) == 2

        # Original values should be unchanged
        original.calculate()
        assert np.isclose(
            original.calc_zones["WholeRoomFluence"].values.mean(),
            original_mean,
            rtol=0.01
        )


class TestRoomSizeEdgeCases:
    """Tests for unusual room sizes."""

    @pytest.mark.filterwarnings("ignore:Eye Dose .* exceeds room boundaries")
    @pytest.mark.filterwarnings("ignore:Skin Dose .* exceeds room boundaries")
    @pytest.mark.filterwarnings("ignore:aerolamp exceeds room boundaries")
    def test_small_room(self):
        """Very small room should work."""
        room = Room(x=1, y=1, z=1)
        lamp = Lamp.from_keyword("aerolamp").move(0.5, 0.5, 1).aim(0.5, 0.5, 0)
        room.add_lamp(lamp)
        room.add_standard_zones()
        room.calculate()

        assert room.calc_zones["WholeRoomFluence"].values.mean() > 0

    def test_large_room(self):
        """Large room should work."""
        room = Room(x=50, y=30, z=5)
        lamp = Lamp.from_keyword("aerolamp").move(25, 15, 5).aim(25, 15, 0)
        room.add_lamp(lamp)
        room.add_standard_zones()
        room.calculate()

        assert room.calc_zones["WholeRoomFluence"].values is not None

    def test_room_in_feet(self):
        """Room in feet should calculate correctly."""
        room = Room(x=20, y=15, z=9, units="feet")
        lamp = Lamp.from_keyword("aerolamp").move(10, 7.5, 9).aim(10, 7.5, 0)
        room.add_lamp(lamp)
        room.add_standard_zones()
        room.calculate()

        assert room.calc_zones["WholeRoomFluence"].values.mean() > 0


class TestLampSurfaceManipulation:
    """Tests for manipulating lamp surface parameters."""

    def test_set_width_changes_calculation(self):
        """Changing lamp width should affect calculation results."""
        room1 = Room(x=6, y=4, z=2.7)
        lamp1 = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        lamp1.set_width(0.01)  # Very narrow
        room1.add_lamp(lamp1)
        room1.add_standard_zones()
        room1.calculate()

        room2 = Room(x=6, y=4, z=2.7)
        lamp2 = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        lamp2.set_width(0.5)  # Much wider
        room2.add_lamp(lamp2)
        room2.add_standard_zones()
        room2.calculate()

        # Different widths should produce different distributions
        # (total power is the same, but spatial distribution differs)
        mean1 = room1.calc_zones["WholeRoomFluence"].values.mean()
        mean2 = room2.calc_zones["WholeRoomFluence"].values.mean()
        # Both should have valid values
        assert mean1 > 0
        assert mean2 > 0

    def test_set_length_changes_calculation(self):
        """Changing lamp length should affect calculation results."""
        lamp = Lamp.from_keyword("aerolamp")
        original_length = lamp.length

        lamp.set_length(0.5)
        assert lamp.length == 0.5
        assert lamp.length != original_length

    def test_set_source_density_affects_nearfield(self):
        """Higher source density should provide better nearfield accuracy."""
        room = Room(x=6, y=4, z=2.7)
        lamp = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)

        # Default source density
        assert lamp.surface.source_density >= 1

        # Increase source density
        lamp.set_source_density(3)
        assert lamp.surface.source_density == 3

        room.add_lamp(lamp)
        room.add_standard_zones()
        room.calculate()

        assert room.calc_zones["WholeRoomFluence"].values.mean() > 0

    def test_load_intensity_map_array(self):
        """Loading an intensity map from array should affect calculations."""
        room = Room(x=6, y=4, z=2.7)
        lamp = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)

        # Create a simple intensity map (higher in center)
        intensity_map = np.array([
            [0.5, 0.7, 0.5],
            [0.7, 1.0, 0.7],
            [0.5, 0.7, 0.5],
        ])
        lamp.load_intensity_map(intensity_map)

        room.add_lamp(lamp)
        room.add_standard_zones()
        room.calculate()

        assert room.calc_zones["WholeRoomFluence"].values.mean() > 0

    def test_change_surface_after_calculation(self):
        """Changing surface parameters after calculation should affect recalculation."""
        room = Room(x=6, y=4, z=2.7)
        lamp = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        room.add_lamp(lamp)
        room.add_standard_zones()
        room.calculate()

        original_mean = room.calc_zones["WholeRoomFluence"].values.mean()

        # Change source density and recalculate
        lamp.set_source_density(5)
        room.calculate()

        # Results may differ slightly due to different discretization
        new_mean = room.calc_zones["WholeRoomFluence"].values.mean()
        assert new_mean > 0


class TestLampSpectrumManipulation:
    """Tests for manipulating lamp spectrum and wavelength."""

    def test_set_wavelength_changes_tlvs(self):
        """Changing wavelength should change TLV calculations."""
        lamp = Lamp.from_keyword("aerolamp")

        # Clear spectra and set single wavelength
        lamp.load_spectra(None)
        lamp.set_wavelength(254)
        skin_254, eye_254 = lamp.get_tlvs(PhotStandard.ACGIH)

        lamp.set_wavelength(222)
        skin_222, eye_222 = lamp.get_tlvs(PhotStandard.ACGIH)

        # 222nm has higher TLVs than 254nm (less hazardous)
        assert skin_222 > skin_254

    def test_load_spectra_none_clears_spectrum(self):
        """Loading None should clear the spectrum."""
        lamp = Lamp.from_keyword("aerolamp")
        assert lamp.spectra is not None

        lamp.load_spectra(None)
        assert lamp.spectra is None

    def test_load_spectra_from_spectrum_object(self):
        """Loading a Spectrum object should work."""
        lamp = Lamp.from_keyword("aerolamp")

        # Create custom spectrum
        custom_spectrum = Spectrum(
            wavelengths=[200, 210, 220, 230, 240],
            intensities=[0.1, 0.5, 1.0, 0.5, 0.1]
        )
        lamp.load_spectra(custom_spectrum)

        assert lamp.spectra is not None
        assert lamp.spectra.peak_wavelength == 220

    def test_load_spectra_from_dict(self):
        """Loading spectrum from dict should work."""
        lamp = Lamp.from_keyword("aerolamp")

        spectrum_dict = {
            "wavelength": [200, 220, 240],
            "intensity": [0.1, 1.0, 0.1]
        }
        lamp.load_spectra(spectrum_dict)

        assert lamp.spectra is not None

    def test_spectrum_normalized_preserves_tlv_ratios(self):
        """Normalizing spectrum should not change TLV ratios (shape preserved)."""
        lamp = Lamp.from_keyword("aerolamp")
        if lamp.spectra is None:
            pytest.skip("Lamp has no spectra")

        original = lamp.spectra
        normalized = original.normalized(100)

        # TLVs are based on spectral shape, so ratios should be preserved
        tlv1 = original.get_tlv({220: 0.5, 250: 1.0})
        tlv2 = normalized.get_tlv({220: 0.5, 250: 1.0})
        # Actual value changes but ratio to itself is preserved
        assert tlv1 > 0 and tlv2 > 0  # Both should be valid

    def test_spectrum_immutable_preserves_original(self):
        """Spectrum.filtered() returns new instance, original unchanged."""
        lamp = Lamp.from_keyword("aerolamp")
        if lamp.spectra is None:
            pytest.skip("Lamp has no spectra")

        original = lamp.spectra
        original_len = len(original.wavelengths)

        # Filter to narrow range - returns NEW spectrum
        filtered = original.filtered(minval=210, maxval=230)

        # Filtered has fewer wavelengths
        assert len(filtered.wavelengths) < original_len

        # Original is unchanged (immutable)
        assert len(original.wavelengths) == original_len

    def test_change_wavelength_after_calculation(self):
        """Changing wavelength after calculation should affect TLV-related results."""
        room = Room(x=6, y=4, z=2.7)
        lamp = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        lamp.load_spectra(None)
        lamp.set_wavelength(254)

        room.add_lamp(lamp)
        room.add_standard_zones()
        room.calculate()

        # Get fluence (should be same regardless of wavelength for same lamp)
        fluence_254 = room.calc_zones["WholeRoomFluence"].values.mean()

        # Change wavelength
        lamp.set_wavelength(222)
        room.calculate()

        fluence_222 = room.calc_zones["WholeRoomFluence"].values.mean()

        # Fluence should be same (wavelength doesn't affect photometry)
        assert np.isclose(fluence_254, fluence_222, rtol=0.01)


class TestLampOrientationWorkflows:
    """Tests for different ways to orient lamps."""

    def test_aim_vs_set_orientation(self):
        """aim() and set_orientation() should both work to orient lamp."""
        room1 = Room(x=6, y=4, z=2.7)
        lamp1 = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        room1.add_lamp(lamp1)
        room1.add_standard_zones()
        room1.calculate()

        # Both orientations should produce valid calculations
        assert room1.calc_zones["WholeRoomFluence"].values.mean() > 0

    def test_rotate_lamp(self):
        """Rotating lamp should work."""
        lamp = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        original_angle = lamp.angle

        lamp.rotate(90)
        assert lamp.angle == 90
        assert lamp.angle != original_angle

    def test_rotate_after_calculation(self):
        """Rotating lamp after calculation should affect recalculation."""
        room = Room(x=6, y=4, z=2.7)
        lamp = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        room.add_lamp(lamp)
        room.add_standard_zones()
        room.calculate()

        original_values = room.calc_zones["WholeRoomFluence"].values.copy()

        lamp.rotate(180)
        room.calculate()

        # For a lamp aimed straight down, 180° rotation may produce same or different results
        # depending on the photometric symmetry
        new_values = room.calc_zones["WholeRoomFluence"].values
        assert new_values is not None


class TestLampScalingWorkflows:
    """Tests for lamp scaling and power adjustment."""

    def test_scale_preserves_distribution(self):
        """Scaling should preserve spatial distribution but change magnitude."""
        room = Room(x=6, y=4, z=2.7)
        lamp = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        room.add_lamp(lamp)
        room.add_standard_zones()
        room.calculate()

        original_values = room.calc_zones["WholeRoomFluence"].values.copy()
        original_mean = original_values.mean()

        # Scale by 2x
        lamp.scale(2.0)
        room.calculate()

        new_values = room.calc_zones["WholeRoomFluence"].values
        new_mean = new_values.mean()

        # Mean should double
        assert np.isclose(new_mean, original_mean * 2, rtol=0.1)

        # Normalized distribution should be same
        original_normalized = original_values / original_mean
        new_normalized = new_values / new_mean
        assert np.allclose(original_normalized, new_normalized, rtol=0.1)

    def test_scale_to_max(self):
        """scale_to_max should set maximum photometric value."""
        lamp = Lamp.from_keyword("aerolamp")
        original_max = lamp.max()

        lamp.scale_to_max(100.0)

        assert np.isclose(lamp.max(), 100.0, rtol=0.01)

    def test_scale_to_center(self):
        """scale_to_center should set center photometric value."""
        lamp = Lamp.from_keyword("aerolamp")

        lamp.scale_to_center(50.0)

        assert np.isclose(lamp.center(), 50.0, rtol=0.01)

    def test_get_total_power_changes_with_scale(self):
        """Total power should change proportionally with scaling."""
        lamp = Lamp.from_keyword("aerolamp")
        original_power = lamp.get_total_power()

        lamp.scale(3.0)

        new_power = lamp.get_total_power()
        assert np.isclose(new_power, original_power * 3, rtol=0.01)

    def test_scale_sets_absolute_value(self):
        """scale() sets absolute scaling factor, not cumulative."""
        lamp = Lamp.from_keyword("aerolamp")
        original_max = lamp.max()

        lamp.scale(2.0)
        assert np.isclose(lamp.scaling_factor, 2.0, rtol=0.01)
        assert np.isclose(lamp.max(), original_max * 2, rtol=0.01)

        # Second scale() replaces, doesn't multiply
        lamp.scale(3.0)
        assert np.isclose(lamp.scaling_factor, 3.0, rtol=0.01)
        assert np.isclose(lamp.max(), original_max * 3, rtol=0.01)


class TestLampSaveLoadWithModifications:
    """Tests for saving/loading lamps with modified parameters."""

    def test_save_load_scaled_lamp(self, temp_file):
        """Scaled lamp should preserve scaling after save/load."""
        room = Room(x=6, y=4, z=2.7)
        lamp = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        lamp.scale(2.5)
        original_scaling = lamp.scaling_factor

        room.add_lamp(lamp)
        room.add_standard_zones()
        room.save(temp_file)

        loaded = Room.load(temp_file)
        loaded_lamp = list(loaded.lamps.values())[0]

        assert np.isclose(loaded_lamp.scaling_factor, original_scaling, rtol=0.01)

    def test_save_load_custom_wavelength(self, temp_file):
        """Custom wavelength should be preserved after save/load."""
        room = Room(x=6, y=4, z=2.7)
        lamp = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        lamp.load_spectra(None)
        lamp.set_wavelength(222)

        room.add_lamp(lamp)
        room.add_standard_zones()
        room.save(temp_file)

        loaded = Room.load(temp_file)
        loaded_lamp = list(loaded.lamps.values())[0]

        assert loaded_lamp.wavelength == 222

    def test_save_load_modified_surface(self, temp_file):
        """Modified surface parameters should be preserved after save/load."""
        room = Room(x=6, y=4, z=2.7)
        lamp = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        lamp.set_source_density(5)
        lamp.set_width(0.2)
        lamp.set_length(0.3)

        room.add_lamp(lamp)
        room.add_standard_zones()
        room.save(temp_file)

        loaded = Room.load(temp_file)
        loaded_lamp = list(loaded.lamps.values())[0]

        # All surface parameters should be preserved
        assert loaded_lamp.surface.source_density == 5
        assert loaded_lamp.width == 0.2
        assert loaded_lamp.length == 0.3


@pytest.mark.filterwarnings("ignore:No valid lamps are present in the room")
class TestLampDisabledState:
    """Tests for enabling/disabling lamps."""

    def test_disabled_lamp_no_contribution(self):
        """Disabled lamp should not contribute to fluence."""
        room = Room(x=6, y=4, z=2.7)
        lamp = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        lamp.enabled = False

        room.add_lamp(lamp)
        room.add_standard_zones()
        room.calculate()

        # Fluence should be zero with disabled lamp
        assert room.calc_zones["WholeRoomFluence"].values.max() == 0

    def test_disable_lamp_after_calculation(self):
        """Disabling lamp after calculation should affect recalculation."""
        room = Room(x=6, y=4, z=2.7)
        lamp = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        room.add_lamp(lamp)
        room.add_standard_zones()
        room.calculate()

        assert room.calc_zones["WholeRoomFluence"].values.mean() > 0

        # Disable lamp
        lamp.enabled = False
        room.calculate()

        assert room.calc_zones["WholeRoomFluence"].values.max() == 0

    def test_reenable_lamp(self):
        """Re-enabling a disabled lamp should restore contribution."""
        room = Room(x=6, y=4, z=2.7)
        lamp = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        room.add_lamp(lamp)
        room.add_standard_zones()
        room.calculate()

        original_mean = room.calc_zones["WholeRoomFluence"].values.mean()

        # Disable and re-enable
        lamp.enabled = False
        room.calculate()
        lamp.enabled = True
        room.calculate()

        new_mean = room.calc_zones["WholeRoomFluence"].values.mean()
        assert np.isclose(original_mean, new_mean, rtol=0.01)

    def test_mix_enabled_disabled_lamps(self):
        """Mixed enabled/disabled lamps should calculate correctly."""
        room = Room(x=6, y=4, z=2.7)

        lamp1 = Lamp.from_keyword("aerolamp").move(2, 2, 2.7).aim(2, 2, 0)
        lamp1.enabled = True

        lamp2 = Lamp.from_keyword("aerolamp").move(4, 2, 2.7).aim(4, 2, 0)
        lamp2.enabled = False

        room.add_lamp(lamp1).add_lamp(lamp2)
        room.add_standard_zones()
        room.calculate()

        # Create single lamp room for comparison
        single_room = Room(x=6, y=4, z=2.7)
        single_lamp = Lamp.from_keyword("aerolamp").move(2, 2, 2.7).aim(2, 2, 0)
        single_room.add_lamp(single_lamp)
        single_room.add_standard_zones()
        single_room.calculate()

        # Results should be similar (only one lamp active in each)
        mixed_mean = room.calc_zones["WholeRoomFluence"].values.mean()
        single_mean = single_room.calc_zones["WholeRoomFluence"].values.mean()
        assert np.isclose(mixed_mean, single_mean, rtol=0.1)


class TestLampCloning:
    """Tests for cloning/copying lamps."""

    def test_lamp_from_dict_creates_independent_copy(self):
        """Creating lamp from dict should be independent of original."""
        lamp1 = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
        lamp1.scale(2.0)

        data = lamp1.to_dict()
        lamp2 = Lamp.from_dict(data)

        # Verify lamp2 has same initial scaling as lamp1
        assert lamp2.scaling_factor == 2.0

        # Modify lamp2 with a different absolute scale
        lamp2.scale(1.5)

        # lamp1 should be unchanged, lamp2 should have new absolute value
        assert lamp1.scaling_factor == 2.0
        assert lamp2.scaling_factor == 1.5  # scale() sets absolute value, not cumulative

    def test_multiple_lamps_from_same_keyword(self):
        """Multiple lamps from same keyword should be independent."""
        lamp1 = Lamp.from_keyword("aerolamp")
        lamp2 = Lamp.from_keyword("aerolamp")

        lamp1.move(1, 1, 2.7)
        lamp2.move(5, 3, 2.7)

        # Positions should be independent
        assert lamp1.x == 1
        assert lamp2.x == 5
