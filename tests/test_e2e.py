"""End-to-end workflow tests for guv_calcs."""

import pytest
import numpy as np
from guv_calcs import Room, Lamp, CalcPlane, CalcVol


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
