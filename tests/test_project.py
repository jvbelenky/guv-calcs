"""Tests for the Project class."""

import pytest
import json
import os
from guv_calcs import Project, Room


class TestProjectCreation:
    """Tests for Project initialization and defaults."""

    def test_default_project(self):
        project = Project()
        assert project.units == "meters"
        assert project.precision == 1
        assert len(project.rooms) == 0

    def test_custom_defaults(self):
        project = Project(standard="ICNIRP", units="feet", precision=3)
        assert "ICNIRP" in str(project.standard).upper()
        assert project.units == "feet"
        assert project.precision == 3


class TestRoomManagement:
    """Tests for adding, creating, accessing, and removing rooms."""

    def test_create_room_applies_defaults(self):
        project = Project(units="feet", precision=3)
        room = project.create_room(room_id="office", x=20, y=15, z=9)
        assert room.units == "feet"
        assert room.precision == 3
        assert "office" in project.rooms

    def test_create_room_kwargs_override_defaults(self):
        project = Project(precision=1)
        room = project.create_room(room_id="lab", precision=4, x=10, y=8, z=3)
        assert room.precision == 4

    def test_add_existing_room(self):
        project = Project()
        room = Room(room_id="lab", x=10, y=8, z=3.0)
        project.add_room(room)
        assert "lab" in project.rooms
        assert project.room("lab") is room

    def test_room_access_by_id(self):
        project = Project()
        project.create_room(room_id="office", x=6, y=4, z=2.7)
        room = project.room("office")
        assert room.x == 6

    def test_remove_room(self):
        project = Project()
        project.create_room(room_id="office", x=6, y=4, z=2.7)
        project.remove_room("office")
        assert len(project.rooms) == 0

    def test_room_id_collision_increment(self):
        project = Project()
        project.create_room(room_id="Room")
        project.create_room(room_id="Room")
        assert len(project.rooms) == 2

    def test_room_independence(self):
        """Modifying one room doesn't affect another."""
        project = Project()
        r1 = project.create_room(room_id="r1", x=6, y=4, z=2.7)
        r2 = project.create_room(room_id="r2", x=10, y=8, z=3.0)
        r1.set_dimensions(x=20)
        assert r2.x == 10


class TestBulkOperations:
    """Tests for bulk calculate and setters."""

    def test_calculate_all(self):
        project = Project()
        r1 = project.create_room(room_id="r1", x=6, y=4, z=2.7)
        r1.place_lamp("aerolamp").add_standard_zones()
        r2 = project.create_room(room_id="r2", x=8, y=6, z=3.0)
        r2.place_lamp("aerolamp").add_standard_zones()
        project.calculate()
        assert r1.calc_zones["WholeRoomFluence"].values is not None
        assert r2.calc_zones["WholeRoomFluence"].values is not None

    def test_set_standard_propagates(self):
        project = Project(standard="ACGIH")
        r1 = project.create_room(room_id="r1", x=6, y=4, z=2.7)
        r1.add_standard_zones()
        project.set_standard("ICNIRP")
        assert "ICNIRP" in str(project.standard).upper()
        assert "ICNIRP" in str(r1.standard).upper()

    def test_set_precision_propagates(self):
        project = Project(precision=1)
        r1 = project.create_room(room_id="r1", x=6, y=4, z=2.7)
        project.set_precision(4)
        assert project.precision == 4
        assert r1.precision == 4

    def test_set_colormap_propagates(self):
        project = Project()
        r1 = project.create_room(room_id="r1", x=6, y=4, z=2.7)
        project.set_colormap("viridis")
        assert project.colormap == "viridis"
        assert r1.colormap == "viridis"

    def test_check_lamps(self):
        project = Project()
        r1 = project.create_room(room_id="r1", x=6, y=4, z=2.7)
        r1.place_lamp("aerolamp").add_standard_zones().calculate()
        results = project.check_lamps()
        assert "r1" in results


class TestSerialization:
    """Tests for save/load round-trips."""

    def test_to_dict_from_dict(self):
        project = Project(precision=3)
        project.create_room(room_id="office", x=6, y=4, z=2.7)
        project.create_room(room_id="lab", x=10, y=8, z=3.0)
        data = project.to_dict()
        loaded = Project.from_dict(data)
        assert len(loaded.rooms) == 2
        assert "office" in loaded.rooms
        assert "lab" in loaded.rooms
        assert loaded.precision == 3

    def test_save_returns_json(self):
        project = Project()
        project.create_room(room_id="office", x=6, y=4, z=2.7)
        result = project.save()
        assert isinstance(result, str)
        data = json.loads(result)
        assert data["format"] == "project"

    def test_save_load_round_trip(self, temp_dir):
        project = Project(precision=2)
        r = project.create_room(room_id="office", x=6, y=4, z=2.7)
        r.place_lamp("aerolamp").add_standard_zones()
        filepath = os.path.join(temp_dir, "test.guv")
        project.save(filepath)

        loaded = Project.load(filepath)
        assert len(loaded.rooms) == 1
        assert "office" in loaded.rooms
        assert loaded.room("office").x == 6
        assert len(loaded.room("office").lamps) == 1
        assert loaded.precision == 2

    def test_load_legacy_single_room(self, temp_dir):
        """Loading a single-room .guv file via Project.load wraps it."""
        room = Room(x=8, y=5, z=3.0)
        room.place_lamp("aerolamp")
        filepath = os.path.join(temp_dir, "legacy.guv")
        room.save(filepath)

        project = Project.load(filepath)
        assert len(project.rooms) == 1
        loaded_room = list(project.rooms.values())[0]
        assert loaded_room.x == 8
        assert len(loaded_room.lamps) == 1

    def test_room_save_load_unchanged(self, temp_dir):
        """Standalone Room save/load still works as before."""
        room = Room(x=6, y=4, z=2.7)
        room.place_lamp("aerolamp").add_standard_zones()
        filepath = os.path.join(temp_dir, "room.guv")
        room.save(filepath)
        loaded = Room.load(filepath)
        assert loaded.x == 6
        assert len(loaded.lamps) == 1


class TestRoomIdentity:
    """Tests for room_id and name on Room."""

    def test_default_room_id(self):
        room = Room()
        assert room.room_id == "Room"
        assert room.name == "Room"

    def test_custom_room_id(self):
        room = Room(room_id="office")
        assert room.room_id == "office"
        assert room.id == "office"
        assert room.name == "office"

    def test_custom_name(self):
        room = Room(room_id="office", name="Main Office")
        assert room.room_id == "office"
        assert room.name == "Main Office"

    def test_room_id_in_repr(self):
        room = Room(room_id="lab")
        assert "lab" in repr(room)

    def test_room_id_in_to_dict(self):
        room = Room(room_id="office", name="Main Office")
        data = room.to_dict()
        assert data["room_id"] == "office"
        assert data["name"] == "Main Office"

    def test_room_id_survives_from_dict(self):
        room = Room(room_id="office", name="Main Office", x=6, y=4, z=2.7)
        data = room.to_dict()
        loaded = Room.from_dict(data)
        assert loaded.room_id == "office"
        assert loaded.name == "Main Office"


class TestExportAndReport:
    """Tests for export_zip and generate_report on Project."""

    def test_export_zip_returns_bytes(self):
        project = Project()
        r = project.create_room(room_id="office", x=6, y=4, z=2.7)
        r.place_lamp("aerolamp").add_standard_zones().calculate()
        result = project.export_zip()
        assert isinstance(result, bytes)

    def test_export_zip_contains_project_guv(self):
        import zipfile, io
        project = Project()
        r = project.create_room(room_id="office", x=6, y=4, z=2.7)
        r.place_lamp("aerolamp").add_standard_zones().calculate()
        zip_bytes = project.export_zip()
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            assert "project.guv" in zf.namelist()

    def test_generate_report_returns_bytes(self):
        project = Project()
        r = project.create_room(room_id="office", x=6, y=4, z=2.7)
        r.place_lamp("aerolamp").add_standard_zones().calculate()
        result = project.generate_report()
        assert isinstance(result, bytes)
        text = result.decode("cp1252")
        assert "office" in text

    def test_generate_report_contains_room_sections(self):
        """Multi-room report labels each room."""
        project = Project()
        r1 = project.create_room(room_id="office", x=6, y=4, z=2.7)
        r1.place_lamp("aerolamp").add_standard_zones().calculate()
        r2 = project.create_room(room_id="lab", x=8, y=6, z=3.0)
        r2.place_lamp("aerolamp").add_standard_zones().calculate()
        result = project.generate_report()
        text = result.decode("cp1252")
        assert "=== Room: office" in text
        assert "=== Room: lab" in text

    def test_generate_report_contains_project_summary(self):
        """Report includes a project summary section."""
        project = Project()
        r = project.create_room(room_id="office", x=6, y=4, z=2.7)
        r.place_lamp("aerolamp").add_standard_zones().calculate()
        result = project.generate_report()
        text = result.decode("cp1252")
        assert "=== Project Summary ===" in text

    def test_generate_report_multi_room_summary(self):
        """Summary lists stats from all rooms."""
        project = Project()
        r1 = project.create_room(room_id="office", x=6, y=4, z=2.7)
        r1.place_lamp("aerolamp").add_standard_zones().calculate()
        r2 = project.create_room(room_id="lab", x=8, y=6, z=3.0)
        r2.place_lamp("aerolamp").add_standard_zones().calculate()
        result = project.generate_report()
        text = result.decode("cp1252")
        # Summary section should reference both rooms
        summary_start = text.index("=== Project Summary ===")
        summary_text = text[summary_start:]
        assert "office" in summary_text
        assert "lab" in summary_text


class TestProjectCopyAndEquality:

    def test_copy_independence(self):
        project = Project()
        project.create_room(room_id="office", x=6, y=4, z=2.7)
        copy = project.copy()
        copy.create_room(room_id="lab", x=10, y=8, z=3.0)
        assert len(project.rooms) == 1
        assert len(copy.rooms) == 2

    def test_equality(self):
        p1 = Project(precision=2)
        p1.create_room(room_id="office", x=6, y=4, z=2.7)
        p2 = Project(precision=2)
        p2.create_room(room_id="office", x=6, y=4, z=2.7)
        assert p1 == p2

    def test_inequality(self):
        p1 = Project(precision=2)
        p1.create_room(room_id="office", x=6, y=4, z=2.7)
        p2 = Project(precision=3)
        p2.create_room(room_id="office", x=6, y=4, z=2.7)
        assert p1 != p2
