"""Shared pytest fixtures for guv_calcs test suite."""

import pytest
import tempfile
import os
from guv_calcs import Room, Lamp, CalcPlane, CalcVol, Polygon2D


# ============== Warning Filters ==============
# These warnings are expected in test scenarios with small rooms

def pytest_configure(config):
    """Configure pytest warning filters for expected warnings."""
    config.addinivalue_line(
        "filterwarnings",
        "ignore:aerolamp exceeds room boundaries:UserWarning"
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:ushio_b1 exceeds room boundaries:UserWarning"
    )


# ============== Room Fixtures ==============

@pytest.fixture
def basic_room():
    """A basic room with default dimensions in meters."""
    return Room(x=6, y=4, z=2.7, units="meters")


@pytest.fixture
def room_feet():
    """A basic room with dimensions in feet."""
    return Room(x=20, y=15, z=9, units="feet")


@pytest.fixture
def room_no_reflectance():
    """A room with reflectance disabled."""
    return Room(x=6, y=4, z=2.7, units="meters", enable_reflectance=False)


# ============== Lamp Fixtures ==============

@pytest.fixture
def basic_lamp():
    """A lamp loaded from the 'aerolamp' keyword."""
    return Lamp.from_keyword("aerolamp")


@pytest.fixture
def positioned_lamp(basic_lamp):
    """A lamp positioned at center-ceiling and aimed downward."""
    return basic_lamp.move(3, 2, 2.7).aim(3, 2, 0)


@pytest.fixture
def lamp_ushio():
    """A lamp loaded from the 'ushio_b1' keyword."""
    return Lamp.from_keyword("ushio_b1")


# ============== Room with Lamp Fixtures ==============

@pytest.fixture
def room_with_lamp(basic_room, positioned_lamp):
    """A room with a single lamp added."""
    basic_room.add_lamp(positioned_lamp)
    return basic_room


@pytest.fixture
def room_with_zones(room_with_lamp):
    """A room with standard zones (SkinLimits, EyeLimits, WholeRoomFluence)."""
    room_with_lamp.add_standard_zones()
    return room_with_lamp


@pytest.fixture
def calculated_room(room_with_zones):
    """A room that has had calculate() called on it."""
    room_with_zones.calculate()
    return room_with_zones


# ============== CalcZone Fixtures ==============

@pytest.fixture
def calc_plane():
    """A basic calculation plane at height 1.8m."""
    return CalcPlane(
        zone_id="TestPlane",
        x1=0, x2=6,
        y1=0, y2=4,
        height=1.8,
        x_spacing=0.5,
        y_spacing=0.5,
    )


@pytest.fixture
def calc_volume():
    """A basic calculation volume covering the full room."""
    return CalcVol(
        zone_id="TestVolume",
        x1=0, x2=6,
        y1=0, y2=4,
        z1=0, z2=2.7,
        x_spacing=0.5,
        y_spacing=0.5,
        z_spacing=0.5,
    )


@pytest.fixture
def calc_plane_dose():
    """A calculation plane configured for dose mode."""
    return CalcPlane(
        zone_id="DosePlane",
        x1=0, x2=6,
        y1=0, y2=4,
        height=1.0,
        dose=True,
        hours=8.0,
    )


# ============== Temporary File Fixtures ==============

@pytest.fixture
def temp_dir():
    """Provide a temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_file(temp_dir):
    """Provide a temporary file path for save/load tests."""
    return os.path.join(temp_dir, "test_room.guv")


# ============== Polygon Room Fixtures ==============

@pytest.fixture
def l_shaped_polygon():
    """An L-shaped polygon floor plan."""
    return Polygon2D(vertices=[(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)])


@pytest.fixture
def polygon_room(l_shaped_polygon):
    """A polygon-based room with L-shaped floor plan."""
    return Room(polygon=l_shaped_polygon, z=2.7, units="meters")
