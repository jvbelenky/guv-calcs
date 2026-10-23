"""A room's results don't depend on the length unit it's expressed in."""

import numpy as np
import pytest
from guv_calcs import Room, Lamp, CalcPlane, CalcVol, Object
from guv_calcs.geometry import Axis1D
from guv_calcs.geometry.polygon import points_in_polygon
from guv_calcs.units import convert_length

L_SHAPE = [(0, 0), (6, 0), (6, 2), (3, 2), (3, 4), (0, 4)]
UNITS = ["feet", "inches", "centimeters", "millimeters", "yards"]


def _room():
    """An L-shaped room in meters with every kind of grid that gets calculated."""
    room = Room(polygon=L_SHAPE, z=2.7, enable_reflectance=True)
    room.add_standard_zones()
    room.add_lamp(Lamp.from_keyword("ushio_b1").move(1.5, 1.0, 2.6))
    room.add_lamp(Lamp.from_keyword("ushio_b1").move(5.1, 0.7, 2.6).aim(5.1, 0.7, 0))
    # point counts that don't divide the spans into round numbers
    room.add_calc_zone(CalcPlane.from_polygon(
        [(0.3, 0.3), (5.7, 0.3), (5.7, 1.7), (0.3, 1.7)], height=0.9,
        num_points=(23, 17), zone_id="Desk",
    ))
    room.add_calc_zone(CalcVol.from_dims(
        dims=room.dim, spacing=(0.37, 0.29, 0.31), zone_id="Coarse",
    ))
    room.add_object(Object.box(0.8, 0.6, 0.75, object_id="Cabinet", R=0.1).move(1.0, 2.5, 0))
    for surface in room.surfaces.values():
        surface.R = 0.2
    return room


@pytest.fixture(scope="module")
def reference():
    room = _room()
    room.calculate()
    return {zid: zone.get_values().copy() for zid, zone in room.calc_zones.items()}


@pytest.mark.slow
@pytest.mark.parametrize("units", UNITS)
def test_results_survive_a_unit_change(reference, units):
    room = _room()
    room.set_units(units)
    room.calculate(hard=True)
    for zid, expected in reference.items():
        np.testing.assert_allclose(
            room.calc_zones[zid].get_values(), expected, rtol=1e-4, atol=1e-6,
            err_msg=f"{zid} in {units}",
        )


@pytest.mark.slow
@pytest.mark.parametrize("units", UNITS)
def test_unit_round_trip_restores_the_grids(units):
    room = _room()
    before = {zid: zone.coords.copy() for zid, zone in room.calc_zones.items()}
    room.set_units(units)
    room.set_units("meters")
    for zid, coords in before.items():
        np.testing.assert_allclose(room.calc_zones[zid].coords, coords, atol=1e-9, err_msg=zid)


@pytest.mark.parametrize("units", UNITS)
def test_point_count_divides_the_span_exactly(units):
    k = convert_length("meters", units, 1.0)
    meters = Axis1D(span=5.1816, num_points_init=25)
    other = Axis1D(span=5.1816 * k, num_points_init=25)
    assert meters.spacing == pytest.approx(5.1816 / 25)
    np.testing.assert_allclose(other.points / k, meters.points, rtol=1e-12)


@pytest.mark.parametrize("scale", [1.0, 1 / 0.3048, 1 / 0.9144, 1000.0])
def test_points_on_an_edge_count_in_every_unit(scale):
    verts = np.array(L_SHAPE, dtype=float) * scale
    # the L's inner corner and points along its inner walls, with float noise
    px = np.array([3.0, 3.0, 4.5, 3.0 + 1e-15]) * scale
    py = np.array([2.0, 3.0, 2.0, 2.5]) * scale
    assert points_in_polygon(px, py, verts, include_boundary=True).all()
