"""Plane plots: heatmap and contours."""

import io
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from guv_calcs import Room, Lamp, CalcPlane, SurfaceGrid, Object
from guv_calcs.calc_zone._plot import _holes_filled, _traced_values


@pytest.fixture(scope="module")
def lit_room():
    room = Room(x=4, y=3, z=2.7, units="meters", enable_reflectance=False)
    room.add_lamp(Lamp.from_keyword("aerolamp").move(2, 1.5, 2.7).aim(2, 1.5, 0))
    grid = SurfaceGrid.from_legacy(mins=(0, 0), maxs=(4, 3), height=1.8, spacing_init=(0.2, 0.2))
    room.add_calc_zone(CalcPlane(zone_id="plane", geometry=grid))
    room.calculate()
    return room


def test_heatmap_by_default(lit_room):
    fig, ax = lit_room.calc_zones["plane"].plot()
    assert len(ax.images) == 1
    assert not ax.collections
    plt.close(fig)


def test_contours_with_named_levels(lit_room):
    zone = lit_room.calc_zones["plane"]
    top = float(np.nanmax(zone.get_values()))
    levels = [top * 0.5, top * 0.1]  # unsorted on purpose
    fig, ax = zone.plot(levels=levels, colors=["#ff0000", None], labels=["half", ""], title="Office")
    # Filled bands and their lines
    assert ax.collections
    assert ax.get_title() == "Office"
    # The colorbar names levels in ascending order: the value where unnamed, the name where named
    cbar_ax = fig.axes[-1]
    ticks = [t.get_text() for t in cbar_ax.get_yticklabels()]
    assert ticks == [f"{top * 0.1:.2f}", "half"]
    plt.close(fig)


def test_lines_only_and_values(lit_room):
    zone = lit_room.calc_zones["plane"]
    top = float(np.nanmax(zone.get_values()))
    fig, ax = zone.plot(levels=[top / 2], filled=False, show_values=True, decimals=1)
    fig.set_size_inches(10, 8)
    # Lines only (no filled bands) and one printed value per cell
    texts = [t.get_text() for t in ax.texts]
    assert f"{top:.1f}" in texts
    assert len(texts) >= zone.get_values().size
    plt.close(fig)


def test_holes_take_their_nearest_value():
    values = np.array([[1.0, np.nan, np.nan], [2.0, np.nan, 4.0]])
    filled = _holes_filled(values)
    assert np.isfinite(filled).all()
    assert filled[0, 1] in (1.0, 2.0, 4.0)
    all_holes = np.full((2, 2), np.nan)
    assert np.isnan(_holes_filled(all_holes)).all()


def test_unlit_cells_are_traced_as_holes():
    # A lit cell beside an obstacle's zeros: the zeros take the lit value, so no band crowds in between
    values = np.array([[8.0, 0.0], [9.0, 0.0]])
    traced, unlit = _traced_values(values)
    assert traced.tolist() == [[8.0, 8.0], [9.0, 9.0]]
    assert unlit.tolist() == [[False, True], [False, True]]
    # An unlit plane is plotted as it is
    traced, unlit = _traced_values(np.zeros((2, 2)))
    assert not unlit.any() and (traced == 0).all()


def test_an_obstacle_is_painted_over_the_bands():
    room = Room(x=4, y=3, z=2.7, units="meters", enable_reflectance=False)
    room.add_lamp(Lamp.from_keyword("aerolamp").move(1.3, 1.2, 2.6).aim(1.3, 1.2, 0))
    room.add_object(Object.box(1.0, 0.8, 2.0, object_id="box", position=(0.8, 0.6, 0)))
    grid = SurfaceGrid.from_legacy(mins=(0, 0), maxs=(4, 3), height=1.8, spacing_init=(0.25, 0.25))
    room.add_calc_zone(CalcPlane(zone_id="plane", geometry=grid))
    room.calculate()
    zone = room.calc_zones["plane"]
    assert (zone.get_values() == 0).any()
    fig, ax = zone.plot(levels=[1, 5])
    # The floor, then the obstacle's cells above every band and line
    over = [im for im in ax.images if im.get_zorder() > max(c.get_zorder() for c in ax.collections)]
    assert len(over) == 1
    plt.close(fig)


def test_bands_reach_the_edges_of_the_plot(lit_room):
    zone = lit_room.calc_zones["plane"]
    low = float(np.nanmin(zone.get_values()))
    fig, ax = zone.plot(levels=[low])
    # One band at the plane's minimum covers the whole plane, edge to edge
    (x0, y0), (x1, y1) = ax.collections[0].get_paths()[0].get_extents().get_points()
    assert (x0, y0, x1, y1) == pytest.approx((0, 0, 4, 3))
    plt.close(fig)


def test_export_zip_uses_plot_kwargs(lit_room):
    zone = lit_room.calc_zones["plane"]
    top = float(np.nanmax(zone.get_values()))
    data = lit_room.export_zip(include_plots=True, plot_kwargs={"plane": {"levels": [top / 2], "title": "Custom"}})
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
    assert f"{zone.name}.png" in names and f"{zone.name} (contours).png" in names


def test_contour_labels_sit_level_on_a_backing_inside_the_plot(lit_room):
    zone = lit_room.calc_zones["plane"]
    top = float(np.nanmax(zone.get_values()))
    fig, ax = zone.plot(levels=[top * 0.3, top * 0.6])
    labels = [t for t in ax.texts if t.get_bbox_patch() is not None]
    assert labels
    for t in labels:
        assert t.get_rotation() == 0
        x, y = t.get_position()
        assert 0 < x < 4 and 0 < y < 3
    plt.close(fig)
