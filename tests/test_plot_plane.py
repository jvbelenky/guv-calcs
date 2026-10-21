"""Plane plots: heatmap and contours."""

import io
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from guv_calcs import Room, Lamp, CalcPlane, SurfaceGrid
from guv_calcs.calc_zone._plot import _holes_filled


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


def test_export_zip_uses_plot_kwargs(lit_room):
    zone = lit_room.calc_zones["plane"]
    top = float(np.nanmax(zone.get_values()))
    data = lit_room.export_zip(include_plots=True, plot_kwargs={"plane": {"levels": [top / 2], "title": "Custom"}})
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert f"{zone.name}.png" in zf.namelist()
