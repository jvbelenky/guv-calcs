from ._zone import CalcZone, CalcPlane, CalcPoint, CalcVol, ZoneView, ZoneResult
from ._io import export_plane, export_volume


def __getattr__(name):
    if name in ("plot_plane", "plot_volume"):
        from ._plot import plot_plane, plot_volume
        return plot_plane if name == "plot_plane" else plot_volume
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
