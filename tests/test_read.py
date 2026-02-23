import numpy as np

from guv_calcs import CalcPlane, CalcVol
from guv_calcs._read import file_to_zone


def test_file_to_zone_roundtrip_plane(tmp_path):
    zone = CalcPlane(
        zone_id="plane",
        x1=0.0,
        x2=1.0,
        y1=0.0,
        y2=2.0,
        num_x=2,
        num_y=3,
        height=1.0,
        offset=False,
    )
    values = np.arange(zone.geometry.num_x * zone.geometry.num_y).reshape(
        zone.geometry.num_x, zone.geometry.num_y
    )
    zone.result.base_values = values

    path = tmp_path / "plane.csv"
    zone.export(fname=path)

    loaded = file_to_zone(path)
    np.testing.assert_allclose(loaded.get_values(), values)


def test_file_to_zone_roundtrip_volume(tmp_path):
    zone = CalcVol(
        zone_id="volume",
        x1=0.0,
        x2=1.0,
        y1=0.0,
        y2=1.0,
        z1=0.0,
        z2=1.0,
        num_x=2,
        num_y=2,
        num_z=2,
        offset=False,
    )
    values = np.arange(
        zone.geometry.num_x * zone.geometry.num_y * zone.geometry.num_z
    ).reshape(zone.geometry.num_x, zone.geometry.num_y, zone.geometry.num_z)
    zone.result.base_values = values

    path = tmp_path / "volume.csv"
    zone.export(fname=path)

    loaded = file_to_zone(path)
    np.testing.assert_allclose(loaded.get_values(), values)
