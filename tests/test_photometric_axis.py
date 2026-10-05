"""Tests for PhotometricAxis (where the beam points in the IES frame)."""

import numpy as np
import pytest
from guv_calcs.lamp import PhotometricAxis

AXES = list(PhotometricAxis)
HORIZONTAL = [a for a in AXES if a.is_horizontal]


class TestMatrix:
    @pytest.mark.parametrize("axis", AXES)
    def test_beam_maps_to_aim(self, axis):
        np.testing.assert_allclose(axis.matrix @ axis.direction, [0, 0, -1], atol=1e-12)

    @pytest.mark.parametrize("axis", AXES)
    def test_is_proper_rotation(self, axis):
        m = axis.matrix
        np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(m), 1.0)

    def test_down_is_identity(self):
        np.testing.assert_allclose(PhotometricAxis.DOWN.matrix, np.eye(3))

    @pytest.mark.parametrize("axis", HORIZONTAL)
    def test_horizontal_zenith_maps_to_local_x(self, axis):
        # local +x is what the pose sends to world-up when banked to 90 deg
        np.testing.assert_allclose(axis.matrix @ [0, 0, 1], [1, 0, 0], atol=1e-12)

    def test_matrix_has_clean_zeros(self):
        m = PhotometricAxis.HORIZONTAL_0.matrix
        assert (m == np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])).all()


class TestParsing:
    def test_default_is_down(self):
        assert PhotometricAxis.from_any(None) is PhotometricAxis.DOWN

    def test_tokens_are_forgiving(self):
        assert PhotometricAxis.from_any("Horizontal-90") is PhotometricAxis.HORIZONTAL_90
        assert PhotometricAxis.from_any(" up ") is PhotometricAxis.UP

    def test_unknown_token_raises(self):
        with pytest.raises(ValueError):
            PhotometricAxis.from_any("sideways")

    def test_phi(self):
        assert PhotometricAxis.HORIZONTAL_270.phi == 270
        assert PhotometricAxis.DOWN.phi is None


class TestExtents:
    def test_down_keeps_order(self):
        assert PhotometricAxis.DOWN.permute_extents(1.0, 2.0, 3.0) == (1.0, 2.0, 3.0)

    def test_horizontal_0(self):
        # ies length (along beam) becomes depth; ies height becomes length
        assert PhotometricAxis.HORIZONTAL_0.permute_extents(1.26, 1.94, 0.42) == pytest.approx((0.42, 1.94, 1.26))

    def test_horizontal_90(self):
        assert PhotometricAxis.HORIZONTAL_90.permute_extents(1.26, 1.94, 0.42) == pytest.approx((0.42, 1.26, 1.94))

    def test_up_keeps_extents(self):
        assert PhotometricAxis.UP.permute_extents(1.0, 2.0, 3.0) == pytest.approx((1.0, 2.0, 3.0))


from guv_calcs import Lamp, Room, WHOLE_ROOM_FLUENCE


class TestLampIntegration:
    def test_default_axis_is_down(self):
        lamp = Lamp.from_keyword("aerolamp")
        assert lamp.photometric_axis is PhotometricAxis.DOWN
        np.testing.assert_allclose(lamp.photometric_axis_matrix, np.eye(3))

    def test_horizontal_lamp_banked_sees_beam_along_aim(self):
        # wall-mounted: at origin, aimed along +x (bank 90, heading 0)
        lamp = Lamp.from_keyword("aerolamp", x=0, y=0, z=0, aimx=1, aimy=0, aimz=0, photometric_axis="horizontal_0")
        th, ph, r = lamp.transform_to_lamp(np.array([[2.0, 0.0, 0.0]]), which="polar")
        assert th[0] == pytest.approx(90.0)
        assert ph[0] % 360 == pytest.approx(0.0, abs=1e-9)
        assert r[0] == pytest.approx(2.0)
        # world up is the ies zenith
        th_up, _, _ = lamp.transform_to_lamp(np.array([[0.0, 0.0, 2.0]]), which="polar")
        assert th_up[0] == pytest.approx(180.0)

    def test_down_lamp_unchanged(self):
        lamp = Lamp.from_keyword("aerolamp", x=0, y=0, z=0, aimx=0, aimy=0, aimz=-1)
        th, _, _ = lamp.transform_to_lamp(np.array([[0.0, 0.0, -2.0]]), which="polar")
        assert th[0] == pytest.approx(0.0)

    def test_world_round_trip(self):
        lamp = Lamp.from_keyword("aerolamp", x=1, y=2, z=3, aimx=4, aimy=2, aimz=3, photometric_axis="horizontal_90")
        pts = np.array([[0.3, -0.2, 0.9], [-1.0, 0.5, 0.1]])
        ies = lamp.transform_to_lamp(pts - lamp.position)  # (3, N)
        back = lamp.transform_to_world(ies.T).T  # pose returns (3, N)
        np.testing.assert_allclose(back, pts, atol=1e-9)

    def test_photometric_coords_web_points_along_aim(self):
        lamp = Lamp.from_keyword("aerolamp", x=0, y=0, z=0, aimx=1, aimy=0, aimz=0, photometric_axis="horizontal_0")
        world = lamp.transform_to_world(lamp.photometric_coords, scale=lamp.values.max())
        # aerolamp is a downlight in its own frame; declared horizontal_0 its
        # file-frame beam (-z) lands on local -x, i.e. world down when banked.
        assert world[2].mean() < 0

    def test_surface_dims_permute_with_axis(self):
        # sterilray ies: width 0.3 (y), length 0.05 (x), height 0 (z)
        lamp = Lamp.from_keyword("sterilray", photometric_axis="horizontal_0")
        assert lamp.length == pytest.approx(0.0)      # ies height -> length
        assert lamp.width == pytest.approx(0.3)       # width stays
        assert lamp.surface.height == pytest.approx(0.05)  # ies length -> depth

    def test_set_axis_rederives_dims_unless_user_set(self):
        lamp = Lamp.from_keyword("sterilray")
        lamp.set_photometric_axis("horizontal_0")
        assert lamp.surface.height == pytest.approx(0.05)
        lamp.set_width(0.4)
        lamp.set_photometric_axis("down")
        assert lamp.width == pytest.approx(0.4)
        assert lamp.surface.height == pytest.approx(0.0)

    def test_calc_state_changes_with_axis(self):
        lamp = Lamp.from_keyword("aerolamp")
        before = lamp.calc_state
        lamp.set_photometric_axis("up")
        assert lamp.calc_state != before

    def test_to_dict_round_trip(self):
        lamp = Lamp.from_keyword("aerolamp", photometric_axis="horizontal_180", photometric_depth=0.04)
        loaded = Lamp.from_dict(lamp.to_dict())
        assert loaded.photometric_axis is PhotometricAxis.HORIZONTAL_180
        assert loaded.fixture.photometric_depth == pytest.approx(0.04)

    def test_legacy_dict_loads_defaults(self):
        data = Lamp.from_keyword("aerolamp").to_dict()
        data.pop("photometric_axis")
        data["fixture"].pop("photometric_depth")
        loaded = Lamp.from_dict(data)
        assert loaded.photometric_axis is PhotometricAxis.DOWN
        assert loaded.fixture.photometric_depth == 0.0

    def test_set_units_converts_depth(self):
        lamp = Lamp.from_keyword("aerolamp", photometric_depth=0.3048)
        lamp.set_units("feet")
        assert lamp.fixture.photometric_depth == pytest.approx(1.0)
        assert lamp.photometric_axis is PhotometricAxis.DOWN

    def test_room_calculates_with_horizontal_lamp(self):
        room = Room(x=4, y=3, z=2.7)
        lamp = Lamp.from_keyword("aerolamp", lamp_id="wall", x=0.05, y=1.5, z=2.3, aimx=4, aimy=1.5, aimz=2.3, photometric_axis="horizontal_0")
        room.add_lamp(lamp)
        room.add_standard_zones()
        room.calculate()
        zone = room.calc_zones[WHOLE_ROOM_FLUENCE]
        assert np.nanmax(zone.values) > 0
