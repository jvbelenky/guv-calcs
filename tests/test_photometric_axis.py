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
