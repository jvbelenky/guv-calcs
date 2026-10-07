"""Tests for the Fixture class."""

import pytest
from dataclasses import replace
from guv_calcs.lamp import Fixture, FixtureShape


class TestFixtureConstruction:
    """Tests for Fixture construction and defaults."""

    def test_default_values(self):
        """Fixture should have sensible defaults."""
        f = Fixture()
        assert f.housing_width == 0.0
        assert f.housing_length == 0.0
        assert f.housing_height == 0.0
        assert f.shape == FixtureShape.RECTANGULAR

    def test_explicit_values(self):
        """Fixture should accept explicit values."""
        f = Fixture(
            housing_width=0.5,
            housing_length=0.3,
            housing_height=0.1,
            shape=FixtureShape.RECTANGULAR,
        )
        assert f.housing_width == 0.5
        assert f.housing_length == 0.3
        assert f.housing_height == 0.1

    def test_has_dimensions_true(self):
        """has_dimensions should be True when any dimension is set."""
        f = Fixture(housing_width=0.1)
        assert f.has_dimensions is True

    def test_has_dimensions_false(self):
        """has_dimensions should be False when all dimensions are zero."""
        f = Fixture()
        assert f.has_dimensions is False


class TestFixtureShape:
    """Tests for FixtureShape enum."""

    def test_from_any_none(self):
        """None should return RECTANGULAR."""
        assert FixtureShape.from_any(None) == FixtureShape.RECTANGULAR

    def test_from_token_aliases(self):
        """Various aliases should parse correctly."""
        assert FixtureShape.from_token("rect") == FixtureShape.RECTANGULAR
        assert FixtureShape.from_token("box") == FixtureShape.RECTANGULAR
        assert FixtureShape.from_token("cylinder") == FixtureShape.CYLINDRICAL
        assert FixtureShape.from_token("sphere") == FixtureShape.SPHERICAL


class TestFixtureSerialization:
    """Tests for Fixture serialization."""

    def test_to_dict(self):
        """to_dict() should include all fields."""
        f = Fixture(
            housing_width=0.5,
            housing_length=0.3,
            housing_height=0.1,
            shape=FixtureShape.CYLINDRICAL,
        )
        d = f.to_dict()
        assert d["housing_width"] == 0.5
        assert d["housing_length"] == 0.3
        assert d["housing_height"] == 0.1
        assert d["shape"] == "cylindrical"

    def test_from_dict_round_trip(self):
        """Fixture should survive to_dict/from_dict round trip."""
        f1 = Fixture(
            housing_width=0.5,
            housing_length=0.3,
            housing_height=0.1,
        )
        d = f1.to_dict()
        f2 = Fixture.from_dict(d)
        assert f2.housing_width == f1.housing_width
        assert f2.housing_length == f1.housing_length
        assert f2.housing_height == f1.housing_height

    def test_from_dict_missing_fields(self):
        """from_dict() should handle missing fields with defaults."""
        d = {"housing_width": 0.5}
        f = Fixture.from_dict(d)
        assert f.housing_width == 0.5
        assert f.housing_length == 0.0
        assert f.shape == FixtureShape.RECTANGULAR

    def test_from_dict_ignores_legacy_fields(self):
        """from_dict() should ignore legacy 'height' and 'mount_type' fields."""
        d = {
            "housing_width": 0.5,
            "housing_length": 0.3,
            "housing_height": 0.1,
            "height": 0.05,  # legacy field
            "mount_type": "pendant",  # legacy field
        }
        f = Fixture.from_dict(d)
        assert f.housing_width == 0.5
        assert f.housing_length == 0.3
        assert f.housing_height == 0.1
        # Should not have height or mount_type attributes
        assert not hasattr(f, 'height') or not f.__dataclass_fields__.get('height')


class TestPhotometricDepth:
    def test_default_zero(self):
        assert Fixture().photometric_depth == 0.0

    def test_round_trip(self):
        f = Fixture(housing_width=0.5, housing_height=0.1, photometric_depth=0.05)
        assert Fixture.from_dict(f.to_dict()) == f

    def test_legacy_dict_without_depth(self):
        f = Fixture.from_dict({"housing_width": 0.5, "housing_length": 0.3, "housing_height": 0.1, "shape": "rectangular"})
        assert f.photometric_depth == 0.0


import itertools
import numpy as np
from guv_calcs.lamp import Fixture, FixtureShape, HousingFace

FACES = ["bottom", "top", "front", "back", "left", "right"]


class TestHousingFace:
    @pytest.mark.parametrize("token,face", [("-z", "bottom"), ("+z", "top"), ("+x", "front"),
                                             ("-x", "back"), ("+y", "left"), ("-y", "right")])
    def test_axis_aliases(self, token, face):
        assert HousingFace.from_token(token) is HousingFace(face)

    def test_default_top(self):
        assert HousingFace.from_any(None) is HousingFace.TOP

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            HousingFace.from_token("sideways")

    @pytest.mark.parametrize("face", FACES)
    def test_opposite(self, face):
        f = HousingFace(face)
        np.testing.assert_array_equal(f.opposite.normal, -f.normal)
        assert f.opposite.opposite is f


class TestMountLayout:
    @pytest.mark.parametrize("face", FACES)
    @pytest.mark.parametrize("roll", [0, 90, 180, 270])
    def test_mount_normal_points_behind_the_emitter(self, face, roll):
        # aim frame: beam on -z, so the mount (wall/ceiling) is on +z
        f = Fixture(mount_face=face, roll=roll)
        np.testing.assert_allclose(f.housing_matrix @ f.mount_face.normal, [0, 0, 1], atol=1e-12)

    def test_defaults(self):
        f = Fixture()
        assert f.mount_face is HousingFace.TOP
        assert f.roll == 0
        assert f.emitter_offset is None
        np.testing.assert_array_equal(f.housing_matrix, np.eye(3))

    def test_roll_normalised_and_validated(self):
        assert Fixture(roll=450).roll == 90
        assert Fixture(roll=-90).roll == 270
        with pytest.raises(ValueError):
            Fixture(roll=45)

    def test_offset_must_have_three_components(self):
        with pytest.raises(ValueError):
            Fixture(emitter_offset=(0.0, 1.0))

    def test_all_24_orientations_distinct(self):
        mats = {Fixture(mount_face=f, roll=r).housing_matrix.round(6).tobytes()
                for f in FACES for r in (0, 90, 180, 270)}
        assert len(mats) == 24

    def test_default_emitter_sits_on_the_front_face(self):
        # classic layout: emitting face at the point, housing behind it
        f = Fixture(housing_width=0.2, housing_length=0.4, housing_height=0.1)
        np.testing.assert_allclose(f.emitter_position(), [0, 0, -0.05], atol=1e-12)
        lo, hi = f.aim_bounds()
        np.testing.assert_allclose(lo, [-0.2, -0.1, 0.0], atol=1e-12)
        np.testing.assert_allclose(hi, [0.2, 0.1, 0.1], atol=1e-12)
        assert f.photometric_depth == 0.0

    def test_default_stays_flush_when_the_housing_grows(self):
        f = replace(Fixture(housing_width=0.2, housing_length=0.2), housing_height=0.2)
        lo, hi = f.aim_bounds()
        assert (lo[2], hi[2]) == pytest.approx((0.0, 0.2))

    def test_offset_from_the_mounting_point(self):
        # light center 6 cm below the ceiling plate of a 10 cm tall housing
        f = Fixture(housing_width=0.2, housing_length=0.2, housing_height=0.1, emitter_offset=(0, 0, -0.06))
        lo, hi = f.aim_bounds()
        assert (lo[2], hi[2]) == pytest.approx((-0.04, 0.06))
        assert f.photometric_depth == pytest.approx(0.04)

    def test_explicit_offset_stays_put_relative_to_the_mount(self):
        f = Fixture(housing_width=0.2, housing_length=0.2, housing_height=0.1, emitter_offset=(0, 0, -0.06))
        g = replace(f, housing_height=0.2)
        lo, hi = g.aim_bounds()
        assert (lo[2], hi[2]) == pytest.approx((-0.14, 0.06))

    def test_legacy_depth(self):
        # 0.8.x: depth d behind the emitting face, housing h behind that face
        f = Fixture(housing_width=0.2, housing_length=0.4, housing_height=0.1, photometric_depth=0.03)
        assert f.emitter_offset == pytest.approx((0.0, 0.0, -0.07))
        assert f.photometric_depth == pytest.approx(0.03)
        lo, hi = f.aim_bounds()
        np.testing.assert_allclose(lo, [-0.2, -0.1, -0.03], atol=1e-12)
        np.testing.assert_allclose(hi, [0.2, 0.1, 0.07], atol=1e-12)

    def test_wall_mount(self):
        # Lumalier-style wall box: back face on the wall, 0.29 deep along the beam
        f = Fixture(housing_length=0.29, housing_width=1.19, housing_height=0.33, mount_face="back")
        lo, hi = f.aim_bounds()
        assert (lo[2], hi[2]) == pytest.approx((0.0, 0.29))
        assert sorted([hi[0] - lo[0], hi[1] - lo[1]]) == pytest.approx([0.33, 1.19])

    def test_corner_order(self):
        f = Fixture(housing_width=0.2, housing_length=0.4, housing_height=0.1)
        hl, hw = 0.2, 0.1
        expected = [[-hl, -hw, 0], [hl, -hw, 0], [hl, hw, 0], [-hl, hw, 0],
                    [-hl, -hw, 0.1], [hl, -hw, 0.1], [hl, hw, 0.1], [-hl, hw, 0.1]]
        np.testing.assert_allclose(f.aim_corners(), expected, atol=1e-12)

    def test_box_contains_the_luminous_opening(self):
        # LM-63: the housing is at least as large as the luminous opening
        f = Fixture(housing_width=0.2, housing_length=0.2, housing_height=0.1)
        lo, hi = f.aim_bounds(luminous=(0.6, 0.3, 0.0))
        np.testing.assert_allclose(lo, [-0.3, -0.15, 0.0], atol=1e-12)
        np.testing.assert_allclose(hi, [0.3, 0.15, 0.1], atol=1e-12)

    def test_luminous_volume_without_a_housing_is_centered_on_the_point(self):
        lo, hi = Fixture().aim_bounds(luminous=(0.0, 0.0, 0.12))
        assert (lo[2], hi[2]) == pytest.approx((-0.06, 0.06))

    def test_default_emitter_puts_a_3d_opening_flush_with_the_front_face(self):
        # UV-FLOW style: 12 cm luminous sides in a 12 cm housing -> photometric center at its middle
        f = Fixture(housing_width=0.58, housing_length=0.58, housing_height=0.12)
        np.testing.assert_allclose(f.emitter_position(luminous_height=0.12), [0, 0, 0], atol=1e-12)
        lo, hi = f.aim_bounds(luminous=(0.58, 0.58, 0.12))
        assert (lo[2], hi[2]) == pytest.approx((-0.06, 0.06))

    def test_explicit_offset_ignores_the_opening_depth(self):
        f = Fixture(housing_height=0.12, emitter_offset=(0, 0, -0.03))
        np.testing.assert_allclose(f.emitter_position(luminous_height=0.12), [0, 0, 0.03], atol=1e-12)

    def test_protruding_emitter(self):
        # emitter 2 cm in front of the front face of a 10 cm housing
        f = Fixture(housing_width=0.2, housing_length=0.2, housing_height=0.1, emitter_offset=(0, 0, -0.12))
        lo, hi = f.aim_bounds()
        assert f.photometric_depth == pytest.approx(-0.02)
        assert (lo[2], hi[2]) == pytest.approx((0.02, 0.12))

    def test_with_photometric_depth_keeps_in_plane_offset(self):
        f = Fixture(housing_height=0.1, emitter_offset=(0.02, 0.0, -0.05))
        g = f.with_photometric_depth(0.01)
        assert g.emitter_offset == pytest.approx((0.02, 0.0, -0.09))

    @pytest.mark.parametrize("face", FACES)
    @pytest.mark.parametrize("roll", [0, 90, 180, 270])
    def test_aim_to_housing_extents_round_trip(self, face, roll):
        f = Fixture(mount_face=face, roll=roll)
        L, W, H = f.aim_to_housing_extents(0.4, 0.2, 0.1)
        g = Fixture(housing_length=L, housing_width=W, housing_height=H, mount_face=face, roll=roll)
        lo, hi = g.aim_bounds()
        assert (hi - lo) == pytest.approx([0.4, 0.2, 0.1])

    def test_dict_round_trip(self):
        f = Fixture(housing_width=0.2, housing_length=0.4, housing_height=0.1,
                    mount_face="left", roll=270, emitter_offset=(0.01, -0.02, 0.03))
        d = f.to_dict()
        assert d["mount_face"] == "left" and d["roll"] == 270
        assert d["emitter_offset"] == [0.01, -0.02, 0.03]
        assert Fixture.from_dict(d) == f

    def test_dict_round_trip_default_offset(self):
        f = Fixture(housing_height=0.1)
        d = f.to_dict()
        assert d["emitter_offset"] is None
        assert Fixture.from_dict(d) == f

    def test_from_dict_legacy(self):
        f = Fixture.from_dict({"housing_width": 0.2, "housing_length": 0.2, "housing_height": 0.1,
                               "photometric_depth": 0.05, "shape": "rectangular"})
        assert f.emitter_offset == pytest.approx((0.0, 0.0, -0.05))
        assert f.photometric_depth == pytest.approx(0.05)
