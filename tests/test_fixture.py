"""Tests for the Fixture class."""

import pytest
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
    @pytest.mark.parametrize("face", FACES)
    def test_rotation_sends_outward_normal_to_beam(self, face):
        f = HousingFace.from_any(face)
        np.testing.assert_allclose(f.rotation @ f.normal, [0, 0, -1], atol=1e-12)

    @pytest.mark.parametrize("token,face", [("-z", "bottom"), ("+z", "top"), ("+x", "front"),
                                             ("-x", "back"), ("+y", "left"), ("-y", "right")])
    def test_axis_aliases(self, token, face):
        assert HousingFace.from_token(token) is HousingFace(face)

    def test_default_bottom(self):
        assert HousingFace.from_any(None) is HousingFace.BOTTOM

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            HousingFace.from_token("sideways")


class TestFixtureLayout:
    def test_defaults(self):
        f = Fixture()
        assert f.beam_face is HousingFace.BOTTOM
        assert f.roll == 0
        assert f.emitter_offset == (0.0, 0.0, 0.0)
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
        mats = {Fixture(beam_face=f, roll=r).housing_matrix.round(6).tobytes()
                for f in FACES for r in (0, 90, 180, 270)}
        assert len(mats) == 24

    def test_legacy_depth_equivalence(self):
        # today's box: z from -d to h-d, xy centered
        f = Fixture(housing_width=0.2, housing_length=0.4, housing_height=0.1, photometric_depth=0.03)
        assert f.emitter_offset == pytest.approx((0.0, 0.0, 0.03))
        assert f.photometric_depth == pytest.approx(0.03)
        lo, hi = f.aim_bounds()
        np.testing.assert_allclose(lo, [-0.2, -0.1, -0.03], atol=1e-12)
        np.testing.assert_allclose(hi, [0.2, 0.1, 0.07], atol=1e-12)

    def test_corner_order_matches_legacy(self):
        f = Fixture(housing_width=0.2, housing_length=0.4, housing_height=0.1)
        c = f.aim_corners()
        hl, hw = 0.2, 0.1
        expected = [[-hl, -hw, 0], [hl, -hw, 0], [hl, hw, 0], [-hl, hw, 0],
                    [-hl, -hw, 0.1], [hl, -hw, 0.1], [hl, hw, 0.1], [-hl, hw, 0.1]]
        np.testing.assert_allclose(c, expected, atol=1e-12)

    def test_surface_height_unions_along_aim(self):
        f = Fixture(housing_width=0.2, housing_length=0.2, housing_height=0.1)
        lo, hi = f.aim_bounds(surface_height=0.3)
        assert lo[2] == pytest.approx(-0.15)
        assert hi[2] == pytest.approx(0.15)

    def test_front_face_box(self):
        # 0.4 L x 0.2 W x 0.1 H box, beam out of +x face, emitter flush:
        # the whole box lies behind the emitter along aim +z, 0.4 deep
        f = Fixture(housing_width=0.2, housing_length=0.4, housing_height=0.1, beam_face="front")
        lo, hi = f.aim_bounds()
        assert lo[2] == pytest.approx(0.0)
        assert hi[2] == pytest.approx(0.4)
        assert sorted([hi[0] - lo[0], hi[1] - lo[1]]) == pytest.approx([0.1, 0.2])

    def test_offset_moves_box_opposite(self):
        base = Fixture(housing_width=0.2, housing_length=0.4, housing_height=0.1)
        moved = Fixture(housing_width=0.2, housing_length=0.4, housing_height=0.1,
                        emitter_offset=(0.15, 0.0, 0.0))
        lo0, _ = base.aim_bounds()
        lo1, _ = moved.aim_bounds()
        assert lo1[0] == pytest.approx(lo0[0] - 0.15)

    def test_protruding_emitter(self):
        # emitter 5 cm in front of the bottom face: box starts 5 cm behind it
        f = Fixture(housing_width=0.2, housing_length=0.2, housing_height=0.1, emitter_offset=(0, 0, -0.05))
        lo, hi = f.aim_bounds()
        assert f.photometric_depth == pytest.approx(-0.05)
        assert lo[2] == pytest.approx(0.05)
        assert hi[2] == pytest.approx(0.15)

    def test_with_photometric_depth_keeps_in_plane_offset(self):
        f = Fixture(housing_height=0.1, beam_face="front", emitter_offset=(0.0, 0.02, 0.03))
        g = f.with_photometric_depth(0.01)
        # front face inward normal is -x
        assert g.emitter_offset == pytest.approx((-0.01, 0.02, 0.03))

    @pytest.mark.parametrize("face", FACES)
    @pytest.mark.parametrize("roll", [0, 90, 180, 270])
    def test_aim_to_housing_extents_round_trip(self, face, roll):
        f = Fixture(beam_face=face, roll=roll)
        L, W, H = f.aim_to_housing_extents(0.4, 0.2, 0.1)
        back = Fixture(housing_length=L, housing_width=W, housing_height=H, beam_face=face, roll=roll)
        lo, hi = back.aim_bounds()
        assert (hi - lo) == pytest.approx([0.4, 0.2, 0.1])

    def test_dict_round_trip(self):
        f = Fixture(housing_width=0.2, housing_length=0.4, housing_height=0.1,
                    beam_face="left", roll=270, emitter_offset=(0.01, -0.02, 0.03))
        d = f.to_dict()
        assert d["beam_face"] == "left" and d["roll"] == 270
        assert d["emitter_offset"] == [0.01, -0.02, 0.03]
        assert Fixture.from_dict(d) == f

    def test_from_dict_legacy(self):
        f = Fixture.from_dict({"housing_width": 0.2, "housing_length": 0.2, "housing_height": 0.1,
                               "photometric_depth": 0.05, "shape": "rectangular"})
        assert f.emitter_offset == pytest.approx((0.0, 0.0, 0.05))
