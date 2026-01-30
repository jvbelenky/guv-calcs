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

    def test_fixture_is_frozen(self):
        """Fixture should be immutable (frozen dataclass)."""
        f = Fixture(housing_width=0.5)
        with pytest.raises(AttributeError):
            f.housing_width = 1.0

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
