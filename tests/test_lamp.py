"""Tests for the Lamp class."""

import pytest
import numpy as np
from guv_calcs import Lamp


class TestLampLoading:
    """Tests for loading lamps from various sources."""

    def test_from_keyword_valid(self):
        """Valid keyword should load lamp successfully."""
        lamp = Lamp.from_keyword("aerolamp")
        assert lamp is not None
        assert lamp.lamp_id == "aerolamp"

    def test_from_keyword_invalid(self):
        """Invalid keyword should raise KeyError."""
        with pytest.raises(KeyError):
            Lamp.from_keyword("nonexistent_lamp")

    def test_from_keyword_type_error(self):
        """Non-string keyword should raise TypeError."""
        with pytest.raises(TypeError):
            Lamp.from_keyword(123)

    def test_from_index_valid(self):
        """Valid index should load lamp successfully."""
        lamp = Lamp.from_index(0)
        assert lamp is not None

    def test_from_index_invalid(self):
        """Out-of-range index should raise IndexError."""
        with pytest.raises(IndexError):
            Lamp.from_index(999)

    def test_from_index_type_error(self):
        """Non-integer index should raise TypeError."""
        with pytest.raises(TypeError):
            Lamp.from_index("0")

    def test_all_keywords_load(self):
        """All valid lamp keywords should load successfully."""
        lamp = Lamp.from_keyword("aerolamp")
        for keyword in lamp.keywords:
            loaded = Lamp.from_keyword(keyword)
            assert loaded is not None
            assert loaded.ies is not None


class TestLampPositioning:
    """Tests for lamp position and orientation."""

    def test_default_position(self):
        """Lamp should have default position of (0, 0, 0)."""
        lamp = Lamp.from_keyword("aerolamp")
        assert lamp.x == 0
        assert lamp.y == 0
        assert lamp.z == 0

    def test_move_updates_position(self, basic_lamp):
        """move() should update lamp position."""
        basic_lamp.move(3, 2, 2.7)
        assert basic_lamp.x == 3
        assert basic_lamp.y == 2
        assert basic_lamp.z == 2.7

    def test_move_returns_self(self, basic_lamp):
        """move() should return self for method chaining."""
        result = basic_lamp.move(1, 2, 3)
        assert result is basic_lamp

    def test_aim_updates_orientation(self, basic_lamp):
        """aim() should update lamp aim point."""
        basic_lamp.move(3, 2, 2.7).aim(3, 2, 0)
        assert basic_lamp.aimx == 3
        assert basic_lamp.aimy == 2
        assert basic_lamp.aimz == 0

    def test_aim_returns_self(self, basic_lamp):
        """aim() should return self for method chaining."""
        # Lamp must be at different position than aim point
        basic_lamp.move(3, 2, 2.7)
        result = basic_lamp.aim(3, 2, 0)
        assert result is basic_lamp

    def test_position_property(self, basic_lamp):
        """position property should return correct values."""
        basic_lamp.move(1, 2, 3)
        pos = basic_lamp.position
        assert pos[0] == 1
        assert pos[1] == 2
        assert pos[2] == 3

    def test_aim_point_property(self, basic_lamp):
        """aim_point property should return correct values."""
        basic_lamp.move(1, 2, 3)  # Move first so aim point is different
        basic_lamp.aim(4, 5, 6)
        aim = basic_lamp.aim_point
        assert aim[0] == 4
        assert aim[1] == 5
        assert aim[2] == 6

    def test_rotate(self, basic_lamp):
        """rotate() should update lamp angle."""
        basic_lamp.rotate(90)
        assert basic_lamp.angle == 90


class TestLampPhotometry:
    """Tests for lamp photometric data."""

    def test_photometry_loaded(self, basic_lamp):
        """Lamp should have photometry data loaded."""
        assert basic_lamp.ies is not None
        assert basic_lamp.photometry is not None

    def test_photometry_values(self, basic_lamp):
        """Photometry values should exist and be positive."""
        values = basic_lamp.values
        assert values is not None
        assert values.size > 0
        assert np.all(values >= 0)

    def test_max_value(self, basic_lamp):
        """max() should return a positive value."""
        max_val = basic_lamp.max()
        assert max_val > 0

    def test_center_value(self, basic_lamp):
        """center() should return a positive value."""
        center_val = basic_lamp.center()
        assert center_val > 0

    def test_total_power(self, basic_lamp):
        """get_total_power() should return a positive value."""
        total = basic_lamp.get_total_power()
        assert total > 0


class TestLampScaling:
    """Tests for lamp scaling functionality."""

    def test_scale_multiplies_values(self, basic_lamp):
        """scale() should multiply photometry values."""
        original_max = basic_lamp.max()
        basic_lamp.scale(2.0)
        assert np.isclose(basic_lamp.max(), original_max * 2.0, rtol=0.01)

    def test_scale_returns_self(self, basic_lamp):
        """scale() should return self for method chaining."""
        result = basic_lamp.scale(1.5)
        assert result is basic_lamp

    def test_scaling_factor_updated(self, basic_lamp):
        """scaling_factor property should be updated after scaling."""
        basic_lamp.scale(2.0)
        assert np.isclose(basic_lamp.scaling_factor, 2.0, rtol=0.01)

    def test_scale_to_max(self, basic_lamp):
        """scale_to_max() should set maximum value."""
        target_max = 100.0
        basic_lamp.scale_to_max(target_max)
        assert np.isclose(basic_lamp.max(), target_max, rtol=0.01)

    def test_scale_to_center(self, basic_lamp):
        """scale_to_center() should set center value."""
        target_center = 50.0
        basic_lamp.scale_to_center(target_center)
        assert np.isclose(basic_lamp.center(), target_center, rtol=0.01)


class TestLampSpectrum:
    """Tests for lamp spectrum handling."""

    def test_spectra_loaded(self, basic_lamp):
        """Lamp loaded from keyword should have spectra."""
        assert basic_lamp.spectra is not None

    def test_wavelength_property(self, basic_lamp):
        """wavelength property should return a value."""
        assert basic_lamp.wavelength is not None

    def test_set_wavelength(self, basic_lamp):
        """set_wavelength() should update wavelength."""
        basic_lamp.load_spectra(None).set_wavelength(254)
        assert basic_lamp.wavelength == 254

    def test_get_tlvs(self, basic_lamp):
        """get_tlvs() should return skin and eye TLV values."""
        from guv_calcs import PhotStandard
        # get_tlvs expects a PhotStandard enum object
        skin_tlv, eye_tlv = basic_lamp.get_tlvs(PhotStandard.ACGIH)
        assert skin_tlv is not None or basic_lamp.spectra is None
        assert eye_tlv is not None or basic_lamp.spectra is None


class TestLampSerialization:
    """Tests for lamp serialization."""

    def test_to_dict(self, basic_lamp):
        """to_dict() should return dict with lamp properties."""
        data = basic_lamp.to_dict()
        assert "lamp_id" in data
        assert "x" in data
        assert "y" in data
        assert "z" in data

    def test_from_dict_round_trip(self, positioned_lamp):
        """Lamp should survive to_dict/from_dict round trip."""
        data = positioned_lamp.to_dict()
        loaded = Lamp.from_dict(data)
        assert loaded.x == positioned_lamp.x
        assert loaded.y == positioned_lamp.y
        assert loaded.z == positioned_lamp.z
        assert loaded.aimx == positioned_lamp.aimx

    def test_equality(self):
        """Two lamps with same properties should be equal."""
        lamp1 = Lamp.from_keyword("aerolamp").move(1, 2, 3)
        lamp2 = Lamp.from_keyword("aerolamp").move(1, 2, 3)
        assert lamp1 == lamp2


class TestLampSurface:
    """Tests for lamp surface properties."""

    def test_surface_dimensions(self, basic_lamp):
        """Lamp should have surface dimensions."""
        assert basic_lamp.width is not None
        assert basic_lamp.length is not None

    def test_set_width(self, basic_lamp):
        """set_width() should update lamp width."""
        basic_lamp.set_width(0.1)
        assert basic_lamp.width == 0.1

    def test_set_length(self, basic_lamp):
        """set_length() should update lamp length."""
        basic_lamp.set_length(0.2)
        assert basic_lamp.length == 0.2

    def test_source_density(self, basic_lamp):
        """Lamp should have source density property."""
        assert basic_lamp.surface.source_density >= 1
