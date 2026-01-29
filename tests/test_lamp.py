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

    def test_user_width_preserved_after_ies_load(self):
        """User-set width should not be overwritten by IES file."""
        lamp = Lamp.from_keyword("aerolamp", width=0.5)
        assert lamp.width == 0.5
        # Verify tracking field is set
        assert lamp.surface._user_width == 0.5

    def test_user_length_preserved_after_ies_load(self):
        """User-set length should not be overwritten by IES file."""
        lamp = Lamp.from_keyword("aerolamp", length=0.8)
        assert lamp.length == 0.8
        # Verify tracking field is set
        assert lamp.surface._user_length == 0.8

    def test_ies_values_used_when_no_user_values(self):
        """IES width/length should be used when user doesn't specify."""
        lamp = Lamp.from_keyword("aerolamp")
        # Should have IES values, not 0.0
        assert lamp.width > 0 or lamp.length > 0  # At least one should be non-zero from IES
        # Tracking fields should be None (user didn't specify)
        assert lamp.surface._user_width is None
        assert lamp.surface._user_length is None

    def test_set_width_after_ies_load_updates_tracking(self, basic_lamp):
        """set_width() after IES load should update tracking field."""
        # Initially tracking is None
        assert basic_lamp.surface._user_width is None
        # Set width explicitly
        basic_lamp.set_width(0.25)
        assert basic_lamp.width == 0.25
        assert basic_lamp.surface._user_width == 0.25

    def test_negative_width_raises_error(self, basic_lamp):
        """Negative width should raise ValueError."""
        with pytest.raises(ValueError, match="width must be non-negative"):
            basic_lamp.set_width(-0.1)

    def test_negative_length_raises_error(self, basic_lamp):
        """Negative length should raise ValueError."""
        with pytest.raises(ValueError, match="length must be non-negative"):
            basic_lamp.set_length(-0.1)

    def test_lamp_surface_to_dict(self, basic_lamp):
        """LampSurface.to_dict() should include all necessary fields."""
        basic_lamp.set_width(0.15)
        surface_dict = basic_lamp.surface.to_dict()

        assert "width" in surface_dict
        assert "length" in surface_dict
        assert "units" in surface_dict
        assert "source_density" in surface_dict
        assert "_user_width" in surface_dict
        assert "_user_length" in surface_dict

        assert surface_dict["width"] == 0.15
        assert surface_dict["_user_width"] == 0.15

    def test_lamp_surface_round_trip(self, basic_lamp):
        """User-set width/length should survive to_dict/from_dict round trip."""
        basic_lamp.set_width(0.2)
        basic_lamp.set_length(0.3)

        data = basic_lamp.to_dict()
        loaded = Lamp.from_dict(data)

        assert loaded.width == 0.2
        assert loaded.length == 0.3
        assert loaded.surface._user_width == 0.2
        assert loaded.surface._user_length == 0.3


class TestIntensityMap:
    """Tests for IntensityMap class."""

    def test_none_source_returns_none_original(self):
        """IntensityMap with None source should have None original."""
        from guv_calcs.intensity_map import IntensityMap
        im = IntensityMap(None)
        assert im.original is None
        assert im.normalized is None

    def test_array_source(self):
        """IntensityMap should accept numpy array."""
        from guv_calcs.intensity_map import IntensityMap
        data = np.array([[1, 2], [3, 4]])
        im = IntensityMap(data)
        assert im.original is not None
        np.testing.assert_array_equal(im.original, data)

    def test_list_source(self):
        """IntensityMap should accept list."""
        from guv_calcs.intensity_map import IntensityMap
        data = [[1, 2], [3, 4]]
        im = IntensityMap(data)
        assert im.original is not None
        np.testing.assert_array_equal(im.original, np.array(data))

    def test_normalized_property(self):
        """Normalized should divide by mean."""
        from guv_calcs.intensity_map import IntensityMap
        data = np.array([[1, 2], [3, 4]])  # mean = 2.5
        im = IntensityMap(data)
        expected = data / data.mean()
        np.testing.assert_array_almost_equal(im.normalized, expected)

    def test_resample_no_map_returns_ones(self):
        """Resample with no map should return array of ones."""
        from guv_calcs.intensity_map import IntensityMap
        im = IntensityMap(None)

        def points_gen(u, v):
            return np.linspace(-1, 1, u), np.linspace(-1, 1, v)

        result = im.resample(3, 4, points_gen)
        assert result.shape == (3, 4)
        np.testing.assert_array_equal(result, np.ones((3, 4)))

    def test_resample_same_shape_returns_normalized(self):
        """Resample with same shape should return normalized map."""
        from guv_calcs.intensity_map import IntensityMap
        data = np.array([[1, 2], [3, 4]])
        im = IntensityMap(data)

        def points_gen(u, v):
            return np.linspace(-1, 1, u), np.linspace(-1, 1, v)

        result = im.resample(2, 2, points_gen)
        np.testing.assert_array_almost_equal(result, im.normalized)

    def test_nan_values_rejected(self):
        """IntensityMap should reject data with NaN values."""
        from guv_calcs.intensity_map import IntensityMap
        data = np.array([[1, np.nan], [3, 4]])
        with pytest.warns(UserWarning, match="invalid values"):
            im = IntensityMap(data)
        assert im.original is None

    def test_invalid_type_rejected(self):
        """IntensityMap should reject invalid types."""
        from guv_calcs.intensity_map import IntensityMap
        with pytest.warns(UserWarning, match="invalid"):
            im = IntensityMap(12345)
        assert im.original is None


class TestLazyCaching:
    """Tests for lazy caching behavior in LampSurface."""

    def test_grid_not_computed_until_accessed(self, basic_lamp):
        """Grid should not be computed until properties are accessed."""
        # After initialization, the grid should be marked dirty but not computed yet
        # We can test this by checking that accessing surface_points triggers computation
        surface = basic_lamp.surface
        surface._grid_dirty = True
        surface._surface_points_cache = None

        # Access should trigger computation
        _ = surface.surface_points
        assert not surface._grid_dirty
        assert surface._surface_points_cache is not None

    def test_position_not_recomputed_if_clean(self, basic_lamp):
        """Position should not be recomputed if cache is clean."""
        surface = basic_lamp.surface
        # Access to ensure computed
        pos1 = surface.position

        # Modify cache directly (would not happen in normal use)
        surface._position_cache = np.array([999, 999, 999])

        # Should return cached value since dirty flag is False
        pos2 = surface.position
        np.testing.assert_array_equal(pos2, np.array([999, 999, 999]))

    def test_set_width_invalidates_grid(self, basic_lamp):
        """Setting width should invalidate grid cache."""
        surface = basic_lamp.surface
        # Ensure grid is computed
        _ = surface.surface_points
        assert not surface._grid_dirty

        # Set width should invalidate
        basic_lamp.set_width(0.5)
        assert surface._grid_dirty

    def test_move_invalidates_position(self, basic_lamp):
        """Moving lamp should invalidate surface position cache."""
        surface = basic_lamp.surface
        # Ensure position is computed
        _ = surface.position
        assert not surface._position_dirty

        # Move lamp should invalidate surface position (via geometry)
        basic_lamp.move(1.0, 2.0, 3.0)
        assert surface._position_dirty
