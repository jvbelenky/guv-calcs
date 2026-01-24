"""Tests for the Spectrum class."""

import pytest
import numpy as np
from guv_calcs import Spectrum, sum_spectrum


class TestSpectrumCreation:
    """Tests for Spectrum initialization."""

    def test_basic_creation(self):
        """Spectrum should be created from wavelengths and intensities."""
        wavelengths = [200, 220, 240, 260, 280, 300]
        intensities = [0.1, 0.5, 1.0, 0.8, 0.3, 0.1]
        spec = Spectrum(wavelengths, intensities)
        assert spec is not None
        assert len(spec.wavelengths) == 6

    def test_arrays_sorted(self):
        """Spectrum should sort wavelengths in ascending order."""
        wavelengths = [300, 200, 250]
        intensities = [0.1, 0.3, 0.5]
        spec = Spectrum(wavelengths, intensities)
        assert spec.wavelengths[0] == 200
        assert spec.wavelengths[-1] == 300

    def test_mismatched_lengths_raises(self):
        """Mismatched wavelength/intensity lengths should raise an error."""
        with pytest.raises((ValueError, IndexError)):
            Spectrum([200, 220, 240], [0.1, 0.5])

    def test_from_dict(self):
        """Spectrum.from_dict should create from dictionary."""
        data = {
            "wavelength": [200, 220, 240],
            "intensity": [0.1, 0.5, 1.0],
        }
        spec = Spectrum.from_dict(data)
        assert len(spec.wavelengths) == 3

    def test_from_source_none(self):
        """Spectrum.from_source(None) should return None."""
        result = Spectrum.from_source(None)
        assert result is None

    def test_from_source_spectrum(self):
        """Spectrum.from_source should return same Spectrum if passed Spectrum."""
        spec = Spectrum([200, 220], [0.1, 0.5])
        result = Spectrum.from_source(spec)
        assert result is spec

    def test_from_source_dict(self):
        """Spectrum.from_source should work with dict."""
        data = {"wv": [200, 220], "int": [0.1, 0.5]}
        spec = Spectrum.from_source(data)
        assert spec is not None

    def test_from_source_tuple(self):
        """Spectrum.from_source should work with (wavelengths, intensities) tuple."""
        spec = Spectrum.from_source(([200, 220], [0.1, 0.5]))
        assert spec is not None


class TestSpectrumProperties:
    """Tests for Spectrum properties."""

    def test_peak_wavelength(self):
        """peak_wavelength should return wavelength of max intensity."""
        spec = Spectrum([200, 220, 240, 260], [0.1, 0.5, 1.0, 0.3])
        assert spec.peak_wavelength == 240

    def test_raw_values_preserved(self):
        """raw_wavelengths and raw_intensities should preserve original values."""
        wavelengths = [200, 220, 240]
        intensities = [0.1, 0.5, 1.0]
        spec = Spectrum(wavelengths, intensities)
        assert np.array_equal(spec.raw_wavelengths, np.array([200, 220, 240]))
        assert np.array_equal(spec.raw_intensities, np.array([0.1, 0.5, 1.0]))


class TestSpectrumOperations:
    """Tests for Spectrum operations."""

    def test_normalize(self):
        """normalize should scale max intensity to specified value."""
        spec = Spectrum([200, 220, 240], [0.1, 0.5, 1.0])
        spec.normalize(100)
        assert max(spec.intensities) == 100

    def test_normalize_default(self):
        """normalize default should scale max to 1."""
        spec = Spectrum([200, 220, 240], [0.1, 0.5, 2.0])
        spec.normalize()
        assert max(spec.intensities) == 1

    def test_filter_range(self):
        """filter should limit wavelengths to specified range."""
        spec = Spectrum([200, 220, 240, 260, 280], [0.1, 0.3, 0.5, 0.3, 0.1])
        spec.filter(minval=210, maxval=270)
        assert min(spec.wavelengths) >= 210
        assert max(spec.wavelengths) <= 270

    def test_revert(self):
        """revert should restore original wavelengths and intensities."""
        spec = Spectrum([200, 220, 240, 260], [0.1, 0.3, 0.5, 0.3])
        spec.filter(minval=210, maxval=250)
        spec.revert()
        assert len(spec.wavelengths) == 4

    def test_sum(self):
        """sum should integrate spectrum."""
        spec = Spectrum([200, 220, 240], [1.0, 1.0, 1.0])
        total = spec.sum()
        assert total > 0

    def test_scale(self):
        """scale should adjust total spectral power."""
        spec = Spectrum([200, 220, 240], [0.1, 0.5, 1.0])
        original_sum = spec.sum()
        target = original_sum * 2
        spec.scale(target)
        assert np.isclose(spec.sum(), target, rtol=0.01)


class TestSpectrumSerialization:
    """Tests for Spectrum serialization."""

    def test_to_dict(self):
        """to_dict should return dict with wavelengths and intensities."""
        spec = Spectrum([200, 220, 240], [0.1, 0.5, 1.0])
        data = spec.to_dict()
        assert "Wavelength" in data
        assert "Unweighted Relative Intensity" in data

    def test_to_dict_as_string(self):
        """to_dict with as_string=True should return string values."""
        spec = Spectrum([200, 220, 240], [0.1, 0.5, 1.0])
        data = spec.to_dict(as_string=True)
        assert isinstance(data["Wavelength"], str)

    def test_to_csv_returns_bytes(self):
        """to_csv without filename should return bytes."""
        spec = Spectrum([200, 220, 240], [0.1, 0.5, 1.0])
        result = spec.to_csv()
        assert isinstance(result, bytes)


class TestSpectrumWeighting:
    """Tests for spectral weighting."""

    def test_weighted_intensities_created(self):
        """Spectrum should have weighted_intensities after creation."""
        spec = Spectrum([200, 220, 240, 260, 280], [0.1, 0.3, 0.5, 0.3, 0.1])
        assert spec.weighted_intensities is not None
        assert len(spec.weighted_intensities) > 0


class TestSumSpectrum:
    """Tests for the sum_spectrum function."""

    def test_sum_spectrum_basic(self):
        """sum_spectrum should integrate wavelength-intensity pairs."""
        wavelengths = np.array([200, 220, 240])
        intensities = np.array([1.0, 1.0, 1.0])
        result = sum_spectrum(wavelengths, intensities)
        # Integration: 1.0 * (220-200) + 1.0 * (240-220) = 40
        assert result == 40

    def test_sum_spectrum_varying_intensity(self):
        """sum_spectrum should handle varying intensities."""
        wavelengths = np.array([200, 220, 240])
        intensities = np.array([0.5, 1.0, 0.5])
        result = sum_spectrum(wavelengths, intensities)
        # Integration: 1.0 * 20 + 0.5 * 20 = 30
        assert result == 30
