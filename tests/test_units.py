"""Tests for unit conversion functions."""

import pytest
import numpy as np
from guv_calcs import convert_length, convert_time, convert_units


class TestLengthConversions:
    """Tests for length unit conversions."""

    def test_meters_to_feet(self):
        """1 meter should equal ~3.28 feet."""
        result = convert_length("meters", "feet", 1.0)
        expected = 1.0 / 0.3048  # ~3.28084
        assert np.isclose(result, expected, rtol=0.0001)

    def test_feet_to_meters(self):
        """1 foot should equal 0.3048 meters."""
        result = convert_length("feet", "meters", 1.0)
        expected = 0.3048
        assert np.isclose(result, expected, rtol=0.0001)

    def test_meters_to_centimeters(self):
        """1 meter should equal 100 centimeters."""
        result = convert_length("meters", "centimeters", 1.0)
        assert np.isclose(result, 100.0, rtol=0.0001)

    def test_centimeters_to_meters(self):
        """100 centimeters should equal 1 meter."""
        result = convert_length("centimeters", "meters", 100.0)
        assert np.isclose(result, 1.0, rtol=0.0001)

    def test_feet_to_inches(self):
        """1 foot should equal 12 inches."""
        result = convert_length("feet", "inches", 1.0)
        assert np.isclose(result, 12.0, rtol=0.0001)

    def test_inches_to_feet(self):
        """12 inches should equal 1 foot."""
        result = convert_length("inches", "feet", 12.0)
        assert np.isclose(result, 1.0, rtol=0.0001)

    def test_same_unit_no_change(self):
        """Converting to same unit should return original value."""
        result = convert_length("meters", "meters", 5.0)
        assert result == 5.0

    def test_multiple_values(self):
        """Should handle multiple values at once."""
        result = convert_length("meters", "feet", 1.0, 2.0, 3.0)
        assert len(result) == 3
        assert np.isclose(result[0], 1.0 / 0.3048, rtol=0.0001)

    def test_unit_aliases(self):
        """Should accept unit aliases like 'm' for 'meters'."""
        result1 = convert_length("m", "ft", 1.0)
        result2 = convert_length("meters", "feet", 1.0)
        assert np.isclose(result1, result2)

    def test_case_insensitive(self):
        """Unit names should be case insensitive."""
        result1 = convert_length("METERS", "FEET", 1.0)
        result2 = convert_length("meters", "feet", 1.0)
        assert np.isclose(result1, result2)


class TestTimeConversions:
    """Tests for time unit conversions."""

    def test_seconds_to_minutes(self):
        """60 seconds should equal 1 minute."""
        result = convert_time("seconds", "minutes", 60.0)
        assert np.isclose(result, 1.0, rtol=0.0001)

    def test_minutes_to_seconds(self):
        """1 minute should equal 60 seconds."""
        result = convert_time("minutes", "seconds", 1.0)
        assert np.isclose(result, 60.0, rtol=0.0001)

    def test_hours_to_seconds(self):
        """1 hour should equal 3600 seconds."""
        result = convert_time("hours", "seconds", 1.0)
        assert np.isclose(result, 3600.0, rtol=0.0001)

    def test_seconds_to_hours(self):
        """3600 seconds should equal 1 hour."""
        result = convert_time("seconds", "hours", 3600.0)
        assert np.isclose(result, 1.0, rtol=0.0001)

    def test_days_to_hours(self):
        """1 day should equal 24 hours."""
        result = convert_time("days", "hours", 1.0)
        assert np.isclose(result, 24.0, rtol=0.0001)

    def test_same_unit_no_change(self):
        """Converting to same unit should return original value."""
        result = convert_time("seconds", "seconds", 100.0)
        assert result == 100.0


class TestConvertUnitsGeneric:
    """Tests for the generic convert_units function."""

    def test_length_conversion(self):
        """convert_units should work for length."""
        result = convert_units("meters", "feet", 1.0, unit_type="length")
        expected = convert_length("meters", "feet", 1.0)
        assert np.isclose(result, expected)

    def test_time_conversion(self):
        """convert_units should work for time."""
        result = convert_units("seconds", "minutes", 60.0, unit_type="time")
        expected = convert_time("seconds", "minutes", 60.0)
        assert np.isclose(result, expected)

    def test_invalid_unit_type(self):
        """convert_units should raise error for invalid unit type."""
        with pytest.raises(ValueError):
            convert_units("meters", "feet", 1.0, unit_type="invalid")


class TestEdgeCases:
    """Tests for edge cases in unit conversions."""

    def test_zero_value(self):
        """Zero should convert to zero."""
        result = convert_length("meters", "feet", 0.0)
        assert result == 0.0

    def test_negative_value(self):
        """Negative values should convert correctly."""
        result = convert_length("meters", "feet", -1.0)
        expected = -1.0 / 0.3048
        assert np.isclose(result, expected, rtol=0.0001)

    def test_large_value(self):
        """Large values should convert correctly."""
        result = convert_length("meters", "feet", 1000000.0)
        expected = 1000000.0 / 0.3048
        assert np.isclose(result, expected, rtol=0.0001)

    def test_none_value(self):
        """None values should be handled."""
        result = convert_length("meters", "feet", None)
        assert result is None

    def test_sigfigs_parameter(self):
        """sigfigs parameter should round result."""
        result = convert_length("meters", "feet", 1.0, sigfigs=2)
        # Should be rounded to 2 decimal places
        assert result == round(result, 2)

    def test_invalid_unit(self):
        """Invalid unit should raise ValueError."""
        with pytest.raises(ValueError):
            convert_length("invalid_unit", "feet", 1.0)
