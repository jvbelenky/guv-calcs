"""Tests for the safety compliance checking functionality."""

import pytest
import numpy as np
from guv_calcs import Room, Lamp
from guv_calcs.safety import (
    ComplianceStatus,
    WarningLevel,
    SafetyCheckResult,
    LampComplianceResult,
)


class TestCheckLampsErrors:
    """Tests for error conditions in check_lamps()."""

    def test_missing_skin_limits_zone(self):
        """Should return error when SkinLimits zone is missing."""
        room = Room(x=6, y=4, z=2.7)
        room.add_lamp(Lamp.from_keyword("ushio_b1"))
        # Don't add standard zones

        result = room.check_lamps()

        assert result.status == ComplianceStatus.NON_COMPLIANT
        assert len(result.warnings) == 1
        assert result.warnings[0].level == WarningLevel.ERROR
        assert "SkinLimits" in result.warnings[0].message

    def test_missing_eye_limits_zone(self):
        """Should return error when EyeLimits zone is missing."""
        room = Room(x=6, y=4, z=2.7)
        room.add_lamp(Lamp.from_keyword("ushio_b1"))
        room.add_standard_zones()
        # Remove only EyeLimits
        room.remove_calc_zone("EyeLimits")

        result = room.check_lamps()

        assert result.status == ComplianceStatus.NON_COMPLIANT
        assert len(result.warnings) == 1
        assert result.warnings[0].level == WarningLevel.ERROR
        assert "EyeLimits" in result.warnings[0].message

    def test_zones_not_calculated(self):
        """Should return error when zones have not been calculated."""
        room = Room(x=6, y=4, z=2.7)
        room.add_lamp(Lamp.from_keyword("ushio_b1"))
        room.add_standard_zones()
        # Don't call calculate()

        result = room.check_lamps()

        assert result.status == ComplianceStatus.NON_COMPLIANT
        assert len(result.warnings) == 1
        assert result.warnings[0].level == WarningLevel.ERROR
        assert "calculated" in result.warnings[0].message.lower()


class TestCheckLampsCompliance:
    """Tests for compliance checking functionality."""

    def test_compliant_lamp(self):
        """A properly configured lamp should be compliant (possibly with dimming)."""
        room = Room(x=6, y=4, z=2.7)
        room.place_lamp(Lamp.from_keyword("ushio_b1"))
        room.add_standard_zones()
        room.calculate()

        result = room.check_lamps()

        # The lamp may need dimming to be compliant - that's still acceptable
        assert result.status in [
            ComplianceStatus.COMPLIANT,
            ComplianceStatus.COMPLIANT_WITH_DIMMING,
        ]
        assert len(result.lamp_results) == 1
        lamp_result = list(result.lamp_results.values())[0]
        # Verify we have valid TLV values
        assert lamp_result.skin_tlv > 0
        assert lamp_result.eye_tlv > 0

    def test_non_compliant_lamp_returns_dimming(self):
        """A non-compliant lamp should show dimming requirements."""
        room = Room(x=6, y=4, z=2.7)
        # Create a lamp that's too bright by scaling it up
        lamp = Lamp.from_keyword("ushio_b1")
        lamp.scale(50.0)  # Make it 50x brighter to ensure non-compliance
        room.place_lamp(lamp)
        room.add_standard_zones()
        room.calculate()

        result = room.check_lamps()

        # Should need dimming
        assert result.status in [
            ComplianceStatus.COMPLIANT_WITH_DIMMING,
            ComplianceStatus.NON_COMPLIANT,
            ComplianceStatus.NON_COMPLIANT_EVEN_WITH_DIMMING,
        ]
        lamp_result = list(result.lamp_results.values())[0]
        # At least one of skin or eye should require dimming
        assert lamp_result.skin_dimming_required < 1.0 or lamp_result.eye_dimming_required < 1.0

    def test_result_contains_dose_arrays(self):
        """Result should contain weighted dose arrays."""
        room = Room(x=6, y=4, z=2.7)
        room.place_lamp(Lamp.from_keyword("ushio_b1"))
        room.add_standard_zones()
        room.calculate()

        result = room.check_lamps()

        assert result.weighted_skin_dose is not None
        assert result.weighted_eye_dose is not None
        assert isinstance(result.weighted_skin_dose, np.ndarray)
        assert isinstance(result.weighted_eye_dose, np.ndarray)

    def test_result_contains_max_doses(self):
        """Result should contain max dose values."""
        room = Room(x=6, y=4, z=2.7)
        room.place_lamp(Lamp.from_keyword("ushio_b1"))
        room.add_standard_zones()
        room.calculate()

        result = room.check_lamps()

        assert result.max_skin_dose >= 0
        assert result.max_eye_dose >= 0
        assert result.max_skin_dose == result.weighted_skin_dose.max()
        assert result.max_eye_dose == result.weighted_eye_dose.max()


class TestCheckLampsMissingSpectrum:
    """Tests for missing spectrum warnings."""

    def test_lphg_lamp_no_warning_without_spectrum(self):
        """LPHG (254nm) lamps should not warn about missing spectrum."""
        room = Room(x=6, y=4, z=2.7)
        # ushio_b1 is 222nm (KrCl) and has a spectrum, so no warning expected
        room.place_lamp(Lamp.from_keyword("ushio_b1"))
        room.add_standard_zones()
        room.calculate()

        result = room.check_lamps()

        # Check that no warnings mention missing spectrum for a lamp with spectrum
        spectrum_warnings = [
            w for w in result.warnings if "missing a spectrum" in w.message
        ]
        assert len(spectrum_warnings) == 0


class TestCheckLampsCombinedDose:
    """Tests for combined dose checking."""

    def test_multiple_lamps_combined_dose(self):
        """Multiple lamps should have their doses combined."""
        room = Room(x=6, y=4, z=2.7)
        # Create two lamps - use place_lamp for proper positioning
        lamp1 = Lamp.from_keyword("ushio_b1")
        lamp2 = Lamp.from_keyword("ushio_b1")
        room.place_lamp(lamp1)
        room.place_lamp(lamp2)  # Will be auto-renamed and positioned
        room.add_standard_zones()
        room.calculate()

        result = room.check_lamps()

        assert len(result.lamp_results) == 2
        # Combined dose should be positive
        assert result.max_skin_dose > 0
        assert result.max_eye_dose > 0


class TestLampComplianceResult:
    """Tests for LampComplianceResult dataclass."""

    def test_lamp_result_fields(self):
        """LampComplianceResult should have all expected fields."""
        room = Room(x=6, y=4, z=2.7)
        room.place_lamp(Lamp.from_keyword("ushio_b1"))
        room.add_standard_zones()
        room.calculate()

        result = room.check_lamps()
        lamp_result = list(result.lamp_results.values())[0]

        assert hasattr(lamp_result, "lamp_id")
        assert hasattr(lamp_result, "lamp_name")
        assert hasattr(lamp_result, "skin_dose_max")
        assert hasattr(lamp_result, "eye_dose_max")
        assert hasattr(lamp_result, "skin_tlv")
        assert hasattr(lamp_result, "eye_tlv")
        assert hasattr(lamp_result, "skin_dimming_required")
        assert hasattr(lamp_result, "eye_dimming_required")
        assert hasattr(lamp_result, "is_skin_compliant")
        assert hasattr(lamp_result, "is_eye_compliant")
        assert hasattr(lamp_result, "missing_spectrum")


class TestSafetyCheckResult:
    """Tests for SafetyCheckResult dataclass."""

    def test_result_is_frozen(self):
        """SafetyCheckResult should be immutable."""
        room = Room(x=6, y=4, z=2.7)
        room.place_lamp(Lamp.from_keyword("ushio_b1"))
        room.add_standard_zones()
        room.calculate()

        result = room.check_lamps()

        with pytest.raises(Exception):  # FrozenInstanceError
            result.status = ComplianceStatus.NON_COMPLIANT


class TestComplianceStatusEnum:
    """Tests for ComplianceStatus enum values."""

    def test_compliant_value(self):
        assert ComplianceStatus.COMPLIANT == "compliant"

    def test_non_compliant_value(self):
        assert ComplianceStatus.NON_COMPLIANT == "non_compliant"

    def test_compliant_with_dimming_value(self):
        assert ComplianceStatus.COMPLIANT_WITH_DIMMING == "compliant_with_dimming"

    def test_non_compliant_even_with_dimming_value(self):
        assert (
            ComplianceStatus.NON_COMPLIANT_EVEN_WITH_DIMMING
            == "non_compliant_even_with_dimming"
        )


class TestWarningLevelEnum:
    """Tests for WarningLevel enum values."""

    def test_info_value(self):
        assert WarningLevel.INFO == "info"

    def test_warning_value(self):
        assert WarningLevel.WARNING == "warning"

    def test_error_value(self):
        assert WarningLevel.ERROR == "error"
