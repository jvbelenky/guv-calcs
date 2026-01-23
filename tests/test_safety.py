"""Tests for the safety module (PhotStandard, get_tlvs)."""

import pytest
import numpy as np
from guv_calcs import PhotStandard, get_tlvs, Spectrum


class TestPhotStandardCreation:
    """Tests for PhotStandard enum."""

    def test_acgih_value(self):
        """ACGIH standard should have correct value."""
        assert PhotStandard.ACGIH.value == "acgih"

    def test_ul8802_value(self):
        """UL8802 standard should have correct value."""
        assert PhotStandard.UL8802.value == "ul8802"

    def test_icnirp_value(self):
        """ICNIRP standard should have correct value."""
        assert PhotStandard.ICNIRP.value == "icnirp"


class TestPhotStandardFromToken:
    """Tests for PhotStandard.from_token parsing."""

    def test_from_token_acgih(self):
        """Should parse ACGIH from various strings."""
        assert PhotStandard.from_token("ACGIH") == PhotStandard.ACGIH
        assert PhotStandard.from_token("acgih") == PhotStandard.ACGIH
        assert PhotStandard.from_token("RP 27.1-22") == PhotStandard.ACGIH

    def test_from_token_ul8802(self):
        """Should parse UL8802 from various strings."""
        assert PhotStandard.from_token("UL8802") == PhotStandard.UL8802
        assert PhotStandard.from_token("ul8802") == PhotStandard.UL8802

    def test_from_token_icnirp(self):
        """Should parse ICNIRP from various strings."""
        assert PhotStandard.from_token("ICNIRP") == PhotStandard.ICNIRP
        assert PhotStandard.from_token("IEC 62471") == PhotStandard.ICNIRP

    def test_from_token_invalid(self):
        """Invalid token should raise ValueError."""
        with pytest.raises(ValueError):
            PhotStandard.from_token("invalid_standard")


class TestPhotStandardFromAny:
    """Tests for PhotStandard.from_any."""

    def test_from_any_passthrough(self):
        """from_any should return same PhotStandard if already PhotStandard."""
        standard = PhotStandard.ACGIH
        assert PhotStandard.from_any(standard) is standard

    def test_from_any_string(self):
        """from_any should parse string to PhotStandard."""
        result = PhotStandard.from_any("ACGIH")
        assert result == PhotStandard.ACGIH


class TestPhotStandardProperties:
    """Tests for PhotStandard properties."""

    def test_label_acgih(self):
        """ACGIH label should be descriptive."""
        label = PhotStandard.ACGIH.label
        assert "ANSI IES RP 27.1-22" in label
        assert "ACGIH" in label

    def test_label_icnirp(self):
        """ICNIRP label should be descriptive."""
        label = PhotStandard.ICNIRP.label
        assert "IEC 62471" in label
        assert "ICNIRP" in label

    def test_str_returns_label(self):
        """str() should return label."""
        assert str(PhotStandard.ACGIH) == PhotStandard.ACGIH.label

    def test_eye_weights(self):
        """eye_weights should return dict of wavelength: weight."""
        weights = PhotStandard.ACGIH.eye_weights
        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_skin_weights(self):
        """skin_weights should return dict of wavelength: weight."""
        weights = PhotStandard.ACGIH.skin_weights
        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_flags_meters(self):
        """flags should return dict with height for meters."""
        flags = PhotStandard.ACGIH.flags(units="meters")
        assert "height" in flags
        assert flags["height"] == 1.8

    def test_flags_feet(self):
        """flags should return dict with height for feet."""
        flags = PhotStandard.ACGIH.flags(units="feet")
        assert "height" in flags
        assert flags["height"] == 5.9

    def test_flags_ul8802(self):
        """UL8802 flags should differ from ACGIH."""
        flags = PhotStandard.UL8802.flags(units="meters")
        assert flags["height"] == 1.9
        assert flags["fov_vert"] == 180


class TestPhotStandardClassMethods:
    """Tests for PhotStandard class methods."""

    def test_dict(self):
        """dict() should return value: label mapping."""
        d = PhotStandard.dict()
        assert "acgih" in d
        assert "ul8802" in d
        assert "icnirp" in d

    def test_labels(self):
        """labels() should return list of label strings."""
        labels = PhotStandard.labels()
        assert len(labels) == 3
        assert all(isinstance(l, str) for l in labels)


class TestGetTlvs:
    """Tests for get_tlvs function."""

    def test_get_tlvs_wavelength_254(self):
        """get_tlvs with 254nm should return skin and eye TLVs."""
        skin, eye = get_tlvs(254, PhotStandard.ACGIH)
        assert skin is not None
        assert eye is not None
        assert skin > 0
        assert eye > 0

    def test_get_tlvs_wavelength_222(self):
        """get_tlvs with 222nm should return different TLVs."""
        skin_254, eye_254 = get_tlvs(254, PhotStandard.ACGIH)
        skin_222, eye_222 = get_tlvs(222, PhotStandard.ACGIH)
        # 222nm has higher TLVs than 254nm
        assert skin_222 > skin_254

    def test_get_tlvs_different_wavelengths(self):
        """Different wavelengths should give different TLVs."""
        skin_254, eye_254 = get_tlvs(254, PhotStandard.ACGIH)
        skin_280, eye_280 = get_tlvs(280, PhotStandard.ACGIH)
        # Different wavelengths have different TLVs
        assert skin_254 != skin_280 or eye_254 != eye_280

    def test_get_tlvs_with_spectrum(self):
        """get_tlvs should work with Spectrum object."""
        spec = Spectrum(
            [200, 210, 220, 230, 240, 250, 260],
            [0.01, 0.1, 0.5, 1.0, 0.5, 0.1, 0.01]
        )
        skin, eye = get_tlvs(spec, PhotStandard.ACGIH)
        assert skin is not None
        assert eye is not None

    def test_get_tlvs_invalid_type(self):
        """get_tlvs with invalid type should raise TypeError."""
        with pytest.raises(TypeError):
            get_tlvs("not_a_number", PhotStandard.ACGIH)
