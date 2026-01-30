from dataclasses import dataclass, field
from enum import StrEnum
import numpy as np
from .lamp.spectrum import Spectrum, log_interp, sum_spectrum
from .io import get_spectral_weightings
from .lamp.lamp_type import GUVType


class PhotStandard(StrEnum):
    ACGIH = "acgih"
    UL8802 = "ul8802"
    ICNIRP = "icnirp"
    # GB28235 = "gb28235" # must be below 5 uW/cm2 everywhere at 2.1 m and below--maybe should correspond to a volume?

    @classmethod
    def from_any(cls, arg) -> "PhotStandard":
        if isinstance(arg, cls):
            return arg
        return cls.from_token(arg)

    @classmethod
    def from_token(cls, token: str) -> "PhotStandard":
        token = str(token).strip().upper()
        if "UL8802" in token:
            return cls.UL8802
        if "ACGIH" in token or "RP 27.1-22" in token:
            return cls.ACGIH
        if "ICNIRP" in token or "IEC 62471" in token:
            return cls.ICNIRP
        # if "GB" in token or "CHINA" in token:
        # return cls.GB28235
        raise ValueError(f"Unknown PhotStandard {token}")

    @property
    def label(self):
        if self is PhotStandard.ACGIH:
            return "ANSI IES RP 27.1-22 (ACGIH Limits)"
        if self is PhotStandard.UL8802:
            return "ANSI IES RP 27.1-22 (ACGIH Limits) - UL8802"
        if self is PhotStandard.ICNIRP:
            return "IEC 62471-6:2022 (ICNIRP Limits)"
        # if self is PhotStandard.GB28235:
        # return "GB 28235 (China)"

    @property
    def eye_weights(self):
        weights = get_spectral_weightings()
        if self in (PhotStandard.ACGIH, PhotStandard.UL8802):
            key = "ANSI IES RP 27.1-22 (Eye)"
        elif self is PhotStandard.ICNIRP:
            key = "IEC 62471-6:2022 (Eye/Skin)"
        return {k: v for k, v in zip(weights["Wavelength (nm)"], weights[key])}

    @property
    def skin_weights(self):
        weights = get_spectral_weightings()
        if self in (PhotStandard.ACGIH, PhotStandard.UL8802):
            key = "ANSI IES RP 27.1-22 (Skin)"
        elif self is PhotStandard.ICNIRP:
            key = "IEC 62471-6:2022 (Eye/Skin)"
        return {k: v for k, v in zip(weights["Wavelength (nm)"], weights[key])}

    def flags(self, units="meters") -> dict:
        if self is PhotStandard.UL8802:
            return {
                "height": 1.9 if units == "meters" else 6.25,
                "skin_horiz": False,
                "eye_vert": False,
                "fov_vert": 180,
            }
        return {
            "height": 1.8 if units == "meters" else 5.9,
            "skin_horiz": True,
            "eye_vert": True,
            "fov_vert": 80,
        }

    @classmethod
    def dict(cls) -> dict:
        return {member.value: member.label for member in cls}

    @classmethod
    def labels(cls) -> list:
        return [member.label for member in cls]

    def __str__(self) -> str:
        return self.label


def get_tlvs(ref, standard=PhotStandard.ACGIH):
    """Get threshold limit values (skin, eye) for a wavelength or spectrum."""
    skin = get_skin_tlv(ref, standard)
    eye = get_eye_tlv(ref, standard)
    return skin, eye
    
def get_skin_tlv(ref, standard=PhotStandard.ACGIH) -> float:
    """Get skin threshold limit value in mJ/cm2/8 hours."""
    return _tlv(ref, get_weights("skin", standard))

def get_eye_tlv(ref, standard=PhotStandard.ACGIH) -> float:
    """Get eye threshold limit value in mJ/cm2/8 hours."""
    return _tlv(ref, get_weights("eye", standard))
    
def get_tlv(ref, standard=PhotStandard.ACGIH, target: str = "eye") -> float:
    """Get threshold limit values (skin or eye) for a wavelength or spectrum."""
    weights = get_weights(target, standard)
    return _tlv(ref, weights)

def get_max_irradiance(ref, standard=PhotStandard.ACGIH, target: str = "eye") -> float:
    """
    Return max allowed irradiance for a wavelength or spectrum (µW/cm²).

    Args:
        ref: Wavelength (nm) or Spectrum object
        standard: Photobiological safety standard
        target: "skin" or "eye"
    """
    weights = get_weights(target, standard)
    tlv = _tlv(ref, weights)
    return tlv / 60 / 60 / 8 * 1000

def get_seconds_to_tlv(
    ref, irradiance: float, standard=PhotStandard.ACGIH, target: str = "eye"
) -> float:
    """
    Return seconds until TLV is reached at given irradiance (µW/cm²).

    Args:
        ref: Wavelength (nm) or Spectrum object
        irradiance: Irradiance in µW/cm²
        standard: Photobiological safety standard
        target: "skin" or "eye"
    """
    weights = get_weights(target, standard)
    wavelengths = list(weights.keys())
    values = list(weights.values())

    if isinstance(ref, (int, float)):
        weighting = log_interp(ref, wavelengths, values)
        weighted_irradiance = irradiance * weighting
    elif isinstance(ref, Spectrum):
        w = log_interp(np.array(ref.wavelengths), wavelengths, values)
        ints = np.array(ref.intensities)
        i_new = ints * irradiance / ref.sum()
        weighted_irradiance = sum_spectrum(np.array(ref.wavelengths), w * i_new)
    else:
        raise TypeError(
            f"Argument `ref` must be either float, int, or Spectrum object, not {type(ref)}"
        )
    return 3000 / weighted_irradiance
    
def get_weights(target: str, standard: PhotStandard):
    if target.lower() not in ["skin", "eye"]:
        raise ValueError(f"weight must be `skin` or `eye` (got {target})")
    return standard.skin_weights if target.lower() == "skin" else standard.eye_weights
    
def _tlv(ref, weights: dict) -> float:
    """Calculate TLV for a single weighting function."""
    wavelengths = list(weights.keys())
    values = list(weights.values())

    if isinstance(ref, (int, float)):
        weighting = log_interp(ref, wavelengths, values)
        return 3 / weighting  # value not to be exceeded in 8 hours
    elif isinstance(ref, Spectrum):
        w = log_interp(np.array(ref.wavelengths), wavelengths, values)
        ints = np.array(ref.intensities)
        i_new = ints / ref.sum()
        s_lambda = sum_spectrum(np.array(ref.wavelengths), w * i_new)
        return 3 / s_lambda
    raise TypeError(
        f"Argument `ref` must be either float, int, or Spectrum object, not {type(ref)}"
    )


# --- Safety compliance checking ---


class ComplianceStatus(StrEnum):
    """Overall compliance status of an installation."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    COMPLIANT_WITH_DIMMING = "compliant_with_dimming"
    NON_COMPLIANT_EVEN_WITH_DIMMING = "non_compliant_even_with_dimming"


class WarningLevel(StrEnum):
    """Severity level for safety warnings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class LampComplianceResult:
    """Compliance result for a single lamp."""

    lamp_id: str
    lamp_name: str
    skin_dose_max: float
    eye_dose_max: float
    skin_tlv: float
    eye_tlv: float
    skin_dimming_required: float  # 1.0 = no dimming, 0.5 = dim to 50%
    eye_dimming_required: float
    is_skin_compliant: bool
    is_eye_compliant: bool
    missing_spectrum: bool


@dataclass(frozen=True)
class SafetyWarning:
    """A warning or error message from safety checking."""

    level: WarningLevel
    message: str
    lamp_id: str | None = None


@dataclass(frozen=True)
class SafetyCheckResult:
    """Complete result from check_lamps() safety analysis."""

    status: ComplianceStatus
    lamp_results: dict[str, LampComplianceResult] = field(default_factory=dict)
    warnings: list[SafetyWarning] = field(default_factory=list)
    weighted_skin_dose: np.ndarray | None = None
    weighted_eye_dose: np.ndarray | None = None
    max_skin_dose: float = 0.0
    max_eye_dose: float = 0.0
    skin_dimming_for_compliance: float | None = None
    eye_dimming_for_compliance: float | None = None


def check_lamps(room) -> SafetyCheckResult:
    """
    Check all lamps for safety compliance with skin and eye TLVs.

    Performs four checks:
    1. Individual lamp compliance - checks if each lamp exceeds skin/eye TLVs
    2. Combined dose compliance - checks if all lamps together exceed limits
    3. Dimmed installation compliance - checks if applying dimming achieves compliance
    4. Missing spectrum warnings - warns if non-LPHG lamps lack spectral data

    Returns a SafetyCheckResult containing compliance status, per-lamp results,
    and any warnings.
    """
    warnings_list: list[SafetyWarning] = []

    # Check for required zones
    if "SkinLimits" not in room.calc_zones:
        return SafetyCheckResult(
            status=ComplianceStatus.NON_COMPLIANT,
            warnings=[
                SafetyWarning(
                    level=WarningLevel.ERROR,
                    message="SkinLimits zone not found. Add standard zones with room.add_standard_zones().",
                )
            ],
        )

    if "EyeLimits" not in room.calc_zones:
        return SafetyCheckResult(
            status=ComplianceStatus.NON_COMPLIANT,
            warnings=[
                SafetyWarning(
                    level=WarningLevel.ERROR,
                    message="EyeLimits zone not found. Add standard zones with room.add_standard_zones().",
                )
            ],
        )

    skin = room.calc_zones["SkinLimits"]
    eye = room.calc_zones["EyeLimits"]
    skin_values = skin.get_values()
    eye_values = eye.get_values()

    # Check if zones have been calculated
    if skin_values is None or eye_values is None:
        return SafetyCheckResult(
            status=ComplianceStatus.NON_COMPLIANT,
            warnings=[
                SafetyWarning(
                    level=WarningLevel.ERROR,
                    message="Zones have not been calculated. Call room.calculate() first.",
                )
            ],
        )

    skindims, eyedims = {}, {}
    lamp_results: dict[str, LampComplianceResult] = {}
    weighted_skin_dose = np.zeros(skin_values.shape)
    weighted_eye_dose = np.zeros(eye_values.shape)

    dimmed_weighted_skin_dose = np.zeros(skin_values.shape)
    dimmed_weighted_eye_dose = np.zeros(eye_values.shape)

    # Check if any individual lamp exceeds the limits
    for lamp_id, lamp in room.lamps.items():
        if lamp_id in eye.lamp_cache.keys() and lamp_id in skin.lamp_cache.keys():
            # Fetch the TLVs for this specific lamp
            tlvs = lamp.get_tlvs(room.standard)
            if tlvs[0] is None or tlvs[1] is None:
                warnings_list.append(
                    SafetyWarning(
                        level=WarningLevel.WARNING,
                        message=f"{lamp.name} has no wavelength or spectrum defined. Cannot calculate TLVs.",
                        lamp_id=lamp_id,
                    )
                )
                continue

            skinmax, eyemax = tlvs

            # These are irradiance values, not dose
            skinrad = skin.lamp_cache[lamp_id].values
            eyerad = eye.lamp_cache[lamp_id].values

            # Convert to dose (mJ/cm2)
            skinvals = skinrad * 3.6 * skin.hours
            eyevals = eyerad * 3.6 * eye.hours

            # Weighting function for this specific lamp
            skinweight, eyeweight = 3 / skinmax, 3 / eyemax

            # Fraction dimming required to be compliant (>1 means already compliant)
            skindim = skinmax / skinvals.max() if skinvals.max() > 0 else float("inf")
            eyedim = eyemax / eyevals.max() if eyevals.max() > 0 else float("inf")

            # Add to total weighted dose
            weighted_skin_dose += skinvals * skinweight
            weighted_eye_dose += eyevals * eyeweight

            skindims[lamp_id] = skindim
            eyedims[lamp_id] = eyedim
            total_dim = min(skindim, eyedim, 1)
            dimmed_weighted_eye_dose += eyevals * eyeweight * total_dim
            dimmed_weighted_skin_dose += skinvals * skinweight * total_dim

            # Check for missing spectrum on non-LPHG lamps
            missing_spectrum = (
                lamp.guv_type != GUVType.LPHG
                and lamp.spectra is None
                and lamp.ies is not None
            )

            # Record per-lamp result
            lamp_results[lamp_id] = LampComplianceResult(
                lamp_id=lamp_id,
                lamp_name=lamp.name,
                skin_dose_max=skinvals.max(),
                eye_dose_max=eyevals.max(),
                skin_tlv=skinmax,
                eye_tlv=eyemax,
                skin_dimming_required=min(skindim, 1.0),
                eye_dimming_required=min(eyedim, 1.0),
                is_skin_compliant=skindim >= 1,
                is_eye_compliant=eyedim >= 1,
                missing_spectrum=missing_spectrum,
            )

            # Individual lamp check - generate warnings
            if min(skindim, eyedim, 1) < 1:
                skindim_pct = round(skindim * 100, 1)
                eyedim_pct = round(eyedim * 100, 1)
                if skindim_pct < 100:
                    msg = f"{lamp.name} must be dimmed to {skindim_pct}% its present power to comply with selected skin TLVs"
                    if eyedim_pct < 100:
                        msg += f" and to {eyedim_pct}% to comply with eye TLVs."
                    warnings_list.append(
                        SafetyWarning(
                            level=WarningLevel.WARNING, message=msg, lamp_id=lamp_id
                        )
                    )
                elif eyedim_pct < 100:
                    msg = f"{lamp.name} must be dimmed to {eyedim_pct}% its present power to comply with selected eye TLVs"
                    warnings_list.append(
                        SafetyWarning(
                            level=WarningLevel.WARNING, message=msg, lamp_id=lamp_id
                        )
                    )

            if missing_spectrum:
                msg = f"{lamp.name} is missing a spectrum. Photobiological safety calculations may be inaccurate."
                warnings_list.append(
                    SafetyWarning(
                        level=WarningLevel.WARNING, message=msg, lamp_id=lamp_id
                    )
                )

    # Check if seemingly-compliant installations actually aren't
    dimvals = list(skindims.values()) + list(eyedims.values())
    DIMMING_NOT_REQUIRED = all(dim >= 1 for dim in dimvals) if dimvals else True
    LAMPS_COMPLIANT = (
        max(weighted_skin_dose.max().round(2), weighted_eye_dose.max().round(2)) <= 3
    )
    DIMMED_LAMPS_COMPLIANT = (
        max(
            dimmed_weighted_skin_dose.max().round(2),
            dimmed_weighted_eye_dose.max().round(2),
        )
        <= 3
    )

    # Calculate dimming needed for overall compliance
    skin_dimming_for_compliance = None
    eye_dimming_for_compliance = None

    if weighted_skin_dose.max() > 3:
        skin_dimming_for_compliance = 3 / weighted_skin_dose.max()
    if weighted_eye_dose.max() > 3:
        eye_dimming_for_compliance = 3 / weighted_eye_dose.max()

    # Combined dose warning
    if DIMMING_NOT_REQUIRED and not LAMPS_COMPLIANT:
        msg = "Though all lamps are individually compliant, dose must be reduced to "
        skindim_pct = round(3 / weighted_skin_dose.max() * 100, 1)
        eyedim_pct = round(3 / weighted_eye_dose.max() * 100, 1)
        if weighted_skin_dose.max() > 3:
            msg += f"{skindim_pct}% its present value to comply with selected skin TLVs"
            if weighted_eye_dose.max() > 3:
                msg += f" and to {eyedim_pct}% to comply with selected eye TLVs."
        elif weighted_eye_dose.max() > 3:
            msg += f"{eyedim_pct}% its present value to comply with selected eye TLVs"
        warnings_list.append(SafetyWarning(level=WarningLevel.WARNING, message=msg))

    # Check if dimming will make the installation compliant
    if not DIMMED_LAMPS_COMPLIANT:
        msg = "Even after applying dimming, this installation may not be compliant. Dose must be reduced to "
        skindim_pct = round(3 / weighted_skin_dose.max() * 100, 1)
        eyedim_pct = round(3 / weighted_eye_dose.max() * 100, 1)
        if dimmed_weighted_skin_dose.max() > 3:
            msg += f"{skindim_pct}% its present value to comply with selected skin TLVs"
            if dimmed_weighted_eye_dose.max() > 3:
                msg += f" and to {eyedim_pct}% to comply with eye TLVs."
        elif dimmed_weighted_eye_dose.max() > 3:
            msg += f"{eyedim_pct}% its present value to comply with selected eye TLVs"
        warnings_list.append(SafetyWarning(level=WarningLevel.ERROR, message=msg))

    # Determine overall status
    if LAMPS_COMPLIANT:
        status = ComplianceStatus.COMPLIANT
    elif DIMMED_LAMPS_COMPLIANT:
        status = ComplianceStatus.COMPLIANT_WITH_DIMMING
    elif not DIMMING_NOT_REQUIRED and DIMMED_LAMPS_COMPLIANT:
        status = ComplianceStatus.COMPLIANT_WITH_DIMMING
    else:
        status = ComplianceStatus.NON_COMPLIANT_EVEN_WITH_DIMMING

    return SafetyCheckResult(
        status=status,
        lamp_results=lamp_results,
        warnings=warnings_list,
        weighted_skin_dose=weighted_skin_dose,
        weighted_eye_dose=weighted_eye_dose,
        max_skin_dose=float(weighted_skin_dose.max()),
        max_eye_dose=float(weighted_eye_dose.max()),
        skin_dimming_for_compliance=skin_dimming_for_compliance,
        eye_dimming_for_compliance=eye_dimming_for_compliance,
    )
