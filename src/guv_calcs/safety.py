from enum import StrEnum
from .spectrum import Spectrum, log_interp
from .io import get_spectral_weightings


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
    skin = _tlv(ref, standard.skin_weights)
    eye = _tlv(ref, standard.eye_weights)
    return skin, eye


def _tlv(ref, weights: dict):
    if isinstance(ref, (int, float)):
        wavelengths = list(weights.keys())
        values = list(weights.values())
        weighting = log_interp(ref, wavelengths, values)
        return 3 / weighting  # value not to be exceeded in 8 hours
    elif isinstance(ref, Spectrum):
        return ref.get_tlv(weights)
    raise TypeError(
        f"Argument `ref` must be either float, int, or Spectrum object, not {type(ref)}"
    )
