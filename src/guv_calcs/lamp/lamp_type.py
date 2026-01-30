from enum import StrEnum
from dataclasses import dataclass, replace
import warnings
from typing import Optional
import re
from .spectrum import Spectrum


class GUVType(StrEnum):
    KRCL = "krcl"
    LPHG = "lphg"
    LED = "led"
    OTHER = "other"

    @classmethod
    def from_any(cls, arg) -> "GUVType | None":
        if arg is None:
            return None
        if isinstance(arg, cls):
            return arg
        return cls.from_token(arg)

    @classmethod
    def from_token(cls, token: str) -> "GUVType":
        token = re.sub(r"[\s_\-()]+", "", str(token).strip().lower())

        aliases = {
            "krcl": cls.KRCL,
            "kryptonchloride": cls.KRCL,
            "kryptonchloride222nm": cls.KRCL,
            "222nm": cls.KRCL,
            "222": cls.KRCL,
            "lowpressuremercury": cls.LPHG,
            "lowpressuremercury254nm": cls.LPHG,
            "lphg": cls.LPHG,
            "254": cls.LPHG,
            "254nm": cls.LPHG,
            "led": cls.LED,
            "none": cls.OTHER,
            "other": cls.OTHER,
        }
        try:
            return aliases[token]
        except KeyError:
            raise ValueError(f"Unknown GUVType {token}")

    @classmethod
    def from_wavelength(cls, wv: float) -> "GUVType":
        if round(wv) == 222:
            return cls.KRCL
        if round(wv) == 254:
            return cls.LPHG
        return cls.OTHER

    @property
    def label(self):
        if self is GUVType.KRCL:
            return "Krypton chloride (222 nm)"
        if self is GUVType.LPHG:
            return "Low-pressure mercury (254 nm)"
        if self is GUVType.LED:
            return "LED"
        if self is GUVType.OTHER:
            return "Other"

    def __str__(self) -> str:
        return self.label

    @property
    def default_wavelength(self) -> Optional[float]:
        if self is GUVType.KRCL:
            return 222.0
        if self is GUVType.LPHG:
            return 254.0
        return None

    @classmethod
    def dict(cls) -> dict:
        return {member.value: member.label for member in cls}


@dataclass(frozen=True)
class LampType:
    spectrum: Spectrum | None = None
    _guv_type: GUVType | None = None
    _wavelength: int | float | None = None

    def __post_init__(self):
        if self._wavelength is not None:
            if not isinstance(self._wavelength, (int, float)):
                raise TypeError(
                    f"Wavelength must be int or float, not {type(self._wavelength)}"
                )
            if self._wavelength != self.wavelength:
                warnings.warn(
                    f"Passed wavelength value {self._wavelength} has been overriden by spectrum or guv_type"
                )

        if self._guv_type is not None and self.spectrum is not None:
            if self._guv_type != self.guv_type:
                warnings.warn(
                    f"Passed guv_type value {self._guv_type} has been overriden by spectrum"
                )

    def __repr__(self):
        spec = True if self.spectrum is not None else False
        return (
            f"spectrum={spec}, guv_type={self.guv_type}, wavelength={self.wavelength}"
        )

    def update(self, **changes):
        return replace(self, **changes)

    @property
    def guv_type(self):
        if self.spectrum is not None:
            wv = self.spectrum.peak_wavelength
            return GUVType.from_wavelength(wv)
        if self._wavelength is not None:
            return GUVType.from_wavelength(self._wavelength)
        return self._guv_type

    @property
    def wavelength(self):
        if self.spectrum is not None:
            return self.spectrum.peak_wavelength
        if self._guv_type is not None:
            return self._guv_type.default_wavelength
        return self._wavelength


class LampUnitType(StrEnum):
    """
    TODO: probably should live in photometry
    units of radiant intensity for GUV lamps
    generally
    """

    MW_PER_SR = "mw/sr"
    UW_PER_CM2 = "uw/cm2"

    @classmethod
    def from_any(cls, arg) -> "LampUnitType":
        if arg is None:
            return cls.MW_PER_SR
        if isinstance(arg, str):
            token = arg.strip().lower()
            if token in ["mw/sr"]:
                return cls.MW_PER_SR
            if token in ["uw/cm2", "uw/cm²"]:
                return cls.UW_PER_CM2
        if isinstance(arg, int):
            if arg == 0:
                return cls.MW_PER_SR
            if arg == 1:
                return cls.UW_PER_CM2
        # warn and return default
        msg = f"Intensity unit {arg} not recognized. Using default value `mW/sr`"
        warnings.warn(msg)
        return cls.MW_PER_SR

    @property
    def label(self) -> str:
        """Nice human-facing text for plots, etc."""
        if self is LampUnitType.MW_PER_SR:
            return "mW/sr"
        if self is LampUnitType.UW_PER_CM2:
            return "µW/cm²"
        # defensive default
        return self.value

    @property
    def factor(self) -> float:
        if self is LampUnitType.MW_PER_SR:
            return 0.1
        if self is LampUnitType.UW_PER_CM2:
            return 1.0

    def __str__(self) -> str:
        return self.label
