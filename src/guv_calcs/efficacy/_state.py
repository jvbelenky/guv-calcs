"""DataState dataclass for InactivationData instance variables."""

from dataclasses import dataclass, field


@dataclass
class DataState:
    """Consolidated state for the InactivationData class."""

    fluence: float | dict | None = None
    volume_m3: float | None = None
    medium: str | list | None = None
    category: str | list | None = None
    wavelength: int | float | list | tuple | None = None
    log: int = 2
    use_metric_units: bool = True
    fluence_wavelengths: list | None = None
    time_cols: dict = field(default_factory=dict)

    def __post_init__(self):
        """Initialize fluence_wavelengths from fluence dict if applicable."""
        if self.fluence_wavelengths is None and isinstance(self.fluence, dict):
            self.fluence_wavelengths = list(self.fluence.keys())
