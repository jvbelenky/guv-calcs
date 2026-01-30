import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from ..io import get_spectral_weightings, load_csv, rows_to_bytes


@dataclass(frozen=True)
class Spectrum:
    """
    Immutable spectral power distribution for a UV lamp.

    Operations like filtered(), scaled(), and normalized() return new instances
    rather than mutating in place. This makes Spectrum safe to share and cache.

    Attributes:
        wavelengths: Wavelength values in nanometers (sorted ascending)
        intensities: Relative intensity values (same length as wavelengths)
        weighted_intensities: Dict of weighted intensity arrays by standard name (lazy)
    """

    wavelengths: tuple[float, ...]
    intensities: tuple[float, ...]
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        # Convert to numpy for processing, then back to tuple for immutability
        wl = np.array(self.wavelengths)
        ints = np.array(self.intensities)

        if len(wl) != len(ints):
            raise ValueError("Number of wavelengths and intensities do not match.")

        # Sort by wavelength
        sortidx = np.argsort(wl)
        wl = wl[sortidx]
        ints = ints[sortidx]

        # Store as tuples (frozen dataclass requires immutable)
        object.__setattr__(self, "wavelengths", tuple(wl.tolist()))
        object.__setattr__(self, "intensities", tuple(ints.tolist()))

    @property
    def weighted_intensities(self) -> dict[str, tuple[float, ...]]:
        """Dict of weighted intensity arrays by standard name (computed lazily)."""
        if "weighted" not in self._cache:
            self._cache["weighted"] = _compute_weighted_spectra(
                np.array(self.wavelengths), np.array(self.intensities)
            )
        return self._cache["weighted"]

    @classmethod
    def from_source(cls, source) -> "Spectrum | None":
        """Best-effort construction from various common representations."""
        if source is None:
            return None
        if isinstance(source, cls):
            return source
        if isinstance(source, dict):
            return cls.from_dict(source)
        if isinstance(source, (str, bytes, Path)):
            return cls.from_file(source)
        if isinstance(source, tuple) and len(source) == 2:
            wavelengths, intensities = source
            return cls(tuple(wavelengths), tuple(intensities))

        warnings.warn(
            f"Datatype {type(source)} not recognized as spectral data source",
            stacklevel=3,
        )
        return None

    @classmethod
    def from_file(cls, filepath):
        """
        Load spectrum from CSV file.

        Expects first column as wavelengths (nm) and second column as intensities.
        """
        csv_data = load_csv(filepath)
        reader = csv.reader(csv_data)
        spectra = []
        for i, row in enumerate(reader):
            try:
                vals = list(map(float, row))
                spectra.append((vals[0], vals[1]))
            except ValueError:
                if i == 0:  # probably a header
                    continue
        if len(spectra) == 0:
            raise ValueError("File contains no valid data.")

        wavelengths = tuple(s[0] for s in spectra)
        intensities = tuple(s[1] for s in spectra)
        return cls(wavelengths, intensities)

    @classmethod
    def from_dict(cls, dct):
        """
        Load spectrum from a dictionary.

        Accepts dicts with wavelength/intensity keys (case-insensitive, partial match)
        or falls back to using first two keys in order.
        """
        wavelengths = None
        intensities = None

        for k, v in dct.items():
            k_lower = k.lower()
            if wavelengths is None and any(
                wk in k_lower for wk in ["wavelength", "wv", "nm"]
            ):
                wavelengths = tuple(v)
            elif intensities is None and any(
                ik in k_lower for ik in ["intensity", "int", "power"]
            ):
                intensities = tuple(v)

        # Fall back to positional for backward compat
        if wavelengths is None or intensities is None:
            keys = list(dct.keys())
            if len(keys) >= 2:
                wavelengths = tuple(dct[keys[0]])
                intensities = tuple(dct[keys[1]])
            else:
                raise ValueError("Dict must have at least 2 keys for wavelengths and intensities")

        return cls(wavelengths, intensities)

    @property
    def peak_wavelength(self) -> float:
        """Wavelength at maximum intensity."""
        idx = self.intensities.index(max(self.intensities))
        return float(round(self.wavelengths[idx]))

    # Immutable operations - return new instances

    def filtered(self, minval: float = None, maxval: float = None) -> "Spectrum":
        """Return a new Spectrum filtered to the specified wavelength range."""
        wl = np.array(self.wavelengths)
        ints = np.array(self.intensities)

        if minval is None:
            minval = wl.min()
        if maxval is None:
            maxval = wl.max()

        mask = (wl >= minval) & (wl <= maxval)
        return Spectrum(tuple(wl[mask].tolist()), tuple(ints[mask].tolist()))

    def scaled(self, power: float, minval: float = None, maxval: float = None) -> "Spectrum":
        """Return a new Spectrum scaled so total power equals the given value."""
        spectral_power = self.sum(minval=minval, maxval=maxval)
        scale_factor = power / spectral_power
        new_intensities = tuple(i * scale_factor for i in self.intensities)
        return Spectrum(self.wavelengths, new_intensities)

    def normalized(self, normval: float = 1) -> "Spectrum":
        """Return a new Spectrum with maximum intensity scaled to the given value."""
        max_int = max(self.intensities)
        scale_factor = normval / max_int
        new_intensities = tuple(i * scale_factor for i in self.intensities)
        return Spectrum(self.wavelengths, new_intensities)

    # Pure computations

    def sum(self, minval: float = None, maxval: float = None, weight: str = None) -> float:
        """Integrate spectrum over wavelength range, optionally using weighted intensities."""
        wl = np.array(self.wavelengths)
        if weight in self.weighted_intensities:
            ints = np.array(self.weighted_intensities[weight])
        else:
            ints = np.array(self.intensities)

        if minval is None:
            minval = wl.min()
        if maxval is None:
            maxval = wl.max()

        mask = (wl >= minval) & (wl <= maxval)
        wl_filt = wl[mask]
        ints_filt = ints[mask]

        return sum_spectrum(wl_filt, ints_filt)

    # Serialization

    def to_dict(self, as_string: bool = False) -> dict:
        """Convert to dictionary representation."""
        spec = {
            "Wavelength": list(self.wavelengths),
            "Unweighted Relative Intensity": list(self.intensities),
        }
        for key, val in self.weighted_intensities.items():
            spec[key] = list(val)

        if as_string:
            for key, val in spec.items():
                spec[key] = ", ".join(map(str, val))
        return spec

    def to_csv(self, fname=None):
        """Export to CSV format."""
        rows = [list(self.to_dict().keys())]
        vals = [list(self.wavelengths), list(self.intensities)]
        vals += [list(v) for v in self.weighted_intensities.values()]
        rows += [list(row) for row in np.array(vals).T]
        csv_bytes = rows_to_bytes(rows)
        if fname is not None:
            with open(fname, "wb") as csvfile:
                csvfile.write(csv_bytes)
        else:
            return csv_bytes

    # Plotting

    def plot(
        self,
        title: str = "",
        fig=None,
        ax=None,
        figsize: tuple = (6.4, 4.8),
        yscale: str = "linear",
        label: str | None = None,
        weights: bool | str = False,
    ):
        """
        Plot the spectrum.

        Args:
            title: Plot title
            fig: Existing matplotlib figure
            ax: Existing matplotlib axes
            figsize: Figure size if creating new figure
            yscale: Y-axis scale ("linear" or "log")
            label: Label for spectrum line (str or True for default)
            weights: Show weighted spectra (True for all, str for specific)
        """
        if fig is None:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = plt.gcf()
        else:
            if ax is None:
                ax = fig.axes[0]

        default_label = "Unweighted Relative Intensity"
        ax.plot(self.wavelengths, self.intensities, label=label or default_label)

        if weights:
            if isinstance(weights, str):
                if weights in self.weighted_intensities:
                    ax.plot(
                        self.wavelengths,
                        self.weighted_intensities[weights],
                        label=weights,
                        alpha=0.7,
                        linestyle="--",
                    )
            elif isinstance(weights, bool):
                for key, val in self.weighted_intensities.items():
                    ax.plot(self.wavelengths, val, label=key, alpha=0.7, linestyle="--")

        if label or weights:
            ax.legend()
        ax.grid(True, which="both", ls="--", c="gray", alpha=0.3)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Relative intensity [%]")
        ax.set_yscale(yscale)
        ax.set_title(title)
        return fig, ax


def _compute_weighted_spectra(wavelengths: np.ndarray, intensities: np.ndarray) -> dict:
    """Compute weighted spectra for all standard weighting functions."""
    weights_dict = get_spectral_weightings()
    keys = list(weights_dict)
    weight_wavelengths = weights_dict[keys[0]]
    weighted_intensities = {}

    for key in keys[1:]:
        weights = log_interp(wavelengths, weight_wavelengths, weights_dict[key])
        weighted_intensity = intensities * weights
        ratio = max(intensities) / max(weighted_intensity)
        weighted_intensities[key] = tuple((weighted_intensity * ratio).tolist())

    return weighted_intensities


def sum_spectrum(wavelength, intensity) -> float:
    """
    Integrate a spectrum using trapezoidal approximation.

    ALWAYS use this when summing across a spectrum!
    """
    wavelength = np.asarray(wavelength)
    intensity = np.asarray(intensity)
    weighted_intensity = [
        intensity[i] * (wavelength[i] - wavelength[i - 1])
        for i in range(1, len(wavelength))
    ]
    return sum(weighted_intensity)


def log_interp(wvs, weight_wvs, weight_values) -> np.ndarray:
    """Log10 interpolation for spectral weighting functions."""
    wvs = np.asarray(wvs)
    weight_wvs = np.asarray(weight_wvs)
    weight_values = np.asarray(weight_values)
    logterp = np.interp(wvs, weight_wvs, np.log10(weight_values))
    return np.power(10, logterp)
