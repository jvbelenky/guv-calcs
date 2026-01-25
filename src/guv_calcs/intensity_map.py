"""Intensity map loading, normalization, and resampling."""

from pathlib import Path
import io
import warnings
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class IntensityMap:
    """
    Manages intensity map loading, normalization, and resampling.

    Attributes:
        original: The raw loaded intensity map (or None)
    """

    def __init__(self, source=None):
        """Load intensity map from various sources."""
        self.original = self._load(source)

    @property
    def normalized(self):
        """Return mean-normalized map, or None if no map."""
        if self.original is not None:
            return self.original / self.original.mean()
        return None

    def resample(self, num_u, num_v, points_generator):
        """
        Resample intensity map to match a grid.

        Args:
            num_u: Number of points in u direction
            num_v: Number of points in v direction
            points_generator: Callable(num_u, num_v) -> (u_points, v_points)

        Returns:
            np.ndarray of shape (num_u, num_v), normalized
        """
        if self.original is None:
            return np.ones((num_u, num_v))

        if self.original.shape == (num_u, num_v):
            return self.normalized

        # Interpolate to new grid
        orig_u, orig_v = self.original.shape
        x_orig, y_orig = points_generator(orig_u, orig_v)

        interpolator = RegularGridInterpolator(
            (x_orig, y_orig),
            self.normalized,
            bounds_error=False,
            fill_value=None,
        )

        x_new, y_new = points_generator(num_u, num_v)
        x_grid, y_grid = np.meshgrid(x_new, y_new)
        points_new = np.array([x_grid.ravel(), y_grid.ravel()]).T

        resampled = interpolator(points_new).reshape(len(x_new), len(y_new)).T
        return resampled / resampled.mean()

    def _load(self, source):
        """Load intensity map from various source types."""
        if source is None:
            return None

        if isinstance(source, (str, Path)):
            if Path(source).is_file():
                data = np.genfromtxt(Path(source), delimiter=",")
            else:
                warnings.warn(
                    f"File {source} not found. intensity_map will not be used.",
                    stacklevel=4,
                )
                return None

        elif isinstance(source, bytes):
            try:
                text = source.decode("utf-8-sig")
                data = np.genfromtxt(io.StringIO(text), delimiter=",")
            except UnicodeDecodeError:
                warnings.warn(
                    "Could not read intensity map file. Intensity map will not be used.",
                    stacklevel=4,
                )
                return None

        elif isinstance(source, (list, np.ndarray)):
            data = np.array(source)

        else:
            warnings.warn(
                f"Argument type {type(source)} for argument intensity_map is invalid. "
                "intensity_map will not be used.",
                stacklevel=4,
            )
            return None

        # Validate for NaN values
        if np.isnan(data).any():
            warnings.warn(
                "File contains invalid values. Intensity map will not be used.",
                stacklevel=4,
            )
            return None

        return data
