"""Filtering utilities for the Data class."""

import pandas as pd

from .constants import COL_CATEGORY, COL_MEDIUM, COL_WAVELENGTH


def filter_by_column(df: pd.DataFrame, col: str, value) -> pd.DataFrame:
    """Filter df by column value. Handles scalar, list, or tuple (min, max) range."""
    if value is None:
        return df
    if isinstance(value, (int, float, str)):
        return df[df[col] == value]
    elif isinstance(value, list):
        return df[df[col].isin(value)]
    elif isinstance(value, tuple) and len(value) == 2:
        return df[(df[col] >= value[0]) & (df[col] <= value[1])]
    return df


def validate_filter(value, valid_values: list, name: str):
    """Validate filter value against valid_values. Normalizes single-item lists to scalars."""
    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        if value not in valid_values:
            raise KeyError(f"{value} is not a valid {name}; must be in {valid_values}")
    elif isinstance(value, list):
        invalid = [v for v in value if v not in valid_values]
        if invalid:
            raise KeyError(f"Invalid {name}(s) {invalid}; must be in {valid_values}")
        if len(value) == 1:
            value = value[0]
    return value


def apply_row_filters(df: pd.DataFrame, medium, category) -> pd.DataFrame:
    """Apply medium/category filters to df. Does NOT apply wavelength filter."""
    df = filter_by_column(df, COL_MEDIUM, medium)
    df = filter_by_column(df, COL_CATEGORY, category)
    return df


def apply_wavelength_filter(df: pd.DataFrame, effective_wavelengths) -> pd.DataFrame:
    """Apply wavelength filter based on effective wavelengths."""
    if effective_wavelengths is None or COL_WAVELENGTH not in df.columns:
        return df
    return filter_by_column(df, COL_WAVELENGTH, effective_wavelengths)


def get_effective_wavelengths(wavelength_filter, fluence_wavelengths) -> list | tuple | None:
    """Get effective wavelengths (merged fluence + user-specified). Returns list, tuple, or None."""
    # If user specified a range (tuple), use it directly
    if isinstance(wavelength_filter, tuple):
        return wavelength_filter

    # Collect wavelengths to include
    wavelengths = set()

    # Always include fluence dict wavelengths if present
    if fluence_wavelengths:
        wavelengths.update(fluence_wavelengths)

    # Add user-specified wavelengths
    if wavelength_filter is not None:
        if isinstance(wavelength_filter, (int, float)):
            wavelengths.add(wavelength_filter)
        elif isinstance(wavelength_filter, list):
            wavelengths.update(wavelength_filter)

    # Return None if no wavelengths to filter by, otherwise sorted list
    if not wavelengths:
        return None
    return sorted(wavelengths)
