"""Filtering utilities for the Data class."""

import re

import pandas as pd

from .constants import COL_CATEGORY, COL_MEDIUM, COL_SPECIES, COL_STRAIN, COL_CONDITION, COL_WAVELENGTH


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


def words_match(query: str, target: str) -> bool:
    """Check if all words in query appear in target (case-insensitive)."""
    query_words = re.findall(r"\w+", query.lower())
    target_lower = target.lower()
    return all(word in target_lower for word in query_words)


def filter_by_words(df: pd.DataFrame, col: str, value: str | None) -> pd.DataFrame:
    """Filter df where all words in value appear in column (case-insensitive)."""
    if value is None:
        return df
    mask = df[col].fillna("").apply(lambda x: words_match(value, x))
    return df[mask]


def apply_row_filters(
    df: pd.DataFrame,
    medium=None,
    category=None,
    species=None,
    strain=None,
    condition=None,
) -> pd.DataFrame:
    """Apply all row-level filters. All words in query must appear in target (case-insensitive)."""
    df = filter_by_words(df, COL_MEDIUM, medium)
    df = filter_by_words(df, COL_CATEGORY, category)
    df = filter_by_words(df, COL_SPECIES, species)
    df = filter_by_words(df, COL_STRAIN, strain)
    df = filter_by_words(df, COL_CONDITION, condition)
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
