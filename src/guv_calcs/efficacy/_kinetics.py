"""Pure utility functions for kinetic parameter handling."""

import re
import warnings
from collections.abc import Callable

import pandas as pd

from .constants import COL_K1, COL_K2, COL_RESISTANT, COL_SPECIES, COL_WAVELENGTH


def species_matches(query: str, target: str) -> bool:
    """Check if all words in query appear in target (case-insensitive)."""
    query_words = re.findall(r"\w+", query.lower())
    target_lower = target.lower()
    return all(word in target_lower for word in query_words)


def parse_resistant(val) -> float:
    """Parse resistant fraction: '0.33%' -> 0.0033, NaN -> 0.0."""
    if pd.isna(val):
        return 0.0
    if isinstance(val, str):
        return float(val.rstrip("%")) / 100
    return float(val)


def extract_kinetic_params(row) -> dict:
    """Extract k1, k2, f from a DataFrame row. NaN -> 0.0."""
    k1 = row[COL_K1] if pd.notna(row[COL_K1]) else 0.0
    k2 = row[COL_K2] if pd.notna(row[COL_K2]) else 0.0
    f = (
        float(row[COL_RESISTANT].rstrip("%")) / 100
        if pd.notna(row[COL_RESISTANT])
        else 0.0
    )
    return {"k1": k1, "k2": k2, "f": f}


def filter_wavelengths(df: pd.DataFrame, fluence_dict: dict) -> tuple[pd.DataFrame, dict]:
    """Filter df to species with data for all wavelengths. Modifies fluence_dict in-place."""
    wavelengths = df[COL_WAVELENGTH].unique()
    remove = []
    for key in fluence_dict.keys():
        if key not in wavelengths:
            msg = f"No data is available for wavelength {key} nm. eACH will be an underestimate."
            warnings.warn(msg, stacklevel=3)
            remove.append(key)
    for key in remove:
        del fluence_dict[key]

    required_wavelengths = fluence_dict.keys()
    filtered_species = df.groupby(COL_SPECIES)[COL_WAVELENGTH].apply(
        lambda x: all(wv in x.values for wv in required_wavelengths)
    )
    valid_species = filtered_species[filtered_species].index
    df = df[df[COL_SPECIES].isin(valid_species)]
    df = df[df[COL_WAVELENGTH].isin(required_wavelengths)]
    return df, fluence_dict


def compute_row(
    row,
    fluence_arg: float | dict,
    function: Callable,
    sigfigs: int = 1,
    **kwargs,
) -> float | None:
    """Apply function to row using kinetic params. Returns None if wavelength not in fluence_dict."""
    if isinstance(fluence_arg, dict):
        fluence = fluence_arg.get(row[COL_WAVELENGTH])
        if fluence is None:
            return None
    elif isinstance(fluence_arg, (float, int)):
        fluence = fluence_arg

    params = extract_kinetic_params(row)
    return round(function(irrad=fluence, **params, **kwargs), sigfigs)
