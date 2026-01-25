"""Column computation utilities for the Data class."""

from itertools import product

import pandas as pd

from .constants import (
    LOG_LABELS,
    COL_CATEGORY,
    COL_SPECIES,
    COL_MEDIUM,
    COL_WAVELENGTH,
    COL_K1,
    COL_EACH,
    COL_CADR_LPS,
    COL_CADR_CFM,
)
from .math import eACH_UV, log1, log2, log3, log4, log5
from ._kinetics import filter_wavelengths, extract_kinetic_params, compute_row


def compute_all_columns(df: pd.DataFrame, fluence, volume_m3=None) -> tuple[pd.DataFrame, dict, list | None]:
    """
    Compute all derived columns for the dataset.

    Returns (df_with_columns, time_cols_dict, fluence_wavelengths).
    Rows with missing k1 are excluded.
    """
    df = df.copy()
    df = df[df[COL_K1].notna()]

    fluence_wavelengths = None
    if isinstance(fluence, dict):
        _, fluence = filter_wavelengths(df, fluence.copy())
        fluence_wavelengths = list(fluence.keys())

    # Calculate time to inactivation for all log levels
    log_funcs = {1: log1, 2: log2, 3: log3, 4: log4, 5: log5}
    for log_level, func in log_funcs.items():
        label = LOG_LABELS[log_level]
        sec_key = f"Seconds to {label} inactivation"
        df[sec_key] = df.apply(compute_row, args=[fluence, func, 0], axis=1)

    # Calculate eACH-UV
    df[COL_EACH] = df.apply(compute_row, args=[fluence, eACH_UV, 1], axis=1)

    # Calculate CADR columns
    add_cadr_columns(df, volume_m3)

    # Calculate all time unit variants
    time_cols = calculate_all_time_columns(df)

    return df, time_cols, fluence_wavelengths


def combine_wavelengths(df: pd.DataFrame, fluence_dict: dict, volume_m3=None) -> tuple[pd.DataFrame, dict]:
    """
    Combine multi-wavelength data into aggregated rows.

    Returns (combined_df, time_cols_dict).
    """
    summed_data = []
    wavelengths = list(fluence_dict.keys())
    group_cols = [COL_SPECIES, COL_MEDIUM]

    for group_key, group in df.groupby(group_cols):
        species, med = group_key
        category = group[COL_CATEGORY].iloc[0]

        # Collect data for each wavelength
        data_by_wv = {}
        for wv in wavelengths:
            wv_rows = group[group[COL_WAVELENGTH] == wv]
            data_by_wv[wv] = [extract_kinetic_params(row) for _, row in wv_rows.iterrows()]

        # Skip if no data for any wavelength
        if any(len(data_by_wv[wv]) == 0 for wv in wavelengths):
            continue

        # Generate all combinations across wavelengths
        for combo in product(*[data_by_wv[wv] for wv in wavelengths]):
            k1_list = [item["k1"] for item in combo]
            k2_list = [item["k2"] for item in combo]
            f_list = [item["f"] for item in combo]
            irrad_list = [fluence_dict[wv] for wv in wavelengths]

            row_data = {
                COL_SPECIES: species,
                COL_CATEGORY: category,
                COL_MEDIUM: med,
                COL_EACH: round(eACH_UV(irrad_list, k1_list, k2_list, f_list), 1),
                f"Seconds to {LOG_LABELS[1]} inactivation": round(log1(irrad_list, k1_list, k2_list, f_list), 0),
                f"Seconds to {LOG_LABELS[2]} inactivation": round(log2(irrad_list, k1_list, k2_list, f_list), 0),
                f"Seconds to {LOG_LABELS[3]} inactivation": round(log3(irrad_list, k1_list, k2_list, f_list), 0),
                f"Seconds to {LOG_LABELS[4]} inactivation": round(log4(irrad_list, k1_list, k2_list, f_list), 0),
                f"Seconds to {LOG_LABELS[5]} inactivation": round(log5(irrad_list, k1_list, k2_list, f_list), 0),
            }
            summed_data.append(row_data)

    result_df = pd.DataFrame(summed_data)
    time_cols = calculate_all_time_columns(result_df)
    add_cadr_columns(result_df, volume_m3)

    return result_df, time_cols


def add_cadr_columns(df: pd.DataFrame, volume_m3: float | None) -> None:
    """Add CADR columns to DataFrame if volume is available and eACH exists."""
    if volume_m3 is not None and COL_EACH in df.columns:
        cubic_feet = volume_m3 * 35.3147
        liters = volume_m3 * 1000
        df[COL_CADR_LPS] = (df[COL_EACH] * liters / 3600).round(1)
        df[COL_CADR_CFM] = (df[COL_EACH] * cubic_feet / 60).round(1)


def calculate_all_time_columns(df: pd.DataFrame) -> dict:
    """Calculate minutes/hours columns for all log levels. Returns time_cols dict."""
    time_cols = {}

    for log_level in [1, 2, 3, 4, 5]:
        label = LOG_LABELS[log_level]
        sec_key = f"Seconds to {label} inactivation"

        if sec_key in df.columns:
            min_key = f"Minutes to {label} inactivation"
            hr_key = f"Hours to {label} inactivation"

            df[min_key] = round(df[sec_key] / 60, 2)
            df[hr_key] = round(df[sec_key] / 3600, 2)

            time_cols[log_level] = {
                "seconds": sec_key,
                "minutes": min_key,
                "hours": hr_key,
            }

    return time_cols
