"""Average value computation utilities for the InactivationData class."""

from collections.abc import Callable

import pandas as pd

from .constants import COL_SPECIES, COL_STRAIN, COL_CONDITION, COL_K1, COL_K2, COL_RESISTANT, COL_WAVELENGTH
from ._kinetics import parse_resistant
from ._filtering import filter_by_words


def collect_parametric_inputs(function, species, strain, medium, condition) -> list:
    """Collect list inputs into parametric tuples for nested dict construction."""
    parametric = []
    if isinstance(function, list):
        parametric.append(("function", function))
    if isinstance(species, list):
        parametric.append(("species", species))
    if isinstance(strain, list):
        parametric.append(("strain", strain))
    if isinstance(medium, list):
        parametric.append(("medium", medium))
    if isinstance(condition, list):
        parametric.append(("condition", condition))
    return parametric


def resolve_function(function, function_map: dict) -> tuple[Callable, str | None]:
    """Resolve function string to callable. Returns (func, func_name)."""
    if isinstance(function, str):
        if function not in function_map:
            raise ValueError(f"Unknown function '{function}'; must be one of {list(function_map.keys())}")
        return function_map[function], function.lower()
    return function, None


def average_value_parametric(
    average_value_func: Callable,
    function,
    parametric: list[tuple[str, list]],
    species,
    strain,
    condition,
    medium,
    **kwargs,
) -> dict:
    """Build nested dict for parametric average_value calls."""
    param_name, param_values = parametric[0]
    remaining_parametric = parametric[1:]

    result = {}
    for val in param_values:
        current_function = val if param_name == "function" else function

        call_kwargs = {
            "species": val if param_name == "species" else species,
            "strain": val if param_name == "strain" else strain,
            "condition": val if param_name == "condition" else condition,
            "medium": val if param_name == "medium" else medium,
            **kwargs,
        }

        if remaining_parametric:
            for remaining_name, remaining_values in remaining_parametric:
                if remaining_name == "function":
                    current_function = remaining_values
                else:
                    call_kwargs[remaining_name] = remaining_values

        sub_result = average_value_func(current_function, **call_kwargs)
        result[val] = sub_result

    return result


def filter_for_average(
    df: pd.DataFrame,
    species: str | None,
    strain: str | None,
    condition: str | None,
    **kwargs,
) -> pd.DataFrame:
    """Apply species/strain/condition filters for average_value."""
    df = filter_by_words(df, COL_SPECIES, species)
    df = filter_by_words(df, COL_STRAIN, strain)
    df = filter_by_words(df, COL_CONDITION, condition)

    for col, value in kwargs.items():
        if col in df.columns:
            df = df[df[col] == value]

    return df


def compute_average_single(
    df: pd.DataFrame,
    func: Callable,
    func_name: str | None,
    fluence: float | dict,
    volume_m3: float | None,
) -> float:
    """Compute average value for single wavelength fluence."""
    if isinstance(fluence, dict):
        fluence_val = list(fluence.values())[0]
    else:
        fluence_val = fluence

    k1_mean = df[COL_K1].mean()
    k2_mean = df[COL_K2].fillna(0).mean()
    f_mean = df[COL_RESISTANT].apply(parse_resistant).mean()

    if func_name in ("cadr_lps",):
        if volume_m3 is None:
            raise ValueError("volume_m3 must be set to compute CADR")
        return func(volume_m3, fluence_val, k1_mean, k2_mean, f_mean)
    elif func_name in ("cadr_cfm",):
        if volume_m3 is None:
            raise ValueError("volume_m3 must be set to compute CADR")
        cubic_feet = volume_m3 * 35.3147
        return func(cubic_feet, fluence_val, k1_mean, k2_mean, f_mean)
    else:
        return func(fluence_val, k1_mean, k2_mean, f_mean)


def compute_average_multiwavelength(
    df: pd.DataFrame,
    func: Callable,
    func_name: str | None,
    fluence: dict,
    volume_m3: float | None,
) -> float:
    """Compute average value for multi-wavelength fluence."""
    wavelengths = list(fluence.keys())
    irrad_list = [fluence[wv] for wv in wavelengths]

    k1_list = []
    k2_list = []
    f_list = []

    for wv in wavelengths:
        wv_df = df[df[COL_WAVELENGTH] == wv]
        if len(wv_df) == 0:
            raise ValueError(f"No data available for wavelength {wv} nm")

        k1_list.append(wv_df[COL_K1].mean())
        k2_list.append(wv_df[COL_K2].fillna(0).mean())
        f_list.append(wv_df[COL_RESISTANT].apply(parse_resistant).mean())

    if func_name in ("cadr_lps",):
        if volume_m3 is None:
            raise ValueError("volume_m3 must be set to compute CADR")
        return func(volume_m3, irrad_list, k1_list, k2_list, f_list)
    elif func_name in ("cadr_cfm",):
        if volume_m3 is None:
            raise ValueError("volume_m3 must be set to compute CADR")
        cubic_feet = volume_m3 * 35.3147
        return func(cubic_feet, irrad_list, k1_list, k2_list, f_list)
    else:
        return func(irrad_list, k1_list, k2_list, f_list)
