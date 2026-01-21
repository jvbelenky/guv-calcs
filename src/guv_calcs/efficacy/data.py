"""Data class for UV efficacy calculations."""

import warnings
import pandas as pd
import numpy as np
from itertools import product

from ..io import get_full_disinfection_table
from .constants import (
    LOG_LABELS,
    COL_CATEGORY,
    COL_SPECIES,
    COL_STRAIN,
    COL_WAVELENGTH,
    COL_K1,
    COL_K2,
    COL_RESISTANT,
    COL_MEDIUM,
    COL_CONDITION,
    COL_REFERENCE,
    COL_LINK,
    COL_EACH,
    COL_CADR_LPS,
    COL_CADR_CFM,
    BASE_DISPLAY_COLS,
)
from .math import eACH_UV, log1, log2, log3, log4, log5
from .plotting import plot as _plot_func

pd.options.mode.chained_assignment = None


# DataFrame-coupled helper functions (not pure math, tied to table structure)

def _compute_row(row, fluence_arg, function, sigfigs=1, **kwargs):
    """
    Apply a function to a single row of the disinfection table.

    fluence_arg may be an int, float, or dict of format {wavelength:fluence}
    """
    if isinstance(fluence_arg, dict):
        fluence = fluence_arg.get(row[COL_WAVELENGTH])
        if fluence is None:
            return None
    elif isinstance(fluence_arg, (float, int)):
        fluence = fluence_arg

    # Fill missing values and convert columns to the correct type
    k1 = row[COL_K1] if pd.notna(row[COL_K1]) else 0.0
    k2 = row[COL_K2] if pd.notna(row[COL_K2]) else 0.0
    f = (
        float(row[COL_RESISTANT].rstrip("%")) / 100
        if pd.notna(row[COL_RESISTANT])
        else 0.0
    )
    return round(function(irrad=fluence, k1=k1, k2=k2, f=f, **kwargs), sigfigs)


def _filter_wavelengths(df, fluence_dict):
    """
    Filter the dataframe only for species which have data for all of the
    wavelengths that are in fluence_dict.

    Returns filtered df and cleaned fluence_dict.
    """
    # remove any wavelengths from the dictionary that aren't in the dataframe
    wavelengths = df[COL_WAVELENGTH].unique()
    remove = []
    for key in fluence_dict.keys():
        if key not in wavelengths:
            msg = f"No data is available for wavelength {key} nm. eACH will be an underestimate."
            warnings.warn(msg, stacklevel=3)
            remove.append(key)
    for key in remove:
        del fluence_dict[key]

    # List of required wavelengths
    required_wavelengths = fluence_dict.keys()
    # Group by Species and filter
    filtered_species = df.groupby(COL_SPECIES)[COL_WAVELENGTH].apply(
        lambda x: all(wavelength in x.values for wavelength in required_wavelengths)
    )
    # Filter the original DataFrame for the Species meeting the condition
    valid_species = filtered_species[filtered_species].index
    df = df[df[COL_SPECIES].isin(valid_species)]
    df = df[df[COL_WAVELENGTH].isin(required_wavelengths)]
    return df, fluence_dict


class Data:
    """
    UV disinfection efficacy data handler.

    Provides access to the disinfection table with optional fluence-based
    computed columns (eACH-UV, time to inactivation).
    """

    @classmethod
    def get_full(cls) -> pd.DataFrame:
        """Return full disinfection table (cached, returns copy)."""
        return get_full_disinfection_table().copy()

    def __init__(
        self,
        fluence: float | dict | None = None,
        volume_m3: float | None = None,
    ):
        """
        Initialize Data with optional fluence and volume for CADR computation.

        Parameters
        ----------
        fluence : float or dict, optional
            Fluence value(s) for computing time/eACH columns.
            Can be a single float or dict mapping wavelength to fluence.
        volume_m3 : float, optional
            Room volume in cubic meters. If provided, CADR columns are computed.
        """

        self._fluence = fluence
        self._volume_m3 = volume_m3

        # Filter/display state (set by subset())
        self._medium = None
        self._category = None
        self._wavelength = None  # User-specified wavelength filter
        self._log = 2  # Log reduction level for time column display (1-5)
        self._use_metric_units = True  # For CADR display (lps vs cfm)

        # Track fluence dict wavelengths separately (always included in display)
        self._fluence_wavelengths = list(fluence.keys()) if isinstance(fluence, dict) else None

        # Load base data (cached)
        self._base_df = self.get_full()
        self._time_cols = {}  # Populated by _compute_all_columns

        # Compute full_df if fluence provided
        if fluence is not None:
            self._full_df = self._compute_all_columns()
        else:
            self._full_df = self._base_df.copy()

    # -------------------------------------------------------------------------
    # New consolidated helper methods
    # -------------------------------------------------------------------------

    def _filter_by_column(self, df: pd.DataFrame, col: str, value) -> pd.DataFrame:
        """
        Filter df by column value (handles str, int, float, list, tuple range).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to filter.
        col : str
            Column name to filter on.
        value : str, int, float, list, or tuple
            Filter value. Tuple (min, max) for range filtering.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame.
        """
        if value is None:
            return df
        if isinstance(value, (int, float, str)):
            return df[df[col] == value]
        elif isinstance(value, list):
            return df[df[col].isin(value)]
        elif isinstance(value, tuple) and len(value) == 2:
            return df[(df[col] >= value[0]) & (df[col] <= value[1])]
        return df

    def _apply_wavelength_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply wavelength filter based on effective wavelengths.

        Uses merged fluence dict wavelengths + user-specified wavelengths.
        """
        effective_wv = self._get_effective_wavelengths()
        if effective_wv is None or COL_WAVELENGTH not in df.columns:
            return df
        return self._filter_by_column(df, COL_WAVELENGTH, effective_wv)

    def _validate_filter(self, value, valid_values: list, name: str):
        """
        Validate filter value and normalize single-item lists to scalars.

        Parameters
        ----------
        value : any
            Value to validate.
        valid_values : list
            List of valid values.
        name : str
            Name of the filter (for error messages).

        Returns
        -------
        Normalized value (single-item list -> scalar).

        Raises
        ------
        KeyError
            If value is not in valid_values.
        """
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

    # -------------------------------------------------------------------------
    # Core computation methods
    # -------------------------------------------------------------------------

    def _compute_all_columns(self) -> pd.DataFrame:
        """
        Compute ALL derived columns for the full dataset.

        Returns DataFrame with all computed columns (per-wavelength rows).
        Rows for wavelengths not in fluence dict will have NaN for computed columns.
        """
        fluence = self._fluence
        df = self._base_df.copy()

        # Handle fluence dict - validate wavelengths exist
        if isinstance(fluence, dict):
            # Validate fluence dict wavelengths exist (warns if any missing)
            # Don't filter df - let _build_display_df handle wavelength filtering
            _, fluence = _filter_wavelengths(df, fluence.copy())
            # Update _fluence_wavelengths with cleaned keys
            self._fluence_wavelengths = list(fluence.keys())

        # Calculate time to inactivation for all log levels (1-5)
        log_funcs = {1: log1, 2: log2, 3: log3, 4: log4, 5: log5}

        for log_level, func in log_funcs.items():
            label = LOG_LABELS[log_level]
            sec_key = f"Seconds to {label} inactivation"
            df[sec_key] = df.apply(_compute_row, args=[fluence, func, 0], axis=1)

        # Calculate eACH-UV for ALL rows (will be NaN for non-Aerosol in display)
        df[COL_EACH] = df.apply(_compute_row, args=[fluence, eACH_UV, 1], axis=1)

        # Calculate CADR if volume provided (both units, display controlled by subset)
        if self._volume_m3 is not None:
            cubic_feet = self._volume_m3 * 35.3147  # 1 m³ = 35.3147 ft³
            liters = self._volume_m3 * 1000
            df[COL_CADR_LPS] = (df[COL_EACH] * liters / 3600).round(1)
            df[COL_CADR_CFM] = (df[COL_EACH] * cubic_feet / 60).round(1)

        # Calculate all time unit variants
        self._time_cols = self._calculate_all_time_columns(df)

        return df

    def _combine_wavelengths(
        self, df: pd.DataFrame, fluence_dict: dict
    ) -> pd.DataFrame:
        """
        Combine multi-wavelength data into aggregated rows.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame already filtered by medium/category (from _apply_row_filters).
        fluence_dict : dict
            Mapping of wavelength -> fluence value.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with one row per (Species, Medium) combination,
            with aggregated eACH-UV and time columns.
        """
        summed_data = []
        wavelengths = list(fluence_dict.keys())

        # Always include Medium in grouping for full_df
        group_cols = [COL_SPECIES, COL_MEDIUM]

        for group_key, group in df.groupby(group_cols):
            species, med = group_key
            category = group[COL_CATEGORY].iloc[0]

            # Collect data for each wavelength
            data_by_wv = {}
            for wv in wavelengths:
                wv_rows = group[group[COL_WAVELENGTH] == wv]
                data_by_wv[wv] = [
                    {
                        "k1": row[COL_K1] if pd.notna(row[COL_K1]) else 0.0,
                        "k2": row[COL_K2] if pd.notna(row[COL_K2]) else 0.0,
                        "f": float(row[COL_RESISTANT].rstrip("%")) / 100 if pd.notna(row[COL_RESISTANT]) else 0.0,
                    }
                    for _, row in wv_rows.iterrows()
                ]

            # Skip if no data for any wavelength
            if any(len(data_by_wv[wv]) == 0 for wv in wavelengths):
                continue

            # Generate all combinations across wavelengths
            for combo in product(*[data_by_wv[wv] for wv in wavelengths]):
                k1_list = [item["k1"] for item in combo]
                k2_list = [item["k2"] for item in combo]
                f_list = [item["f"] for item in combo]
                irrad_list = [fluence_dict[wv] for wv in wavelengths]

                # Calculate all log levels
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

        # Add minutes/hours columns (uses same column names as per-wavelength)
        self._calculate_all_time_columns(result_df)

        return result_df

    def _calculate_all_time_columns(self, df) -> dict:
        """
        Calculate all time columns for log1-5 in seconds/minutes/hours.
        Stores all in df, returns dict of column names by log level.
        """
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

    # -------------------------------------------------------------------------
    # Filter methods
    # -------------------------------------------------------------------------

    def subset(
        self,
        medium: str | list | None = None,
        category: str | list | None = None,
        wavelength: int | float | list | tuple | None = None,
        log: int | None = None,
        use_metric: bool | None = None,
    ) -> "Data":
        """
        Set filters for display. Returns self for chaining.

        Parameters
        ----------
        medium : str or list, optional
            Filter by medium ("Aerosol", "Surface", "Liquid").
        category : str or list, optional
            Filter by category ("Virus", "Bacteria", etc.).
        wavelength : int, float, list, or tuple, optional
            Filter by wavelength. Tuple (min, max) for range.
        log : int, optional
            Log reduction level for time column display (1-5).
            1=90%, 2=99%, 3=99.9%, 4=99.99%, 5=99.999%. Default is 2.
        use_metric : bool, optional
            If True, display CADR in lps; if False, display in cfm. Default True.

        Returns
        -------
        Data
            Self, for method chaining.
        """
        # Validate and set medium filter
        if medium is not None:
            self._medium = self._validate_filter(medium, self.get_valid_mediums(), "medium")

        # Validate and set category filter
        if category is not None:
            self._category = self._validate_filter(category, self.get_valid_categories(), "category")

        # Validate and set wavelength filter
        # When fluence is a dict, user-specified wavelengths are ADDED to the fluence
        # dict wavelengths (fluence wavelengths are always included in display)
        if wavelength is not None:
            valid_wavelengths = self.get_valid_wavelengths()
            if isinstance(wavelength, (int, float)):
                if wavelength not in valid_wavelengths:
                    raise KeyError(f"{wavelength} is not a valid wavelength; must be in {valid_wavelengths}")
            elif isinstance(wavelength, list):
                invalid = [w for w in wavelength if w not in valid_wavelengths]
                if invalid:
                    raise KeyError(f"Invalid wavelength(s) {invalid}; must be in {valid_wavelengths}")
                # Normalize single-item lists
                if len(wavelength) == 1:
                    wavelength = wavelength[0]
            self._wavelength = wavelength

        # Validate and set log level
        if log is not None:
            if log not in [1, 2, 3, 4, 5]:
                raise ValueError(f"log must be 1, 2, 3, 4, or 5; got {log}")
            self._log = log

        # Set CADR display units
        if use_metric is not None:
            self._use_metric_units = use_metric

        return self

    def _get_effective_wavelengths(self) -> list | tuple | None:
        """
        Get effective wavelengths for display (merged fluence + user-specified).

        Returns list of wavelengths, tuple (range), or None if no filter.
        Fluence dict wavelengths are always included when present.
        """
        # If user specified a range (tuple), use it directly
        # (fluence wavelengths should already be in the range or user knows what they're doing)
        if isinstance(self._wavelength, tuple):
            return self._wavelength

        # Collect wavelengths to include
        wavelengths = set()

        # Always include fluence dict wavelengths if present
        if self._fluence_wavelengths:
            wavelengths.update(self._fluence_wavelengths)

        # Add user-specified wavelengths
        if self._wavelength is not None:
            if isinstance(self._wavelength, (int, float)):
                wavelengths.add(self._wavelength)
            elif isinstance(self._wavelength, list):
                wavelengths.update(self._wavelength)

        # Return None if no wavelengths to filter by, otherwise sorted list
        if not wavelengths:
            return None
        return sorted(wavelengths)

    def _apply_row_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply medium/category filters to df.

        Does NOT apply wavelength filter (wavelength filtering happens separately
        based on context - per-wavelength vs combined data).
        """
        df = self._filter_by_column(df, COL_MEDIUM, self._medium)
        df = self._filter_by_column(df, COL_CATEGORY, self._category)
        return df

    # -------------------------------------------------------------------------
    # Display/output methods
    # -------------------------------------------------------------------------

    def _build_display_df(self) -> pd.DataFrame:
        """
        Build display DataFrame by applying filters and selecting appropriate columns.
        Uses internal filter state (_medium, _category) and effective wavelengths
        (merged from fluence dict wavelengths + user-specified wavelengths).
        """
        df = self._apply_row_filters(self._full_df.copy())
        df = self._apply_wavelength_filter(df)
        return self._select_display_columns(df)

    def _select_display_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select which columns to display based on context.
        Always shows base columns, optionally adds CADR, eACH-UV, and one time column.
        """
        df = df.copy()

        # Build ordered list of columns to display
        display_cols = []

        # Add efficacy columns (CADR, eACH-UV) at the front if applicable
        # Show eACH-UV only when medium filter is exactly "Aerosol"
        show_each = self._medium == "Aerosol" and COL_EACH in df.columns

        # Show CADR only when CADR columns exist AND medium is "Aerosol"
        show_cadr = self._medium == "Aerosol" and COL_CADR_LPS in df.columns

        if show_cadr:
            # Use appropriate CADR unit based on stored unit preference
            if self._use_metric_units:
                display_cols.append(COL_CADR_LPS)
            else:
                display_cols.append(COL_CADR_CFM)

        if show_each:
            display_cols.append(COL_EACH)

        # Add ONE time column if fluence was provided (based on _log level)
        if self._fluence is not None and self._time_cols and self._log in self._time_cols:
            # Use _select_time_display_columns to get the best unit for this log level
            time_col, _ = self._select_time_display_columns(
                df, self._time_cols, log_level=self._log
            )
            if time_col and time_col in df.columns:
                display_cols.append(time_col)

        # Add ALL base columns (always shown)
        for col in BASE_DISPLAY_COLS:
            if col in df.columns:
                display_cols.append(col)

        # Filter to only columns that exist
        display_cols = [c for c in display_cols if c in df.columns]

        # Remove duplicates while preserving order
        seen = set()
        final_cols = []
        for c in display_cols:
            if c not in seen:
                seen.add(c)
                final_cols.append(c)

        result = df[final_cols].copy()

        # Fill NaN with spaces only for string columns
        for col in result.columns:
            if result[col].dtype == object:
                result[col] = result[col].fillna(" ")

        # Apply shared output column selection (drop single-value cols)
        return self._select_output_columns(result)

    def _select_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply shared column selection: drop single-value cols, select best time col."""
        df = df.copy()

        # Drop single-value columns
        for col in [COL_MEDIUM, COL_CATEGORY, COL_WAVELENGTH]:
            if col in df.columns and len(df[col].unique()) == 1:
                df = df.drop(columns=[col])

        # Keep only the relevant time column for current log level
        # Only attempt if we have time_cols info and the seconds column exists in df
        if self._time_cols and self._log in self._time_cols:
            sec_col = self._time_cols[self._log].get("seconds")
            if sec_col and sec_col in df.columns:
                time_col, _ = self._select_time_display_columns(df, self._time_cols, log_level=self._log)
                # Drop all other time columns
                all_time_cols = []
                for log_cols in self._time_cols.values():
                    all_time_cols.extend(log_cols.values())
                for col in all_time_cols:
                    if col in df.columns and col != time_col:
                        df = df.drop(columns=[col])

        return df

    def _select_time_display_columns(self, df, time_cols, log_level=2, left_axis=None, right_axis=None):
        """
        Select which time columns to display based on value ranges or user preference.

        Parameters
        ----------
        df : DataFrame
            Data to analyze for automatic unit selection.
        time_cols : dict
            Dictionary mapping log levels to unit column names.
        log_level : int
            Log reduction level (1, 2, or 3).
        left_axis : str, optional
            User-specified left axis unit: "seconds", "minutes", or "hours".
        right_axis : str or False, optional
            User-specified right axis unit, or False to disable right axis.

        Returns tuple of (left_col, right_col) for display.

        Automatic selection logic (when no user preference):
        - If median < 100 seconds: seconds on left only (no right axis)
        - If median < 6000 seconds (100 min): minutes on left, seconds on right
        - Otherwise: hours on left, minutes on right
        """
        if log_level not in time_cols:
            return None, None

        cols = time_cols[log_level]
        sec_key = cols["seconds"]
        min_key = cols["minutes"]
        hr_key = cols["hours"]

        unit_to_key = {"seconds": sec_key, "minutes": min_key, "hours": hr_key}

        # User-specified axes override automatic selection
        if left_axis is not None:
            left_key = unit_to_key.get(left_axis.lower())
            if left_key is None:
                raise ValueError(f"Invalid left_axis '{left_axis}'. Must be 'seconds', 'minutes', or 'hours'.")

            if right_axis is False:
                return left_key, None
            elif right_axis is not None:
                right_key = unit_to_key.get(right_axis.lower())
                if right_key is None:
                    raise ValueError(f"Invalid right_axis '{right_axis}'. Must be 'seconds', 'minutes', 'hours', or False.")
                return left_key, right_key
            else:
                # User specified left only - use standard pairing for right
                if left_axis.lower() == "hours":
                    return left_key, min_key
                elif left_axis.lower() == "minutes":
                    return left_key, sec_key
                else:  # seconds
                    return left_key, None

        if len(df) == 0:
            return min_key, sec_key

        # Automatic selection based on median value
        median_seconds = df[sec_key].median()

        if median_seconds < 100:
            # Seconds is best - seconds on left only (no right axis)
            return sec_key, None
        elif median_seconds < 6000:  # Less than 100 minutes
            # Minutes is best - minutes on left, seconds on right
            return min_key, sec_key
        else:
            # Hours is best - hours on left, minutes on right
            return hr_key, min_key

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def display_df(self) -> pd.DataFrame:
        """Return filtered DataFrame with context-appropriate columns."""
        df = self._build_display_df()
        return df.sort_values(COL_SPECIES)

    def table(self) -> pd.DataFrame:
        """Return filtered DataFrame with context-appropriate columns."""
        return self.display_df

    @property
    def df(self) -> pd.DataFrame:
        """Alias for display_df (for backwards compatibility)."""
        return self.display_df

    @property
    def base_df(self) -> pd.DataFrame:
        """Return the raw CSV data (no computed columns)."""
        return self._base_df

    @property
    def full_df(self) -> pd.DataFrame:
        """Return all computed columns (when fluence provided)."""
        return self._full_df

    @property
    def combined_df(self) -> pd.DataFrame | None:
        """Combined multi-wavelength df for plotting, computed on demand."""
        if not isinstance(self._fluence, dict) or len(self._fluence) <= 1:
            return None
        filtered = self._apply_row_filters(self._full_df)
        result = self._combine_wavelengths(filtered, self._fluence)
        result = self._select_output_columns(result)
        return result.sort_values(COL_SPECIES)

    @property
    def medium(self):
        return self._medium

    @property
    def category(self):
        return self._category

    @property
    def wavelength(self):
        """Return effective wavelengths (merged fluence + user-specified)."""
        return self._get_effective_wavelengths()

    @property
    def fluence(self):
        return self._fluence

    @property
    def log(self):
        return self._log

    @property
    def keys(self):
        return self.display_df.keys()

    @property
    def categories(self):
        df = self.display_df
        if COL_CATEGORY not in df.columns:
            return None
        return sorted(df[COL_CATEGORY].unique())

    @property
    def mediums(self):
        df = self.display_df
        if COL_MEDIUM not in df.columns:
            return None
        return sorted(df[COL_MEDIUM].unique())

    @property
    def wavelengths(self):
        df = self.display_df
        if COL_WAVELENGTH not in df.columns:
            return None
        return sorted(df[COL_WAVELENGTH].unique())

    # -------------------------------------------------------------------------
    # Class methods for validation
    # -------------------------------------------------------------------------

    @classmethod
    def get_valid_categories(cls) -> list[str]:
        """Return list of valid category values without instantiating."""
        return sorted(cls.get_full()[COL_CATEGORY].unique())

    @classmethod
    def get_valid_mediums(cls) -> list[str]:
        """Return list of valid medium values without instantiating."""
        return sorted(cls.get_full()[COL_MEDIUM].unique())

    @classmethod
    def get_valid_wavelengths(cls) -> list[float]:
        """Return list of valid wavelength values without instantiating."""
        return sorted(cls.get_full()[COL_WAVELENGTH].unique())

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def plot(self, **kwargs):
        """Plot inactivation data. See plotting.plot for full documentation."""
        return _plot_func(self, **kwargs)

    def _get_key(self, substring):
        """Get column key containing substring from display_df."""
        df = self.display_df
        index = np.array([substring in key for key in df.keys()])
        return df.keys()[index][0] if sum(index) > 0 else None


