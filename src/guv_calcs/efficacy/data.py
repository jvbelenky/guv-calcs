import re
import warnings
from collections.abc import Callable
from itertools import product

import pandas as pd

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
    COL_EACH,
    COL_CADR_LPS,
    COL_CADR_CFM,
    BASE_DISPLAY_COLS,
)
from .math import CADR_CFM, CADR_LPS, eACH_UV, log1, log2, log3, log4, log5
from .plotting import plot_swarm, plot_survival
from .utils import auto_select_time_columns

pd.options.mode.chained_assignment = None

# Function name mapping for average_value()
FUNCTION_MAP = {
    "log1": log1,
    "log2": log2,
    "log3": log3,
    "log4": log4,
    "log5": log5,
    "each": eACH_UV,
    "each-uv": eACH_UV,
    "each_uv": eACH_UV,
    "eACH_UV": eACH_UV,
    "cadr-lps": CADR_LPS,
    "CADR-LPS": CADR_LPS,
    "cadr_lps": CADR_LPS,
    "CADR_LPS": CADR_LPS,
    "cadr-cfm": CADR_CFM,
    "CADR-CFM": CADR_CFM,
    "cadr_cfm": CADR_CFM,
    "CADR_CFM": CADR_CFM,
}


class Data:
    """
    UV disinfection efficacy data handler.

    Provides access to the disinfection table with optional fluence-based
    computed columns (eACH-UV, time to inactivation).
    """

    # =========================================================================
    # Class methods
    # =========================================================================

    @classmethod
    def get_full(cls) -> pd.DataFrame:
        """Return full disinfection table (cached, returns copy)."""
        return get_full_disinfection_table().copy()

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

    # =========================================================================
    # Constructor
    # =========================================================================

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
        self._fluence_wavelengths = (
            list(fluence.keys()) if isinstance(fluence, dict) else None
        )

        # Load base data (cached)
        self._base_df = self.get_full()
        self._time_cols = {}  # Populated by _compute_all_columns
        self._combined_full_df = None  # Lazily computed by combined_full_df property

        # Compute full_df if fluence provided
        if fluence is not None:
            self._full_df = self._compute_all_columns()
        else:
            self._full_df = self._base_df.copy()

    # =========================================================================
    # Public API
    # =========================================================================

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
            self._medium = self._validate_filter(
                medium, self.get_valid_mediums(), "medium"
            )

        # Validate and set category filter
        if category is not None:
            self._category = self._validate_filter(
                category, self.get_valid_categories(), "category"
            )

        # Validate and set wavelength filter
        # When fluence is a dict, user-specified wavelengths are ADDED to the fluence
        # dict wavelengths (fluence wavelengths are always included in display)
        if wavelength is not None:
            valid_wavelengths = self.get_valid_wavelengths()
            if isinstance(wavelength, (int, float)):
                if wavelength not in valid_wavelengths:
                    raise KeyError(
                        f"{wavelength} is not a valid wavelength; must be in {valid_wavelengths}"
                    )
            elif isinstance(wavelength, list):
                invalid = [w for w in wavelength if w not in valid_wavelengths]
                if invalid:
                    raise KeyError(
                        f"Invalid wavelength(s) {invalid}; must be in {valid_wavelengths}"
                    )
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

    def table(self) -> pd.DataFrame:
        """Return filtered DataFrame with context-appropriate columns."""
        return self.display_df

    def plot(self, **kwargs):
        """Plot inactivation data. See plotting.plot for full documentation."""
        return plot_swarm(self, **kwargs)

    def plot_survival(self, **kwargs):
        """Plot survival fraction over time. See plotting.plot_survival for full documentation."""
        # Only use data's fluence if not explicitly provided
        if "fluence" not in kwargs and self.fluence is not None:
            kwargs["fluence"] = self.fluence
        return plot_survival(self, **kwargs)

    def save(self, filepath: str, **kwargs) -> None:
        """
        Save the current display_df to a CSV file.
        """
        self.display_df.to_csv(filepath, index=False, **kwargs)

    def average_value(
        self,
        function: str | Callable,
        species: str | list | None = None,
        strain: str | list | None = None,
        condition: str | list | None = None,
        medium: str | list | None = None,
        **kwargs,
    ) -> float | dict | None:
        """
        Compute a derived value from averaged k parameters.

        Filters the data by the specified criteria (in addition to existing
        medium/category/wavelength filters), averages k1, k2, f across matching
        rows, then applies the specified function.

        When any filter parameter is a list, returns a (nested) dict keyed by
        those values instead of a single float.

        function : str or callable
            One of: 'log1', 'log2', 'log3', 'log4', 'log5', 'eACH_UV',
            'CADR_LPS', 'CADR_CFM', or a callable with signature
            f(irrad, k1, k2, f) -> float
        species : str or list, optional
            Filter by species name (smart partial match - all query words
            must appear in target, e.g., "e. coli" matches "Escherichia coli").
            If a list, returns dict keyed by species values.
        strain : str or list, optional
            Filter by strain (partial match, case-insensitive).
            If a list, returns dict keyed by strain values.
        condition : str or list, optional
            Filter by condition (partial match, case-insensitive).
            If a list, returns dict keyed by condition values.
        medium : str or list, optional
            Filter by medium ("Aerosol", "Surface", "Liquid"). Overrides
            instance medium filter if provided.
            If a list, returns dict keyed by medium values.
        **kwargs
            Additional column filters as column_name=value (exact match)

        Returns
        -------
        float, dict, or None
            The computed value using averaged k parameters, or None if
            fluence is not set or no rows match the filters. Returns a
            (nested) dict when any filter parameter is a list.

        Examples
        --------
        >>> data = Data(fluence=0.5)
        >>> data.average_value("log2", species="E. coli")
        1.23

        >>> data.average_value("log2", species=["E. coli", "Staph"])
        {"E. coli": 1.23, "Staph": 4.56}

        >>> data.average_value("log2", species=["E. coli"], medium=["Aerosol", "Surface"])
        {"E. coli": {"Aerosol": 1.23, "Surface": 2.34}}
        """
        if self._fluence is None:
            warnings.warn("fluence must be set to compute average_value", stacklevel=2)
            return None

        # Check for parametric (list) inputs
        # Order: species -> strain -> medium -> condition (condition depends on medium)
        parametric = []
        if isinstance(species, list):
            parametric.append(("species", species))
        if isinstance(strain, list):
            parametric.append(("strain", strain))
        if isinstance(medium, list):
            parametric.append(("medium", medium))
        if isinstance(condition, list):
            parametric.append(("condition", condition))

        # If any parametric inputs, build nested dict recursively
        if parametric:
            return self._average_value_parametric(
                function, parametric, species, strain, condition, medium, **kwargs
            )

        # Non-parametric case: return single value
        return self._average_value_single(
            function, species, strain, condition, medium, **kwargs
        )

    def _average_value_parametric(
        self,
        function: str | Callable,
        parametric: list[tuple[str, list]],
        species: str | list | None,
        strain: str | list | None,
        condition: str | list | None,
        medium: str | list | None,
        **kwargs,
    ) -> dict:
        """
        Build nested dict for parametric average_value calls.

        Iterates over the first parametric dimension, recursively calling
        average_value for each value in that dimension.
        """
        param_name, param_values = parametric[0]
        remaining_parametric = parametric[1:]

        result = {}
        for val in param_values:
            # Build kwargs for recursive call with this dimension fixed
            call_kwargs = {
                "species": val if param_name == "species" else species,
                "strain": val if param_name == "strain" else strain,
                "condition": val if param_name == "condition" else condition,
                "medium": val if param_name == "medium" else medium,
                **kwargs,
            }

            # If there are remaining parametric dimensions, rebuild the list args
            # so the recursive call continues to iterate over them
            if remaining_parametric:
                for remaining_name, remaining_values in remaining_parametric:
                    call_kwargs[remaining_name] = remaining_values

            sub_result = self.average_value(function, **call_kwargs)
            result[val] = sub_result

        return result

    def _average_value_single(
        self,
        function: str | Callable,
        species: str | None = None,
        strain: str | None = None,
        condition: str | None = None,
        medium: str | None = None,
        **kwargs,
    ) -> float | None:
        """
        Compute a single derived value from averaged k parameters.

        This is the core implementation called by average_value for non-parametric cases.
        """
        # Start with filtered data
        if medium is not None:
            # Override medium filter - start from _full_df and apply filters manually
            df = self._full_df.copy()
            df = self._filter_by_column(df, COL_MEDIUM, medium)
            df = self._filter_by_column(df, COL_CATEGORY, self._category)
            df = self._apply_wavelength_filter(df)
        else:
            # Use existing filters
            df = self.full_df.copy()

        # Apply species filter (smart partial match)
        if species is not None:
            mask = df[COL_SPECIES].apply(lambda x: self._species_matches(species, x))
            df = df[mask]

        # Apply partial match filters (case-insensitive contains)
        if strain is not None:
            strain_lower = strain.lower()
            df = df[df[COL_STRAIN].fillna("").str.lower().str.contains(strain_lower)]

        if condition is not None:
            condition_lower = condition.lower()
            df = df[
                df[COL_CONDITION].fillna("").str.lower().str.contains(condition_lower)
            ]

        # Apply additional exact match filters from kwargs
        for col, value in kwargs.items():
            if col in df.columns:
                df = df[df[col] == value]

        if len(df) == 0:
            warnings.warn("No rows match the specified filters", stacklevel=2)
            return None

        # Resolve function
        if isinstance(function, str):
            if function not in FUNCTION_MAP:
                valid_funcs = list(FUNCTION_MAP.keys())
                raise ValueError(
                    f"Unknown function '{function}'; must be one of {valid_funcs}"
                )
            func = FUNCTION_MAP[function]
            func_name = function.lower()
        else:
            func = function
            func_name = None

        # Handle multi-wavelength fluence
        if isinstance(self._fluence, dict) and len(self._fluence) > 1:
            return self._compute_average_multiwavelength(df, func, func_name)
        else:
            return self._compute_average_single(df, func, func_name)

    # =========================================================================
    # Public properties - data access
    # =========================================================================

    @property
    def display_df(self) -> pd.DataFrame:
        """Return filtered DataFrame with context-appropriate columns."""
        df = self._apply_row_filters(self._full_df.copy())
        df = self._apply_wavelength_filter(df)
        df = self._select_display_columns(df)
        return df.sort_values(COL_SPECIES)

    @property
    def base_df(self) -> pd.DataFrame:
        """Return the raw CSV data (no computed columns)."""
        return self._base_df

    @property
    def full_df(self) -> pd.DataFrame:
        """Return all computed columns (when fluence provided), filtered by medium/category."""
        df = self._apply_row_filters(self._full_df.copy())
        return self._apply_wavelength_filter(df)

    @property
    def combined_full_df(self) -> pd.DataFrame | None:
        """
        Full combined DataFrame with ALL columns (computed lazily).

        This stores all computed columns for multi-wavelength data.
        Use combined_df for display with context-appropriate columns.
        """
        if not isinstance(self._fluence, dict) or len(self._fluence) <= 1:
            return None
        if self._combined_full_df is None:
            # Compute combined df from full_df (all wavelengths, all rows)
            # Note: We don't filter here - filtering is applied when accessing combined_df
            self._combined_full_df = self._combine_wavelengths(
                self._full_df, self._fluence
            )
        return self._combined_full_df

    @property
    def combined_df(self) -> pd.DataFrame | None:
        """Combined multi-wavelength df with context-appropriate columns for display."""
        if self.combined_full_df is None:
            return None
        # Apply row filters and column selection to the full combined df
        filtered = self._apply_row_filters(self.combined_full_df.copy())
        result = self._select_display_columns(filtered)
        return result.sort_values(COL_SPECIES)

    # =========================================================================
    # Public properties - filter state
    # =========================================================================

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

    # =========================================================================
    # Public properties - metadata
    # =========================================================================

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
    def species(self):
        df = self.display_df
        if COL_SPECIES not in df.columns:
            return None
        return sorted(df[COL_SPECIES].unique())
        
    @property
    def strains(self):
        df = self.display_df
        if COL_STRAIN not in df.columns:
            return None
        return sorted(df[COL_STRAIN].unique())

    @property
    def mediums(self):
        df = self.display_df
        if COL_MEDIUM not in df.columns:
            return None
        return sorted(df[COL_MEDIUM].unique())
        
    @property
    def conditions(self):
        df = self.display_df
        if COL_CONDITION not in df.columns:
            return None
        return sorted(df[COL_CONDITION].unique())

    @property
    def wavelengths(self):
        df = self.display_df
        if COL_WAVELENGTH not in df.columns:
            return None
        return sorted(df[COL_WAVELENGTH].unique())
        
    # =========================================================================
    # Private computation methods
    # =========================================================================

    @staticmethod
    def _species_matches(query: str, target: str) -> bool:
        """Check if all words in query appear in target (case-insensitive)."""
        query_words = re.findall(r"\w+", query.lower())
        target_lower = target.lower()
        return all(word in target_lower for word in query_words)

    @staticmethod
    def _parse_resistant(val) -> float:
        """Parse '0.33%' -> 0.0033, NaN -> 0.0"""
        if pd.isna(val):
            return 0.0
        if isinstance(val, str):
            return float(val.rstrip("%")) / 100
        return float(val)

    @staticmethod
    def _extract_kinetic_params(row):
        """
        Extract k1, k2, f kinetic parameters from a dataframe row.

        Returns dict with k1, k2, f values (handling NaN and % parsing).
        """
        k1 = row[COL_K1] if pd.notna(row[COL_K1]) else 0.0
        k2 = row[COL_K2] if pd.notna(row[COL_K2]) else 0.0
        f = (
            float(row[COL_RESISTANT].rstrip("%")) / 100
            if pd.notna(row[COL_RESISTANT])
            else 0.0
        )
        return {"k1": k1, "k2": k2, "f": f}

    @staticmethod
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

        params = Data._extract_kinetic_params(row)
        return round(function(irrad=fluence, **params, **kwargs), sigfigs)

    @staticmethod
    def _filter_wavelengths(df, fluence_dict):
        """
        Filter dataframe for species with data for all wavelengths in fluence_dict.

        Returns filtered df and cleaned fluence_dict.
        """
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

    def _compute_average_single(
        self, df: pd.DataFrame, func: Callable, func_name: str | None
    ) -> float:
        """Compute average value for single wavelength fluence."""
        # Get fluence value
        if isinstance(self._fluence, dict):
            fluence = list(self._fluence.values())[0]
        else:
            fluence = self._fluence

        # Average k parameters
        k1_mean = df[COL_K1].mean()
        k2_mean = df[COL_K2].fillna(0).mean()
        f_mean = df[COL_RESISTANT].apply(self._parse_resistant).mean()

        # Call function with appropriate signature
        if func_name in ("cadr_lps",):
            if self._volume_m3 is None:
                raise ValueError("volume_m3 must be set to compute CADR")
            return func(self._volume_m3, fluence, k1_mean, k2_mean, f_mean)
        elif func_name in ("cadr_cfm",):
            if self._volume_m3 is None:
                raise ValueError("volume_m3 must be set to compute CADR")
            cubic_feet = self._volume_m3 * 35.3147
            return func(cubic_feet, fluence, k1_mean, k2_mean, f_mean)
        else:
            return func(fluence, k1_mean, k2_mean, f_mean)

    def _compute_average_multiwavelength(
        self, df: pd.DataFrame, func: Callable, func_name: str | None
    ) -> float:
        """Compute average value for multi-wavelength fluence."""
        wavelengths = list(self._fluence.keys())
        irrad_list = [self._fluence[wv] for wv in wavelengths]

        # Average k parameters per wavelength
        k1_list = []
        k2_list = []
        f_list = []

        for wv in wavelengths:
            wv_df = df[df[COL_WAVELENGTH] == wv]
            if len(wv_df) == 0:
                raise ValueError(f"No data available for wavelength {wv} nm")

            k1_list.append(wv_df[COL_K1].mean())
            k2_list.append(wv_df[COL_K2].fillna(0).mean())
            f_list.append(wv_df[COL_RESISTANT].apply(self._parse_resistant).mean())

        # Call function with lists
        if func_name in ("cadr_lps",):
            if self._volume_m3 is None:
                raise ValueError("volume_m3 must be set to compute CADR")
            return func(self._volume_m3, irrad_list, k1_list, k2_list, f_list)
        elif func_name in ("cadr_cfm",):
            if self._volume_m3 is None:
                raise ValueError("volume_m3 must be set to compute CADR")
            cubic_feet = self._volume_m3 * 35.3147
            return func(cubic_feet, irrad_list, k1_list, k2_list, f_list)
        else:
            return func(irrad_list, k1_list, k2_list, f_list)

    def _compute_all_columns(self) -> pd.DataFrame:
        """
        Compute ALL derived columns for the full dataset.

        Returns DataFrame with all computed columns (per-wavelength rows).
        Rows with missing k1 values are excluded since no calculations are possible.
        """
        fluence = self._fluence
        df = self._base_df.copy()

        # Filter out rows with missing k1 values - can't compute anything without k1
        df = df[df[COL_K1].notna()]

        # Handle fluence dict - validate wavelengths exist
        if isinstance(fluence, dict):
            # Validate fluence dict wavelengths exist (warns if any missing)
            # Don't filter df - let _build_display_df handle wavelength filtering
            _, fluence = self._filter_wavelengths(df, fluence.copy())
            # Update _fluence_wavelengths with cleaned keys
            self._fluence_wavelengths = list(fluence.keys())

        # Calculate time to inactivation for all log levels (1-5)
        log_funcs = {1: log1, 2: log2, 3: log3, 4: log4, 5: log5}

        for log_level, func in log_funcs.items():
            label = LOG_LABELS[log_level]
            sec_key = f"Seconds to {label} inactivation"
            df[sec_key] = df.apply(self._compute_row, args=[fluence, func, 0], axis=1)

        # Calculate eACH-UV for ALL rows (will be NaN for non-Aerosol in display)
        df[COL_EACH] = df.apply(self._compute_row, args=[fluence, eACH_UV, 1], axis=1)

        # Calculate CADR columns
        self._add_cadr_columns(df)

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
                    self._extract_kinetic_params(row) for _, row in wv_rows.iterrows()
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
                    f"Seconds to {LOG_LABELS[1]} inactivation": round(
                        log1(irrad_list, k1_list, k2_list, f_list), 0
                    ),
                    f"Seconds to {LOG_LABELS[2]} inactivation": round(
                        log2(irrad_list, k1_list, k2_list, f_list), 0
                    ),
                    f"Seconds to {LOG_LABELS[3]} inactivation": round(
                        log3(irrad_list, k1_list, k2_list, f_list), 0
                    ),
                    f"Seconds to {LOG_LABELS[4]} inactivation": round(
                        log4(irrad_list, k1_list, k2_list, f_list), 0
                    ),
                    f"Seconds to {LOG_LABELS[5]} inactivation": round(
                        log5(irrad_list, k1_list, k2_list, f_list), 0
                    ),
                }
                summed_data.append(row_data)

        result_df = pd.DataFrame(summed_data)

        # Add minutes/hours columns (uses same column names as per-wavelength)
        self._calculate_all_time_columns(result_df)

        # Add CADR columns
        self._add_cadr_columns(result_df)

        return result_df

    def _add_cadr_columns(self, df: pd.DataFrame) -> None:
        """Add CADR columns to DataFrame if volume is available and eACH exists."""
        if self._volume_m3 is not None and COL_EACH in df.columns:
            cubic_feet = self._volume_m3 * 35.3147  # 1 m³ = 35.3147 ft³
            liters = self._volume_m3 * 1000
            df[COL_CADR_LPS] = (df[COL_EACH] * liters / 3600).round(1)
            df[COL_CADR_CFM] = (df[COL_EACH] * cubic_feet / 60).round(1)

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

    # =========================================================================
    # Private filter methods
    # =========================================================================

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
                raise KeyError(
                    f"{value} is not a valid {name}; must be in {valid_values}"
                )
        elif isinstance(value, list):
            invalid = [v for v in value if v not in valid_values]
            if invalid:
                raise KeyError(
                    f"Invalid {name}(s) {invalid}; must be in {valid_values}"
                )
            if len(value) == 1:
                value = value[0]
        return value

    def _apply_row_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply medium/category filters to df.

        Does NOT apply wavelength filter (wavelength filtering happens separately
        based on context - per-wavelength vs combined data).
        """
        df = self._filter_by_column(df, COL_MEDIUM, self._medium)
        df = self._filter_by_column(df, COL_CATEGORY, self._category)
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

    # =========================================================================
    # Private display methods
    # =========================================================================

    def _select_display_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select which columns to display based on context.
        Always shows base columns, optionally adds CADR, eACH-UV, and time columns.
        Drops single-value columns (medium/category/wavelength) for cleaner display.
        """
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

        # Add time columns if fluence was provided (based on _log level)
        if (
            self._fluence is not None
            and self._time_cols
            and self._log in self._time_cols
        ):
            primary, secondary = auto_select_time_columns(
                df, self._time_cols, self._log
            )
            if primary and primary in df.columns:
                display_cols.append(primary)
            if secondary and secondary in df.columns:
                display_cols.append(secondary)

        # Add base columns, skipping single-value columns
        for col in BASE_DISPLAY_COLS:
            if col in df.columns:
                if col in (COL_MEDIUM, COL_CATEGORY, COL_WAVELENGTH):
                    # Skip single-value columns for cleaner display
                    if len(df[col].unique()) == 1:
                        continue
                display_cols.append(col)

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

        return result
