import numbers
import warnings
from collections.abc import Callable

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
    COL_EACH,
    COL_CADR_LPS,
    COL_CADR_CFM,
    BASE_DISPLAY_COLS,
)
from .math import CADR_CFM, CADR_LPS, eACH_UV, log1, log2, log3, log4, log5
from .plotting import plot_swarm, plot_survival, plot_wavelength
from .utils import auto_select_time_columns
from ._filtering import (
    apply_row_filters,
    apply_wavelength_filter,
    get_effective_wavelengths,
)
from ._computation import (
    compute_all_columns,
    combine_wavelengths,
)
from ._averaging import (
    collect_parametric_inputs,
    resolve_function,
    average_value_parametric,
    filter_for_average,
    compute_average_single,
    compute_average_multiwavelength,
)
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


class InactivationData:
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
        Initialize with optional fluence and volume for CADR computation.

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
        self._species = None
        self._strain = None
        self._condition = None
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
            self._full_df, self._time_cols, fluence_wavelengths = compute_all_columns(
                self._base_df, self._fluence, self._volume_m3
            )
            if fluence_wavelengths is not None:
                self._fluence_wavelengths = fluence_wavelengths
        else:
            self._full_df = self._base_df.copy()

    # =========================================================================
    # Public API
    # =========================================================================

    def subset(
        self,
        medium: str | list | None = None,
        category: str | list | None = None,
        species: str | None = None,
        strain: str | None = None,
        condition: str | None = None,
        wavelength: int | float | list | tuple | None = None,
        log: int | None = None,
        use_metric: bool | None = None,
    ) -> "InactivationData":
        """
        Set filters for display. Returns self for chaining.

        medium : str or list, optional
            Filter by medium ("Aerosol", "Surface", "Liquid"). Case-insensitive,
            accepts aliases: "air" for Aerosol, "water" for Liquid.
        category : str or list, optional
            Filter by category ("Virus", "Bacteria", etc.). Case-insensitive,
            accepts aliases: "virus" for Viruses, "spores" for Bacterial spores.
        species : str, optional
            Filter by species name. Partial matching (all words must appear).
        strain : str, optional
            Filter by strain. Substring matching.
        condition : str, optional
            Filter by condition. Substring matching.
        wavelength : int, float, list, or tuple, optional
            Filter by wavelength. Tuple (min, max) for range.
        log : int, optional
            Log reduction level for time column display (1-5).
            1=90%, 2=99%, 3=99.9%, 4=99.99%, 5=99.999%. Default is 2.
        use_metric : bool, optional
            If True, display CADR in lps; if False, display in cfm. Default True.
        """
        # Set row filters (all use case-insensitive substring matching)
        if medium is not None:
            self._medium = medium
        if category is not None:
            self._category = category
        if species is not None:
            self._species = species
        if strain is not None:
            self._strain = strain
        if condition is not None:
            self._condition = condition

        # Validate and set wavelength filter
        # When fluence is a dict, user-specified wavelengths are ADDED to the fluence
        # dict wavelengths (fluence wavelengths are always included in display)
        if wavelength is not None:
            valid_wavelengths = self.get_valid_wavelengths()
            if isinstance(wavelength, numbers.Real):
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

    def table(self, species=None, strain=None, condition=None, medium=None, category=None) -> pd.DataFrame:
        """Return filtered DataFrame with context-appropriate columns."""
        df = self._get_filtered_df(species, strain, condition, medium, category)
        df = self._select_display_columns(df)
        for col in [COL_SPECIES, COL_STRAIN, COL_CONDITION]:
            if col in df.columns:
                return df.sort_values(col)
        return df

    def plot(self, **kwargs):
        """Plot inactivation data. See plotting.plot for full documentation."""
        return plot_swarm(self, **kwargs)

    def plot_survival(self, **kwargs):
        """Plot survival fraction over time. See plotting.plot_survival for full documentation."""
        # Only use data's fluence if not explicitly provided
        if "fluence" not in kwargs and self.fluence is not None:
            kwargs["fluence"] = self.fluence
        return plot_survival(self, **kwargs)

    def plot_wavelength(self, **kwargs):
        """Plot wavelength vs k-values. See plotting.plot_wavelength for full documentation."""
        return plot_wavelength(self, **kwargs)

    def save(self, filepath: str, **kwargs) -> None:
        """
        Save the current display_df to a CSV file.
        """
        self.display_df.to_csv(filepath, index=False, **kwargs)

    def average_value(
        self,
        function: str | Callable | list,
        species: str | list | None = None,
        strain: str | list | None = None,
        condition: str | list | None = None,
        medium: str | list | None = None,
        category: str | list | None = None,
        **kwargs,
    ) -> float | dict | None:
        """
        Compute a derived value from averaged k parameters.

        Filters data, averages k1/k2/f across matching rows, applies function.
        List inputs return nested dicts keyed by those values.
        """
        if self._fluence is None:
            warnings.warn("fluence must be set to compute average_value", stacklevel=2)
            return None

        parametric = collect_parametric_inputs(function, species, strain, medium, condition, category)
        if parametric:
            return average_value_parametric(
                self.average_value, function, parametric, species, strain, condition, medium, category, **kwargs
            )

        return self._average_value_single(function, species, strain, condition, medium, category, **kwargs)

    def _average_value_single(
        self,
        function: str | Callable,
        species: str | None = None,
        strain: str | None = None,
        condition: str | None = None,
        medium: str | None = None,
        category: str | None = None,
        **kwargs,
    ) -> float | None:
        """Compute a single derived value from averaged k parameters."""
        df = self._get_filtered_df(species, strain, condition, medium, category)

        df = filter_for_average(df, **kwargs)
        if len(df) == 0:
            warnings.warn("No rows match the specified filters", stacklevel=2)
            return None

        func, func_name = resolve_function(function, FUNCTION_MAP)

        if isinstance(self._fluence, dict) and len(self._fluence) > 1:
            return compute_average_multiwavelength(df, func, func_name, self._fluence, self._volume_m3)
        return compute_average_single(df, func, func_name, self._fluence, self._volume_m3)

    # =========================================================================
    # Public properties - data access
    # =========================================================================

    @property
    def display_df(self) -> pd.DataFrame:
        """Return filtered DataFrame with context-appropriate columns."""
        df = self._get_filtered_df()
        df = self._select_display_columns(df)
        for col in [COL_SPECIES, COL_STRAIN, COL_CONDITION]:
            if col in df.columns:
                return df.sort_values(col)
        return df

    @property
    def base_df(self) -> pd.DataFrame:
        """Return the raw CSV data (no computed columns)."""
        return self._base_df

    @property
    def full_df(self) -> pd.DataFrame:
        """Return all computed columns (when fluence provided), filtered."""
        return self._get_filtered_df()

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
            self._combined_full_df, _ = combine_wavelengths(
                self._full_df, self._fluence, self._volume_m3
            )
        return self._combined_full_df

    @property
    def combined_df(self) -> pd.DataFrame | None:
        """Combined multi-wavelength df with context-appropriate columns for display."""
        if self.combined_full_df is None:
            return None
        filtered = self._get_filtered_df(df=self.combined_full_df.copy())
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
        return get_effective_wavelengths(self._wavelength, self._fluence_wavelengths)

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
        df = self.full_df
        if COL_CATEGORY not in df.columns:
            return None
        values = df[COL_CATEGORY].dropna().unique()
        return sorted(values) if len(values) > 0 else None

    @property
    def species(self):
        df = self.full_df
        if COL_SPECIES not in df.columns:
            return None
        values = df[COL_SPECIES].dropna().unique()
        return sorted(values) if len(values) > 0 else None

    @property
    def strains(self):
        df = self.full_df
        if COL_STRAIN not in df.columns:
            return None
        values = df[COL_STRAIN].dropna().unique()
        return sorted(values) if len(values) > 0 else None

    @property
    def mediums(self):
        df = self.full_df
        if COL_MEDIUM not in df.columns:
            return None
        values = df[COL_MEDIUM].dropna().unique()
        return sorted(values) if len(values) > 0 else None

    @property
    def conditions(self):
        df = self.full_df
        if COL_CONDITION not in df.columns:
            return None
        values = df[COL_CONDITION].dropna().unique()
        return sorted(values) if len(values) > 0 else None

    @property
    def wavelengths(self):
        df = self.display_df
        if COL_WAVELENGTH not in df.columns:
            return None
        return sorted(df[COL_WAVELENGTH].unique())
        
    # =========================================================================
    # Private computation methods
    # =========================================================================

    # =========================================================================
    # Private filter methods - delegate to _filtering module
    # =========================================================================

    def _get_filtered_df(self, species=None, strain=None, condition=None, medium=None, category=None, df=None):
        """Apply filters, merging call-time overrides with instance state."""
        if df is None:
            df = self._full_df.copy()
        df = apply_row_filters(
            df,
            medium=medium if medium is not None else self._medium,
            category=category if category is not None else self._category,
            species=species if species is not None else self._species,
            strain=strain if strain is not None else self._strain,
            condition=condition if condition is not None else self._condition,
        )
        return self._apply_wavelength_filter(df)

    def _apply_wavelength_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply wavelength filter based on effective wavelengths."""
        return apply_wavelength_filter(df, get_effective_wavelengths(self._wavelength, self._fluence_wavelengths))

    # =========================================================================
    # Private display methods
    # =========================================================================

    def _select_display_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select columns to display based on context."""
        display_cols = []

        # Show eACH/CADR only when data is exclusively Aerosol
        is_aerosol_only = (
            COL_MEDIUM in df.columns
            and len(df) > 0
            and (df[COL_MEDIUM] == "Aerosol").all()
        )
        show_each = is_aerosol_only and COL_EACH in df.columns
        show_cadr = is_aerosol_only and COL_CADR_LPS in df.columns

        if show_cadr:
            display_cols.append(COL_CADR_LPS if self._use_metric_units else COL_CADR_CFM)
        if show_each:
            display_cols.append(COL_EACH)

        if self._fluence is not None and self._time_cols and self._log in self._time_cols:
            primary, secondary = auto_select_time_columns(df, self._time_cols, self._log)
            if primary and primary in df.columns:
                display_cols.append(primary)
            if secondary and secondary in df.columns:
                display_cols.append(secondary)

        for col in BASE_DISPLAY_COLS:
            if col in df.columns:
                # Skip columns with single value (uninformative for display)
                if len(df[col].unique()) == 1:
                    continue
                display_cols.append(col)

        seen = set()
        final_cols = [c for c in display_cols if not (c in seen or seen.add(c))]
        result = df[final_cols].copy()

        for col in result.columns:
            if result[col].dtype == object:
                result[col] = result[col].fillna(" ")

        return result
