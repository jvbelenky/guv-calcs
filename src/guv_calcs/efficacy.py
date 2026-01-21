import pandas as pd
import warnings
import math
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import numpy as np
from .io import get_full_disinfection_table
from .units import LengthUnits

pd.options.mode.chained_assignment = None


def get_disinfection_table(fluence=None):
    """Return disinfection table with optional fluence-based computed columns."""
    return Data(fluence=fluence).df


class Data:

    @classmethod
    def get_full(cls) -> pd.DataFrame:
        """Return full disinfection table (cached, returns copy)."""
        return get_full_disinfection_table().copy()

    def __init__(self, fluence: float | dict | None = None):
        """
        Initialize Data with optional fluence computation.

        Parameters
        ----------
        fluence : float or dict, optional
            Fluence value(s) for computing time/eACH columns.
            Can be a single float or dict mapping wavelength to fluence.
        """
        self._fluence = fluence
        self._use_metric_units = True  # For CADR display (lps vs cfm)

        # Filter state (set by subset())
        self._medium = None
        self._category = None
        self._wavelength = None  # User-specified wavelength filter
        self._log = 2  # Log reduction level for time column display (1-5)

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

    def _compute_all_columns(self, room: "Room | None" = None) -> pd.DataFrame:
        """
        Compute ALL derived columns for the full dataset.

        Parameters
        ----------
        room : Room, optional
            Room object for CADR calculations. Only used transiently.

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
        log_labels = {1: "90%", 2: "99%", 3: "99.9%", 4: "99.99%", 5: "99.999%"}

        for log_level, func in log_funcs.items():
            label = log_labels[log_level]
            sec_key = f"Seconds to {label} inactivation"
            df[sec_key] = df.apply(_compute_row, args=[fluence, func, 0], axis=1)

        # Calculate eACH-UV for ALL rows (will be NaN for non-Aerosol in display)
        df["eACH-UV"] = df.apply(_compute_row, args=[fluence, eACH_UV, 1], axis=1)

        # Calculate CADR if room provided
        if room is not None:
            cadr_lps, _ = _get_cadr(df["eACH-UV"], room, units="lps")
            cadr_cfm, _ = _get_cadr(df["eACH-UV"], room, units="cfm")
            df["CADR-UV [lps]"] = cadr_lps.round(1)
            df["CADR-UV [cfm]"] = cadr_cfm.round(1)

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
        group_cols = ["Species", "Medium"]

        for group_key, group in df.groupby(group_cols):
            species, med = group_key
            category = group["Category"].iloc[0]

            # Collect data for each wavelength
            data_by_wv = {}
            for wv in wavelengths:
                wv_rows = group[group["wavelength [nm]"] == wv]
                data_by_wv[wv] = [
                    {
                        "k1": row["k1 [cm2/mJ]"] if pd.notna(row["k1 [cm2/mJ]"]) else 0.0,
                        "k2": row["k2 [cm2/mJ]"] if pd.notna(row["k2 [cm2/mJ]"]) else 0.0,
                        "f": float(row["% resistant"].rstrip("%")) / 100 if pd.notna(row["% resistant"]) else 0.0,
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
                    "Species": species,
                    "Category": category,
                    "Medium": med,
                    "eACH-UV": round(eACH_UV(irrad_list, k1_list, k2_list, f_list), 1),
                    "Seconds to 90% inactivation": round(log1(irrad_list, k1_list, k2_list, f_list), 0),
                    "Seconds to 99% inactivation": round(log2(irrad_list, k1_list, k2_list, f_list), 0),
                    "Seconds to 99.9% inactivation": round(log3(irrad_list, k1_list, k2_list, f_list), 0),
                    "Seconds to 99.99% inactivation": round(log4(irrad_list, k1_list, k2_list, f_list), 0),
                    "Seconds to 99.999% inactivation": round(log5(irrad_list, k1_list, k2_list, f_list), 0),
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
        log_labels = {1: "90%", 2: "99%", 3: "99.9%", 4: "99.99%", 5: "99.999%"}
        time_cols = {}

        for log_level in [1, 2, 3, 4, 5]:
            label = log_labels[log_level]
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

    def subset(
        self,
        medium: str | list | None = None,
        category: str | list | None = None,
        wavelength: int | float | list | tuple | None = None,
        log: int | None = None,
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

        Returns
        -------
        Data
            Self, for method chaining.
        """
        # Validate and set medium filter
        if medium is not None:
            valid_mediums = self.get_valid_mediums()
            if isinstance(medium, str):
                if medium not in valid_mediums:
                    raise KeyError(f"{medium} is not a valid medium; must be in {valid_mediums}")
            elif isinstance(medium, list):
                invalid = [m for m in medium if m not in valid_mediums]
                if invalid:
                    raise KeyError(f"Invalid medium(s) {invalid}; must be in {valid_mediums}")
            # Normalize single-item lists
            if isinstance(medium, list) and len(medium) == 1:
                medium = medium[0]
            self._medium = medium

        # Validate and set category filter
        if category is not None:
            valid_categories = self.get_valid_categories()
            if isinstance(category, str):
                if category not in valid_categories:
                    raise KeyError(f"{category} is not a valid category; must be in {valid_categories}")
            elif isinstance(category, list):
                invalid = [c for c in category if c not in valid_categories]
                if invalid:
                    raise KeyError(f"Invalid category(s) {invalid}; must be in {valid_categories}")
            # Normalize single-item lists
            if isinstance(category, list) and len(category) == 1:
                category = category[0]
            self._category = category

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
            if isinstance(wavelength, list) and len(wavelength) == 1:
                wavelength = wavelength[0]
            self._wavelength = wavelength

        # Validate and set log level
        if log is not None:
            if log not in [1, 2, 3, 4, 5]:
                raise ValueError(f"log must be 1, 2, 3, 4, or 5; got {log}")
            self._log = log

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
        if self._medium is not None:
            if isinstance(self._medium, str):
                df = df[df["Medium"] == self._medium]
            elif isinstance(self._medium, list):
                df = df[df["Medium"].isin(self._medium)]

        if self._category is not None:
            if isinstance(self._category, str):
                df = df[df["Category"] == self._category]
            elif isinstance(self._category, list):
                df = df[df["Category"].isin(self._category)]

        return df

    def _build_display_df(self) -> pd.DataFrame:
        """
        Build display DataFrame by applying filters and selecting appropriate columns.
        Uses internal filter state (_medium, _category) and effective wavelengths
        (merged from fluence dict wavelengths + user-specified wavelengths).
        """
        df = self._apply_row_filters(self._full_df.copy())

        # Get effective wavelengths (merged fluence + user-specified)
        effective_wv = self._get_effective_wavelengths()

        # Only filter by wavelength if the column exists and we have wavelengths to filter by
        if effective_wv is not None and "wavelength [nm]" in df.columns:
            if isinstance(effective_wv, (int, float)):
                df = df[df["wavelength [nm]"] == effective_wv]
            elif isinstance(effective_wv, list):
                df = df[df["wavelength [nm]"].isin(effective_wv)]
            elif isinstance(effective_wv, tuple) and len(effective_wv) == 2:
                wv_min, wv_max = effective_wv
                df = df[(df["wavelength [nm]"] >= wv_min) & (df["wavelength [nm]"] <= wv_max)]

        return self._select_display_columns(df)

    def _select_display_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select which columns to display based on context.
        Always shows base columns, optionally adds CADR, eACH-UV, and one time column.
        """
        df = df.copy()

        # Base columns that should ALWAYS appear
        base_cols = [
            "Category",
            "Species",
            "Strain",
            "wavelength [nm]",
            "k1 [cm2/mJ]",
            "k2 [cm2/mJ]",
            "% resistant",
            "Medium",
            "Condition",
            "Reference",
            "Link",
        ]

        # Build ordered list of columns to display
        display_cols = []

        # Add efficacy columns (CADR, eACH-UV) at the front if applicable
        # Show eACH-UV only when medium filter is exactly "Aerosol"
        show_each = self._medium == "Aerosol" and "eACH-UV" in df.columns

        # Show CADR only when CADR columns exist AND medium is "Aerosol"
        show_cadr = (self._medium == "Aerosol" and "CADR-UV [lps]" in df.columns)

        if show_cadr:
            # Use appropriate CADR unit based on stored unit preference
            if self._use_metric_units:
                display_cols.append("CADR-UV [lps]")
            else:
                display_cols.append("CADR-UV [cfm]")

        if show_each:
            display_cols.append("eACH-UV")

        # Add ONE time column if fluence was provided (based on _log level)
        if self._fluence is not None and self._time_cols and self._log in self._time_cols:
            # Use _select_time_display_columns to get the best unit for this log level
            time_col, _ = self._select_time_display_columns(
                df, self._time_cols, log_level=self._log
            )
            if time_col and time_col in df.columns:
                display_cols.append(time_col)

        # Add ALL base columns (always shown)
        for col in base_cols:
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
        for col in ["Medium", "Category", "wavelength [nm]"]:
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

    @property
    def display_df(self) -> pd.DataFrame:
        """Return filtered DataFrame with context-appropriate columns."""
        df = self._build_display_df()
        return df.sort_values("Species")

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
        return result.sort_values("Species")

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
        if "Category" not in df.columns:
            return None
        return sorted(df["Category"].unique())

    @property
    def mediums(self):
        df = self.display_df
        if "Medium" not in df.columns:
            return None
        return sorted(df["Medium"].unique())

    @property
    def wavelengths(self):
        df = self.display_df
        if "wavelength [nm]" not in df.columns:
            return None
        return sorted(df["wavelength [nm]"].unique())

    @classmethod
    def get_valid_categories(cls) -> list[str]:
        """Return list of valid category values without instantiating."""
        return sorted(cls.get_full()["Category"].unique())

    @classmethod
    def get_valid_mediums(cls) -> list[str]:
        """Return list of valid medium values without instantiating."""
        return sorted(cls.get_full()["Medium"].unique())

    @classmethod
    def get_valid_wavelengths(cls) -> list[float]:
        """Return list of valid wavelength values without instantiating."""
        return sorted(cls.get_full()["wavelength [nm]"].unique())

    @classmethod
    def with_room(
        cls,
        room: "Room",
        zone_id: str = "WholeRoomFluence",
    ) -> "Data":
        """
        Create Data instance from a Room's fluence values.

        Parameters
        ----------
        room : Room
            Room object with calculated fluence values.
        zone_id : str, optional
            Zone ID to get fluence from. Default is "WholeRoomFluence".

        Returns
        -------
        Data
            Data instance with room-based CADR calculations available.

        Notes
        -----
        Use .subset() to filter results:
            Data.with_room(room).subset(medium="Aerosol", category="Virus")
        """
        fluence_dict = room.fluence_dict(zone_id)
        if len(fluence_dict) == 0:
            msg = "Fluence not available; returning base table."
            warnings.warn(msg, stacklevel=2)
            return cls()

        # Create instance with fluence
        data = cls(fluence=fluence_dict)
        # Set unit preference based on room units (for CADR display)
        data._use_metric_units = room.dim.units in [LengthUnits.METERS, LengthUnits.CENTIMETERS]
        # Compute full_df with CADR columns (room used transiently, not stored)
        data._full_df = data._compute_all_columns(room=room)
        return data

    def plot(self, title=None, figsize=None, air_changes=None, mode="default", log=2, yscale="auto",
             left_axis=None, right_axis=None):
        """
        Plot inactivation data for all species as a violin and scatter plot.

        Parameters
        ----------
        title : str, optional
            Plot title. Auto-generated if not provided.
        figsize : tuple, optional
            Figure size (width, height). If None, calculated dynamically based on
            number of species with minimum (8, 5).
        air_changes : float, optional
            If provided, draws a horizontal line at this value.
        mode : str, optional
            "default" - shows CADR/eACH-UV (with room), eACH-UV (with fluence), or k1 (no fluence)
            "time" - if fluence is provided, shows time to inactivation instead
        log : int, optional
            Log reduction level for time mode: 1 (90%), 2 (99%), 3 (99.9%),
            4 (99.99%), or 5 (99.999%). Default is 2 (99% inactivation).
        yscale : str, optional
            Y-axis scale: "auto" (default), "linear", or "log".
            "auto" uses log scale if data spans >3 orders of magnitude.
        left_axis : str, optional
            For time mode: specify left y-axis unit ("seconds", "minutes", or "hours").
            If specified, automatically enables time mode. Automatic selection picks
            units to keep values readable (not in multi-thousands).
        right_axis : str or False, optional
            For time mode: specify right y-axis unit, or False to disable right axis.
            If specified, automatically enables time mode.
        """
        # Check if user specified time axis - auto-enable time mode
        time_axis_specified = left_axis is not None or right_axis is not None
        if time_axis_specified and mode == "default":
            mode = "time"

        # Build plotting df: need all time columns, don't apply _select_output_columns
        # (that's for user-facing output, not plotting)
        if isinstance(self._fluence, dict) and len(self._fluence) > 1:
            # Multi-wavelength: use combined data
            filtered = self._apply_row_filters(self._full_df)
            df = self._combine_wavelengths(filtered, self._fluence)
        else:
            # Single wavelength: apply row+wavelength filters, keep all columns
            df = self._apply_row_filters(self._full_df.copy())
            effective_wv = self._get_effective_wavelengths()
            if effective_wv is not None and "wavelength [nm]" in df.columns:
                if isinstance(effective_wv, (int, float)):
                    df = df[df["wavelength [nm]"] == effective_wv]
                elif isinstance(effective_wv, list):
                    df = df[df["wavelength [nm]"].isin(effective_wv)]
                elif isinstance(effective_wv, tuple) and len(effective_wv) == 2:
                    wv_min, wv_max = effective_wv
                    df = df[(df["wavelength [nm]"] >= wv_min) & (df["wavelength [nm]"] <= wv_max)]

        # Dynamic figsize based on number of species
        if figsize is None:
            n_species = df["Species"].nunique()
            width = max(8, n_species * 0.25)
            figsize = (width, 5)

        # Determine which columns are available
        eachkey = self._get_key("eACH")
        cadrkey = self._get_key("CADR")
        kkey = self._get_key("k1")

        # Flag for time mode (dual axis)
        use_time_mode = False
        log_level = log  # Store for title generation

        # Get time columns for the specified log level
        # Use row-filtered _full_df (not column-filtered df) since we need all time columns
        # to determine best unit via median calculation
        def get_time_labels(log_lvl):
            if hasattr(self, '_time_cols') and log_lvl in self._time_cols:
                time_df = self._apply_row_filters(self._full_df)
                return self._select_time_display_columns(
                    time_df, self._time_cols, log_level=log_lvl, left_axis=left_axis, right_axis=right_axis
                )
            return None, None

        # Determine what to plot based on mode and available data
        if mode == "time":
            left_label, right_label = get_time_labels(log)
            if left_label is None:
                raise ValueError(f"Time to inactivation (log{log}) not available. Provide fluence to calculate it.")
            use_time_mode = True
        else:
            # Default mode: eACH/CADR if available, otherwise k1, otherwise time
            if eachkey and cadrkey:
                left_label = eachkey
                right_label = cadrkey
            elif eachkey:
                left_label = eachkey
                right_label = None
            elif kkey:
                left_label = kkey
                right_label = None
            else:
                # Fall back to time if fluence was provided but eACH/k1 not available
                left_label, right_label = get_time_labels(log)
                if left_label is not None:
                    use_time_mode = True
                else:
                    raise ValueError("No plottable data available (need eACH-UV, k1, or time to inactivation)")

        # Determine hue (colors) and style (shapes) based on available columns
        has_medium = "Medium" in df.columns and len(df["Medium"].unique()) > 1
        has_category = "Category" in df.columns and len(df["Category"].unique()) > 1
        has_wavelength = "wavelength [nm]" in df.columns and len(df["wavelength [nm]"].unique()) > 1

        # Category ordering for grouping
        category_order = ["Bacteria", "Viruses", "Bacterial spores", "Fungi", "Protists"]
        medium_order = ["Aerosol", "Surface", "Liquid"]

        # Helper: map wavelength to consistent color (violet=200nm to red=310nm)
        # Custom rainbow colormap with good contrast on white backgrounds
        def wavelength_to_color(wv, wv_min=200, wv_max=310):
            from matplotlib.colors import LinearSegmentedColormap
            # Define colors: violet -> blue -> teal -> green -> orange -> red
            # All colors chosen for good visibility on white
            colors = [
                (0.5, 0.0, 0.8),    # violet (200nm)
                (0.2, 0.3, 0.9),    # blue (220nm)
                (0.0, 0.6, 0.7),    # teal (240nm)
                (0.1, 0.7, 0.3),    # green (260nm)
                (0.9, 0.5, 0.0),    # orange (280nm)
                (0.85, 0.1, 0.1),   # red (310nm)
            ]
            cmap = LinearSegmentedColormap.from_list("uv_rainbow", colors)
            norm = (wv - wv_min) / (wv_max - wv_min)
            norm = max(0, min(1, norm))  # Clamp to [0, 1]
            return cmap(norm)

        # Hue (colors): use for wavelength if multiple wavelengths present
        use_wavelength_colors = has_wavelength
        wv_col = None
        wv_order = None
        palette = None
        hue_col = None
        hue_order = None

        # Helper to format wavelength: 222.0 → "222", 194.1 → "194.1"
        def format_wv(wv):
            if wv == int(wv):
                return str(int(wv))
            return str(wv)

        if use_wavelength_colors:
            df = df.copy()
            unique_wvs = sorted(df["wavelength [nm]"].unique())
            n_unique = len(unique_wvs)

            if n_unique > 6:
                # Use binned ranges for legend, but color by actual wavelength
                df["wavelength range"] = df["wavelength [nm]"].apply(
                    lambda x: f"{int(x // 10 * 10)}-{int(x // 10 * 10 + 10)} nm"
                )
                wv_col = "wavelength range"
                wv_order = sorted(df[wv_col].unique(), key=lambda x: int(x.split("-")[0]))
                # Compute average wavelength per bucket for legend colors
                bucket_avg_wv = df.groupby("wavelength range")["wavelength [nm]"].mean().to_dict()
                palette = {bucket: wavelength_to_color(bucket_avg_wv[bucket]) for bucket in wv_order}
            else:
                # Use formatted wavelengths for cleaner legend display
                df["wavelength"] = df["wavelength [nm]"].apply(format_wv)
                wv_col = "wavelength"
                # Build order based on numeric sort, then map to formatted strings
                wv_order = [format_wv(wv) for wv in unique_wvs]
                palette = {format_wv(wv): wavelength_to_color(wv) for wv in unique_wvs}

            hue_col = wv_col
            hue_order = wv_order
        elif has_category:
            # Use category colors when wavelength colors aren't needed
            # (colors only, no legend needed since categories are labeled on plot)
            hue_col = "Category"
            hue_order = [cat for cat in category_order if cat in df["Category"].unique()]
            palette = None  # Use seaborn default category colors

        # Style (shapes): use for Medium if not filtered, OR use for wavelength if
        # single medium specified (colorblind-friendly: both color and shape for wavelength)
        if has_medium:
            style = "Medium"
            style_order = [m for m in medium_order if m in df["Medium"].unique()]
        elif use_wavelength_colors:
            # Single medium specified but multiple wavelengths - use shape for wavelength too
            style = wv_col
            style_order = wv_order
        else:
            style = None
            style_order = None

        # Category grouping: if category not filtered, sort by category and prepare labels
        if has_category:
            df = df.copy() if not use_wavelength_colors else df
            df["_cat_order"] = df["Category"].apply(
                lambda x: category_order.index(x) if x in category_order else 99
            )
            df = df.sort_values(["_cat_order", "Species"])

        # When no hue (single wavelength, no multiple wavelengths), use a consistent color
        if hue_col is None:
            default_palette = sns.color_palette()
            if self.category is not None and self.category in category_order:
                color = default_palette[category_order.index(self.category)]
            else:
                color = default_palette[0]
        else:
            color = None

        fig, ax1 = plt.subplots(figsize=figsize)

        # Build plot kwargs
        violin_kwargs = dict(
            data=df,
            x="Species",
            y=left_label,
            inner=None,
            ax=ax1,
            alpha=0.4,
            legend=False,
            )

        # Scatter plot: shows individual points colored by hue, shaped by style
        scatter_kwargs = dict(
            data=df,
            x="Species",
            y=left_label,
            ax=ax1,
            s=80,
            alpha=0.8,
        )

        if hue_col is not None:
            scatter_kwargs.update(hue=hue_col, hue_order=hue_order)
            if palette is not None:
                scatter_kwargs["palette"] = palette
        else:
            scatter_kwargs["color"] = color

        # Violin coloring: use gray for wavelength colors (scatter shows colors),
        # but apply category colors to violins when category is the hue
        if use_wavelength_colors:
            violin_kwargs["color"] = "lightgray"
        elif hue_col == "Category":
            violin_kwargs.update(hue=hue_col, hue_order=hue_order)
        else:
            violin_kwargs["color"] = color

        # Always add style (shapes) if available, regardless of hue
        if style is not None:
            scatter_kwargs.update(style=style, style_order=style_order)

        sns.violinplot(**violin_kwargs)
        sns.scatterplot(**scatter_kwargs)

        ax1.set_ylabel(left_label)
        ax1.set_xlabel(None)

        # Determine yscale if auto
        if yscale == "auto":
            y_data = df[left_label].dropna()
            y_min, y_max = y_data.min(), y_data.max()
            # Use log scale if data spans >3 orders of magnitude and min > 0
            if y_min > 0 and y_max / y_min > 1000:
                yscale = "log"
            else:
                yscale = "linear"

        ax1.set_yscale(yscale)
        if yscale == "linear":
            ax1.set_ylim(bottom=0)
        ax1.grid("--")
        ax1.set_xticks(ax1.get_xticks())
        ax1.set_xticklabels(
            ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )

        if right_label is not None:
            ax2 = ax1.twinx()
            ax2.set_ylabel(right_label)
            ax2.set_yscale(yscale)
            # Link right axis to left axis with appropriate conversion factor
            # (they're the same data in different units)
            left_data = df[left_label].dropna()
            right_data = df[right_label].dropna()
            if len(left_data) > 0 and len(right_data) > 0:
                # Calculate conversion factor from left to right units
                conversion = (right_data / left_data).median()
                left_min, left_max = ax1.get_ylim()
                ax2.set_ylim(bottom=left_min * conversion, top=left_max * conversion)

        # Add air changes line if provided and showing eACH
        if air_changes is not None and left_label == eachkey:
            yval = air_changes
            if yval < 0.1 * ax1.get_ylim()[1]:
                yval += 0.05 * ax1.get_ylim()[1]

            ax1.axhline(y=air_changes, color="red", linestyle="--", linewidth=1.5)
            ac = int(air_changes) if int(air_changes) == air_changes else round(air_changes, 2)
            string = f"{ac} air change\nfrom ventilation" if ac == 1 else f"{ac} air changes\nfrom ventilation"
            ax1.text(
                1.01,
                yval,
                string,
                color="red",
                va="center",
                ha="left",
                transform=ax1.get_yaxis_transform(),
            )

        # Set title - use ax.set_title for single line (no whitespace), suptitle for multi-line
        final_title = title or self._generate_title(left_label, right_label, use_time_mode, log_level)
        if "\n" in final_title:
            fig.suptitle(final_title)
        else:
            ax1.set_title(final_title)

        # Add category separators and labels when category is not filtered
        if has_category:
            # Get species order from x-axis
            species_order = [t.get_text() for t in ax1.get_xticklabels()]
            # Map species to category
            species_to_cat = df.groupby("Species")["Category"].first().to_dict()

            # Find category boundaries
            prev_cat = None
            boundaries = []
            for i, species in enumerate(species_order):
                cat = species_to_cat.get(species)
                if cat != prev_cat and prev_cat is not None:
                    boundaries.append((i - 0.5, cat))
                if i == 0:
                    boundaries.append((-0.5, cat))
                prev_cat = cat

            # Draw vertical lines
            for x_pos, cat in boundaries:
                if x_pos > -0.5:  # Don't draw line at the very start
                    ax1.axvline(x=x_pos, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)

            # Move species labels down to make room for category labels above
            ax1.tick_params(axis='x', pad=18)

            # Add category labels centered between plot and species labels
            fig.subplots_adjust(bottom=0.35)
            for i, (x_pos, cat) in enumerate(boundaries):
                # Find the end of this category section
                if i < len(boundaries) - 1:
                    next_x = boundaries[i + 1][0]
                else:
                    next_x = len(species_order) - 0.5
                mid_x = (x_pos + next_x) / 2
                ax1.text(mid_x, -0.06, cat, transform=ax1.get_xaxis_transform(),
                        ha="center", va="center", fontsize=12)

        # Position legend - default to right side
        # Don't show legend for category colors (already labeled on plot)
        show_legend = (wv_col is not None) or (style is not None and hue_col != "Category")
        has_right_axis = right_label is not None

        # Choose legend position based on where data is on the RIGHT side of the plot
        # (since legend goes on the right)
        species_order = df["Species"].unique()
        n_species = len(species_order)
        # Get data for the rightmost ~25% of species
        right_species = species_order[-(max(1, n_species // 4)):]
        right_data = df[df["Species"].isin(right_species)][left_label].dropna()

        y_min, y_max = ax1.get_ylim()
        if yscale == "log" and y_min > 0:
            y_midpoint = math.sqrt(y_min * y_max)  # Geometric mean (visual midpoint)
        else:
            y_midpoint = (y_min + y_max) / 2  # Arithmetic mean

        # Put legend opposite to where right-side data is concentrated
        right_median = right_data.median() if len(right_data) > 0 else y_midpoint
        inside_loc = "lower right" if right_median > y_midpoint else "upper right"

        if show_legend:
            handles, labels = ax1.get_legend_handles_labels()
            n_entries = len(labels)
            # If right axis exists and legend is small, put inside to avoid overlap
            if has_right_axis and n_entries <= 6:
                ax1.legend(loc=inside_loc, framealpha=0.9)
            else:
                # Default: put legend on right side outside plot
                fig.subplots_adjust(right=0.75)
                ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        elif style is not None:
            # Show legend for medium shapes only (no category colors in legend)
            handles, labels = ax1.get_legend_handles_labels()
            # Filter to only show style (Medium) entries, not hue (Category)
            medium_handles = []
            medium_labels = []
            for h, l in zip(handles, labels):
                if l in medium_order:
                    medium_handles.append(h)
                    medium_labels.append(l)
            if medium_handles:
                n_entries = len(medium_handles)
                # Put inside on right if small enough, otherwise outside
                if n_entries <= 6:
                    ax1.legend(medium_handles, medium_labels, loc=inside_loc, framealpha=0.9)
                else:
                    fig.subplots_adjust(right=0.85)
                    ax1.legend(medium_handles, medium_labels, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        elif hue_col == "Category":
            # Category colors but no legend needed
            ax1.legend().set_visible(False)

        return fig

    def _get_key(self, substring):
        df = self.display_df
        index = np.array([substring in key for key in df.keys()])
        return df.keys()[index][0] if sum(index) > 0 else None

    def _generate_title(self, left_label, right_label, use_time_mode=False, log_level=2):

        # Use generic "Time to X% inactivation" for time mode
        if use_time_mode:
            log_labels = {1: "90%", 2: "99%", 3: "99.9%", 4: "99.99%", 5: "99.999%"}
            title = f"Time to {log_labels.get(log_level, '99%')} inactivation"
        elif "k1" in left_label:
            # Check if no filters applied
            no_filters = (self.medium is None and self.category is None
                         and self.wavelength is None and self.fluence is None)
            if no_filters:
                title = "UVC susceptibility constants for all data"
            else:
                title = "UVC susceptibility constants"
        else:
            title = left_label
            if right_label is not None:
                title += "/" + right_label

        # Add wavelength context
        # Multi-wavelength fluence dict: use fluence keys (plot uses combined_df)
        is_multiwavelength = isinstance(self._fluence, dict) and len(self._fluence) > 1
        if is_multiwavelength:
            guv_types = ", ".join(["GUV-" + str(int(wv)) for wv in self._fluence.keys()])
            title += f" from {guv_types}"
        elif self.wavelength is not None:
            if isinstance(self.wavelength, (int, float)):
                if self.fluence is not None:
                    title += f" from GUV-{int(self.wavelength)}"
                else:
                    title += f" at {int(self.wavelength)} nm"
            elif isinstance(self.wavelength, list):
                if self.fluence is not None:
                    guv_str = ", ".join(f"GUV-{int(w)}" for w in self.wavelength)
                    title += f" from {guv_str}"
                else:
                    wv_str = ", ".join(str(int(w)) for w in self.wavelength)
                    title += f" at {wv_str} nm"
            elif isinstance(self.wavelength, tuple):
                if self.fluence is not None:
                    title += f" from GUV-{int(self.wavelength[0])}-{int(self.wavelength[1])}"
                else:
                    title += f" at {int(self.wavelength[0])}-{int(self.wavelength[1])} nm"

        # Add medium ("in Medium" or "on Surface") and/or category
        if self.medium is not None:
            if isinstance(self.medium, list):
                title += f" in {', '.join(self.medium)}"
            elif self.medium == "Surface":
                title += " on Surface"
            else:
                title += f" in {self.medium}"
        if self.category is not None:
            # Always use "for" with categories
            cat_str = ', '.join(self.category) if isinstance(self.category, list) else self.category
            title += f" for {cat_str}"

        if self.fluence is not None:
            if isinstance(self.fluence, dict):
                if len(self.fluence) > 1:
                    f = [round(val, 2) for val in self.fluence.values()]
                    title += f"\nwith average fluence rates: {f} uW/cm²"
                else:
                    # Single-wavelength dict - extract the value
                    val = list(self.fluence.values())[0]
                    title += f"\nwith average fluence rate {round(val, 2)} uW/cm²"
            else:
                title += f"\nwith average fluence rate {round(self.fluence, 2)} uW/cm²"
        return title


def _compute_row(row, fluence_arg, function, sigfigs=1, **kwargs):
    """
    fluence_arg may be an int, float, or dict of format {wavelength:fluence}
    """
    if isinstance(fluence_arg, dict):
        fluence = fluence_arg.get(row["wavelength [nm]"])
        if fluence is None:
            return None
    elif isinstance(fluence_arg, (float, int)):
        fluence = fluence_arg

    # Fill missing values and convert columns to the correct type
    k1 = row["k1 [cm2/mJ]"] if pd.notna(row["k1 [cm2/mJ]"]) else 0.0
    k2 = row["k2 [cm2/mJ]"] if pd.notna(row["k2 [cm2/mJ]"]) else 0.0
    f = (
        float(row["% resistant"].rstrip("%")) / 100
        if pd.notna(row["% resistant"])
        else 0.0
    )
    return round(function(irrad=fluence, k1=k1, k2=k2, f=f, **kwargs), sigfigs)


def _get_cadr(eACH, room, units=None):
    """
    Compute clean air delivery rate from room volume and eACH value.

    Parameters
    ----------
    eACH : float or Series
        Equivalent air changes per hour from UV.
    room : Room
        Room object with dimension information.
    units : str, optional
        Force output units: "lps" or "cfm". If None, uses room units.

    Returns
    -------
    tuple
        (CADR value(s), column key string)
    """
    # Get volumes (compute cfm from cubic_meters to avoid units.py bug with numpy arrays)
    cubic_meters = room.dim.cubic_meters
    cubic_feet = cubic_meters * 35.3147  # 1 cubic meter = 35.3147 cubic feet

    if units == "lps":
        return eACH * cubic_meters * 1000 / 60 / 60, "CADR-UV [lps]"
    elif units == "cfm":
        return eACH * cubic_feet / 60, "CADR-UV [cfm]"
    elif room.dim.units in [LengthUnits.METERS, LengthUnits.CENTIMETERS]:
        return eACH * cubic_meters * 1000 / 60 / 60, "CADR-UV [lps]"
    else:
        return eACH * cubic_feet / 60, "CADR-UV [cfm]"


def _filter_wavelengths(df, fluence_dict):
    """
    filter the dataframe only for species which have data for all of the
    wavelengths that are in fluence_dict
    """
    # remove any wavelengths from the dictionary that aren't in the dataframe
    wavelengths = df["wavelength [nm]"].unique()
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
    filtered_species = df.groupby("Species")["wavelength [nm]"].apply(
        lambda x: all(wavelength in x.values for wavelength in required_wavelengths)
    )
    # Filter the original DataFrame for the Species meeting the condition
    valid_species = filtered_species[filtered_species].index
    df = df[df["Species"].isin(valid_species)]
    df = df[df["wavelength [nm]"].isin(required_wavelengths)]
    return df, fluence_dict


def sum_multiwavelength_data(df, room=None):
    """for a dataframe that has more than one wavelength, sum the values to get
    the total eACH from UV"""
    # Group by Species
    summed_data = []
    for species, group in df.groupby("Species"):
        # Group by wavelength for the current species
        each_by_wavelength = (
            group.groupby("wavelength [nm]")["eACH-UV"].apply(list).to_dict()
        )

        # Generate all combinations of eACH-UV values across wavelengths
        each_combinations = product(*each_by_wavelength.values())

        # Sum combinations and preserve category information
        for comb in each_combinations:
            summed_data.append(
                {
                    "Species": species,
                    "Category": group["Category"].iloc[
                        0
                    ],  # Assume all rows for a species have the same category
                    "eACH-UV": sum(comb),
                }
            )
    df = pd.DataFrame(summed_data)
    if room is not None:
        cadr, cadr_key = _get_cadr(df["eACH-UV"], room)
        df[cadr_key] = cadr
    return df

def CADR_CFM(cubic_feet, irrad, k1, k2=0, f=0):
    return eACH_UV(irrad=irrad, k1=k1, k2=k2, f=f) * cubic_feet / 60


def CADR_LPS(cubic_meters, irrad, k1, k2=0, f=0):
    return eACH_UV(irrad=irrad, k1=k1, k2=k2, f=f) * cubic_meters * 1000 / 60 / 60


def eACH_UV(irrad, k1, k2=0, f=0):
    """
    Calculate equivalent air changes per hour from UV.

    For multi-wavelength: pass lists for irrad, k1, k2, f (same length).
    The eACH values are additive across wavelengths.
    """
    if isinstance(irrad, (list, tuple, np.ndarray)):
        return sum(eACH_UV(i, k, kk, ff) for i, k, kk, ff in zip(irrad, k1, k2, f))
    return (k1 * (1 - f) + k2 - k2 * (1 - f)) * irrad * 3.6


def log1(irrad, k1, k2=0, f=0, **kwargs):
    return seconds_to_S(0.1, irrad=irrad, k1=k1, k2=k2, f=f, **kwargs)


def log2(irrad, k1, k2=0, f=0, **kwargs):
    return seconds_to_S(0.01, irrad=irrad, k1=k1, k2=k2, f=f, **kwargs)


def log3(irrad, k1, k2=0, f=0, **kwargs):
    return seconds_to_S(0.001, irrad=irrad, k1=k1, k2=k2, f=f, **kwargs)


def log4(irrad, k1, k2=0, f=0, **kwargs):
    """Time to 99.99% inactivation (4-log reduction)."""
    return seconds_to_S(0.0001, irrad=irrad, k1=k1, k2=k2, f=f, **kwargs)


def log5(irrad, k1, k2=0, f=0, **kwargs):
    """Time to 99.999% inactivation (5-log reduction)."""
    return seconds_to_S(0.00001, irrad=irrad, k1=k1, k2=k2, f=f, **kwargs)


def seconds_to_S(S, irrad, k1, k2=0, f=0, tol=1e-10, max_iter=100):
    """
    Calculate time in seconds to reach survival fraction S.

    S: float, (0,1) - surviving fraction
    irrad: float or list - fluence/irradiance in uW/cm2
    k1: float or list - first susceptibility value, cm2/mJ
    k2: float or list - second susceptibility value, cm2/mJ
    f: float or list - (0,1) resistant fraction
    tol: float, numerical tolerance
    max_iter: maximum number of iterations to wait for solution to converge

    For multi-wavelength: pass lists for irrad, k1, k2, f (same length).
    The k*irrad values are summed, and f values are averaged.
    """
    # Handle multi-wavelength case
    if isinstance(irrad, (list, tuple, np.ndarray)):
        k1_irrad = sum(k * i for k, i in zip(k1, irrad))
        k2_irrad = sum(k * i for k, i in zip(k2, irrad))
        f_eff = sum(f) / len(f)
    else:
        k1_irrad = k1 * irrad
        k2_irrad = k2 * irrad
        f_eff = f

    def S_of_t(t):
        return (1 - f_eff) * math.exp(-k1_irrad / 1000 * t) + f_eff * math.exp(
            -k2_irrad * t
        )

    # Bracket the root
    t_low = 0.0
    t_high = 1.0
    while S_of_t(t_high) > S:
        t_high *= 2.0
    # Bisection
    for _ in range(max_iter):
        t_mid = 0.5 * (t_low + t_high)
        if S_of_t(t_mid) > S:
            t_low = t_mid
        else:
            t_high = t_mid
        if abs(t_high - t_low) < tol:
            break
    return 0.5 * (t_low + t_high)
