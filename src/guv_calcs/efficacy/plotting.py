import warnings
import math
import numbers
import re
import textwrap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from .constants import (
    LOG_LABELS,
    CATEGORY_ORDER,
    MEDIUM_ORDER,
    COL_CATEGORY,
    COL_SPECIES,
    COL_STRAIN,
    COL_CONDITION,
    COL_WAVELENGTH,
    COL_MEDIUM,
    COL_K1,
    COL_K2,
    COL_RESISTANT,
    COL_EACH,
    COL_CADR_LPS,
    COL_CADR_CFM,
    TIME_UNIT_ALIASES,
    TIME_THRESHOLD_SECONDS,
    TIME_THRESHOLD_MINUTES,
)
from .utils import (
    auto_select_time_columns,
    wavelength_to_color,
    format_wavelength,
    parse_axis_input,
    get_compatible_group,
    is_time_column,
)
from .math import seconds_to_S, survival_fraction
from ._kinetics import parse_resistant

__all__ = ["plot_swarm", "plot_survival", "plot_wavelength"]


# =============================================================================
# Main plot function
# =============================================================================


def plot_swarm(
    data,
    title=None,
    figsize=None,
    air_changes=None,
    mode="default",
    log=2,
    yscale="auto",
    left_axis=None,
    right_axis=None,
    time_units=None,
):
    """
    Plot inactivation data for all species as a violin and scatter plot.

    Parameters
    ----------
    data : InactivationData
        Data instance to plot.
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
    time_units : str, optional
        Time unit for display: "seconds", "minutes", or "hours".
        If specified, enables time mode and shows the specified unit on left axis.
        Can be combined with `log` parameter to control which log level.
    """
    # Build plotting DataFrame
    df = _build_plot_df(data)

    # Determine x-axis column based on data granularity
    # Hierarchy: Species → Strain → Condition
    x_col, n_groups = _determine_x_axis(df)

    # Dynamic figsize based on number of groups
    if figsize is None:
        width = max(8, n_groups * 0.25)
        figsize = (width, 5)

    # Determine columns to plot
    left_label, right_label, use_time_mode, effective_log = _determine_plot_columns(
        data, df, mode, log, left_axis, right_axis, time_units
    )

    # Filter out rows with NaN in the plotted column
    df = df[df[left_label].notna()]

    # Configure hue and style (modifies df in place for wavelength columns)
    df = df.copy()  # Avoid modifying original
    config = _configure_hue_style(df, data)

    hue_col = config["hue_col"]
    hue_order = config["hue_order"]
    palette = config["palette"]
    color = config["color"]
    style = config["style"]
    style_order = config["style_order"]
    use_wavelength_colors = config["use_wavelength_colors"]
    wv_col = config["wv_col"]

    # Check if category is not filtered (for separators)
    has_category = COL_CATEGORY in df.columns and len(df[COL_CATEGORY].unique()) > 1

    # Sort x-axis alphabetically
    x_order = sorted(df[x_col].dropna().unique())
    df = df.sort_values(x_col)

    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)

    # Build violin plot kwargs
    violin_kwargs = dict(
        data=df,
        x=x_col,
        y=left_label,
        order=x_order,
        inner=None,
        ax=ax1,
        alpha=0.4,
        legend=False,
    )

    # Violin coloring: use gray for wavelength colors (scatter shows colors),
    # but apply category colors to violins when category is the hue
    if use_wavelength_colors:
        violin_kwargs["color"] = "lightgray"
    elif hue_col == COL_CATEGORY:
        violin_kwargs.update(hue=hue_col, hue_order=hue_order)
    else:
        violin_kwargs["color"] = color

    # Build scatter plot kwargs
    scatter_kwargs = dict(
        data=df,
        x=x_col,
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

    # Always add style (shapes) if available, regardless of hue
    if style is not None:
        scatter_kwargs.update(style=style, style_order=style_order)

    # Create plots
    sns.violinplot(**violin_kwargs)
    sns.scatterplot(**scatter_kwargs)

    # Configure axes
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

    # Configure right axis if needed
    if right_label is not None:
        ax2 = ax1.twinx()
        ax2.set_ylabel(right_label)
        ax2.set_yscale(yscale)
        # Link right axis to left axis with appropriate conversion factor
        left_data = df[left_label].dropna()
        right_data = df[right_label].dropna()
        if len(left_data) > 0 and len(right_data) > 0:
            # Calculate conversion factor from left to right units
            conversion = (right_data / left_data).median()
            left_min, left_max = ax1.get_ylim()
            ax2.set_ylim(bottom=left_min * conversion, top=left_max * conversion)

    # Add air changes line if provided and showing eACH
    if air_changes is not None and "eACH" in left_label:
        yval = air_changes
        if yval < 0.1 * ax1.get_ylim()[1]:
            yval += 0.05 * ax1.get_ylim()[1]

        ax1.axhline(y=air_changes, color="red", linestyle="--", linewidth=1.5)
        ac = (
            int(air_changes)
            if int(air_changes) == air_changes
            else round(air_changes, 2)
        )
        string = (
            f"{ac} air change\nfrom ventilation"
            if ac == 1
            else f"{ac} air changes\nfrom ventilation"
        )
        ax1.text(
            1.01,
            yval,
            string,
            color="red",
            va="center",
            ha="left",
            transform=ax1.get_yaxis_transform(),
        )

    # Add category separators if showing species and category is not filtered
    if has_category and x_col == COL_SPECIES:
        species_order = [t.get_text() for t in ax1.get_xticklabels()]
        _add_category_separators(ax1, fig, df, species_order)

    # Position legend (may adjust axes via subplots_adjust)
    show_legend = (wv_col is not None) or (
        style is not None and hue_col != COL_CATEGORY
    )
    has_right_axis = right_label is not None
    _position_legend(
        ax1,
        fig,
        df,
        x_col,
        left_label,
        yscale,
        show_legend,
        has_right_axis,
        wv_col,
        style,
        hue_col,
    )

    # Set title AFTER legend positioning so axes position is finalized
    final_title = title or _generate_title(
        data, left_label, right_label, use_time_mode, effective_log
    )
    final_title = _wrap_title(final_title, fig)
    # Position title closer to plot if single line; leave room if wrapped
    num_lines = final_title.count("\n") + 1
    title_y = 0.93 if num_lines == 1 else 0.98
    # Center title on plot area, not figure (which includes legend)
    ax_pos = ax1.get_position()
    title_x = ax_pos.x0 + ax_pos.width / 2
    fig.suptitle(final_title, x=title_x, y=title_y, ha="center")

    return fig


# =============================================================================
# Data organizing utilities
# =============================================================================


def _build_plot_df(data):
    """Build DataFrame for plotting (uses full DFs, not display DFs)."""
    if data.combined_full_df is not None:
        return data._get_filtered_df(df=data.combined_full_df.copy())
    return data._get_filtered_df()


def _determine_x_axis(df):
    """
    Determine the x-axis column based on data granularity.

    Returns (x_col, n_groups) where x_col is the column to use and n_groups
    is the number of unique values in that column.

    Hierarchy: Species → Strain → Condition
    - Multiple species: use Species
    - Single species, multiple strains: use Strain
    - Single species, single strain, multiple conditions: use Condition
    """
    n_species = df[COL_SPECIES].nunique()

    if n_species > 1:
        return COL_SPECIES, n_species

    # Single species - try strain
    if COL_STRAIN in df.columns:
        n_strains = df[COL_STRAIN].nunique()
        if n_strains > 1:
            return COL_STRAIN, n_strains

    # Single species and strain - try condition
    if COL_CONDITION in df.columns:
        n_conditions = df[COL_CONDITION].nunique()
        if n_conditions > 1:
            return COL_CONDITION, n_conditions

    # Fall back to species even if single
    return COL_SPECIES, n_species


def _determine_plot_columns(
    data, df, mode, log, left_axis, right_axis, time_units=None
):
    """
    Determine which columns to plot based on mode and available data.

    Returns (left_label, right_label, use_time_mode, effective_log).

    Supports forgiving user input parsing:
    - "each", "eACH", "EACH" -> COL_EACH
    - "cadr", "CADR" -> COL_CADR_LPS or COL_CADR_CFM based on use_metric_units
    - "k1", "K1" -> COL_K1
    - "log1", "90%", "0.9", 0.9 -> time column for log1
    - "log2", "99%", "0.99" -> time column for log2
    - etc.
    - "seconds", "minutes", "hours" -> time column in that unit

    Priority for default mode: eACH+CADR > eACH > k1 > time.
    Time mode requires fluence; falls back to k1 with warning if unavailable.
    """
    time_cols = data._time_cols
    use_metric = data._use_metric_units

    # Handle time_units parameter - enables time mode with specific unit
    if time_units is not None:
        time_units_lower = time_units.lower()
        valid_units = {"seconds", "minutes", "hours", "sec", "min", "hr", "s", "m", "h"}
        if time_units_lower not in valid_units:
            raise ValueError(
                f"Invalid time_units '{time_units}'. Must be 'seconds', 'minutes', or 'hours'."
            )
        # Normalize to standard names
        unit_map = {
            "sec": "seconds",
            "s": "seconds",
            "min": "minutes",
            "m": "minutes",
            "hr": "hours",
            "h": "hours",
        }
        time_units = unit_map.get(time_units_lower, time_units_lower)

        # If left_axis specifies a log level, use that; otherwise use the log parameter
        left_parsed, left_log = parse_axis_input(left_axis, time_cols, use_metric, log)
        effective_log = left_log or log

        # Build the time column name directly
        if effective_log in time_cols and time_units in time_cols[effective_log]:
            time_col = time_cols[effective_log][time_units]
            if time_col in df.columns and df[time_col].notna().any():
                return time_col, None, True, effective_log

        warnings.warn(
            f"Time column for {time_units} at log{effective_log} not available.",
            stacklevel=3,
        )

    # Parse user-specified axes
    left_parsed, left_log = parse_axis_input(left_axis, time_cols, use_metric, log)
    right_parsed, right_log = parse_axis_input(right_axis, time_cols, use_metric, log)

    # If user specified a log level through axis input, use it
    effective_log = left_log or right_log or log

    # Auto-enable time mode if user specified time-related axis
    if left_parsed is not None and is_time_column(left_parsed, time_cols):
        mode = "time"
    elif right_parsed is not None and is_time_column(right_parsed, time_cols):
        mode = "time"

    # Check what columns are available
    # eACH and CADR only make sense when data is exclusively Aerosol
    is_aerosol = COL_MEDIUM in df.columns and len(df) > 0 and (df[COL_MEDIUM] == "Aerosol").all()
    has_each = is_aerosol and COL_EACH in df.columns and df[COL_EACH].notna().any()
    cadr_col = COL_CADR_LPS if use_metric else COL_CADR_CFM
    has_cadr = is_aerosol and cadr_col in df.columns and df[cadr_col].notna().any()
    has_k1 = COL_K1 in df.columns and df[COL_K1].notna().any()
    has_time = data._fluence is not None and time_cols

    # If user specified either axis, use what they specified
    if left_parsed is not None or right_parsed is not None:
        # Validate left axis if specified
        if left_parsed is not None:
            if left_parsed not in df.columns or not df[left_parsed].notna().any():
                warnings.warn(
                    f"Requested column '{left_parsed}' not available in data. Using defaults.",
                    stacklevel=3,
                )
                left_parsed = None

        # Validate right axis if specified
        if right_parsed is not None:
            if right_parsed not in df.columns or not df[right_parsed].notna().any():
                warnings.warn(
                    f"Right axis column '{right_parsed}' not available.", stacklevel=3
                )
                right_parsed = None

        # Check compatibility if both axes specified
        if left_parsed is not None and right_parsed is not None:
            left_group = get_compatible_group(left_parsed, time_cols)
            if right_parsed not in left_group:
                warnings.warn(
                    f"Columns '{left_parsed}' and '{right_parsed}' are not linearly "
                    f"related and cannot be co-plotted. Showing only left axis.",
                    stacklevel=3,
                )
                right_parsed = None

        # If left axis specified, use it (with optional right)
        if left_parsed is not None:
            use_time = is_time_column(left_parsed, time_cols)
            return left_parsed, right_parsed, use_time, effective_log

        # If only right axis specified, use it as left (no auto-fill)
        if right_parsed is not None:
            use_time = is_time_column(right_parsed, time_cols)
            return right_parsed, None, use_time, effective_log

    # Time mode (explicit)
    if mode == "time":
        if has_time:
            left, right = auto_select_time_columns(df, time_cols, effective_log)
            return left, right, True, effective_log
        warnings.warn(
            "Time mode requested but fluence not provided. Showing k1.", stacklevel=3
        )
        return COL_K1, None, False, effective_log

    # Default mode: prefer eACH/CADR > eACH > k1 > time
    if has_each and has_cadr:
        return COL_EACH, cadr_col, False, effective_log
    if has_each:
        return COL_EACH, None, False, effective_log
    if has_k1:
        return COL_K1, None, False, effective_log
    if has_time:
        left, right = auto_select_time_columns(df, time_cols, effective_log)
        return left, right, True, effective_log

    raise ValueError(
        "No plottable data available (need eACH-UV, k1, or time to inactivation)"
    )


# =============================================================================
# Plotting utilities
# =============================================================================


def _configure_hue_style(df, data):
    """
    Configure hue (colors) and style (shapes) for plotting.

    Returns
    -------
    dict
        Configuration with keys: hue_col, hue_order, palette, color,
        style, style_order, use_wavelength_colors, wv_col, wv_order
    """
    has_medium = COL_MEDIUM in df.columns and len(df[COL_MEDIUM].unique()) > 1
    has_category = COL_CATEGORY in df.columns and len(df[COL_CATEGORY].unique()) > 1
    has_wavelength = (
        COL_WAVELENGTH in df.columns and len(df[COL_WAVELENGTH].unique()) > 1
    )

    config = {
        "hue_col": None,
        "hue_order": None,
        "palette": None,
        "color": None,
        "style": None,
        "style_order": None,
        "use_wavelength_colors": has_wavelength,
        "wv_col": None,
        "wv_order": None,
    }

    if has_wavelength:
        unique_wvs = sorted(df[COL_WAVELENGTH].unique())
        n_unique = len(unique_wvs)

        if n_unique > 6:
            # Use binned ranges for legend, but color by actual wavelength
            df["wavelength range"] = df[COL_WAVELENGTH].apply(
                lambda x: f"{int(x // 10 * 10)}-{int(x // 10 * 10 + 10)} nm"
            )
            wv_col = "wavelength range"
            wv_order = sorted(df[wv_col].unique(), key=lambda x: int(x.split("-")[0]))
            # Compute average wavelength per bucket for legend colors
            bucket_avg_wv = (
                df.groupby("wavelength range")[COL_WAVELENGTH].mean().to_dict()
            )
            palette = {
                bucket: wavelength_to_color(bucket_avg_wv[bucket])
                for bucket in wv_order
            }
        else:
            # Use formatted wavelengths for cleaner legend display
            df["wavelength"] = df[COL_WAVELENGTH].apply(format_wavelength)
            wv_col = "wavelength"
            # Build order based on numeric sort, then map to formatted strings
            wv_order = [format_wavelength(wv) for wv in unique_wvs]
            palette = {
                format_wavelength(wv): wavelength_to_color(wv) for wv in unique_wvs
            }

        config["hue_col"] = wv_col
        config["hue_order"] = wv_order
        config["palette"] = palette
        config["wv_col"] = wv_col
        config["wv_order"] = wv_order

    elif has_category:
        # Use category colors when wavelength colors aren't needed
        config["hue_col"] = COL_CATEGORY
        config["hue_order"] = [
            cat for cat in CATEGORY_ORDER if cat in df[COL_CATEGORY].unique()
        ]
        # palette = None uses seaborn default category colors

    # Style (shapes): use for Medium if not filtered, OR use for wavelength if
    # single medium specified (colorblind-friendly: both color and shape for wavelength)
    if has_medium:
        config["style"] = COL_MEDIUM
        config["style_order"] = [
            m for m in MEDIUM_ORDER if m in df[COL_MEDIUM].unique()
        ]
    elif has_wavelength:
        # Single medium specified but multiple wavelengths - use shape for wavelength too
        config["style"] = config["wv_col"]
        config["style_order"] = config["wv_order"]

    # Category grouping: if category not filtered, sort by category
    if has_category:
        df["_cat_order"] = df[COL_CATEGORY].apply(
            lambda x: CATEGORY_ORDER.index(x) if x in CATEGORY_ORDER else 99
        )
        df.sort_values(["_cat_order", COL_SPECIES], inplace=True)

    # When no hue (single wavelength, no multiple wavelengths), use a consistent color
    if config["hue_col"] is None:
        default_palette = sns.color_palette()
        if data.category is not None and data.category in CATEGORY_ORDER:
            config["color"] = default_palette[CATEGORY_ORDER.index(data.category)]
        else:
            config["color"] = default_palette[0]

    return config


def _generate_title(data, left_label, right_label, use_time_mode, log_level):
    """Generate plot title based on data state and plot mode."""
    # Determine stem based on mode
    if use_time_mode:
        stem = f"Time to {LOG_LABELS.get(log_level, '99%')} inactivation"
    elif "k1" in left_label:
        no_filters = (
            data.medium is None
            and data.category is None
            and data.wavelength is None
            and data.fluence is None
        )
        stem = (
            "UVC susceptibility constants for all data"
            if no_filters
            else "UVC susceptibility constants"
        )
    else:
        stem = left_label
        if right_label is not None:
            stem += "/" + right_label

    # Determine fluence value (extract from dict if needed)
    fluence = None
    if data.fluence is not None:
        if isinstance(data.fluence, dict):
            if len(data.fluence) == 1:
                fluence = list(data.fluence.values())[0]
            else:
                fluence = sum(data.fluence.values())  # Total for display
        else:
            fluence = data.fluence

    # Get actual values from data for title
    # Show value(s) if filtered (fewer than all possible values)
    all_species = data.get_full()[COL_SPECIES].unique()
    all_categories = data.get_full()[COL_CATEGORY].unique()

    species_list = data.species or []
    species_val = species_list[0] if len(species_list) == 1 else None

    categories = data.categories or []
    # Only show category if it's been filtered down
    category_val = categories[0] if len(categories) == 1 and len(categories) < len(all_categories) else None

    strains = data.strains or []
    strain_val = strains[0] if len(strains) == 1 else None

    return _build_title(
        data,
        stem,
        fluence=fluence,
        fluence_dict=data._fluence if isinstance(data._fluence, dict) else None,
        category=category_val,
        species=species_val,
        strain=strain_val,
    )


def _get_wavelength_str(data, fluence_dict=None):
    """
    Get wavelength string like 'GUV-222' or 'GUV-222, GUV-254'.

    Parameters
    ----------
    data : InactivationData
        Data instance for wavelength info.
    fluence_dict : dict, optional
        Multi-wavelength fluence dict (for wavelength extraction).

    Returns
    -------
    str
        Formatted wavelength string, or empty string if no wavelength info.
    """
    # Multi-wavelength from explicit dict
    if fluence_dict and len(fluence_dict) > 1:
        return ", ".join(f"GUV-{int(wv)}" for wv in fluence_dict.keys())

    # Multi-wavelength from data
    if isinstance(data._fluence, dict) and len(data._fluence) > 1:
        return ", ".join(f"GUV-{int(wv)}" for wv in data._fluence.keys())

    # Single wavelength
    if data.wavelength is not None:
        if isinstance(data.wavelength, numbers.Real):
            return f"GUV-{int(data.wavelength)}"
        elif isinstance(data.wavelength, list) and len(data.wavelength) == 1:
            return f"GUV-{int(data.wavelength[0])}"

    return ""


def _build_title(
    data,
    stem,
    fluence=None,
    fluence_dict=None,
    suffix="",
    fluence_label="at",
    category=None,
    species=None,
    strain=None,
):
    """
    Build plot title with pattern:
    - Single wavelength: {stem} {medium} by {wavelength} {fluence_label} {fluence} for {filters} {suffix}
    - Multi-wavelength: {stem} {medium} by {wavelengths}\\nwith {rate_term} {fluences} for {filters} {suffix}
    """
    parts = [stem]

    # Medium: "in Aerosol" or "on Surface" - use actual data values
    mediums = data.mediums
    if mediums and len(mediums) == 1:
        medium = mediums[0]
        if medium == "Surface":
            parts.append("on Surface")
        else:
            parts.append(f"in {medium}")

    # Check if multi-wavelength
    is_multi_wavelength = (fluence_dict and len(fluence_dict) > 1) or (
        isinstance(data._fluence, dict) and len(data._fluence) > 1
    )

    # Wavelength: "by GUV-222" or "by GUV-222, GUV-254"
    wv_str = _get_wavelength_str(data, fluence_dict)
    if wv_str:
        parts.append(f"by {wv_str}")

    # Fluence handling
    if is_multi_wavelength:
        # Multi-wavelength: newline then "with fluence rates [5, 3] µW/cm²"
        fd = fluence_dict if fluence_dict else data._fluence
        fluence_values = [round(v, 2) for v in fd.values()]
        is_surface = mediums and len(mediums) == 1 and mediums[0] == "Surface"
        rate_term = "irradiances" if is_surface else "fluence rates"
        # Join first part, then add newline, then fluence + rest
        first_line = " ".join(parts)
        second_line_parts = [f"with {rate_term} {fluence_values} µW/cm²"]
        filter_str = _build_filter_str(category, species, strain)
        if filter_str:
            second_line_parts.append(f"for {filter_str}")
        if suffix:
            second_line_parts.append(suffix)
        return first_line + "\n" + " ".join(second_line_parts)
    else:
        # Single wavelength
        if fluence is not None:
            parts.append(f"{fluence_label} {round(fluence, 2)} µW/cm²")

        # Filters: "for Bacteria", "for E. coli", etc.
        filter_str = _build_filter_str(category, species, strain)
        if filter_str:
            parts.append(f"for {filter_str}")

        # Suffix: "(95% CI)"
        if suffix:
            parts.append(suffix)

        return " ".join(parts)


def _build_filter_str(category=None, species=None, strain=None):
    """Build filter description string for title."""
    parts = []
    if species is not None:
        parts.append(species)
    elif category is not None:
        cat_str = ", ".join(category) if isinstance(category, list) else category
        parts.append(cat_str)
    if strain is not None:
        parts.append(f"({strain})")
    return " ".join(parts)


def _wrap_title(title, fig, chars_per_inch=12):
    """
    Wrap title text to fit the figure width.

    Parameters
    ----------
    title : str
        The title string to wrap.
    fig : matplotlib.figure.Figure
        The figure to fit the title to.
    chars_per_inch : float, optional
        Approximate characters per inch at default font size. Default is 12.

    Returns
    -------
    str
        Title with newlines inserted for wrapping.
    """
    fig_width = fig.get_figwidth()
    wrap_width = int(fig_width * chars_per_inch)

    # Split on existing newlines first, then wrap each line separately
    result_lines = []
    for line in title.split("\n"):
        # Replace spaces within parentheses/brackets with non-breaking spaces to keep units together
        protected = re.sub(
            r"[\(\[]([^\)\]]+)[\)\]]",
            lambda m: m.group(0)[0] + m.group(1).replace(" ", "\xa0") + m.group(0)[-1],
            line,
        )

        # Wrap the line
        wrapped = textwrap.wrap(protected, width=wrap_width, break_on_hyphens=False)

        # Restore regular spaces and add to result
        result_lines.extend(w.replace("\xa0", " ") for w in wrapped)

    return "\n".join(result_lines)


def _add_category_separators(ax, fig, df, species_order):
    """Add vertical lines and category labels between categories on the plot."""
    # Map species to category
    species_to_cat = df.groupby(COL_SPECIES)[COL_CATEGORY].first().to_dict()

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
            ax.axvline(x=x_pos, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)

    # Move species labels down to make room for category labels above
    ax.tick_params(axis="x", pad=18)

    # Add category labels centered between plot and species labels
    fig.subplots_adjust(bottom=0.35)
    for i, (x_pos, cat) in enumerate(boundaries):
        # Find the end of this category section
        if i < len(boundaries) - 1:
            next_x = boundaries[i + 1][0]
        else:
            next_x = len(species_order) - 0.5
        mid_x = (x_pos + next_x) / 2
        ax.text(
            mid_x,
            -0.06,
            cat,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=12,
        )


def _position_legend(
    ax, fig, df, x_col, left_label, yscale, show_legend, has_right_axis, wv_col, style, hue_col
):
    """Position legend based on data distribution."""
    # Get x-axis group order
    x_order = df[x_col].unique()
    n_groups = len(x_order)

    # Get data for the rightmost ~25% of groups
    right_groups = x_order[-(max(1, n_groups // 4)) :]
    right_data = df[df[x_col].isin(right_groups)][left_label].dropna()

    y_min, y_max = ax.get_ylim()
    if yscale == "log" and y_min > 0:
        y_midpoint = math.sqrt(y_min * y_max)  # Geometric mean (visual midpoint)
    else:
        y_midpoint = (y_min + y_max) / 2  # Arithmetic mean

    # Put legend opposite to where right-side data is concentrated
    right_median = right_data.median() if len(right_data) > 0 else y_midpoint
    inside_loc = "lower right" if right_median > y_midpoint else "upper right"

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        n_entries = len(labels)
        # If right axis exists and legend is small, put inside to avoid overlap
        if has_right_axis and n_entries <= 6:
            ax.legend(loc=inside_loc, framealpha=0.9)
        else:
            # Default: put legend on right side outside plot
            fig.subplots_adjust(right=0.75)
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    elif style is not None:
        # Show legend for medium shapes only (no category colors in legend)
        handles, labels = ax.get_legend_handles_labels()
        # Filter to only show style (Medium) entries, not hue (Category)
        medium_handles = []
        medium_labels = []
        for h, l in zip(handles, labels):
            if l in MEDIUM_ORDER:
                medium_handles.append(h)
                medium_labels.append(l)
        if medium_handles:
            n_entries = len(medium_handles)
            # Put inside on right if small enough, otherwise outside
            if n_entries <= 6:
                ax.legend(medium_handles, medium_labels, loc=inside_loc, framealpha=0.9)
            else:
                fig.subplots_adjust(right=0.85)
                ax.legend(
                    medium_handles,
                    medium_labels,
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                    borderaxespad=0,
                )
    elif hue_col == COL_CATEGORY:
        # Category colors but no legend needed
        ax.legend().set_visible(False)


# =============================================================================
# Survival fraction plot
# =============================================================================


def plot_survival(
    data,
    fluence=None,
    species=None,
    labels=None,
    title=None,
    time_units=None,
    xrange=None,
    yrange=(0, 1),
    figsize=(6.4, 4.8),
    show_ci=None,
):
    """
    Plot survival fraction over time with optional 95% CI bands.

    Parameters
    ----------
    data : InactivationData
        Data instance to plot from.
    fluence : float or list of float, optional
        Irradiance value(s) in µW/cm². If not provided, uses data's fluence.
        Provide a list for multiple curves at different irradiances.
    species : str or list of str, optional
        Species name(s) to plot. If None and fluence is a single value, plots
        all available species. Required when fluence is a list.
    labels : list of str, optional
        Custom labels for legend. If not provided, uses species names or
        fluence values.
    title : str, optional
        Custom plot title. If not provided, auto-generates based on mode.
    time_units : str, optional
        Time unit for x-axis: 'seconds', 'minutes', or 'hours'.
        If not provided, auto-selects based on data range.
    xrange : tuple, optional
        (min, max) tuple for x-axis limits.
    yrange : tuple, optional
        (min, max) tuple for y-axis limits. Default is (0, 1).
    figsize : tuple, optional
        Figure size (width, height). Default is (6.4, 4.8).
    show_ci : bool, optional
        Whether to show 95% CI bands. If None (default), auto-determined based
        on number of species: enabled for ≤4 species, disabled for >4.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.

    Notes
    -----
    Two modes are supported:
    - Single species + multiple irradiances: one curve per irradiance
    - Single irradiance + multiple species: one curve per species

    The survival model uses the full two-phase equation:
        S(t) = (1-f)*exp(-k1*I/1000*t) + f*exp(-k2*I/1000*t)

    95% CI bands are calculated using k1 ± 1.96*SEM propagated through
    the survival formula.
    """
    # Get fluence from data if not provided
    if fluence is None:
        if data._fluence is None:
            raise ValueError(
                "fluence must be provided either to InactivationData() or to plot_survival()"
            )
        fluence = data._fluence

    # Determine if multi-wavelength mode (dict fluence with multiple wavelengths)
    multi_wavelength = isinstance(fluence, dict) and len(fluence) > 1

    # Handle dict fluence - extract single value or keep as dict for multi-wavelength
    if isinstance(fluence, dict):
        if len(fluence) == 1:
            fluence = list(fluence.values())[0]
        # else: keep as dict for multi-wavelength processing

    # For multi-wavelength, fluence_list is used differently
    if multi_wavelength:
        fluence_dict = fluence
        wavelengths = list(fluence_dict.keys())
        fluence_list = [sum(fluence_dict.values())]  # Total fluence for display
        multi_fluence = False
    else:
        fluence_dict = None
        wavelengths = None
        # Normalize fluence to list
        fluence_list = [fluence] if isinstance(fluence, numbers.Real) else list(fluence)
        multi_fluence = len(fluence_list) > 1

    # Validate: species is required when fluence is a list (multiple single-wavelength fluences)
    if multi_fluence and species is None:
        raise ValueError(
            "species must be specified when fluence is a list. "
            "Provide a single species name to plot multiple irradiance curves."
        )

    # Build the plot dataframe
    # For multi-wavelength, use combined_full_df which pre-filters to species with data at all wavelengths
    if multi_wavelength:
        if data.combined_full_df is not None:
            combined_df = data._get_filtered_df(df=data.combined_full_df.copy())
        else:
            combined_df = None
        # Also get full df for k-value extraction per wavelength
        df = data._get_filtered_df()
    else:
        df = _build_plot_df(data)
        combined_df = None

    # Normalize species to list
    if species is None:
        if multi_wavelength and combined_df is not None:
            # Use species from combined_df (already filtered to those with data at all wavelengths)
            species_list = list(combined_df[COL_SPECIES].unique())
        else:
            species_list = list(df[COL_SPECIES].unique())
        if len(species_list) == 0:
            raise ValueError("No species found in filtered data")
    elif isinstance(species, str):
        species_list = [species]
    else:
        species_list = list(species)

    multi_species = len(species_list) > 1

    if multi_fluence and multi_species:
        raise ValueError(
            "Cannot plot multiple species with multiple fluence values. "
            "Use either a single species with multiple fluences, or multiple species with a single fluence."
        )

    # Filter to requested species
    df = df[df[COL_SPECIES].isin(species_list)]
    if len(df) == 0:
        raise ValueError(f"No data found for species: {species_list}")

    # Get k1, k2, f values for each species
    species_data = {}
    skipped_species = []

    for sp in species_list:
        sp_df = df[df[COL_SPECIES] == sp]

        if multi_wavelength:
            # Multi-wavelength: get k values per wavelength
            irrad_list = []
            k1_list = []
            k2_list = []
            f_list = []
            k1_sem_list = []

            missing_wv = False
            for wv in wavelengths:
                wv_df = sp_df[sp_df[COL_WAVELENGTH] == wv]
                k1_vals = wv_df[COL_K1].dropna().values

                if len(k1_vals) == 0:
                    missing_wv = True
                    break

                k2_vals = (
                    wv_df[COL_K2].fillna(0).values
                    if COL_K2 in wv_df.columns
                    else np.zeros(len(k1_vals))
                )
                f_vals = (
                    wv_df[COL_RESISTANT].apply(parse_resistant).values
                    if COL_RESISTANT in wv_df.columns
                    else np.zeros(len(k1_vals))
                )

                irrad_list.append(fluence_dict[wv])
                k1_list.append(np.mean(k1_vals))
                k2_list.append(np.mean(k2_vals))
                f_list.append(np.mean(f_vals))
                k1_sem_list.append(
                    np.std(k1_vals, ddof=1) / np.sqrt(len(k1_vals))
                    if len(k1_vals) > 1
                    else 0
                )

            if missing_wv:
                skipped_species.append(sp)
                warnings.warn(
                    f"Species '{sp}' missing data for one or more wavelengths, skipping.",
                    stacklevel=2,
                )
                continue

            species_data[sp] = {
                "irrad_list": irrad_list,
                "k1_list": k1_list,
                "k2_list": k2_list,
                "f_list": f_list,
                "k1_sem_list": k1_sem_list,
                "multi_wavelength": True,
            }
        else:
            # Single wavelength: get mean k values
            k1_vals = sp_df[COL_K1].dropna().values
            k2_vals = (
                sp_df[COL_K2].fillna(0).values
                if COL_K2 in sp_df.columns
                else np.zeros(len(k1_vals))
            )
            f_vals = (
                sp_df[COL_RESISTANT].apply(parse_resistant).values
                if COL_RESISTANT in sp_df.columns
                else np.zeros(len(k1_vals))
            )

            if len(k1_vals) == 0:
                skipped_species.append(sp)
                warnings.warn(
                    f"No k1 values found for species '{sp}', skipping.", stacklevel=2
                )
                continue

            species_data[sp] = {
                "k1_mean": np.mean(k1_vals),
                "k1_sem": np.std(k1_vals, ddof=1) / np.sqrt(len(k1_vals))
                if len(k1_vals) > 1
                else 0,
                "k2_mean": np.mean(k2_vals),
                "f_mean": np.mean(f_vals),
                "multi_wavelength": False,
            }

    # Update species_list to only include those with data
    species_list = [sp for sp in species_list if sp in species_data]

    if len(species_list) == 0:
        raise ValueError("No species with valid k1 data found.")

    # Auto-determine show_ci if not explicitly set
    if show_ci is None:
        show_ci = len(species_list) <= 4

    # Determine number of curves and labels
    if multi_fluence:
        n_curves = len(fluence_list)
        if labels is None:
            labels = [f"{f} µW/cm²" for f in fluence_list]
    else:
        n_curves = len(species_list)
        if labels is None:
            labels = species_list.copy()

    if labels is not None and len(labels) != n_curves:
        raise ValueError(
            f"Number of labels ({len(labels)}) must match number of curves ({n_curves})"
        )

    # Calculate time range for plotting
    # For single mode, extend to 99.9% to show all inactivation lines
    # For multi mode, use 99% as before
    single_mode = len(species_list) == 1 and not multi_fluence
    target_survival = 0.001 if single_mode else 0.01

    max_time = 0
    for sp in species_list:
        sd = species_data[sp]
        if sd["multi_wavelength"]:
            t_target = seconds_to_S(
                target_survival,
                sd["irrad_list"],
                sd["k1_list"],
                sd["k2_list"],
                sd["f_list"],
            )
            max_time = max(max_time, t_target)
        else:
            for f in fluence_list:
                t_target = seconds_to_S(
                    target_survival, f, sd["k1_mean"], sd["k2_mean"], sd["f_mean"]
                )
                max_time = max(max_time, t_target)

    # Auto-select time units based on max_time (in seconds)
    if time_units is None:
        if max_time < TIME_THRESHOLD_SECONDS:
            time_units = "seconds"
        elif max_time < TIME_THRESHOLD_MINUTES:
            time_units = "minutes"
        else:
            time_units = "hours"

    # Normalize time units
    time_units = TIME_UNIT_ALIASES.get(time_units.lower(), time_units.lower())

    # Time divisor for conversion
    time_divisors = {"seconds": 1, "minutes": 60, "hours": 3600}
    time_div = time_divisors.get(time_units, 1)

    # Create time array
    if xrange is not None:
        t_min, t_max = xrange[0] * time_div, xrange[1] * time_div
    else:
        t_min, t_max = 0, max_time * 1.1  # Add 10% padding

    t_seconds = np.linspace(t_min, t_max, 500)
    t_display = t_seconds / time_div

    fig, ax = plt.subplots(figsize=figsize)

    # Color palette - use tab20 for more colors, supports up to 20 curves
    colors = plt.cm.tab20.colors

    # Plot each curve
    if multi_fluence:
        # Single species, multiple fluences
        sp = species_list[0]
        sd = species_data[sp]
        for i, f in enumerate(fluence_list):
            label = labels[i] if labels else f"{f} µW/cm²"
            color = colors[i % len(colors)]
            _plot_survival_with_ci(
                ax, t_seconds, t_display, f, sd, show_ci, label, color
            )
    elif single_mode:
        # Single species, single fluence - special mode with inactivation lines
        sp = species_list[0]
        sd = species_data[sp]
        f = fluence_list[0]
        _plot_survival_with_ci(
            ax, t_seconds, t_display, f, sd, show_ci, label=None, color="black"
        )

        # Calculate times to log reductions and add vertical lines
        log_colors = {"90%": "blue", "99%": "green", "99.9%": "orange"}
        log_targets = {"90%": 0.1, "99%": 0.01, "99.9%": 0.001}

        for log_label, target in log_targets.items():
            if sd.get("multi_wavelength", False):
                t_log = seconds_to_S(
                    target, sd["irrad_list"], sd["k1_list"], sd["k2_list"], sd["f_list"]
                )
            else:
                t_log = seconds_to_S(
                    target, f, sd["k1_mean"], sd["k2_mean"], sd["f_mean"]
                )
            t_log_display = t_log / time_div
            # Only draw line if it's within the plot range
            if t_log_display <= t_display[-1]:
                ax.axvline(
                    x=t_log_display,
                    label=f"{log_label} Inactivation",
                    linestyle="--",
                    linewidth=1,
                    color=log_colors[log_label],
                )
    else:
        # Multiple species, single fluence
        f = fluence_list[0]
        for i, sp in enumerate(species_list):
            sd = species_data[sp]
            label = labels[i] if labels else sp
            color = colors[i % len(colors)]
            _plot_survival_with_ci(
                ax, t_seconds, t_display, f, sd, show_ci, label, color
            )

    # Set axis labels
    ax.set_xlabel(f"Time ({time_units})")
    ax.set_ylabel("Survival fraction")

    # Set axis limits with padding
    if yrange is not None:
        ax.set_ylim(yrange)
    if xrange is not None:
        ax.set_xlim(xrange)
    else:
        # Add padding to x-axis so data doesn't touch edges
        ax.set_xlim(t_display[0], t_display[-1])

    # Add margins so data doesn't touch plot edges
    ax.margins(x=0.02, y=0.02)

    ax.grid(True, linestyle="--", alpha=0.7)

    # Position legend - inside plot for single mode, outside for multi
    if single_mode:
        ax.legend(loc="best")
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    # Generate title if not provided
    # Set title (wrap to figure width)
    if title is None:
        title = _generate_survival_title(
            data,
            species_list,
            fluence_list,
            multi_fluence,
            multi_wavelength,
            fluence_dict if multi_wavelength else None,
            show_ci=show_ci,
        )
    title = _wrap_title(title, fig)
    ax.set_title(title)

    # Adjust layout to make room for legend
    fig.tight_layout()
    if not single_mode:
        fig.subplots_adjust(right=0.75)

    return fig


def _plot_survival_with_ci(
    ax, t_seconds, t_display, fluence, species_data, show_ci, label, color
):
    """
    Plot a single survival curve with optional 95% CI band.

    Handles both single-wavelength and multi-wavelength species_data.
    """
    sd = species_data

    if sd.get("multi_wavelength", False):
        # Multi-wavelength mode
        S_mean = survival_fraction(
            t_seconds, sd["irrad_list"], sd["k1_list"], sd["k2_list"], sd["f_list"]
        )

        # CI for multi-wavelength: vary each k1 by its SEM
        if show_ci and any(sem > 0 for sem in sd["k1_sem_list"]):
            k1_lo = [k - 1.96 * sem for k, sem in zip(sd["k1_list"], sd["k1_sem_list"])]
            k1_hi = [k + 1.96 * sem for k, sem in zip(sd["k1_list"], sd["k1_sem_list"])]
            S_lo = survival_fraction(
                t_seconds, sd["irrad_list"], k1_hi, sd["k2_list"], sd["f_list"]
            )
            S_hi = survival_fraction(
                t_seconds, sd["irrad_list"], k1_lo, sd["k2_list"], sd["f_list"]
            )
            ax.fill_between(t_display, S_lo, S_hi, alpha=0.2, color=color)
    else:
        # Single-wavelength mode
        S_mean = survival_fraction(
            t_seconds, fluence, sd["k1_mean"], sd["k2_mean"], sd["f_mean"]
        )

        if show_ci and sd["k1_sem"] > 0:
            k1_lo = sd["k1_mean"] - 1.96 * sd["k1_sem"]
            k1_hi = sd["k1_mean"] + 1.96 * sd["k1_sem"]
            S_lo = survival_fraction(
                t_seconds, fluence, k1_hi, sd["k2_mean"], sd["f_mean"]
            )
            S_hi = survival_fraction(
                t_seconds, fluence, k1_lo, sd["k2_mean"], sd["f_mean"]
            )
            ax.fill_between(t_display, S_lo, S_hi, alpha=0.2, color=color)

    ax.plot(t_display, S_mean, label=label, color=color, linewidth=2)


def _generate_survival_title(
    data,
    species_list,
    fluence_list,
    multi_fluence,
    multi_wavelength=False,
    fluence_dict=None,
    show_ci=True,
):
    """Generate auto title for survival plot."""
    single_mode = len(species_list) == 1 and not multi_fluence

    # Determine the stem based on mode
    if single_mode or multi_fluence:
        species = species_list[0]
        stem = f"Estimated {species} reduction"
    else:
        stem = "Estimated reduction"

    # For multi-wavelength, show total fluence
    if multi_wavelength and fluence_dict:
        total_fluence = sum(fluence_dict.values())
    else:
        total_fluence = fluence_list[0]

    # multi_fluence mode: no fluence in title (multiple curves show different fluences)
    fluence_value = None if multi_fluence else total_fluence

    # Only include CI suffix if CI bands are shown
    suffix = "(95% CI)" if show_ci else ""

    return _build_title(
        data, stem, fluence=fluence_value, fluence_dict=fluence_dict, suffix=suffix
    )


# =============================================================================
# Wavelength vs K-value plot
# =============================================================================


def plot_wavelength(
    data,
    y="k1",
    title=None,
    figsize=(8, 5),
    show_fit=True,
    fit_type="smooth",
    smoothing=None,
    yscale="linear",
):
    """
    Plot wavelength vs k-values (susceptibility constants).

    Parameters
    ----------
    data : InactivationData
        Data instance to plot.
    y : str, optional
        Which k-value to plot: "k1" (default) or "k2".
    title : str, optional
        Plot title. Auto-generated if not provided.
    figsize : tuple, optional
        Figure size (width, height). Default is (8, 5).
    show_fit : bool, optional
        Whether to show best-fit line. Default is True.
    fit_type : str, optional
        Type of fit: "smooth" (default) or "linear".
    smoothing : float, optional
        Smoothing factor. Higher values = smoother curve. Auto-determined if None.
    yscale : str, optional
        Y-axis scale: "linear" (default) or "log".

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.

    Notes
    -----
    Marker colors and shapes are determined dynamically:
    - Multiple mediums: color by medium
    - Multiple categories: color by category
    - Multiple species: color by species (if few), shape by species (if many)
    - Single filter applied: single color, shape varies by remaining dimension
    """
    # Build plotting DataFrame
    df = data._get_filtered_df()

    # Determine y column
    y_col = COL_K1 if y.lower() == "k1" else COL_K2 if y.lower() == "k2" else y
    if y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' not found in data")

    # Filter out rows with NaN in wavelength or y column
    df = df[[COL_WAVELENGTH, y_col, COL_SPECIES, COL_MEDIUM, COL_CATEGORY]].dropna(
        subset=[COL_WAVELENGTH, y_col]
    )

    if len(df) == 0:
        raise ValueError("No data available for plotting")

    # Determine hue and style based on data dimensions
    hue_col, style_col, palette = _determine_wavelength_plot_style(df, data)

    fig, ax = plt.subplots(figsize=figsize)

    # Build scatter kwargs
    scatter_kwargs = dict(
        data=df,
        x=COL_WAVELENGTH,
        y=y_col,
        ax=ax,
        s=80,
        alpha=0.7,
    )

    if hue_col is not None:
        scatter_kwargs["hue"] = hue_col
        if palette is not None:
            scatter_kwargs["palette"] = palette

    if style_col is not None:
        scatter_kwargs["style"] = style_col

    sns.scatterplot(**scatter_kwargs)

    # Add best-fit lines if requested (one per group)
    if show_fit and len(df) > 3:
        _add_grouped_fit_lines(ax, df, COL_WAVELENGTH, y_col, hue_col, fit_type, smoothing)

    # Configure axes
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(y_col)
    ax.set_yscale(yscale)
    ax.grid(True, linestyle="--", alpha=0.7)

    # Position legend
    if hue_col is not None or style_col is not None:
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) <= 10:
            ax.legend(loc="best")
        else:
            fig.subplots_adjust(right=0.75)
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    # Set title
    if title is None:
        title = _generate_wavelength_title(data, y_col)
    title = _wrap_title(title, fig)
    ax.set_title(title)

    fig.tight_layout()

    return fig


def _determine_wavelength_plot_style(df, data):
    """
    Determine hue and style columns for wavelength plot.

    Returns (hue_col, style_col, palette).
    """
    n_mediums = df[COL_MEDIUM].nunique()
    n_categories = df[COL_CATEGORY].nunique()
    n_species = df[COL_SPECIES].nunique()

    # Priority: medium > category > species for hue
    # Use style for secondary dimension

    if n_mediums > 1:
        # Color by medium, shape by category if multiple
        hue_col = COL_MEDIUM
        style_col = COL_CATEGORY if n_categories > 1 else None
        palette = None  # Use seaborn defaults
    elif n_categories > 1:
        # Color by category, shape by species if many
        hue_col = COL_CATEGORY
        style_col = COL_SPECIES if n_species > 1 and n_species <= 6 else None
        palette = None
    elif n_species > 1:
        # Single medium and category, multiple species
        if n_species <= 10:
            hue_col = COL_SPECIES
            style_col = None
            palette = None
        else:
            # Too many species for colors, use shapes
            hue_col = None
            style_col = COL_SPECIES
            palette = None
    else:
        # Single everything
        hue_col = None
        style_col = None
        palette = None

    return hue_col, style_col, palette


def _add_grouped_fit_lines(ax, df, x_col, y_col, hue_col, fit_type="smooth", smoothing=None):
    """Add best-fit lines for each group in the data."""
    from scipy.interpolate import UnivariateSpline

    # Get the color palette used by seaborn
    if hue_col is not None:
        groups = df[hue_col].unique()
        palette = sns.color_palette()
        color_map = {group: palette[i % len(palette)] for i, group in enumerate(groups)}
    else:
        groups = [None]
        color_map = {None: "red"}

    for group in groups:
        if group is not None:
            group_df = df[df[hue_col] == group]
            color = color_map[group]
        else:
            group_df = df
            color = "red"

        x = group_df[x_col].values
        y = group_df[y_col].values

        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 4:
            continue

        # Sort by x and aggregate duplicate x values (average y)
        sort_idx = np.argsort(x_clean)
        x_sorted = x_clean[sort_idx]
        y_sorted = y_clean[sort_idx]

        # Aggregate duplicate x values by averaging y
        unique_x, inverse = np.unique(x_sorted, return_inverse=True)
        avg_y = np.array([y_sorted[inverse == i].mean() for i in range(len(unique_x))])

        x_line = np.linspace(unique_x.min(), unique_x.max(), 100)

        if fit_type == "linear":
            coeffs = np.polyfit(unique_x, avg_y, 1)
            poly = np.poly1d(coeffs)
            y_line = poly(x_line)
        else:
            # Try multiple smoothing values and pick the best valid one
            y_line = _find_best_spline_fit(unique_x, avg_y, x_line, smoothing)

        # Plot fit line (no label, just matching color)
        ax.plot(x_line, y_line, "-", linewidth=2, alpha=0.7, color=color)


def _find_best_spline_fit(x, y, x_line, smoothing=None):
    """Find a smooth fit that follows the data points naturally."""
    from scipy.interpolate import PchipInterpolator
    from scipy.signal import savgol_filter

    if len(x) < 2:
        return np.full_like(x_line, np.mean(y))

    if len(x) == 2:
        coeffs = np.polyfit(x, y, 1)
        return np.poly1d(coeffs)(x_line)

    try:
        # First interpolate with PCHIP to get a curve through the points
        pchip = PchipInterpolator(x, y)
        y_line = pchip(x_line)

        # Then apply Savitzky-Golay filter to smooth out spikiness
        # Window must be odd and less than data length
        window = min(31, len(y_line) // 3)
        if window % 2 == 0:
            window += 1
        window = max(5, window)

        y_smooth = savgol_filter(y_line, window, polyorder=3)

        # Clip negatives
        return np.maximum(y_smooth, 0)
    except Exception:
        # Fallback to simple polynomial
        coeffs = np.polyfit(x, y, min(3, len(x) - 1))
        return np.maximum(np.poly1d(coeffs)(x_line), 0)


def _generate_wavelength_title(data, y_col):
    """Generate title for wavelength plot."""
    stem = f"{y_col} vs Wavelength"

    # Get actual filter values from data
    mediums = data.mediums
    categories = data.categories
    species_list = data.species

    parts = [stem]

    # Add filter info
    if mediums and len(mediums) == 1:
        medium = mediums[0]
        if medium == "Surface":
            parts.append("on Surface")
        else:
            parts.append(f"in {medium}")

    if categories and len(categories) == 1:
        parts.append(f"for {categories[0]}")
    elif species_list and len(species_list) == 1:
        parts.append(f"for {species_list[0]}")

    return " ".join(parts)
