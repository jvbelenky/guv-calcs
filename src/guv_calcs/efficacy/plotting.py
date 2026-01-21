import warnings
import math
import matplotlib.pyplot as plt
import seaborn as sns

from .constants import (
    LOG_LABELS,
    CATEGORY_ORDER,
    MEDIUM_ORDER,
    COL_CATEGORY,
    COL_SPECIES,
    COL_WAVELENGTH,
    COL_MEDIUM,
    COL_K1,
    COL_EACH,
    COL_CADR_LPS,
    COL_CADR_CFM,
)
from .utils import (
    auto_select_time_columns,
    wavelength_to_color,
    format_wavelength,
    parse_axis_input,
    get_compatible_group,
    is_time_column,
)


# =============================================================================
# Main plot function
# =============================================================================

def plot(data, title=None, figsize=None, air_changes=None, mode="default", log=2,
         yscale="auto", left_axis=None, right_axis=None, time_units=None):
    """
    Plot inactivation data for all species as a violin and scatter plot.

    Parameters
    ----------
    data : Data
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

    # Dynamic figsize based on number of species
    if figsize is None:
        n_species = df[COL_SPECIES].nunique()
        width = max(8, n_species * 0.25)
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

    hue_col = config['hue_col']
    hue_order = config['hue_order']
    palette = config['palette']
    color = config['color']
    style = config['style']
    style_order = config['style_order']
    use_wavelength_colors = config['use_wavelength_colors']
    wv_col = config['wv_col']

    # Check if category is not filtered (for separators)
    has_category = COL_CATEGORY in df.columns and len(df[COL_CATEGORY].unique()) > 1

    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)

    # Build violin plot kwargs
    violin_kwargs = dict(
        data=df,
        x=COL_SPECIES,
        y=left_label,
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
        x=COL_SPECIES,
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

    # Set title
    final_title = title or _generate_title(data, left_label, right_label, use_time_mode, effective_log)
    if "\n" in final_title:
        fig.suptitle(final_title)
    else:
        ax1.set_title(final_title)

    # Add category separators if category is not filtered
    if has_category:
        species_order = [t.get_text() for t in ax1.get_xticklabels()]
        _add_category_separators(ax1, fig, df, species_order)

    # Position legend
    show_legend = (wv_col is not None) or (style is not None and hue_col != COL_CATEGORY)
    has_right_axis = right_label is not None
    _position_legend(ax1, fig, df, left_label, yscale, show_legend, has_right_axis,
                     wv_col, style, hue_col)

    return fig


# =============================================================================
# Data organizing utilities
# =============================================================================

def _build_plot_df(data):
    """Build DataFrame for plotting (uses full DFs, not display DFs)."""
    if data.combined_full_df is not None:
        # Use full combined df with all columns, apply row filters
        return data._apply_row_filters(data.combined_full_df.copy())
    # Use full_df with all columns, apply row and wavelength filters
    df = data._apply_row_filters(data._full_df.copy())
    return data._apply_wavelength_filter(df)


def _determine_plot_columns(data, df, mode, log, left_axis, right_axis, time_units=None):
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
            raise ValueError(f"Invalid time_units '{time_units}'. Must be 'seconds', 'minutes', or 'hours'.")
        # Normalize to standard names
        unit_map = {"sec": "seconds", "s": "seconds", "min": "minutes", "m": "minutes", "hr": "hours", "h": "hours"}
        time_units = unit_map.get(time_units_lower, time_units_lower)

        # If left_axis specifies a log level, use that; otherwise use the log parameter
        left_parsed, left_log = parse_axis_input(left_axis, time_cols, use_metric, log)
        effective_log = left_log or log

        # Build the time column name directly
        if effective_log in time_cols and time_units in time_cols[effective_log]:
            time_col = time_cols[effective_log][time_units]
            if time_col in df.columns and df[time_col].notna().any():
                return time_col, None, True, effective_log

        warnings.warn(f"Time column for {time_units} at log{effective_log} not available.", stacklevel=3)

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
    # eACH and CADR only make sense for Aerosol medium
    is_aerosol = data._medium == "Aerosol"
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
                    stacklevel=3
                )
                left_parsed = None

        # Validate right axis if specified
        if right_parsed is not None:
            if right_parsed not in df.columns or not df[right_parsed].notna().any():
                warnings.warn(
                    f"Right axis column '{right_parsed}' not available.",
                    stacklevel=3
                )
                right_parsed = None

        # Check compatibility if both axes specified
        if left_parsed is not None and right_parsed is not None:
            left_group = get_compatible_group(left_parsed, time_cols)
            if right_parsed not in left_group:
                warnings.warn(
                    f"Columns '{left_parsed}' and '{right_parsed}' are not linearly "
                    f"related and cannot be co-plotted. Showing only left axis.",
                    stacklevel=3
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
        warnings.warn("Time mode requested but fluence not provided. Showing k1.", stacklevel=3)
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

    raise ValueError("No plottable data available (need eACH-UV, k1, or time to inactivation)")


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
    has_wavelength = COL_WAVELENGTH in df.columns and len(df[COL_WAVELENGTH].unique()) > 1

    config = {
        'hue_col': None,
        'hue_order': None,
        'palette': None,
        'color': None,
        'style': None,
        'style_order': None,
        'use_wavelength_colors': has_wavelength,
        'wv_col': None,
        'wv_order': None,
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
            bucket_avg_wv = df.groupby("wavelength range")[COL_WAVELENGTH].mean().to_dict()
            palette = {bucket: wavelength_to_color(bucket_avg_wv[bucket]) for bucket in wv_order}
        else:
            # Use formatted wavelengths for cleaner legend display
            df["wavelength"] = df[COL_WAVELENGTH].apply(format_wavelength)
            wv_col = "wavelength"
            # Build order based on numeric sort, then map to formatted strings
            wv_order = [format_wavelength(wv) for wv in unique_wvs]
            palette = {format_wavelength(wv): wavelength_to_color(wv) for wv in unique_wvs}

        config['hue_col'] = wv_col
        config['hue_order'] = wv_order
        config['palette'] = palette
        config['wv_col'] = wv_col
        config['wv_order'] = wv_order

    elif has_category:
        # Use category colors when wavelength colors aren't needed
        config['hue_col'] = COL_CATEGORY
        config['hue_order'] = [cat for cat in CATEGORY_ORDER if cat in df[COL_CATEGORY].unique()]
        # palette = None uses seaborn default category colors

    # Style (shapes): use for Medium if not filtered, OR use for wavelength if
    # single medium specified (colorblind-friendly: both color and shape for wavelength)
    if has_medium:
        config['style'] = COL_MEDIUM
        config['style_order'] = [m for m in MEDIUM_ORDER if m in df[COL_MEDIUM].unique()]
    elif has_wavelength:
        # Single medium specified but multiple wavelengths - use shape for wavelength too
        config['style'] = config['wv_col']
        config['style_order'] = config['wv_order']

    # Category grouping: if category not filtered, sort by category
    if has_category:
        df["_cat_order"] = df[COL_CATEGORY].apply(
            lambda x: CATEGORY_ORDER.index(x) if x in CATEGORY_ORDER else 99
        )
        df.sort_values(["_cat_order", COL_SPECIES], inplace=True)

    # When no hue (single wavelength, no multiple wavelengths), use a consistent color
    if config['hue_col'] is None:
        default_palette = sns.color_palette()
        if data.category is not None and data.category in CATEGORY_ORDER:
            config['color'] = default_palette[CATEGORY_ORDER.index(data.category)]
        else:
            config['color'] = default_palette[0]

    return config


def _generate_title(data, left_label, right_label, use_time_mode, log_level):
    """Generate plot title based on data state and plot mode."""
    # Use generic "Time to X% inactivation" for time mode
    if use_time_mode:
        title = f"Time to {LOG_LABELS.get(log_level, '99%')} inactivation"
    elif "k1" in left_label:
        # Check if no filters applied
        no_filters = (data.medium is None and data.category is None
                     and data.wavelength is None and data.fluence is None)
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
    is_multiwavelength = isinstance(data._fluence, dict) and len(data._fluence) > 1
    if is_multiwavelength:
        guv_types = ", ".join(["GUV-" + str(int(wv)) for wv in data._fluence.keys()])
        title += f" from {guv_types}"
    elif data.wavelength is not None:
        if isinstance(data.wavelength, (int, float)):
            if data.fluence is not None:
                title += f" from GUV-{int(data.wavelength)}"
            else:
                title += f" at {int(data.wavelength)} nm"
        elif isinstance(data.wavelength, list):
            if data.fluence is not None:
                guv_str = ", ".join(f"GUV-{int(w)}" for w in data.wavelength)
                title += f" from {guv_str}"
            else:
                wv_str = ", ".join(str(int(w)) for w in data.wavelength)
                title += f" at {wv_str} nm"
        elif isinstance(data.wavelength, tuple):
            if data.fluence is not None:
                title += f" from GUV-{int(data.wavelength[0])}-{int(data.wavelength[1])}"
            else:
                title += f" at {int(data.wavelength[0])}-{int(data.wavelength[1])} nm"

    # Add medium ("in Medium" or "on Surface") and/or category
    if data.medium is not None:
        if isinstance(data.medium, list):
            title += f" in {', '.join(data.medium)}"
        elif data.medium == "Surface":
            title += " on Surface"
        else:
            title += f" in {data.medium}"
    if data.category is not None:
        # Always use "for" with categories
        cat_str = ', '.join(data.category) if isinstance(data.category, list) else data.category
        title += f" for {cat_str}"

    if data.fluence is not None:
        # Use "irradiance" for Surface, "average fluence rate" for Aerosol/Liquid
        rate_term = "irradiance" if data.medium == "Surface" else "average fluence rate"
        if isinstance(data.fluence, dict):
            if len(data.fluence) > 1:
                f = [round(val, 2) for val in data.fluence.values()]
                title += f"\nwith {rate_term}s: {f} µW/cm²"
            else:
                # Single-wavelength dict - extract the value
                val = list(data.fluence.values())[0]
                title += f"\nwith {rate_term} {round(val, 2)} µW/cm²"
        else:
            title += f"\nwith {rate_term} {round(data.fluence, 2)} µW/cm²"
    return title


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
    ax.tick_params(axis='x', pad=18)

    # Add category labels centered between plot and species labels
    fig.subplots_adjust(bottom=0.35)
    for i, (x_pos, cat) in enumerate(boundaries):
        # Find the end of this category section
        if i < len(boundaries) - 1:
            next_x = boundaries[i + 1][0]
        else:
            next_x = len(species_order) - 0.5
        mid_x = (x_pos + next_x) / 2
        ax.text(mid_x, -0.06, cat, transform=ax.get_xaxis_transform(),
                ha="center", va="center", fontsize=12)


def _position_legend(ax, fig, df, left_label, yscale, show_legend, has_right_axis,
                     wv_col, style, hue_col):
    """Position legend based on data distribution."""
    # Get species order from x-axis
    species_order = df[COL_SPECIES].unique()
    n_species = len(species_order)

    # Get data for the rightmost ~25% of species
    right_species = species_order[-(max(1, n_species // 4)):]
    right_data = df[df[COL_SPECIES].isin(right_species)][left_label].dropna()

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
                ax.legend(medium_handles, medium_labels, bbox_to_anchor=(1.02, 1),
                          loc="upper left", borderaxespad=0)
    elif hue_col == COL_CATEGORY:
        # Category colors but no legend needed
        ax.legend().set_visible(False)
