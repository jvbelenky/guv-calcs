from matplotlib.colors import LinearSegmentedColormap
from .constants import (
    COL_CADR_LPS,
    COL_CADR_CFM,
    AXIS_ALIASES,
    LOG_ALIASES,
    TIME_UNIT_ALIASES,
    RATE_COLS,
)


def auto_select_time_columns(df, time_cols, log):
    """
    Auto-select the best time columns based on data range.

    Returns (primary_col, secondary_col) where secondary may be None.

    Selection logic:
    - If median < 100 seconds: seconds only
    - If median < 6000 seconds: minutes primary, seconds secondary
    - Otherwise: hours primary, minutes secondary
    """
    if log not in time_cols:
        return None, None

    cols = time_cols[log]
    sec_key = cols["seconds"]
    min_key = cols["minutes"]
    hr_key = cols["hours"]

    if len(df) == 0 or sec_key not in df.columns:
        return min_key, sec_key

    median_seconds = df[sec_key].median()

    if median_seconds < 100:
        return sec_key, None
    elif median_seconds < 6000:
        return min_key, sec_key
    else:
        return hr_key, min_key


def wavelength_to_color(wv, wv_min=200, wv_max=310):
    """
    Map wavelength to color on UV rainbow scale.

    Custom rainbow colormap with good contrast on white backgrounds.
    Violet (200nm) -> Blue -> Teal -> Green -> Orange -> Red (310nm)
    """
    colors = [
        (0.5, 0.0, 0.8),  # violet (200nm)
        (0.2, 0.3, 0.9),  # blue (220nm)
        (0.0, 0.6, 0.7),  # teal (240nm)
        (0.1, 0.7, 0.3),  # green (260nm)
        (0.9, 0.5, 0.0),  # orange (280nm)
        (0.85, 0.1, 0.1),  # red (310nm)
    ]
    cmap = LinearSegmentedColormap.from_list("uv_rainbow", colors)
    norm = (wv - wv_min) / (wv_max - wv_min)
    norm = max(0, min(1, norm))
    return cmap(norm)


def format_wavelength(wv):
    """Format wavelength: 222.0 -> '222', 194.1 -> '194.1'."""
    if wv == int(wv):
        return str(int(wv))
    return str(wv)


def parse_axis_input(value, time_cols, use_metric_units=True, log_level=None):
    """
    Parse user-friendly axis input to actual column name.

    value : str, int, float, or None
        User input like "each", "log3", "99.9%", 0.999, etc.
    time_cols : dict
        Dict mapping log level -> {unit: column_name}.
    use_metric_units : bool
        For CADR, use lps (True) or cfm (False).
    log_level : int, optional
        Default log level for time unit specification (e.g., "minutes" uses this log level).

    Returns:
    tuple
        (column_name, resolved_log_level) or (None, None) if not found.
        resolved_log_level is set when the input specified a log level.
    """
    if value is None:
        return None, None

    resolved_log = None

    # Handle numeric input (survival fraction like 0.9, 0.99, etc. or percent like 99.9)
    if isinstance(value, (int, float)):
        # Could be survival fraction (0.9, 0.99) or percent without % (99, 99.9)
        if value < 1:
            # Survival fraction: 0.9 -> log1, 0.99 -> log2, etc.
            resolved_log = _survival_fraction_to_log(value)
        elif value >= 90:
            # Percent like 99.9 -> convert to survival fraction first
            survival = value / 100
            resolved_log = _survival_fraction_to_log(survival)

        if resolved_log and resolved_log in time_cols:
            # Auto-select best time unit
            return _auto_select_time_unit(time_cols, resolved_log), resolved_log
        return None, None

    # Handle string input
    value_str = str(value).lower().strip().rstrip("%")

    # Direct column aliases (each, k1, k2)
    if value_str in AXIS_ALIASES:
        return AXIS_ALIASES[value_str], None

    # CADR special case (depends on units)
    if value_str == "cadr":
        return COL_CADR_LPS if use_metric_units else COL_CADR_CFM, None

    # Log level aliases (log1, 90, 0.9, etc.)
    if value_str in LOG_ALIASES:
        resolved_log = LOG_ALIASES[value_str]
        if resolved_log in time_cols:
            return _auto_select_time_unit(time_cols, resolved_log), resolved_log
        return None, resolved_log

    # Time unit aliases (seconds, minutes, hours) - use default log level
    if value_str in TIME_UNIT_ALIASES:
        unit = TIME_UNIT_ALIASES[value_str]
        effective_log = log_level if log_level else 2  # Default to log2 (99%)
        if effective_log in time_cols and unit in time_cols[effective_log]:
            return time_cols[effective_log][unit], effective_log
        return None, effective_log

    # Not recognized
    return None, None


def _survival_fraction_to_log(survival):
    """
    Convert survival fraction to log level.

    0.9 -> 1, 0.99 -> 2, 0.999 -> 3, 0.9999 -> 4, 0.99999 -> 5

    Uses midpoints between log levels as thresholds:
    - < 0.95 -> log1 (90%)
    - < 0.995 -> log2 (99%)
    - < 0.9995 -> log3 (99.9%)
    - < 0.99995 -> log4 (99.99%)
    - else -> log5 (99.999%)
    """
    if survival >= 1 or survival <= 0:
        return None

    # Map survival fractions to log levels using midpoint thresholds
    thresholds = [
        (0.95, 1),  # < 0.95 -> log1 (90%)
        (0.995, 2),  # < 0.995 -> log2 (99%)
        (0.9995, 3),  # < 0.9995 -> log3 (99.9%)
        (0.99995, 4),  # < 0.99995 -> log4 (99.99%)
    ]

    for threshold, log_level in thresholds:
        if survival < threshold:
            return log_level
    return 5  # >= 0.99995 -> log5 (99.999%)


def _auto_select_time_unit(time_cols, log_level):
    """
    Auto-select best time unit column based on typical data ranges.

    Returns the column name for the most appropriate time unit.
    This is a default selection; actual smart selection based on data
    happens in auto_select_time_columns().
    """
    if log_level not in time_cols:
        return None
    # Default to minutes as a reasonable starting point
    # (actual selection will be refined by auto_select_time_columns with real data)
    cols = time_cols[log_level]
    return cols.get("minutes") or cols.get("seconds") or cols.get("hours")


def get_compatible_group(col, time_cols):
    """
    Get the colinear column group for a column.

    Colinear columns are linearly related and can share axes on a plot.
    Note: k1 and k2 are NOT colinear - they are independent kinetic parameters.
    """
    # Check rate group (eACH, CADR variants - all derived from same calculation)
    if col in RATE_COLS:
        return RATE_COLS

    # Check time groups (each log level has seconds/minutes/hours variants)
    for log_level, cols in time_cols.items():
        time_group = set(cols.values())  # {seconds_col, minutes_col, hours_col}
        if col in time_group:
            return time_group

    # Unknown column - only compatible with itself
    return {col}


def is_time_column(col, time_cols):
    """True if col is a time column, False otherwise."""
    for cols in time_cols.values():
        if col in cols.values():
            return True
    return False
