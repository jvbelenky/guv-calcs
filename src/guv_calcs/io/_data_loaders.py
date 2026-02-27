"""Package data accessors and CSV loading utilities."""

import pathlib
from pathlib import Path
import io
import csv
from functools import cache

import numpy as np
import pandas as pd
from importlib import resources


@cache
def get_full_disinfection_table():
    """Fetch all inactivation constant data without any filtering."""
    fname = "UVC Inactivation Constants.csv"
    path = resources.files("guv_calcs.data").joinpath(fname)
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0", "Index"])
    return df


def get_spectral_weightings():
    """Return a dict of all the relevant spectral weightings by wavelength."""
    fname = "UV Spectral Weighting Curves.csv"
    path = resources.files("guv_calcs.data").joinpath(fname)
    with path.open("rb") as file:
        weights = file.read()

    csv_data = load_csv(weights)
    reader = csv.reader(csv_data, delimiter=",")
    headers = next(reader, None)  # get headers

    data = {}
    for header in headers:
        data[header] = []
    for row in reader:
        for header, value in zip(headers, row):
            data[header].append(float(value))

    spectral_weightings = {}
    for i, (key, val) in enumerate(data.items()):
        spectral_weightings[key] = np.array(val)
    return spectral_weightings


def load_csv(datasource):
    """Load csv data from either path or bytes."""
    if isinstance(datasource, (str, pathlib.PurePath)):
        filepath = Path(datasource)
        filetype = filepath.suffix.lower()
        if filetype != ".csv":
            raise TypeError("Currently, only .csv files are supported")
        csv_data = open(datasource, mode="r")
    elif isinstance(datasource, bytes):
        # Convert bytes to a string using io.StringIO to simulate a file
        csv_data = io.StringIO(datasource.decode("utf-8"), newline="")
    else:
        raise TypeError(f"File type {type(datasource)} not valid")
    return csv_data


def load_spectrum_file(datasource, *, all_columns=False):
    """Load spectrum data from a file path or bytes.

    Supports CSV (.csv), Excel (.xls, .xlsx) files. Automatically detects
    the row where numeric data begins, skipping arbitrary header rows.

    Args:
        datasource: str/Path to a file, or bytes of file content.
        all_columns: If True, return all numeric columns as separate series
            with auto-detected labels. If False (default), return only the
            first two columns as (wavelength, intensity) pairs.

    Returns:
        If all_columns is False:
            list[tuple[float, float]]: (wavelength, intensity) pairs.
        If all_columns is True:
            dict with keys:
              - wavelengths: list[float]
              - series: list[dict] each with 'label', 'intensities', 'peak_wavelength'
    """
    df = _read_datasource_to_dataframe(datasource)

    if all_columns:
        return _extract_all_spectra_from_dataframe(df)
    return _extract_spectrum_from_dataframe(df)


def _read_datasource_to_dataframe(datasource) -> pd.DataFrame:
    """Read a file path or bytes into a DataFrame."""
    if isinstance(datasource, (str, pathlib.PurePath)):
        filepath = Path(datasource)
        ext = filepath.suffix.lower()
        if ext == ".csv":
            return pd.read_csv(filepath, header=None)
        elif ext in (".xls", ".xlsx"):
            return pd.read_excel(filepath, header=None)
        else:
            raise TypeError(
                f"Unsupported file extension '{ext}'. "
                "Supported formats: .csv, .xls, .xlsx"
            )
    elif isinstance(datasource, bytes):
        return _read_bytes_to_dataframe(datasource)
    else:
        raise TypeError(f"Unsupported data source type: {type(datasource)}")


def _read_bytes_to_dataframe(data: bytes) -> pd.DataFrame:
    """Try to parse bytes as CSV, then Excel (.xlsx), then old Excel (.xls)."""
    parse_errors = []

    # Try CSV first — use sep="," explicitly so rows without commas
    # (metadata headers) still produce multiple columns filled with NaN
    try:
        text = data.decode("utf-8", errors="replace")
        # Detect the max number of commas in any line to set column count
        lines = text.splitlines()
        max_fields = max((line.count(",") + 1 for line in lines), default=0)
        if max_fields >= 2:
            df = pd.read_csv(
                io.BytesIO(data), header=None, sep=",",
                names=range(max_fields), on_bad_lines="skip"
            )
            return df
    except (pd.errors.ParserError, ValueError, TypeError) as exc:
        parse_errors.append(f"csv parse failed: {exc}")

    # Try xlsx (openpyxl engine)
    try:
        df = pd.read_excel(io.BytesIO(data), header=None, engine="openpyxl")
        if df.shape[1] >= 2:
            return df
    except (ValueError, OSError, ImportError, TypeError) as exc:
        parse_errors.append(f"xlsx parse failed: {exc}")

    # Try xls (xlrd engine)
    try:
        df = pd.read_excel(io.BytesIO(data), header=None, engine="xlrd")
        if df.shape[1] >= 2:
            return df
    except (ValueError, OSError, ImportError, TypeError) as exc:
        parse_errors.append(f"xls parse failed: {exc}")

    detail = "; ".join(parse_errors) if parse_errors else "no parser accepted the input"
    raise ValueError(f"Could not parse file as CSV or Excel format: {detail}")


def _find_data_start_row(df: pd.DataFrame, min_cols: int = 2) -> int:
    """Find the first row where at least `min_cols` columns are numeric."""
    for i in range(len(df)):
        numeric_count = 0
        for j in range(min(min_cols, df.shape[1])):
            try:
                float(df.iloc[i, j])
                numeric_count += 1
            except (ValueError, TypeError):
                break
        if numeric_count >= min_cols:
            return i
    raise ValueError("No numeric data found in file")


def _extract_all_spectra_from_dataframe(df: pd.DataFrame) -> dict:
    """Extract all numeric columns as separate series from a DataFrame.

    Returns a dict with:
      - wavelengths: list[float]
      - series: list[dict] each with 'label', 'intensities', 'peak_wavelength'
    """
    if df.shape[1] < 2:
        raise ValueError("Data must have at least 2 columns")

    start_row = _find_data_start_row(df)

    # Auto-detect labels from header rows above the data start.
    # Scan backwards from start_row for the last row with string values in data columns.
    labels = []
    if start_row > 0:
        for row_idx in range(start_row - 1, -1, -1):
            row_labels = []
            has_strings = False
            for col_idx in range(1, df.shape[1]):
                val = df.iloc[row_idx, col_idx]
                if isinstance(val, str) and val.strip():
                    row_labels.append(val.strip())
                    has_strings = True
                else:
                    row_labels.append(None)
            if has_strings:
                labels = row_labels
                break

    # Extract numeric data from start_row onward
    data = df.iloc[start_row:].copy()
    wavelengths = pd.to_numeric(data.iloc[:, 0], errors="coerce")

    series = []
    for col_idx in range(1, df.shape[1]):
        col_data = pd.to_numeric(data.iloc[:, col_idx], errors="coerce")
        # Only include columns that have at least some numeric data
        valid_mask = wavelengths.notna() & col_data.notna()
        if valid_mask.sum() == 0:
            continue

        wl = wavelengths[valid_mask].tolist()
        intensities = col_data[valid_mask].tolist()

        # Determine label
        label_idx = col_idx - 1
        if label_idx < len(labels) and labels[label_idx] is not None:
            label = labels[label_idx]
        else:
            label = f"Column {col_idx}"

        # Find peak wavelength (wavelength at max intensity)
        max_idx = intensities.index(max(intensities))
        peak_wavelength = wl[max_idx]

        series.append({
            "label": label,
            "intensities": intensities,
            "peak_wavelength": peak_wavelength,
        })

    if not series:
        raise ValueError("No numeric data columns found in file")

    # Use wavelengths from the first valid mask (all series share wavelengths)
    first_valid = wavelengths.notna()
    for col_idx in range(1, df.shape[1]):
        col_data = pd.to_numeric(data.iloc[:, col_idx], errors="coerce")
        first_valid = first_valid & col_data.notna()
    # Use the wavelengths from the broadest set (first series)
    col1 = pd.to_numeric(data.iloc[:, 1], errors="coerce")
    wl_mask = wavelengths.notna() & col1.notna()
    final_wavelengths = wavelengths[wl_mask].tolist()

    return {
        "wavelengths": final_wavelengths,
        "series": series,
    }


def _extract_spectrum_from_dataframe(df: pd.DataFrame) -> list[tuple[float, float]]:
    """Extract (wavelength, intensity) pairs from a DataFrame.

    Scans for the first row where columns 0 and 1 are both numeric,
    then takes all subsequent numeric rows.
    """
    if df.shape[1] < 2:
        raise ValueError("Data must have at least 2 columns")

    # Find the first row where both columns 0 and 1 are numeric
    start_row = None
    for i in range(len(df)):
        try:
            float(df.iloc[i, 0])
            float(df.iloc[i, 1])
            start_row = i
            break
        except (ValueError, TypeError):
            continue

    if start_row is None:
        raise ValueError("No numeric data found in file")

    # Take columns 0 and 1 from start_row onward
    subset = df.iloc[start_row:, :2].copy()
    subset.columns = [0, 1]
    subset[0] = pd.to_numeric(subset[0], errors="coerce")
    subset[1] = pd.to_numeric(subset[1], errors="coerce")
    subset = subset.dropna()

    if len(subset) == 0:
        raise ValueError("No numeric data found in file")

    return list(zip(subset[0].tolist(), subset[1].tolist()))
