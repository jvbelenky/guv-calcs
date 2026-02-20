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


def load_spectrum_file(datasource):
    """Load spectrum data from a file path or bytes.

    Supports CSV (.csv), Excel (.xls, .xlsx) files. Automatically detects
    the row where numeric data begins, skipping arbitrary header rows.

    Args:
        datasource: str/Path to a file, or bytes of file content.

    Returns:
        list[tuple[float, float]]: (wavelength, intensity) pairs.
    """
    if isinstance(datasource, (str, pathlib.PurePath)):
        filepath = Path(datasource)
        ext = filepath.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(filepath, header=None)
        elif ext in (".xls", ".xlsx"):
            df = pd.read_excel(filepath, header=None)
        else:
            raise TypeError(
                f"Unsupported file extension '{ext}'. "
                "Supported formats: .csv, .xls, .xlsx"
            )
    elif isinstance(datasource, bytes):
        df = _read_bytes_to_dataframe(datasource)
    else:
        raise TypeError(f"Unsupported data source type: {type(datasource)}")

    return _extract_spectrum_from_dataframe(df)


def _read_bytes_to_dataframe(data: bytes) -> pd.DataFrame:
    """Try to parse bytes as CSV, then Excel (.xlsx), then old Excel (.xls)."""
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
    except Exception:
        pass

    # Try xlsx (openpyxl engine)
    try:
        df = pd.read_excel(io.BytesIO(data), header=None, engine="openpyxl")
        if df.shape[1] >= 2:
            return df
    except Exception:
        pass

    # Try xls (xlrd engine)
    try:
        df = pd.read_excel(io.BytesIO(data), header=None, engine="xlrd")
        if df.shape[1] >= 2:
            return df
    except Exception:
        pass

    raise ValueError("Could not parse file as CSV or Excel format")


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
