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
