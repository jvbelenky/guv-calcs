import warnings
from importlib import resources
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import csv
import numpy as np
from .io import load_csv

pd.options.mode.chained_assignment = None


def get_full_disinfection_table():
    """fetch all inactivation constant data without any filtering"""
    fname = "UVC Inactivation Constants.csv"
    path = resources.files("guv_calcs.data").joinpath(fname)
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0", "Index"])
    df = df.rename(
        columns={
            "Medium (specific)": "Condition",
            "Full Citation": "Reference",
            "Organism": "Category",
            "URL": "Link",
        }
    )
    return df


def get_disinfection_table(fluence=None, wavelength=None, room=None):
    """
    Retrieve and format inactivation data.

    Fluence: numeric or dict
        If dict, should take form { wavelength : fluence}
        If None, eACH and CADR will not be computed
    wavelength: int or float
        If None, all wavelengths will be returned
    room: guv_calcs.room.Room object
        If None, CADR will not be computed.
    """

    df = get_full_disinfection_table()
    df = df[df["Medium"] == "Aerosol"]

    newkeys = []

    if wavelength is not None:
        valid_wavelengths = df["wavelength [nm]"].unique()
        if wavelength not in valid_wavelengths:
            msg = f"No data is available for wavelength {wavelength} nm."
            warnings.warn(msg, stacklevel=3)
        df = df[df["wavelength [nm]"] == wavelength]
    else:
        newkeys += ["wavelength [nm]"]
        df["wavelength [nm]"] = df["wavelength [nm]"].astype(int)
    if fluence is not None:
        if isinstance(fluence, dict):
            df, fluence = _filter_wavelengths(df, fluence)
        df["eACH-UV"] = df.apply(_compute_eACH_UV, args=[fluence], axis=1)
        newkeys += ["eACH-UV"]
        if room is not None:
            df = _get_cadr(df, room)
            newkeys += ["CADR-UV [cfm]", "CADR-UV [lps]"]

    newkeys += [
        "Category",
        "Species",
        "Strain",
        "k1 [cm2/mJ]",
        "k2 [cm2/mJ]",
        "% resistant",
        "Condition",
        "Reference",
        "Link",
    ]
    df = df[newkeys].fillna(" ")
    df = df.sort_values("Species")

    return df


def plot_disinfection_data(df, fluence_dict=None, room=None, title=None):
    """
    Plot eACH/CADR for all species as violin (kde) plot and swarmplot.

    df: pd.DataFrame
        To be generated with the get_disinfection_table function
    fluence_dict: dict
        Optional. Dictionary of form {wavelegnth: fluence}. Used to generate
        automatic title if one is not provided.
    room: guv_calcs.room.Room
        Optional. Used to calculate CADR for multi-wavelength dataframes.
    title: str
        Optional title string.
    """

    if "wavelength [nm]" in df.keys() and len(df["wavelength [nm]"].unique()) > 1:
        if "eACH-UV" in df.keys():
            df = sum_multiwavelength_data(df, room)
            style = None
        else:
            style = "wavelength [nm]"
    else:
        style = None

    categories = ["Bacteria", "Virus", "Fungi", "Protists", "Bacterial spores"]
    hue_order = []
    for val in categories:
        if val in df["Category"].unique():
            hue_order += [val]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    if "eACH-UV" in df.keys():
        key = "eACH-UV"
    else:
        key = "k1 [cm2/mJ]"
    sns.violinplot(
        data=df,
        x="Species",
        y=key,
        hue="Category",
        hue_order=hue_order,
        inner=None,
        ax=ax1,
        alpha=0.5,
        legend=False,
    )
    sns.scatterplot(
        data=df,
        x="Species",
        y=key,
        hue="Category",
        hue_order=hue_order,
        style=style,
        ax=ax1,
        s=100,
        alpha=0.7,
    )
    ax1.set_ylabel(key)
    ax1.set_xlabel(None)
    # ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(bottom=0)
    ax1.grid("--")
    ax1.set_xticks(ax1.get_xticks())
    ax1.set_xticklabels(
        ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )

    if "CADR-UV [cfm]" in df.keys():
        # add a second axis to show CADR
        ax2 = ax1.twinx()
        sns.violinplot(
            data=df,
            x="Species",
            y="CADR-UV [cfm]",
            ax=ax2,
            alpha=0,
            legend=False,
            inner=None,
        )
        ax2.set_ylabel("CADR-UV [cfm]")
        ax2.set_ylim(bottom=0)

    if room is not None:
        yval = room.air_changes
        if yval < 0.1 * ax1.get_ylim()[1]:  # avoid overlapping with bottom ticks
            yval += 0.05 * ax1.get_ylim()[1]

        ax1.axhline(y=room.air_changes, color="red", linestyle="--", linewidth=1.5)
        if int(room.air_changes) == room.air_changes:
            ac = int(room.air_changes)
        else:
            ac = round(room.air_changes, 2)
        if ac == 1:
            string = f"{ac} air change\nfrom ventilation"
        else:
            string = f"{ac} air changes\nfrom ventilation"
        ax1.text(
            1.01,
            yval,
            string,
            color="red",
            va="center",
            ha="left",
            transform=ax1.get_yaxis_transform(),
        )

    if title is not None:
        fig.suptitle(title)
    elif fluence_dict is not None:
        fig.suptitle(_generate_title(df, fluence_dict))

    return fig


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
        df = _get_cadr(df, room)
    return df


def _generate_title(df, fluence_dict):
    """convenience function; generates automatic title for plot_disinfection_data"""
    title = ""
    guv_types = ", ".join(["GUV-" + str(wv) for wv in fluence_dict.keys()])
    fluences = ", ".join([str(round(val, 2)) for val in fluence_dict.values()])
    if "eACH-UV" in df.keys():
        title += "eACH"
        if "CADR-UV [cfm]" in df.keys():
            title += "/CADR"
        title += f" from {guv_types}"
        if isinstance(fluence_dict, dict) and len(fluence_dict) > 1:
            title += f"\nwith average fluence rates: [{fluences}] uW/cm²"
        else:
            title += f" with average fluence rate {fluences} uW/cm²"
    else:
        title += f"K1 susceptibility values for {guv_types}"
    return title


def _get_cadr(df, room):
    """
    calculate CADR from a dataframe which contains a key called `eACH-UV`
    """
    volume = room.volume
    # convert to cubic feet for cfm
    if room.dim.units == "meters":
        volume = volume / (0.3048 ** 3)
    cadr_uv_cfm = df["eACH-UV"] * volume / 60
    cadr_uv_lps = cadr_uv_cfm * 0.47195
    df["CADR-UV [cfm]"] = cadr_uv_cfm.round(1)
    df["CADR-UV [lps]"] = cadr_uv_lps.round(1)
    return df


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


def _compute_eACH_UV(row, fluence_arg):
    """
    compute equivalent air changes per hour for every row in the disinfection
    rate database.
    fluence_arg may be an int, float, or dict of format {wavelength:fluence}
    """
    # Extract fluence for the given wavelength
    if isinstance(fluence_arg, dict):
        fluence = fluence_arg[row["wavelength [nm]"]]
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

    # Compute eACH-UV
    eACH = (k1 * (1 - f) + k2 - k2 * (1 - f)) * fluence * 3.6
    return round(eACH, 1)


def get_spectral_weightings():
    """
    Return a dict of all the relevant spectral weightings by wavelength
    """

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
