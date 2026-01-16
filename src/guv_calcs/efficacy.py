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


def get_disinfection_table(fluence=None, wavelength=None, medium=None, category=None):
    return Data(
        medium=medium, category=category, wavelength=wavelength, fluence=fluence
    ).df


class Data:
    def __init__(
        self,
        medium: str | None = None,
        category: str | None = None,
        wavelength: int | float | None = None,
        fluence: float | None = None,
    ):
        self._base_df = get_full_disinfection_table()
        self._medium = medium
        self._category = category
        self._wavelength = wavelength
        self._fluence = fluence

        keys = [
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

        df = self._base_df
        if medium is not None:
            if medium not in self.mediums:
                raise KeyError(
                    f"{medium} is not a valid medium; must be in {self.mediums}"
                )
            df = df[df["Medium"] == medium]
            keys.remove("Medium")

        if category is not None:
            if category not in self.categories:
                raise KeyError(
                    f"{category} is not a valid category; must be in {self.categories}"
                )
            df = df[df["Category"] == category]
            keys.remove("Category")

        if wavelength is not None:
            if wavelength not in self.wavelengths:
                raise KeyError(
                    f"{wavelength} is not a valid wavelength; must be in {self.wavelengths}"
                )
            df = df[df["wavelength [nm]"] == wavelength]
            keys.remove("wavelength [nm]")

        if fluence is not None:
            key = "Seconds to 99% inactivation"
            df[key] = df.apply(_compute_row, args=[fluence, log2, 0], axis=1)
            if sum(df[key] > 60) / len(df[key]) > 0.5:
                newkey = "Minutes to 99% inactivation"
                df[newkey] = round(df[key] / 60, 1)
                if sum(df[newkey] > 60) / len(df[newkey]) > 0.75:
                    finalkey = "Hours to 99% inactivation"
                    df[finalkey] = round(df[newkey] / 60, 1)
                    keys = [finalkey] + keys
                else:
                    keys = [newkey] + keys
            else:
                keys = [key] + keys

            if medium == "Aerosol":
                df["eACH-UV"] = df.apply(
                    _compute_row, args=[fluence, eACH_UV, 1], axis=1
                )
                keys = ["eACH-UV"] + keys
            # todo: add log-N inactivations

        df = df[keys].fillna(" ")
        df = df.sort_values("Species")
        self._df = df

    @property
    def df(self):
        return self._df

    @property
    def table(self):
        """alias"""
        return self.df

    @property
    def base_df(self):
        return self._base_df

    @property
    def medium(self):
        return self._medium

    @property
    def category(self):
        return self._category

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def fluence(self):
        return self._fluence

    @property
    def keys(self):
        return self._df.keys()

    @property
    def categories(self):
        return sorted(self._base_df["Category"].unique())

    @property
    def mediums(self):
        return sorted(self._base_df["Medium"].unique())

    @property
    def wavelengths(self):
        return sorted(self._base_df["wavelength [nm]"].unique())

    @classmethod
    def with_room(cls, room, zone_id="WholeRoomFluence", category=None):
        fluence_dict = room.fluence_dict(zone_id)
        if len(fluence_dict) == 0:
            msg = "Fluence value not available; returning full disinfection data table."
            warnings.warn(msg, stacklevel=3)
            return cls(category=category)

        data = cls(medium="Aerosol", category=category, fluence=fluence_dict)

        df, fluence_dict = _filter_wavelengths(data.df, fluence_dict)
        data._fluence = fluence_dict

        # add cadr
        cadr, cadr_key = _get_cadr(df["eACH-UV"], room)
        keys = [cadr_key] + list(df.keys())
        df[cadr_key] = cadr.round(1)

        df = df[keys]  # rearrange
        if len(fluence_dict) == 1:  # drop wavelength column if unnecessary
            df = df.drop(columns="wavelength [nm]")
        data._df = df
        return data

    def plot(
        self, title=None, left_axis=None, right_axis=None, figsize=(8, 5), room=None
    ):
        """
        plot inactivation data for all species as a violin (kde) and swarm plot
        """
        df = self._df
        if "wavelength [nm]" in df.keys() and len(df["wavelength [nm]"].unique()) > 1:
            if "eACH-UV" in df.keys():
                df = sum_multiwavelength_data(df, room=room)
                style = None
            else:
                style = "wavelength [nm]"
        else:
            style = None

        kkey = self._get_key("k1")
        timekey = self._get_key("inactivation")
        eachkey = self._get_key("eACH")
        cadrkey = self._get_key("CADR")

        right_label = right_axis or cadrkey or eachkey or timekey
        left_label = kkey  # default left axis
        for key in [left_axis, eachkey, timekey]:
            if key != right_label and key is not None:
                left_label = key

        category_order = ["Bacteria", "Virus", "Fungi", "Protists", "Bacterial spores"]
        categories = list(df["Category"].unique())
        hue_order = [cat for cat in category_order if cat in categories]

        fig, ax1 = plt.subplots(figsize=(8, 5))

        sns.violinplot(
            data=df,
            x="Species",
            y=left_label,
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
            y=left_label,
            hue="Category",
            hue_order=hue_order,
            style=style,
            ax=ax1,
            s=100,
            alpha=0.7,
        )

        if right_label is not None:
            # add a second axis
            ax2 = ax1.twinx()
            sns.violinplot(
                data=df,
                x="Species",
                y=right_label,
                ax=ax2,
                alpha=0,
                legend=False,
                inner=None,
            )
            ax2.set_ylabel(right_label)
            ax2.set_ylim(bottom=0)

        # add ACH linestyle
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

        fig.suptitle(title or self._generate_title())

    def _get_key(self, substring):
        index = np.array([substring in key for key in self._df.keys()])
        return self._df.keys()[index][0] if sum(index) > 0 else None

    def _generate_title(self):
        if self.wavelength is None:
            wvs = self.df["wavelength [nm]"].unique()
        else:
            wvs = [self.wavelength]
        guv_types = ", ".join(["GUV-" + str(wv) for wv in wvs])
        if isinstance(self.fluence, dict):
            fluence_dict = self.fluence


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


def _get_cadr(eACH, room):
    """compute clean air delivery rate from room volume and eACH value"""
    if room.dim.units in [LengthUnits.METERS, LengthUnits.CENTIMETERS]:
        return eACH * room.dim.cubic_meters * 1000 / 60 / 60, "CADR-UV [lps]"
    else:
        return eACH * room.dim.cubic_feet / 60, "CADR-UV [cfm]"


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
        cadr, cadr_key = _get_cadr(df["eACH_UV"], room)
        df[cadr_key] = cadr
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


def CADR_CFM(cubic_feet, irrad, k1, k2=0, f=0):
    return eACH_UV(irrad=irrad, k1=k1, k2=k2, f=f) * cubic_feet / 60


def CADR_LPS(cubic_meters, irrad, k1, k2=0, f=0):
    return eACH_UV(irrad=irrad, k1=k1, k2=k2, f=f) * cubic_meters * 1000 / 60 / 60


def eACH_UV(irrad, k1, k2=0, f=0):
    return (k1 * (1 - f) + k2 - k2 * (1 - f)) * irrad * 3.6


def log1(irrad, k1, k2=0, f=0, **kwargs):
    return seconds_to_S(0.1, irrad=irrad, k1=k1, k2=k2, f=f, **kwargs)


def log2(irrad, k1, k2=0, f=0, **kwargs):
    return seconds_to_S(0.01, irrad=irrad, k1=k1, k2=k2, f=f, **kwargs)


def log3(irrad, k1, k2=0, f=0, **kwargs):
    return seconds_to_S(0.001, irrad=irrad, k1=k1, k2=k2, f=f, **kwargs)


def seconds_to_S(S, irrad, k1, k2=0, f=0, tol=1e-10, max_iter=100):
    """
    S: float, (0,1) - surviving fraction
    irrad: float, fluence/irradiance in uW/cm2
    k1: float, first susceptibility value, cm2/mJ
    k2: float, second susceptibility value, cm2/mJ
    f: float, (0,1) - resistant fraction
    tol: float, numerical tolerance
    max_iter: maximum number of iterations to wait for solution to converge
    """

    def S_of_t(t):
        return (1 - f) * math.exp(-k1 * irrad / 1000 * t) + f * math.exp(
            -k2 * irrad * t
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


# def get_disinfection_table(fluence=None, wavelength=None, medium="Aerosol", room=None):
# """
# Retrieve and format inactivation data.

# Fluence: numeric or dict
# If dict, should take form { wavelength : fluence}
# If None, eACH and CADR will not be computed
# wavelength: int or float
# If None, all wavelengths will be returned
# room: guv_calcs.Room object
# If None, CADR will not be computed.
# """

# df = get_full_disinfection_table()
# df = df[df["Medium"] == medium]

# newkeys = []

# if wavelength is not None:
# valid_wavelengths = df["wavelength [nm]"].unique()
# if wavelength not in valid_wavelengths:
# msg = f"No data is available for wavelength {wavelength} nm."
# warnings.warn(msg, stacklevel=3)
# df = df[df["wavelength [nm]"] == wavelength]
# else:
# newkeys += ["wavelength [nm]"]
# df["wavelength [nm]"] = df["wavelength [nm]"].astype(int)

# if fluence is not None:
# if isinstance(fluence, dict):
# df, fluence = filter_wavelengths(df, fluence)
# df["eACH-UV"] = df.apply(_compute_eACH_UV, args=[fluence], axis=1)
# newkeys += ["eACH-UV"]
# if room is not None:
# df = _get_cadr(df, room)
# newkeys += ["CADR-UV [cfm]", "CADR-UV [lps]"]

# newkeys += [
# "Category",
# "Species",
# "Strain",
# "k1 [cm2/mJ]",
# "k2 [cm2/mJ]",
# "% resistant",
# "Condition",
# "Reference",
# "Link",
# ]
# df = df[newkeys].fillna(" ")
# df = df.sort_values("Species")

# return df


# def _get_cadr(df, room):
# """
# calculate CADR from a dataframe which contains a key called `eACH-UV`
# """
# # volume = room.volume
# # # convert to cubic feet for cfm
# # if room.dim.units == "meters":
# # volume = volume / (0.3048 ** 3)
# cadr_uv_cfm = df["eACH-UV"] * room.dim.cubic_feet / 60
# cadr_uv_lps = df["eACH-UV"] * room.dim.cubic_meters * 1000 / 60 / 60
# df["CADR-UV [cfm]"] = cadr_uv_cfm.round(1)
# df["CADR-UV [lps]"] = cadr_uv_lps.round(1)
# return df


# def _compute_eACH_UV(row, fluence_arg):
# """
# compute equivalent air changes per hour for every row in the disinfection
# rate database.
# fluence_arg may be an int, float, or dict of format {wavelength:fluence}
# """
# # Extract fluence for the given wavelength
# if isinstance(fluence_arg, dict):
# fluence = fluence_arg[row["wavelength [nm]"]]
# elif isinstance(fluence_arg, (float, int)):
# fluence = fluence_arg

# # Fill missing values and convert columns to the correct type
# k1 = row["k1 [cm2/mJ]"] if pd.notna(row["k1 [cm2/mJ]"]) else 0.0
# k2 = row["k2 [cm2/mJ]"] if pd.notna(row["k2 [cm2/mJ]"]) else 0.0
# f = (
# float(row["% resistant"].rstrip("%")) / 100
# if pd.notna(row["% resistant"])
# else 0.0
# )
# return round(get_eACH_UV(irrad=fluence, k1=k1, k2=k2, f=f), 1)


def plot_disinfection_data(df, fluence_dict=None, room=None, title=None):
    """
    Plot eACH/CADR for all species as violin (kde) plot and swarmplot.

    df: pd.DataFrame
        To be generated with the get_disinfection_table function
    fluence_dict: dict
        Optional. Dictionary of form {wavelength: fluence}. Used to generate
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
