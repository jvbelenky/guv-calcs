import warnings
import pathlib
from pathlib import Path
import datetime
import json
import zipfile
import io
import csv
from functools import cache
from packaging.version import Version
import pandas as pd
import numpy as np
import plotly.io as pio
from plotly.graph_objs._figure import Figure as plotly_fig
from matplotlib.figure import Figure as mpl_fig
from importlib import resources
from .room_dims import RoomDimensions

# -------------- Loading a room from file -------------------


def parse_guv_file(filedata):
    """Parse a .guv file from various input types (path, string, bytes, dict)."""
    if isinstance(filedata, dict):
        return filedata
    if isinstance(filedata, (str, bytes, bytearray)):
        try:
            return json.loads(filedata)
        except json.JSONDecodeError:
            # Not JSON string, try as file path
            return _load_guv_file(Path(filedata))
    if isinstance(filedata, pathlib.PurePath):
        return _load_guv_file(filedata)
    raise TypeError(f"Cannot load room from {type(filedata).__name__}")


def _load_guv_file(path):
    """Load JSON from a .guv file path."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    if path.suffix.lower() != ".guv":
        raise ValueError(f"Please provide a valid .guv file (got {path.suffix}): {path}")
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in {path}: {e}") from e


def save_room_data(room, fname):
    """save all relevant parameters to a json file"""
    savedata = {}
    version = get_version(Path(__file__).parent / "_version.py")
    savedata["guv-calcs_version"] = version

    now = datetime.datetime.now()
    now_local = datetime.datetime.now(now.astimezone().tzinfo)
    timestamp = now_local.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    savedata["timestamp"] = timestamp

    savedata["data"] = room.to_dict()
    if fname is not None:
        filename = _check_savefile(fname, ".guv")
        with open(filename, "w") as json_file:
            json.dump(savedata, json_file, indent=4)
    else:
        return json.dumps(savedata, indent=4)


def _check_savefile(filename, ext):
    """
    enforce that a savefile has the correct extension
    """

    if not ext.startswith("."):
        ext = "." + ext

    if isinstance(filename, str):
        if not filename.lower().endswith(ext):
            filename += ext
    elif isinstance(filename, pathlib.PurePath):
        if not filename.suffix == ext:
            filename = filename.parent / (filename.name + ext)
    return filename


def export_room_zip(
    room,
    fname=None,
    include_plots=False,
    include_lamp_files=False,
    include_lamp_plots=False,
):

    """
    write the room project file and all results files to a zip file. Optionally include
    extra files like lamp ies files, spectra files, and plots.
    """

    # save base project
    data_dict = {"room.guv": room.save()}

    # save all results
    for zone_id, zone in room.calc_zones.items():
        data_dict[zone.name + ".csv"] = zone.export()
        if include_plots:
            if zone.dose:
                title = f"{zone.hours} Hour Dose"
            else:
                title = "Irradiance"
            if zone.calctype == "Plane":
                # Save the figure to a BytesIO object
                title += f" ({zone.height} m)"
                fig, ax = zone.plot_plane(title=title)
                img_bytes = fig_to_bytes(fig)
                if img_bytes is not None:
                    data_dict[zone.name + ".png"] = img_bytes
            elif zone.calctype == "Volume":
                fig = zone.plot_volume()
                img_bytes = fig_to_bytes(fig)
                if img_bytes is not None:
                    data_dict[zone.name + ".png"] = img_bytes

    # save lamp files if indicated to
    for lamp_id, lamp in room.lamps.items():
        if lamp.ies is not None:
            if include_lamp_files:
                data_dict[lamp.name + ".ies"] = lamp.save_ies()
            if include_lamp_plots:
                ies_fig, ax = lamp.plot_ies(title=lamp.name)
                img_bytes = fig_to_bytes(ies_fig)
                if img_bytes is not None:
                    data_dict[lamp.name + "_ies.png"] = img_bytes
        if lamp.spectrum is not None:
            if include_lamp_plots:
                linfig, _ = lamp.spectrum.plot(
                    title=lamp.name, yscale="linear", weights=True, label=True
                )
                logfig, _ = lamp.spectrum.plot(
                    title=lamp.name, yscale="log", weights=True, label=True
                )
                linkey = lamp.name + "_spectra_linear.png"
                logkey = lamp.name + "_spectra_log.png"
                lin_bytes = fig_to_bytes(linfig)
                log_bytes = fig_to_bytes(logfig)
                if lin_bytes is not None:
                    data_dict[linkey] = lin_bytes
                if log_bytes is not None:
                    data_dict[logkey] = log_bytes

    zip_buffer = io.BytesIO()
    # Create a zip file within this BytesIO object
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Loop through the dictionary, adding each string/byte stream to the zip
        for filename, content in data_dict.items():
            # Ensure the content is in bytes
            if isinstance(content, str):
                content = content.encode("utf-8")
            # Add the file to the zip; writing the bytes to a BytesIO object for the file
            file_buffer = io.BytesIO(content)
            zip_file.writestr(filename, file_buffer.getvalue())
    zip_bytes = zip_buffer.getvalue()

    if fname is not None:
        with open(fname, "wb") as f:
            f.write(zip_bytes)
    else:
        return zip_bytes


def _build_room_rows(room):
    """Build the row data for a single room's report."""
    precision = room.precision if room.precision > 3 else 3

    def fmt(v):
        return round(v, precision) if isinstance(v, (int, float)) else v

    # ───  Room parameters  ───────────────────────────────
    rows = [["Room Parameters"]]
    rows += [["", "Dimensions", "x", "y", "z", "units"]]
    d = room.dim
    rows += [["", "", fmt(d.x), fmt(d.y), fmt(d.z), d.units]]
    vol_units = "ft 3" if room.units == "feet" else "m 3"
    rows += [["", "Volume", fmt(room.volume), vol_units]]
    rows += [[""]]

    # ───  Reflectance  ──────────────────────────────────
    rows += [["", "Reflectance"]]
    rows += [["", "", "Floor", "Ceiling", "North", "South", "East", "West", "Enabled"]]
    rows += [
        ["", "", *room.ref_manager.reflectances.values(), room.ref_manager.enabled]
    ]
    rows += [[""]]

    # ───  Luminaires  ───────────────────────────────────
    if room.lamps:
        rows += [["Luminaires"]]
        rows += [["", "", "", "Surface Position", "", "", "Aim"]]
        rows += [
            [
                "",
                "ID",
                "Name",
                "x",
                "y",
                "z",
                "x",
                "y",
                "z",
                "Orientation",
                "Tilt",
                "Surface Length",
                "Surface Width",
                "Scaling factor",
            ]
        ]
        for lamp in room.lamps.values():
            rows += [
                [
                    "",
                    lamp.lamp_id,
                    lamp.name,
                    fmt(lamp.x),
                    fmt(lamp.y),
                    fmt(lamp.z),
                    fmt(lamp.aimx),
                    fmt(lamp.aimy),
                    fmt(lamp.aimz),
                    fmt(lamp.heading),
                    fmt(lamp.bank),
                    fmt(lamp.surface.length),
                    fmt(lamp.surface.width),
                    fmt(lamp.scaling_factor),
                ]
            ]
        rows += [[""]]

    # ----- Calc zones ------------------------
    zones = [z for z in room.calc_zones.values() if z.values is not None]

    # ----- Calc planes -----------------------
    planes = [z for z in zones if z.calctype == "Plane"]
    if planes:
        rows += [["Calculation Planes"]]
        rows += [
            [
                "",
                "ID",
                "Name",
                "x1",
                "x2",
                "y1",
                "y2",
                "height",
                "Vertical irradiance",
                "Horizontal irradiance",
                "Vertical field of view",
                "Horizontal field of view",
                "Dose",
                "Dose Hours",
            ]
        ]
        for pl in planes:
            rows += [
                [
                    "",
                    pl.zone_id,
                    pl.name,
                    fmt(pl.x1),
                    fmt(pl.x2),
                    fmt(pl.y1),
                    fmt(pl.y2),
                    fmt(pl.height),
                    pl.vert,
                    pl.horiz,
                    pl.fov_vert,
                    pl.fov_horiz,
                    pl.dose,
                    pl.hours if pl.dose else "",
                ]
            ]
        rows += [[""]]

    # ------ Calc volumes ----------------------
    vols = [z for z in zones if z.calctype == "Volume"]
    if vols:
        rows += [["Calculation Volumes"]]
        rows += [
            [
                "",
                "ID",
                "Name",
                "x1",
                "x2",
                "y1",
                "y2",
                "z1",
                "z2",
                "Dose",
                "Dose Hours",
            ]
        ]
        for v in vols:
            rows += [
                [
                    "",
                    v.zone_id,
                    v.name,
                    fmt(v.x1),
                    fmt(v.x2),
                    fmt(v.y1),
                    fmt(v.y2),
                    fmt(v.z1),
                    fmt(v.z2),
                    v.dose,
                    v.hours if v.dose else "",
                ]
            ]
        rows += [[""]]

    # --------- Statistics -----------------
    if zones:
        rows += [["Statistics"]]
        rows += [
            ["", "Calculation Zone", "Avg", "Max", "Min", "Max/Min", "Avg/Min", "Units"]
        ]
        for zone in zones:
            values = zone.get_values()
            avg = values.mean()
            mx = values.max()
            mn = values.min()
            mxmin = mx / mn if mn != 0 else float("inf")
            avgmin = avg / mn if mn != 0 else float("inf")
            rows += [
                [
                    "",
                    zone.name,
                    round(avg, precision),
                    round(mx, precision),
                    round(mn, precision),
                    round(mxmin, precision),
                    round(avgmin, precision),
                    zone.units,
                ]
            ]
        rows += [[""]]

    return rows


def generate_report(self, fname=None):
    """Dump a one-file CSV snapshot of the current room."""
    rows = _build_room_rows(self)
    rows += [[f"Generated {datetime.datetime.now().isoformat(timespec='seconds')}"]]
    csv_bytes = rows_to_bytes(rows)

    if fname is not None:
        with open(fname, "wb") as csvfile:
            csvfile.write(csv_bytes)
    else:
        return csv_bytes


# ------- package data loading ---------------


@cache
def get_full_disinfection_table():
    """fetch all inactivation constant data without any filtering"""
    fname = "UVC Inactivation Constants.csv"
    path = resources.files("guv_calcs.data").joinpath(fname)
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0", "Index"])
    return df


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


# ------- Conversions to bytes ---------------

# Track whether we've already warned about missing Chrome
_chrome_warning_shown = False


def fig_to_bytes(fig):
    """
    Convert a matplotlib or plotly figure to PNG bytes.

    For plotly figures, this requires Chrome to be installed (used by kaleido).
    If Chrome is not available, returns None and issues a warning.
    """
    global _chrome_warning_shown

    if isinstance(fig, mpl_fig):
        buf = io.BytesIO()
        fig.savefig(
            buf, format="png"
        )  # You can change the format as needed (e.g., 'jpeg', 'pdf')
        buf.seek(0)  # Rewind the buffer
        byt = buf.getvalue()
    elif isinstance(fig, plotly_fig):
        try:
            byt = pio.to_image(fig, format="png", scale=1)
        except Exception as e:
            # kaleido v1 requires Chrome to be installed
            if not _chrome_warning_shown:
                warnings.warn(
                    "Could not export plotly figure to image. "
                    "This requires Chrome/Chromium to be installed for kaleido. "
                    f"Error: {e}"
                )
                _chrome_warning_shown = True
            return None
    else:
        raise TypeError("This figure type cannot be converted to bytes")
    return byt


def rows_to_bytes(rows, encoding="cp1252"):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerows(rows)

    # Get the CSV data from buffer, convert to bytes
    csv_data = buffer.getvalue()
    csv_bytes = csv_data.encode(encoding)  # encode to bytes
    return csv_bytes


# ----------- misc io ----------


def load_csv(datasource):
    """load csv data from either path or bytes"""
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


def get_version(path) -> dict:

    version = {}
    with open(path) as f:
        exec(f.read(), version)
    return version["__version__"]


# -------------- Project IO -------------------


def _make_envelope(format_key=None):
    """Build the common save envelope (version + timestamp)."""
    savedata = {}
    savedata["guv-calcs_version"] = get_version(Path(__file__).parent / "_version.py")
    now = datetime.datetime.now()
    now_local = datetime.datetime.now(now.astimezone().tzinfo)
    savedata["timestamp"] = now_local.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    if format_key:
        savedata["format"] = format_key
    return savedata


def save_project_data(project, fname):
    """Save a Project to a .guv file (or return JSON string if fname is None)."""
    savedata = _make_envelope("project")
    savedata["data"] = project.to_dict()
    if fname is not None:
        filename = _check_savefile(fname, ".guv")
        with open(filename, "w") as f:
            json.dump(savedata, f, indent=4)
    else:
        return json.dumps(savedata, indent=4)


def load_project(cls, filedata):
    """Load a Project from a .guv file, JSON string, bytes, or dict.

    If the file is a legacy single-room format (no "format" key),
    wraps it in a one-room Project.
    """
    from .room import Room

    load_data = parse_guv_file(filedata)

    saved_version = load_data.get("guv-calcs_version", "0.0.0")
    current_version = get_version(Path(__file__).parent / "_version.py")
    if saved_version != current_version:
        warnings.warn(
            f"File was saved with guv-calcs {saved_version}, "
            f"current version is {current_version}"
        )

    fmt = load_data.get("format")
    data = load_data.get("data", load_data)

    if fmt == "project":
        return cls.from_dict(dict(data))

    # Legacy single-room format
    room = Room.load(filedata)
    project = cls(
        standard=room.standard,
        units=room.units,
        precision=room.precision,
        colormap=room.colormap,
        enable_reflectance=room.ref_manager.enabled,
        reflectance_max_num_passes=room.ref_manager.max_num_passes,
        reflectance_threshold=room.ref_manager.threshold,
    )
    project.rooms.add(room, on_collision="overwrite")
    return project


def export_project_zip(project, fname=None, **kwargs):
    """Export a Project as a zip with per-room subdirectories."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Project-level .guv
        project_json = project.save()
        zf.writestr("project.guv", project_json.encode("utf-8"))

        for room_id, room in project.rooms.items():
            room_zip_bytes = export_room_zip(room, **kwargs)
            # Nest room zip contents under room_id/ prefix
            with zipfile.ZipFile(io.BytesIO(room_zip_bytes), "r") as room_zf:
                for name in room_zf.namelist():
                    zf.writestr(f"{room_id}/{name}", room_zf.read(name))

    zip_bytes = zip_buffer.getvalue()
    if fname is not None:
        with open(fname, "wb") as f:
            f.write(zip_bytes)
    else:
        return zip_bytes


def _build_project_summary(project):
    """Build a cross-room summary table."""
    rows = [["=== Project Summary ==="]]
    header = [
        "",
        "Room ID",
        "Room Name",
        "Calculation Zone",
        "Avg",
        "Max",
        "Min",
        "Max/Min",
        "Avg/Min",
        "Units",
    ]
    has_zones = False
    data_rows = []
    for room_id, room in project.rooms.items():
        zones = [z for z in room.calc_zones.values() if z.values is not None]
        for zone in zones:
            has_zones = True
            values = zone.get_values()
            avg = values.mean()
            mx = values.max()
            mn = values.min()
            mxmin = mx / mn if mn != 0 else float("inf")
            avgmin = avg / mn if mn != 0 else float("inf")
            precision = room.precision if room.precision > 3 else 3
            data_rows.append(
                [
                    "",
                    room_id,
                    room.name,
                    zone.name,
                    round(avg, precision),
                    round(mx, precision),
                    round(mn, precision),
                    round(mxmin, precision),
                    round(avgmin, precision),
                    zone.units,
                ]
            )

    if has_zones:
        rows.append(header)
        rows += data_rows
    else:
        rows.append(["", "No computed zones found."])
    rows.append([""])
    return rows


def generate_project_report(project, fname=None):
    """Generate a combined CSV report across all rooms."""
    all_rows = []
    for room_id, room in project.rooms.items():
        all_rows.append([f"=== Room: {room_id} ({room.name}) ==="])
        all_rows += _build_room_rows(room)
        all_rows.append([""])

    all_rows += _build_project_summary(project)
    all_rows += [[f"Generated {datetime.datetime.now().isoformat(timespec='seconds')}"]]

    csv_bytes = rows_to_bytes(all_rows)
    if fname is not None:
        with open(fname, "wb") as f:
            f.write(csv_bytes)
    else:
        return csv_bytes
