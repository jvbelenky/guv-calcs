import warnings
import pathlib
from pathlib import Path
import datetime
import json
import zipfile
import io
import csv
import plotly.io as pio
from plotly.graph_objs._figure import Figure as plotly_fig
from matplotlib.figure import Figure as mpl_fig

# -------------- Loading a room from file -------------------


def load_room(filedata):
    """
    TODO: wow this is hideous this should really just be a class method
    load a room object from json filedata
    """
    from .room import Room

    load_data = _parse_loadfile(filedata)
    saved_version = load_data["guv-calcs_version"]
    current_version = get_version(Path(__file__).parent / "_version.py")
    if saved_version != current_version:
        warnings.warn(
            f"This file was saved with guv-calcs {saved_version}, while you have {current_version} installed."
        )
    room_dict = load_data["data"]
    return Room.from_dict(room_dict)


def _parse_loadfile(filedata):
    """
    validate and parse a loadfile
    """

    if isinstance(filedata, (str, bytes or bytearray)):
        try:
            dct = json.loads(filedata)
        except json.JSONDecodeError:
            path = Path(filedata)
            dct = _load_file(path)
    elif isinstance(filedata, pathlib.PurePath):
        dct = _load_file(filedata)

    return dct


def _load_file(path):
    """load json from a path"""
    if path.is_file():
        if path.suffix.lower() != ".guv":
            raise ValueError("Please provide a valid .guv file")
        with open(path, "r") as json_file:
            try:
                dct = json.load(json_file)
            except json.JSONDecodeError:
                raise ValueError(".guv file is malformed")
    else:
        raise FileNotFoundError("Please provide a valid .guv file")
    return dct


def save_room(room, fname):
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
    for zone_id, zone in room.scene.calc_zones.items():
        if zone.calctype != "Zone":
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
                    data_dict[zone.name + ".png"] = fig_to_bytes(fig)
                elif zone.calctype == "Volume":
                    fig = zone.plot_volume()
                    data_dict[zone.name + ".png"] = fig_to_bytes(fig)

    # save lamp files if indicated to
    for lamp_id, lamp in room.scene.lamps.items():
        if lamp.filedata is not None:
            if include_lamp_files:
                data_dict[lamp.name + ".ies"] = lamp.save_ies()
            if include_lamp_plots:
                ies_fig, ax = lamp.plot_ies(title=lamp.name)
                data_dict[lamp.name + "_ies.png"] = fig_to_bytes(ies_fig)
        if lamp.spectra is not None:
            if include_lamp_plots:
                linfig, _ = lamp.spectra.plot(
                    title=lamp.name, yscale="linear", weights=True, label=True
                )
                logfig, _ = lamp.spectra.plot(
                    title=lamp.name, yscale="log", weights=True, label=True
                )
                linkey = lamp.name + "_spectra_linear.png"
                logkey = lamp.name + "_spectra_log.png"
                data_dict[linkey] = fig_to_bytes(linfig)
                data_dict[logkey] = fig_to_bytes(logfig)

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


def generate_report(self, fname=None):
    """
    Dump a one file CSV with a snapshot of the current room.
    Sections are separated by blank lines so Excel shows each
    header group clearly.
    """
    precision = self.precision if self.precision > 3 else 3
    fmt = lambda v: round(v, precision) if isinstance(v, (int, float)) else v
    # ───  Room parameters  ───────────────────────────────
    rows = [["Room Parameters"]]
    rows += [["", "Dimensions", "x", "y", "z", "units"]]
    d = self.dim
    rows += [["", "", fmt(d.x), fmt(d.y), fmt(d.z), d.units]]
    vol_units = "ft 3" if self.units == "feet" else "m 3"
    rows += [["", "Volume", fmt(self.volume), vol_units]]
    rows += [[""]]

    # ───  Reflectance  ──────────────────────────────────
    rows += [["", "Reflectance"]]
    rows += [["", "", "Floor", "Ceiling", "North", "South", "East", "West", "Enabled"]]
    rows += [["", "", *self.ref_manager.reflectances.values(), self.enable_reflectance]]
    rows += [[""]]

    # ───  Luminaires  ───────────────────────────────────
    if self.scene.lamps:
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
                "Fixture Depth",
            ]
        ]
        for lamp in self.scene.lamps.values():
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
                    fmt(lamp.surface.depth),
                ]
            ]
        rows += [[""]]

    # ───  Calculation planes  ───────────────────────────
    planes = [z for z in self.scene.calc_zones.values() if z.calctype == "Plane"]
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

    # ───  Calculation volumes  ──────────────────────────
    vols = [z for z in self.scene.calc_zones.values() if z.calctype == "Volume"]
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

    # ───  Statistics  ──────────────────────────
    zones = [z for z in self.scene.calc_zones.values() if z.calctype != "Zone"]
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
            mxmin = mx / mn
            avgmin = avg / mn
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

    # footer
    rows += [[f"Generated {datetime.datetime.now().isoformat(timespec='seconds')}"]]
    csv_bytes = rows_to_bytes(rows)

    if fname is not None:
        with open(fname, "wb") as csvfile:
            csvfile.write(csv_bytes)
    else:
        return csv_bytes


# ------- Conversions to bytes ---------------


def fig_to_bytes(fig):
    if isinstance(fig, mpl_fig):
        buf = io.BytesIO()
        fig.savefig(
            buf, format="png"
        )  # You can change the format as needed (e.g., 'jpeg', 'pdf')
        buf.seek(0)  # Rewind the buffer
        byt = buf.getvalue()
    elif isinstance(fig, plotly_fig):
        byt = pio.to_image(fig, format="png", scale=1)
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
