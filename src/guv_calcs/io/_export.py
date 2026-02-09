"""Zip export and figure/CSV byte-conversion utilities."""

import warnings
import io
import zipfile
import csv
import functools

import plotly.io as pio
from plotly.graph_objs._figure import Figure as plotly_fig
from matplotlib.figure import Figure as mpl_fig


def export_room_zip(
    room,
    fname=None,
    include_plots=False,
    include_lamp_files=False,
    include_lamp_plots=False,
):
    """Write the room project file and all results files to a zip file.

    Optionally include extra files like lamp ies files, spectra files, and plots.
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


@functools.lru_cache(maxsize=1)
def _warn_chrome_missing(error_msg):
    """Issue a one-time warning about missing Chrome for kaleido."""
    warnings.warn(
        "Could not export plotly figure to image. "
        "This requires Chrome/Chromium to be installed for kaleido. "
        f"Error: {error_msg}"
    )


def fig_to_bytes(fig):
    """Convert a matplotlib or plotly figure to PNG bytes.

    For plotly figures, this requires Chrome to be installed (used by kaleido).
    If Chrome is not available, returns None and issues a warning.
    """
    if isinstance(fig, mpl_fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)  # Rewind the buffer
        byt = buf.getvalue()
    elif isinstance(fig, plotly_fig):
        try:
            byt = pio.to_image(fig, format="png", scale=1)
        except Exception as e:
            # kaleido v1 requires Chrome to be installed
            _warn_chrome_missing(str(e))
            return None
    else:
        raise TypeError("This figure type cannot be converted to bytes")
    return byt


def rows_to_bytes(rows, encoding="cp1252"):
    """Convert a list of row lists to CSV bytes."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerows(rows)

    # Get the CSV data from buffer, convert to bytes
    csv_data = buffer.getvalue()
    csv_bytes = csv_data.encode(encoding)  # encode to bytes
    return csv_bytes
