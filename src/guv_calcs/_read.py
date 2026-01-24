from .calc_zone import CalcVol, CalcPlane
from pathlib import Path
import numpy as np


def read_fluence(lines):
    points = []
    values = []
    dose = False
    for i, line in enumerate(lines):
        if "dose" in line:
            dose = True
        if i in [15, 16, 17]:
            lst = line.split("\n")[0].split(",")
            vals = [float(val) for val in lst]
            points.append(np.array(vals))
        elif i > 17:
            try:
                lst = line.split("\n")[0].split(",")
                vals = np.array([float(val) for val in lst])
                values.append(vals)
            except ValueError:
                # these are just blank lines, skip and move on
                continue
    xp, yp, zp = points
    arr = np.array(values)
    arr = arr.reshape(len(zp), len(yp), len(xp)).transpose(2, 1, 0)

    X, Y, Z = [grid.reshape(-1) for grid in np.meshgrid(*points, indexing="ij")]
    coords = np.array((X, Y, Z)).T
    return arr, coords, points, dose


def _get_coords(xpoints, ypoints, height):
    X, Y = [grid.reshape(-1) for grid in np.meshgrid(xpoints, ypoints, indexing="ij")]
    Z = np.full(X.shape, height)
    return X, Y, Z


def read_irradiance(lines):
    # if there are no blank lines in the file will throw a StopIteration error,
    split_idx = next(i for i, line in enumerate(lines) if line.strip() == "")
    xp = list(map(float, lines[0].lstrip(" ,").rstrip("\n").split(",")))
    values = []
    yp = []
    for line in lines[1:split_idx]:
        row = list(map(float, line.rstrip("\n").split(",")))
        yp.append(row[0])
        values.append(row[1:])
    values2 = []
    for line in lines[split_idx + 1 :]:
        row = list(map(float, line.lstrip(" ,").rstrip("\n").split(",")))
        values2.append(row)
    xp = np.array(xp)
    yp = np.array(yp)
    values = np.array(values).T
    values2 = np.array(values2)

    if len(np.unique(values2.reshape(-1))) == 1:
        ref_surface = "xy"
        xpoints = xp
        ypoints = yp[::-1]
        height = np.unique(values2)[0]
        X, Y, Z = _get_coords(xpoints, ypoints, height)
        coords = np.stack([X, Y, Z], axis=-1)
    elif len(np.unique(yp)) == 1:
        ref_surface = "xz"
        xpoints = xp
        ypoints = values2.T[0][::-1]
        height = np.unique(yp)[0]
        X, Y, Z = _get_coords(xpoints, ypoints, height)
        coords = np.stack([X, Z, Y], axis=-1)
    elif len(np.unique(xp)) == 1:
        ref_surface = "yz"
        xpoints = yp[::-1]
        ypoints = values2.T[0][::-1]
        height = np.unique(xp)[0]
        X, Y, Z = _get_coords(xpoints, ypoints, height)
        coords = np.stack([Z, Y, X], axis=-1)
    points = [xpoints, ypoints]

    return values, coords, points, height, ref_surface


def read_export_file(file_path):
    """
    read a fluence export csv and return the array of fluence
    measurements, as well as the real x y and z coordinates of those measurements
    """
    with open(file_path, "r", encoding="cp1252") as f:
        lines = f.readlines()

    if "Data format notes" in lines[0]:
        values, coords, points, dose = read_fluence(lines)
    else:
        values, coords, points, height, ref_surface = read_irradiance(lines)
    return values, coords


def _get_spacing(pts):
    spacings = []
    for i in range(1, len(pts)):
        spacings.append(round(pts[i] - pts[i - 1], 5))
    return max(set(spacings), key=spacings.count)


def file_to_zone(file_path):
    """
    read a previously exported file into a calculation zone object
    RISKY!!! use at own risk, much data will be missing
    """
    path = Path(file_path)
    with open(path, "r", encoding="cp1252") as f:
        lines = f.readlines()
    if "Data format notes" in lines[0]:
        values, coords, points, dose = read_fluence(lines)
        xp, yp, zp = points
        x_spacing = _get_spacing(xp)
        y_spacing = _get_spacing(yp)
        z_spacing = _get_spacing(zp)
        x1, x2 = xp[0], xp[-1]
        y1, y2 = yp[0], yp[-1]
        z1, z2 = zp[0], zp[-1]
        zone = CalcVol(
            zone_id=path.stem,
            x1=x1,
            x2=x2,
            y1=y1,
            y2=y2,
            z1=z1,
            z2=z2,
            x_spacing=x_spacing,
            y_spacing=y_spacing,
            z_spacing=z_spacing,
            dose=dose,
            offset=False,
        )
        zone.values = values
        zone.coords = coords
    else:
        values, coords, points, height, ref_surface = read_irradiance(lines)
        xp, yp = points
        x_spacing = _get_spacing(xp)
        y_spacing = _get_spacing(yp)
        x1, x2 = xp[0], xp[-1]
        y1, y2 = yp[0], yp[-1]
        try:
            zone = CalcPlane(
                zone_id=path.stem,
                x1=x1,
                x2=x2,
                y1=y1,
                y2=y2,
                height=height,
                x_spacing=x_spacing,
                y_spacing=y_spacing,
                ref_surface=ref_surface,
                offset=False,
            )
            zone.values = values
            zone.coords = coords
        except ValueError:
            if ref_surface == "yz":
                msg = "Calculation planes derived from Acuity Visual with the `yz` reference plane are bugged and cannot be read."
                raise ValueError(msg)
            else:
                raise
    return zone
