import numpy as np
from .io import rows_to_bytes


def export_plane(zone, fname=None):

    num_x, num_y = zone.geometry.num_x, zone.geometry.num_y  # tmp

    values = zone.get_values()
    if values is None:
        vals = [[-1] * num_y] * num_x
    elif values.shape != (num_x, num_y):
        vals = [[-1] * num_y] * num_x
    else:
        vals = values
    zvals = zone.geometry.coords.T[2].reshape(num_x, num_y).T[::-1]

    xpoints = zone.geometry.coords.T[0].reshape(num_x, num_y).T[0].tolist()
    ypoints = zone.geometry.coords.T[1].reshape(num_x, num_y)[0].tolist()

    if len(np.unique(xpoints)) == 1 and len(np.unique(ypoints)) == 1:
        xpoints = zone.geometry.coords.T[0].reshape(num_x, num_y)[0].tolist()
        ypoints = zone.geometry.coords.T[1].reshape(num_x, num_y).T[0].tolist()
        vals = np.array(vals).T.tolist()
        zvals = zvals.T.tolist()

    rows = [[""] + xpoints]

    rows += np.concatenate(([np.flip(ypoints)], vals)).T.tolist()
    rows += [""]
    # zvals

    rows += [[""] + list(line) for line in zvals]
    return to_csv(rows=rows, fname=fname)


def export_volume(zone, fname=None):

    header = """Data format notes:
        
    Data consists of numZ horizontal grids of fluence rate values; each grid contains numX by numY points.

    numX; numY; numZ are given on the first line of data.
    The next line contains numX values; indicating the X-coordinate of each grid column.
    The next line contains numY values; indicating the Y-coordinate of each grid row.
    The next line contains numZ values; indicating the Z-coordinate of each horizontal grid.
    A blank line separates the position data from the first horizontal grid of fluence rate values.
    A blank line separates each subsequent horizontal grid of fluence rate values.

    fluence rate values are given in µW/cm²
    """

    lines = header.split("\n")
    rows = [[line] for line in lines]
    rows += [zone.geometry.num_points]
    rows += zone.geometry.points
    values = zone.get_values()
    for i in range(zone.geometry.num_z):
        rows += [""]
        if values is None:
            rows += [[""] * zone.geometry.num_x] * zone.geometry.num_y
        elif values.shape != (
            zone.geometry.num_x,
            zone.geometry.num_y,
            zone.geometry.num_z,
        ):
            rows += [[""] * zone.geometry.num_x] * zone.geometry.num_y
        else:
            rows += values.T[i].tolist()

    return to_csv(rows=rows, fname=fname)


def to_csv(rows, fname=None):

    # rows = zone._write_rows()  # implemented in subclass
    csv_bytes = rows_to_bytes(rows)

    if fname is not None:
        with open(fname, "wb") as csvfile:
            csvfile.write(csv_bytes)
    return csv_bytes
