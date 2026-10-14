import numpy as np
from ..io import rows_to_bytes


def export_plane(zone, fname=None):

    geom = zone.geometry
    num_x, num_y = geom.num_x, geom.num_y
    # polygon planes are written on their full bounding grid, with the cells
    # outside the outline left blank
    coords = geom.coords_full

    values = zone.get_values()
    if values is None:
        vals = [[-1] * num_y for _ in range(num_x)]
    elif not geom.is_rectangular and np.size(values) == geom.total_points:
        vals = geom.values_to_full_grid(values, fill=np.nan).reshape(num_x, num_y)
    elif values.shape != (num_x, num_y):
        vals = [[-1] * num_y for _ in range(num_x)]
    else:
        vals = values
    zvals = coords.T[2].reshape(num_x, num_y).T[::-1]

    xpoints = coords.T[0].reshape(num_x, num_y).T[0].tolist()
    ypoints = coords.T[1].reshape(num_x, num_y)[0].tolist()

    if len(np.unique(xpoints)) == 1 and len(np.unique(ypoints)) == 1:
        xpoints = coords.T[0].reshape(num_x, num_y)[0].tolist()
        ypoints = coords.T[1].reshape(num_x, num_y).T[0].tolist()
        vals = np.array(vals).T.tolist()
        zvals = zvals.T.tolist()

    rows = [[""] + xpoints]

    rows += _blank_nans(np.concatenate(([np.flip(ypoints)], vals)).T.tolist())
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

    fluence rate values are given in uW/cm2
    """

    geom = zone.geometry
    shape = (geom.num_x, geom.num_y, geom.num_z)
    lines = header.split("\n")
    rows = [[line] for line in lines]
    rows += [shape]
    # polygon volumes are written on their full bounding grid, with the
    # points outside the outline left blank
    rows += [np.unique(geom.coords_full.T[0])]
    rows += [np.unique(geom.coords_full.T[1])]
    rows += [np.unique(geom.coords_full.T[2])]
    values = zone.get_values()
    if (values is not None and not geom.is_rectangular
            and np.size(values) == geom.total_points):
        values = geom.values_to_full_grid(values, fill=np.nan).reshape(shape)
    for i in range(geom.num_z):
        rows += [""]
        if values is None:
            rows += [[""] * geom.num_x for _ in range(geom.num_y)]
        elif values.shape != shape:
            rows += [[""] * geom.num_x for _ in range(geom.num_y)]
        else:
            rows += _blank_nans(values.T[i].tolist())

    return to_csv(rows=rows, fname=fname)


def export_point(zone, fname=None):
    pos = zone.geometry.position
    aim = zone.geometry.aim_point
    normal = zone.geometry.normal_direction
    values = zone.get_values()
    val = "" if values is None else values.flat[0]
    rows = [
        ["Zone ID", zone.zone_id],
        ["Name", zone.name],
        ["Position", *pos],
        ["Aim Point", *aim],
        ["Normal", *normal],
        ["Value", val],
        ["Units", zone.value_units],
    ]
    return to_csv(rows=rows, fname=fname)


def _blank_nans(rows):
    """Write NaN cells (outside a polygon's outline) as empty strings."""
    return [["" if isinstance(v, float) and v != v else v for v in row] for row in rows]


def to_csv(rows, fname=None):

    # rows = zone._write_rows()  # implemented in subclass
    csv_bytes = rows_to_bytes(rows)

    if fname is not None:
        with open(fname, "wb") as csvfile:
            csvfile.write(csv_bytes)
    return csv_bytes
