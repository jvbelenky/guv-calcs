"""CSV report generation for rooms and projects."""

import datetime
import numbers

from ._export import rows_to_bytes


def _zone_stats(zone, precision):
    """Compute avg/max/min/ratios for a calculation zone."""
    s = zone.get_statistics()
    return (
        round(s["mean"], precision),
        round(s["max"], precision),
        round(s["min"], precision),
        round(s["max_min"], precision),
        round(s["avg_min"], precision),
    )


def _build_room_rows(room):
    """Build the row data for a single room's report."""
    precision = room.precision if room.precision > 3 else 3

    def fmt(v):
        return round(v, precision) if isinstance(v, numbers.Real) else v

    # ───  Room parameters  ───────────────────────────────
    rows = [["Room Parameters"]]
    rows += [["", "Dimensions", "x", "y", "z", "units"]]
    d = room.dim
    rows += [["", "", fmt(d.x), fmt(d.y), fmt(d.z), d.units]]
    # x/y above are bounding-box extents; a floor plan that does not start at
    # the origin (a traced room, or Room(..., origin=)) also reports where it sits
    x_min, y_min, _, _ = d.polygon.bounding_box
    if x_min != 0 or y_min != 0:
        rows += [["", "Origin", "x", "y"]]
        rows += [["", "", fmt(x_min), fmt(y_min)]]
    if d.is_polygon:
        rows += [["", "Floor Plan", "vertex", "x", "y"]]
        for i, (vx, vy) in enumerate(d.polygon.vertices):
            rows += [["", "", i, fmt(vx), fmt(vy)]]
    abbr = d.units.abbreviation
    area_units = f"{abbr} 2"
    vol_units = f"{abbr} 3"
    rows += [["", "Floor area", fmt(d.polygon.area), area_units]]
    rows += [["", "Volume", fmt(room.volume), vol_units]]
    rows += [[""]]

    # ───  Reflectance  ──────────────────────────────────
    rows += [["", "Reflectance"]]
    labels = [k.replace("_", " ").title() for k in room.surfaces]
    rows += [["", "", *labels, "Enabled"]]
    rows += [
        ["", "", *[v.R for v in room.surfaces.values()], room.ref_manager.enabled]
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

    # ───  Objects (obstacles)  ─────────────────────────
    if room.objects:
        rows += [["Objects"]]
        rows += [["", "", "", "", "Size", "", "", "Base centre", "", "", "Rotation"]]
        rows += [
            [
                "",
                "ID",
                "Name",
                "Shape",
                "Width",
                "Length",
                "Height",
                "x",
                "y",
                "z",
                "Yaw",
                "Pitch",
                "Roll",
                "Reflectance",
                "Transmittance",
                "Enabled",
            ]
        ]
        for obj in room.objects.values():
            data = obj.to_dict()
            shape = data["shape"]
            rows += [
                [
                    "",
                    obj.id,
                    obj.name,
                    shape["type"],
                    fmt(obj.width),
                    fmt(obj.length),
                    fmt(obj.height),
                    fmt(obj.x),
                    fmt(obj.y),
                    fmt(obj.z),
                    fmt(data["yaw"]),
                    fmt(data["pitch"]),
                    fmt(data["roll"]),
                    obj.R,
                    obj.T,
                    obj.enabled,
                ]
            ]
            if shape["type"] == "extrusion":
                rows += [["", "", "Footprint", "vertex", "x", "y"]]
                for i, (vx, vy) in enumerate(shape["polygon"]["vertices"]):
                    rows += [["", "", "", i, fmt(vx), fmt(vy)]]
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
                "Exposure Time",
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
                    pl.exposure_time if pl.dose else "",
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
                "Exposure Time",
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
                    v.exposure_time if v.dose else "",
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
            avg, mx, mn, mxmin, avgmin = _zone_stats(zone, precision)
            rows += [
                [
                    "",
                    zone.name,
                    avg,
                    mx,
                    mn,
                    mxmin,
                    avgmin,
                    zone.value_units,
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
        precision = room.precision if room.precision > 3 else 3
        for zone in zones:
            has_zones = True
            avg, mx, mn, mxmin, avgmin = _zone_stats(zone, precision)
            data_rows.append(
                [
                    "",
                    room_id,
                    room.name,
                    zone.name,
                    avg,
                    mx,
                    mn,
                    mxmin,
                    avgmin,
                    zone.value_units,
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
