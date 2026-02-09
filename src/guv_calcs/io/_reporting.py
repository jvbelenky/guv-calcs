"""CSV report generation for rooms and projects."""

import datetime

from ._export import rows_to_bytes


def _zone_stats(zone, precision):
    """Compute avg/max/min/ratios for a calculation zone."""
    values = zone.get_values()
    avg = values.mean()
    mx = values.max()
    mn = values.min()
    mxmin = mx / mn if mn != 0 else float("inf")
    avgmin = avg / mn if mn != 0 else float("inf")
    return (
        round(avg, precision),
        round(mx, precision),
        round(mn, precision),
        round(mxmin, precision),
        round(avgmin, precision),
    )


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
