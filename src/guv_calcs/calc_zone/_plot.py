import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objs as go
import numpy as np
from scipy.ndimage import distance_transform_edt
import warnings
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers '3d' projection


def plot_volume(zone, title=None, colormap=None):
    """
    Plot the fluence values as an isosurface using Plotly.

    Args:
        zone: CalcVol instance
        title: Optional plot title
        colormap: Plotly colorscale name (defaults to "plasma")
    """
    if colormap is None:
        colormap = "plasma"

    fig = go.Figure()

    if zone.values is None:
        warnings.warn("No values calculated for this volume.")
    else:
        values = zone.get_values().flatten()
        isomin = values.mean() / 2

        fig.add_trace(
            go.Isosurface(
                x=zone.coords.T[0],
                y=zone.coords.T[1],
                z=zone.coords.T[2],
                value=values,
                isomin=isomin,
                surface_count=3,
                opacity=0.25,
                showscale=False,
                colorbar=None,
                colorscale=colormap,
                caps=dict(x_show=False, y_show=False, z_show=False),
                name=zone.name + " Values",
            )
        )
    fig.update_layout(
        title=dict(
            text=zone.name if title is None else title,
            x=0.5,  # center horizontally
            y=0.85,  # lower this value to move the title down (default is 0.95)
            xanchor="center",
            yanchor="top",
            font=dict(size=18),
        ),
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"
        ),
        height=450,
    )
    fig.update_scenes(camera_projection_type="orthographic")
    return fig


def plot_plane(
    zone,
    fig=None,
    ax=None,
    vmin=None,
    vmax=None,
    title=None,
    colormap=None,
    levels=None,
    colors=None,
    labels=None,
    filled=True,
    contour_labels=True,
    show_values=False,
    decimals=2,
    floor_color="#eeeeee",
):
    """
    Plot the image of the radiation pattern.
    Works for any plane orientation (axis-aligned or arbitrary).

    Args:
        zone: CalcPlane instance
        colormap: Matplotlib colormap name (defaults to "plasma")
        levels: contour levels; given, the plane is drawn as contours (filled
            bands between ascending levels) instead of a heatmap
        colors: one color per level (defaults to evenly spaced along the colormap)
        labels: one name per level, drawn on its line and in the colorbar in
            place of the value (None or "" keeps the value)
        filled: fill the bands between levels; otherwise draw lines only
        contour_labels: label each line
        show_values: print each cell's value where it fits
        decimals: decimals for values, and for levels without a name
        floor_color: color below the lowest level
    """
    if colormap is None:
        colormap = "plasma"
    if fig is None:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()
    else:
        if ax is None:
            ax = fig.axes[0]

    title = "" if title is None else title
    values = zone.get_values()

    if values is not None:
        geom = zone.geometry
        if not geom.is_rectangular:
            # polygon planes: plot the full bounding grid, NaN (blank) outside
            values = geom.values_to_full_grid(values, fill=np.nan).reshape(
                geom.num_x, geom.num_y)
        values = np.asarray(values, dtype=float)

        u_label, v_label, extent, v_positive = _plane_axes(geom)

        # Transpose so rows=v, cols=u
        plot_values = values.T

        # Flip if v points in negative direction (so positive is at top)
        if not v_positive:
            plot_values = plot_values[::-1]

        if levels is not None and len(levels) > 0:
            _draw_contours(
                fig, ax, plot_values, extent, levels, colors, labels, filled,
                contour_labels, decimals, floor_color, colormap, zone.value_units,
            )
        else:
            vmin = np.nanmin(values) if vmin is None else vmin
            vmax = np.nanmax(values) if vmax is None else vmax
            img = ax.imshow(
                plot_values,
                extent=extent,
                vmin=vmin,
                vmax=vmax,
                cmap=colormap,
                origin='lower',
                aspect='equal'
            )
            cbar = fig.colorbar(img, pad=0.03)
            cbar.set_label(zone.value_units, loc="center")

        if show_values:
            _draw_cell_values(fig, ax, plot_values, extent, decimals, levels, colors, colormap, vmin, vmax)

        ax.set_xlabel(u_label)
        ax.set_ylabel(v_label)
        ax.set_title(title)

    return fig, ax


def _plane_axes(geom):
    """Axis labels, imshow extent and v direction of a plane's grid."""
    u_hat = getattr(geom, 'u_hat', None)
    v_hat = getattr(geom, 'v_hat', None)
    if u_hat is None or v_hat is None:
        # Fallback for geometries without basis vectors
        return 'X', 'Y', [geom.x1, geom.x2, geom.y1, geom.y2], True

    def get_axis_label(vec):
        abs_vec = np.abs(vec)
        idx = int(np.argmax(abs_vec))
        if abs_vec[idx] > 0.9:
            return ['X', 'Y', 'Z'][idx]
        return None

    # mins/maxs are already the correct 2D bounds
    extent = [geom.mins[0], geom.maxs[0], geom.mins[1], geom.maxs[1]]
    # Whether v points in the positive direction of its dominant axis
    v_idx = int(np.argmax(np.abs(v_hat)))
    return get_axis_label(u_hat) or 'U', get_axis_label(v_hat) or 'V', extent, bool(v_hat[v_idx] > 0)


def _cell_centers(extent, shape):
    """Coordinates of the cell centres of an imshow image (rows=v from the bottom)."""
    nv, nu = shape
    du = (extent[1] - extent[0]) / nu
    dv = (extent[3] - extent[2]) / nv
    return (extent[0] + (np.arange(nu) + 0.5) * du, extent[2] + (np.arange(nv) + 0.5) * dv)


def _holes_filled(values):
    """`values` with each NaN given its nearest valid cell's value.

    Contouring the filled grid runs each level's line up to a hole (outside a
    polygon room, inside an obstacle) instead of stopping half a cell short of
    it; the holes are painted back on top afterwards.
    """
    holes = np.isnan(values)
    if not holes.any() or holes.all():
        return values
    idx = distance_transform_edt(holes, return_distances=False, return_indices=True)
    return values[tuple(idx)]


def _traced_values(values):
    """The grid to contour, and the unlit cells to paint back over it.

    Unlit cells (exactly zero: inside an obstacle, or in its shadow) sit a
    cliff away from their lit neighbours; interpolating across it would crowd
    every band into that one cell, so they're traced as holes, like NaN ones.
    """
    unlit = values == 0
    traced = _holes_filled(np.where(unlit, np.nan, values))
    if np.isnan(traced).all():
        return _holes_filled(values), np.zeros_like(unlit)
    return traced, unlit


def _level_colors(levels, colors, colormap):
    """One color per level: those given, else evenly spaced along the colormap."""
    n = len(levels)
    cmap = plt.get_cmap(colormap)
    defaults = [mcolors.to_hex(cmap(0.5 if n == 1 else i / (n - 1))) for i in range(n)]
    if colors is None:
        return defaults
    return [c if c else defaults[i] for i, c in enumerate(list(colors)[:n])] + defaults[len(colors):]


def _level_text(level, label, decimals):
    return label if label else f"{level:.{decimals}f}"


def _draw_contours(fig, ax, values, extent, levels, colors, labels, filled,
                   contour_labels, decimals, floor_color, colormap, units):
    """Filled bands (or lines) at `levels`, holes blank, with a banded colorbar."""
    order = np.argsort(levels)
    levels = [float(levels[i]) for i in order]
    colors = _level_colors(levels, None if colors is None else [list(colors)[i] if i < len(colors) else None for i in order], colormap)
    labels = [None if labels is None or i >= len(labels) else labels[i] for i in order]

    holes = np.isnan(values)
    traced, unlit = _traced_values(values)
    x, y = _cell_centers(extent, values.shape)

    # Below the lowest level, inside the room
    ax.imshow(np.ma.masked_array(np.zeros_like(values), holes), extent=extent, origin="lower",
              cmap=mcolors.ListedColormap([floor_color]), aspect="equal", interpolation="nearest")
    top = max(np.nanmax(traced), levels[-1]) * 1.001 + 1e-12
    if filled:
        ax.contourf(x, y, traced, levels=levels + [top], colors=colors, extend="neither")
    lines = ax.contour(
        x, y, traced, levels=levels,
        colors=[_darker(c) for c in colors] if filled else colors,
        linewidths=0.8 if filled else 1.5,
    )
    # Paint the holes back over the bands and lines (an image otherwise draws beneath them)
    for cells, color in ((unlit, floor_color), (holes, "white")):
        if cells.any():
            ax.imshow(np.ma.masked_array(np.zeros_like(values), ~cells), extent=extent, origin="lower",
                      cmap=mcolors.ListedColormap([color]), aspect="equal", interpolation="nearest", zorder=2.5)
    if contour_labels:
        fmt = {lvl: _level_text(lvl, lab, decimals) for lvl, lab in zip(levels, labels)}
        texts = ax.clabel(lines, levels=[lvl for lvl in lines.levels if lvl in fmt], fmt=fmt,
                          fontsize=8, inline=True, inline_spacing=8, colors="black")
        # Level and on a light backing, like the app's; none clipped at the plot's edge
        margin_x = (extent[1] - extent[0]) * 0.06
        margin_y = (extent[3] - extent[2]) * 0.04
        for text in texts:
            tx, ty = text.get_position()
            if not (extent[0] + margin_x < tx < extent[1] - margin_x and extent[2] + margin_y < ty < extent[3] - margin_y):
                text.remove()
                continue
            text.set_rotation(0)
            text.set_bbox(dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # Colorbar: a floor band, then one band per level, labelled at the level
    bounds = list(range(len(levels) + 2))
    norm = mcolors.BoundaryNorm(bounds, len(levels) + 1)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=mcolors.ListedColormap([floor_color] + colors))
    cbar = fig.colorbar(mappable, ax=ax, pad=0.03, boundaries=bounds, ticks=bounds[1:-1])
    cbar.ax.set_yticklabels([_level_text(lvl, lab, decimals) for lvl, lab in zip(levels, labels)])
    cbar.set_label(units, loc="center")


def _darker(color):
    r, g, b = mcolors.to_rgb(color)
    return (r * 0.55, g * 0.55, b * 0.55)


def _draw_cell_values(fig, ax, values, extent, decimals, levels, colors, colormap, vmin, vmax):
    """Each cell's value, in black or white against its color, sized to fit the cell."""
    nv, nu = values.shape
    bbox = ax.get_position()
    width_in, height_in = fig.get_size_inches()
    cell_w = bbox.width * width_in * 72 / nu
    cell_h = bbox.height * height_in * 72 / nv
    widest = max((len(f"{v:.{decimals}f}") for v in values[np.isfinite(values)]), default=1)
    size = min(cell_w / (widest * 0.62), cell_h * 0.7, 9)
    if size < 3:
        return
    x, y = _cell_centers(extent, values.shape)
    if levels is not None and len(levels):
        order = np.argsort(levels)
        lv = [float(levels[i]) for i in order]
        lc = _level_colors(lv, None if colors is None else [list(colors)[i] if i < len(colors) else None for i in order], colormap)
    else:
        cmap = plt.get_cmap(colormap)
        lo = np.nanmin(values) if vmin is None else vmin
        hi = np.nanmax(values) if vmax is None else vmax
    for j in range(nv):
        for i in range(nu):
            v = values[j, i]
            if not np.isfinite(v):
                continue
            if levels is not None and len(levels):
                band = np.searchsorted(lv, v, side="right") - 1
                rgb = mcolors.to_rgb(lc[band]) if band >= 0 else (0.93, 0.93, 0.93)
            else:
                rgb = cmap((v - lo) / ((hi - lo) or 1))[:3]
            dark = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] < 0.5
            ax.text(x[i], y[j], f"{v:.{decimals}f}", ha="center", va="center", fontsize=size,
                    color="white" if dark else "black")


def plot_point(zone, fig=None, ax=None, title=None, arrow_length=0.5, colormap=None):
    """Plot a CalcPoint as a 3D scatter marker with a normal arrow."""
    if colormap is None:
        colormap = "plasma"

    if fig is None and ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    elif fig is None:
        fig = plt.gcf()
    elif ax is None:
        ax = fig.add_subplot(111, projection="3d")

    pos = zone.geometry.position
    normal = zone.geometry.normal_direction
    values = zone.get_values()

    # scatter the point
    if values is not None:
        cmap = plt.get_cmap(colormap)
        color = cmap(0.5)
    else:
        color = "gray"
    ax.scatter(*pos, c=[color], s=80, depthshade=False, zorder=5)

    # normal arrow
    ax.quiver(
        pos[0], pos[1], pos[2],
        normal[0] * arrow_length,
        normal[1] * arrow_length,
        normal[2] * arrow_length,
        color="red", arrow_length_ratio=0.2,
    )

    # expand axes to include arrow tip with padding
    tip = [pos[i] + normal[i] * arrow_length for i in range(3)]
    margin = arrow_length * 0.3
    for i, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        lo = min(pos[i], tip[i]) - margin
        hi = max(pos[i], tip[i]) + margin
        setter(lo, hi)

    # annotate value
    if values is not None:
        label = f"{values.flat[0]:.2f} {zone.value_units}"
    else:
        label = "no value"
    ax.text(pos[0], pos[1], pos[2], f"  {label}", fontsize=8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(zone.name if title is None else title)

    return fig, ax
