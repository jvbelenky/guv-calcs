import matplotlib.pyplot as plt
import plotly.graph_objs as go
import warnings


def plot_volume(zone, title=None):
    """
    Plot the fluence values as an isosurface using Plotly.
    """

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
                colorscale=zone.colormap,
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


def plot_plane(zone, fig=None, ax=None, vmin=None, vmax=None, title=None):
    """
    TODO: extent will not work correctly for non-axis-aligned planes
    Plot the image of the radiation pattern
    """
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
        vmin = values.min() if vmin is None else vmin
        vmax = values.max() if vmax is None else vmax
        extent = [
            zone.geometry.x1,
            zone.geometry.x2,
            zone.geometry.y1,
            zone.geometry.y2,
        ]

        ref_surface = getattr(zone.geometry, 'ref_surface', None)
        direction = getattr(zone.geometry, 'direction', 1)
        values = values.T
        # XZ planes with direction=1 have v pointing in -Z, so don't flip
        # Other planes need the flip to orient correctly
        if not (ref_surface == 'xz' and direction == 1):
            values = values[::-1]
        img = ax.imshow(values, extent=extent, vmin=vmin, vmax=vmax, cmap=zone.colormap)
        cbar = fig.colorbar(img, pad=0.03)
        ax.set_title(title)
        cbar.set_label(zone.units, loc="center")
    return fig, ax
