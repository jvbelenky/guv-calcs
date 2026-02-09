import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np
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
    Plot the image of the radiation pattern.
    Works for any plane orientation (axis-aligned or arbitrary).
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

        geom = zone.geometry

        # Get basis vectors if available
        u_hat = getattr(geom, 'u_hat', None)
        v_hat = getattr(geom, 'v_hat', None)

        # Determine axis labels and extent from geometry
        if u_hat is not None and v_hat is not None:
            # Derive axis labels from basis vectors
            def get_axis_label(vec):
                abs_vec = np.abs(vec)
                idx = int(np.argmax(abs_vec))
                if abs_vec[idx] > 0.9:
                    return ['X', 'Y', 'Z'][idx]
                return None

            u_label = get_axis_label(u_hat) or 'U'
            v_label = get_axis_label(v_hat) or 'V'

            # Use mins/maxs for extent (they're already correct 2D bounds)
            mins = geom.mins
            maxs = geom.maxs
            extent = [mins[0], maxs[0], mins[1], maxs[1]]

            # Determine if v points in positive direction of its dominant axis
            abs_v = np.abs(v_hat)
            v_idx = int(np.argmax(abs_v))
            v_positive = v_hat[v_idx] > 0
        else:
            # Fallback for geometries without basis vectors
            u_label = 'X'
            v_label = 'Y'
            extent = [geom.x1, geom.x2, geom.y1, geom.y2]
            v_positive = True

        # Transpose so rows=v, cols=u
        plot_values = values.T

        # Flip if v points in negative direction (so positive is at top)
        if not v_positive:
            plot_values = plot_values[::-1]

        img = ax.imshow(
            plot_values,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            cmap=zone.colormap,
            origin='lower',
            aspect='equal'
        )

        cbar = fig.colorbar(img, pad=0.03)
        ax.set_xlabel(u_label)
        ax.set_ylabel(v_label)
        ax.set_title(title)
        cbar.set_label(zone.units, loc="center")

    return fig, ax
