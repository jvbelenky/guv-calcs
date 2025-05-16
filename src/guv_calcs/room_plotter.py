import plotly.graph_objs as go
import numpy as np
from scipy.spatial import Delaunay
from .trigonometry import to_polar
from .calc_zone import CalcPlane, CalcVol
from .units import convert_units


class RoomPlotter:
    def __init__(self, room):
        self.room = room

    def plotly(
        self,
        fig=None,
        select_id=None,
        title="",
    ):
        """plot all"""
        if fig is None:
            fig = go.Figure()

        # first delete any extraneous traces
        lamp_ids = list(self.room.lamps.keys())
        aim_ids = [lampid + "_aim" for lampid in lamp_ids]
        zone_ids = list(self.room.calc_zones.keys())
        for active_ids in [lamp_ids, aim_ids, zone_ids]:
            self._remove_traces_by_ids(fig, active_ids)

        # plot lamps
        for lamp_id, lamp in self.room.lamps.items():
            if lamp.filedata is not None:
                fig = self._plot_lamp(lamp=lamp, fig=fig, select_id=select_id)
        for zone_id, zone in self.room.calc_zones.items():
            if isinstance(zone, CalcPlane):
                if zone.show_values and zone.values.sum() > 0:
                    fig = self._plot_plane_values(zone=zone, fig=fig)
                else:
                    fig = self._plot_plane(zone=zone, fig=fig, select_id=select_id)
            elif isinstance(zone, CalcVol):
                fig = self._plot_vol(zone=zone, fig=fig, select_id=select_id)

        x, y, z = self.room.dim.x, self.room.dim.y, self.room.dim.z

        # set views
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(range=[0, x]),
                yaxis=dict(range=[0, y]),
                zaxis=dict(range=[0, z]),
                aspectratio=dict(
                    x=x / z,
                    y=y / z,
                    z=1,
                ),
            ),
            height=750,
            autosize=False,
            margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=0),
            legend=dict(
                x=0,
                y=1,
                yanchor="top",
                xanchor="left",
            ),
        )
        fig.add_annotation(
            text=f"Units: {self.room.units}",
            xref="paper",
            yref="paper",
            x=0,
            y=0,
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=12, color="gray"),
            bgcolor="rgba(255,255,255,0.5)",
            borderpad=4,
        )
        fig.update_scenes(camera_projection_type="orthographic")
        return fig

    def _set_color(self, select_id, label, enabled):
        if not enabled:
            color = "#d1d1d1"  # grey
        elif select_id is not None and select_id == label:
            color = "#cc61ff"  # purple
        else:
            color = "#5e8ff7"  # blue
        return color

    def _update_trace_by_id(self, fig, trace_id, **updates):
        # Iterate through all traces
        for trace in fig.data:
            # Check if trace customdata matches the trace_id
            if trace.customdata and trace.customdata[0] == trace_id:
                # Update trace properties based on additional keyword arguments
                for key, value in updates.items():
                    setattr(trace, key, value)

    def _remove_traces_by_ids(self, fig, active_ids):
        # Convert fig.data, which is a tuple, to a list to allow modifications
        traces = list(fig.data)

        # Iterate in reverse to avoid modifying the list while iterating
        for i in reversed(range(len(traces))):
            trace = traces[i]
            # Check if the trace's customdata is set and its ID is not in the list of active IDs
            if trace.customdata and trace.customdata[0] not in active_ids:
                del traces[i]  # Remove the trace from the list
        fig.data = traces

    def _plot_lamp(self, lamp, fig, select_id=None, color="#cc61ff"):
        """plot lamp as a photometric web"""

        init_scale = convert_units(self.room.units, "meters", lamp.values.max())
        coords = lamp.transform_to_world(lamp.photometric_coords, scale=init_scale)
        scale = lamp.get_total_power() / 120
        coords = (coords.T - lamp.position) * scale + lamp.surface.position
        x, y, z = coords.T

        Theta, Phi, R = to_polar(*lamp.photometric_coords.T)
        tri = Delaunay(np.column_stack((Theta.flatten(), Phi.flatten())))
        lampcolor = self._set_color(select_id, label=lamp.lamp_id, enabled=lamp.enabled)

        lamptrace = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=tri.simplices[:, 0],
            j=tri.simplices[:, 1],
            k=tri.simplices[:, 2],
            color=lampcolor,
            opacity=0.4,
            name=lamp.name,
            customdata=["lamp_" + lamp.lamp_id],
            legendgroup="lamp_" + lamp.lamp_id,
            # legendgroup="lamps",
            # legendgrouptitle_text="Lamps",
            showlegend=True,
        )
        xi, yi, zi = lamp.surface.position
        xia, yia, zia = lamp.aim_point
        aimtrace = go.Scatter3d(
            x=[xi, xia],
            y=[yi, yia],
            z=[zi, zia],
            mode="lines",
            line=dict(color="black", width=2, dash="dash"),
            name=lamp.name,
            customdata=["lamp_" + lamp.lamp_id + "_aim"],
            legendgroup="lamp_" + lamp.lamp_id,
            # legendgroup="lamps",
            # legendgrouptitle_text="Lamps",
            showlegend=False,
        )
        xs, ys, zs = lamp.surface.surface_points.T
        surfacetrace = go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            marker=dict(size=2, color=lampcolor),
            opacity=0.9,
            name=lamp.name,
            customdata=["lamp_" + lamp.lamp_id + "_surface"],
            legendgroup="lamp_" + lamp.lamp_id,
            # legendgroup="lamps",
            # legendgrouptitle_text="Lamps",
            showlegend=False,
        )

        traces = [trace.customdata[0] for trace in fig.data]
        if lamptrace.customdata[0] not in traces:
            fig.add_trace(lamptrace)
            fig.add_trace(aimtrace)
            fig.add_trace(surfacetrace)
        else:
            self._update_trace_by_id(
                fig,
                lamp.lamp_id,
                x=x,
                y=y,
                z=z,
                i=tri.simplices[:, 0],
                j=tri.simplices[:, 1],
                k=tri.simplices[:, 2],
                color=lampcolor,
                name=lamp.name,
            )

            self._update_trace_by_id(
                fig,
                lamp.lamp_id + "_aim",
                x=[xi, xia],
                y=[yi, yia],
                z=[zi, zia],
            )

            self._update_trace_by_id(fig, lamp.lamp_id + "_surface", x=xs, y=ys, z=zs)
        return fig

    def _plot_plane(self, zone, fig, select_id=None):
        """plot a calculation plane"""
        zonecolor = self._set_color(select_id, zone.zone_id, zone.enabled)
        x, y, z = zone.coords.T
        zonetrace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(size=2, color=zonecolor),
            opacity=0.5,
            # legendgroup="planes",
            # legendgrouptitle_text="Calculation Planes",
            showlegend=True,
            name=zone.name,
            customdata=["zone_" + zone.zone_id],
        )
        traces = [trace.name for trace in fig.data]
        if zonetrace.name not in traces:
            fig.add_trace(zonetrace)
        else:
            self._update_trace_by_id(
                fig,
                zone.zone_id,
                x=x,
                y=y,
                z=z,
                marker=dict(size=2, color=zonecolor),
            )

        return fig

    def _plot_plane_values(self, zone, fig):
        if zone.ref_surface == "xy":
            X, Y = np.meshgrid(zone.xp, zone.yp, indexing="ij")
            Z = np.full_like(X, zone.height)
        elif zone.ref_surface == "xz":
            X, Z = np.meshgrid(zone.xp, zone.yp, indexing="ij")
            Y = np.full_like(X, zone.height)
        elif zone.ref_surface == "yz":
            Y, Z = np.meshgrid(zone.xp, zone.yp, indexing="ij")
            X = np.full_like(X, zone.height)
        zone_value_trace = go.Surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=zone.values,
            colorscale=self.room.scene.colormap,
            showscale=False,
            # colorbar=None,
            # legendgroup="planes",
            # legendgrouptitle_text="Calculation Planes",
            showlegend=True,
            name=zone.name,
            customdata=["zone_" + zone.zone_id],
        )
        traces = [trace.name for trace in fig.data]
        if zone_value_trace.name not in traces:
            fig.add_trace(zone_value_trace)
        else:
            self._update_trace_by_id(
                fig,
                zone.zone_id,
                x=X,
                y=Y,
                z=Z,
                surfacecolor=zone.values,
            )
        return fig

    def _plot_vol(self, zone, fig, select_id=None):
        # Define the vertices of the rectangular prism
        (x1, y1, z1), (x2, y2, z2) = zone.dimensions
        vertices = [
            (x1, y1, z1),  # 0
            (x2, y1, z1),  # 1
            (x2, y2, z1),  # 2
            (x1, y2, z1),  # 3
            (x1, y1, z2),  # 4
            (x2, y1, z2),  # 5
            (x2, y2, z2),  # 6
            (x1, y2, z2),  # 7
        ]

        # Define edges by vertex indices
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # Bottom face
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # Top face
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # Side edges
        ]

        # Create lists for x, y, z coordinates
        x_coords = []
        y_coords = []
        z_coords = []

        # Append coordinates for each edge, separated by None to create breaks
        for v1, v2 in edges:
            x_coords.extend([vertices[v1][0], vertices[v2][0], None])
            y_coords.extend([vertices[v1][1], vertices[v2][1], None])
            z_coords.extend([vertices[v1][2], vertices[v2][2], None])

        zonecolor = self._set_color(select_id, label=zone.zone_id, enabled=zone.enabled)
        # Create a single trace for all edges
        zonetrace = go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode="lines",
            line=dict(color=zonecolor, width=5, dash="dot"),
            # legendgroup="volumes",
            # legendgrouptitle_text="Calculation Volumes",
            name=zone.name,
            customdata=["zone_" + zone.zone_id],
        )
        traces = [trace.name for trace in fig.data]
        if zonetrace.name not in traces:
            fig.add_trace(zonetrace)
        else:
            self._update_trace_by_id(
                fig,
                zone.zone_id,
                x=x_coords,
                y=y_coords,
                z=z_coords,
                line=dict(color=zonecolor, width=5, dash="dot"),
            )
        # fluence isosurface
        if zone.values is not None and zone.show_values:
            X, Y, Z = np.meshgrid(*zone.points, indexing="ij")
            x, y, z = X.flatten(), Y.flatten(), Z.flatten()
            values = zone.values.flatten()
            isomin = zone.values.mean() / 2
            if zone.name + " Values" not in traces:
                zone_value_trace = go.Isosurface(
                    x=x,
                    y=y,
                    z=z,
                    value=values,
                    isomin=isomin,
                    surface_count=3,
                    opacity=0.25,
                    showscale=False,
                    colorbar=None,
                    colorscale=self.room.scene.colormap,
                    name=zone.name + " Values",
                    customdata=["zone_" + zone.zone_id + "_values"],
                    # legendgroup="volumes",
                    # legendgrouptitle_text="Calculation Volumes",
                    showlegend=True,
                )
                fig.add_trace(zone_value_trace)
            else:
                self._update_trace_by_id(
                    fig,
                    zone.zone_id,
                    x=x,
                    y=y,
                    z=z,
                    values=values,
                    isomin=isomin,
                )

        return fig
