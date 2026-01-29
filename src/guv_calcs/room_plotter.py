import plotly.graph_objs as go
import numpy as np
from scipy.spatial import Delaunay
from .trigonometry import to_polar
from .calc_zone import CalcPlane, CalcVol
from .poly_grid import PolygonGrid, PolygonVolGrid
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
        """
        plot all

        TODO:
        - Cleaner color selection
        - maybe cleaner update control?
        """
        if fig is None:
            fig = go.Figure()

        # first delete any extraneous traces
        lamp_ids = list(self.room.lamps.keys())
        aim_ids = [lampid + "_aim" for lampid in lamp_ids]
        surface_ids = [lampid + "_surface" for lampid in lamp_ids]
        fixture_ids = [lampid + "_fixture" for lampid in lamp_ids]
        zone_ids = list(self.room.calc_zones.keys())
        for active_ids in [lamp_ids, aim_ids, surface_ids, fixture_ids, zone_ids]:
            self._remove_traces_by_ids(fig, active_ids)

        # plot lamps
        for lamp_id, lamp in self.room.lamps.items():
            if lamp.ies is not None:
                fig = self._plot_lamp(lamp=lamp, fig=fig, select_id=select_id)
        for zone_id, zone in self.room.calc_zones.items():
            if isinstance(zone, CalcPlane):
                if zone.show_values and zone.values is not None:
                    fig = self._plot_plane_values(zone=zone, fig=fig)
                else:
                    fig = self._plot_plane(zone=zone, fig=fig, select_id=select_id)
            elif isinstance(zone, CalcVol):
                fig = self._plot_vol(zone=zone, fig=fig, select_id=select_id)

        # for filter_id, filt in self.room.filters.items():
        # fig = self._plot_filter(filt=filt, fig=fig, select_id=select_id)

        # for obs_id, obs in self.room.obstacles.items():
        # fig = self._plot_obstacle(obs=obs, fig=fig)

        # Draw room outline for polygon rooms
        if self.room.is_polygon:
            fig = self._plot_polygon_room_outline(fig=fig)

        # Get axis ranges - for polygon rooms, use bounding box coordinates
        if self.room.is_polygon:
            x_min, y_min, x_max, y_max = self.room.polygon.bounding_box
            x_range = [x_min, x_max]
            y_range = [y_min, y_max]
            x_span = x_max - x_min
            y_span = y_max - y_min
        else:
            x_range = [0, self.room.dim.x]
            y_range = [0, self.room.dim.y]
            x_span = self.room.dim.x
            y_span = self.room.dim.y

        z = self.room.dim.z

        # set views
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(range=x_range),
                yaxis=dict(range=y_range),
                zaxis=dict(range=[0, z]),
                aspectratio=dict(
                    x=x_span / z,
                    y=y_span / z,
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
        scale = lamp.get_total_power() / 100
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

        # Fixture housing box (only if fixture has dimensions)
        fixturetrace = None
        if lamp.fixture.has_dimensions:
            # Get bounding box corners and convert to room units if needed
            corners = lamp.geometry.get_bounding_box_corners()
            # Corners are in surface.units, convert to room.units if different
            if lamp.surface.units != self.room.units:
                corners = convert_units(
                    lamp.surface.units, self.room.units, corners
                )
            xf, yf, zf = self._get_box_coords_from_corners(corners)
            fixturetrace = go.Scatter3d(
                x=xf,
                y=yf,
                z=zf,
                mode="lines",
                line=dict(color=lampcolor, width=3),
                opacity=0.7,
                name=lamp.name + " fixture",
                customdata=["lamp_" + lamp.lamp_id + "_fixture"],
                legendgroup="lamp_" + lamp.lamp_id,
                showlegend=False,
            )

        traces = [trace.customdata[0] for trace in fig.data]
        if lamptrace.customdata[0] not in traces:
            fig.add_trace(lamptrace)
            fig.add_trace(aimtrace)
            fig.add_trace(surfacetrace)
            if fixturetrace is not None:
                fig.add_trace(fixturetrace)
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

            if fixturetrace is not None:
                if "lamp_" + lamp.lamp_id + "_fixture" in traces:
                    self._update_trace_by_id(
                        fig,
                        lamp.lamp_id + "_fixture",
                        x=xf,
                        y=yf,
                        z=zf,
                        line=dict(color=lampcolor, width=3),
                    )
                else:
                    fig.add_trace(fixturetrace)
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
        # Check if this is a polygon grid (irregular points)
        is_irregular = isinstance(zone.geometry, PolygonGrid)

        if is_irregular:
            # Use Mesh3d with Delaunay triangulation for irregular grids
            x, y, z = zone.coords.T
            values = zone.values.flatten() if zone.values.ndim > 1 else zone.values

            # Create 2D triangulation based on the plane orientation
            if isinstance(zone.geometry, PolygonGrid):
                # Floor/ceiling - triangulate in xy plane
                tri = Delaunay(np.column_stack((x, y)))
                simplices = tri.simplices

                # Filter out triangles outside the polygon (handles concave shapes)
                centroids = np.column_stack((x, y))[simplices].mean(axis=1)
                inside = zone.geometry.polygon.contains_points(centroids)
                simplices = simplices[inside]
            else:
                # Wall - triangulate in local uv space
                # Project points onto the wall plane for triangulation
                origin = np.array(zone.geometry.origin)
                u_hat = zone.geometry.u_hat
                v_hat = zone.geometry.v_hat
                pts = zone.coords - origin
                u_coords = pts @ u_hat
                v_coords = pts @ v_hat
                tri = Delaunay(np.column_stack((u_coords, v_coords)))
                simplices = tri.simplices

            zone_value_trace = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=simplices[:, 0],
                j=simplices[:, 1],
                k=simplices[:, 2],
                intensity=values,
                colorscale=self.room.scene.colormap,
                showscale=False,
                showlegend=True,
                name=zone.name,
                customdata=["zone_" + zone.zone_id],
            )
        else:
            # Regular grid - use Surface plot
            x, y, z = zone.coords.T.reshape(3, *zone.num_points)
            zone_value_trace = go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=zone.values,
                colorscale=self.room.scene.colormap,
                showscale=False,
                showlegend=True,
                name=zone.name,
                customdata=["zone_" + zone.zone_id],
            )

        traces = [trace.name for trace in fig.data]
        if zone_value_trace.name not in traces:
            fig.add_trace(zone_value_trace)
        else:
            if is_irregular:
                self._update_trace_by_id(
                    fig,
                    zone.zone_id,
                    x=x,
                    y=y,
                    z=z,
                    intensity=values,
                )
            else:
                self._update_trace_by_id(
                    fig,
                    zone.zone_id,
                    x=x,
                    y=y,
                    z=z,
                    surfacecolor=zone.values,
                )
        return fig

    def _plot_obstacle(self, obs, fig):
        x_coords, y_coords, z_coords = self._get_box_coords(obs.lo, obs.hi)
        obs_trace = go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode="lines",
            line=dict(color="#000000", width=3),
            name=obs.name,
            customdata=["obstacle_" + obs.obs_id],
        )
        traces = [trace.name for trace in fig.data]
        if obs_trace.name not in traces:
            fig.add_trace(obs_trace)
        else:
            self._update_trace_by_id(
                fig,
                obs.obs_id,
                x=x_coords,
                y=y_coords,
                z=z_coords,
            )
        return fig

    def _plot_vol(self, zone, fig, select_id=None):

        # Check if this is a polygon volume
        if isinstance(zone.geometry, PolygonVolGrid):
            x_coords, y_coords, z_coords = self._get_polygon_vol_coords(zone.geometry)
        else:
            x_coords, y_coords, z_coords = self._get_box_coords(
                *np.array(zone.dimensions).T
            )
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
            values = zone.values.flatten()

            # For polygon volumes, use full grid with -inf masking for proper isosurface
            if isinstance(zone.geometry, PolygonVolGrid):
                coords_full = zone.geometry.coords_full
                values_full = zone.geometry.values_to_full_grid(values)
                # Filter out -inf values for mean calculation
                valid_mask = np.isfinite(values_full)
                isomin = values_full[valid_mask].mean() / 2

                if zone.name + " Values" not in traces:
                    zone_value_trace = go.Isosurface(
                        x=coords_full.T[0],
                        y=coords_full.T[1],
                        z=coords_full.T[2],
                        value=values_full,
                        isomin=isomin,
                        surface_count=3,
                        opacity=0.25,
                        showscale=False,
                        colorbar=None,
                        colorscale=self.room.scene.colormap,
                        caps=dict(x_show=False, y_show=False, z_show=False),
                        name=zone.name + " Values",
                        customdata=["zone_" + zone.zone_id + "_values"],
                        showlegend=True,
                    )
                    fig.add_trace(zone_value_trace)
                else:
                    self._update_trace_by_id(
                        fig,
                        zone.zone_id + "_values",
                        x=coords_full.T[0],
                        y=coords_full.T[1],
                        z=coords_full.T[2],
                        value=values_full,
                        isomin=isomin,
                    )
            else:
                isomin = zone.values.mean() / 2
                if zone.name + " Values" not in traces:
                    zone_value_trace = go.Isosurface(
                        x=zone.coords.T[0],
                        y=zone.coords.T[1],
                        z=zone.coords.T[2],
                        value=values,
                        isomin=isomin,
                        surface_count=3,
                        opacity=0.25,
                        showscale=False,
                        colorbar=None,
                        colorscale=self.room.scene.colormap,
                        name=zone.name + " Values",
                        customdata=["zone_" + zone.zone_id + "_values"],
                        showlegend=True,
                    )
                    fig.add_trace(zone_value_trace)
                else:
                    self._update_trace_by_id(
                        fig,
                        zone.zone_id,
                        x=zone.coords.T[0],
                        y=zone.coords.T[1],
                        z=zone.coords.T[2],
                        values=values,
                        isomin=isomin,
                    )

        return fig

    def _get_box_coords(self, lo, hi):
        (x1, y1, z1), (x2, y2, z2) = lo, hi
        # Define the vertices of the rectangular prism
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

        return x_coords, y_coords, z_coords

    def _get_box_coords_from_corners(self, corners):
        """Generate wireframe coordinates from 8 corner vertices.

        corners: (8, 3) array with vertices ordered as:
            0: (-x, -y, z_min), 1: (+x, -y, z_min), 2: (+x, +y, z_min), 3: (-x, +y, z_min)
            4: (-x, -y, z_max), 5: (+x, -y, z_max), 6: (+x, +y, z_max), 7: (-x, +y, z_max)
        """
        # Define edges by vertex indices (same as _get_box_coords)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # Side edges
        ]

        x_coords, y_coords, z_coords = [], [], []
        for v1, v2 in edges:
            x_coords.extend([corners[v1, 0], corners[v2, 0], None])
            y_coords.extend([corners[v1, 1], corners[v2, 1], None])
            z_coords.extend([corners[v1, 2], corners[v2, 2], None])

        return x_coords, y_coords, z_coords

    def _get_polygon_vol_coords(self, geometry):
        """Get coordinates for drawing a polygon volume outline."""
        polygon = geometry.polygon
        z_height = geometry.z_height

        # Get polygon vertices and close the polygon
        verts = list(polygon.vertices)
        verts.append(verts[0])

        # Floor outline (z=0)
        x_floor = [v[0] for v in verts]
        y_floor = [v[1] for v in verts]
        z_floor = [0.0] * len(verts)

        # Ceiling outline (z=z_height)
        x_ceil = [v[0] for v in verts]
        y_ceil = [v[1] for v in verts]
        z_ceil = [z_height] * len(verts)

        # Vertical edges at each vertex
        x_vert, y_vert, z_vert = [], [], []
        for vx, vy in polygon.vertices:
            x_vert.extend([vx, vx, None])
            y_vert.extend([vy, vy, None])
            z_vert.extend([0.0, z_height, None])

        # Combine all lines
        x_coords = x_floor + [None] + x_ceil + [None] + x_vert
        y_coords = y_floor + [None] + y_ceil + [None] + y_vert
        z_coords = z_floor + [None] + z_ceil + [None] + z_vert

        return x_coords, y_coords, z_coords

    def _plot_polygon_room_outline(self, fig):
        """Draw the outline of a polygon room (floor, ceiling, and vertical edges)."""
        polygon = self.room.polygon
        z = self.room.dim.z

        # Get polygon vertices
        verts = list(polygon.vertices)
        verts.append(verts[0])  # Close the polygon

        # Floor outline
        x_floor = [v[0] for v in verts]
        y_floor = [v[1] for v in verts]
        z_floor = [0.0] * len(verts)

        # Ceiling outline
        x_ceil = [v[0] for v in verts]
        y_ceil = [v[1] for v in verts]
        z_ceil = [z] * len(verts)

        # Vertical edges at each vertex
        x_vert, y_vert, z_vert = [], [], []
        for vx, vy in polygon.vertices:
            x_vert.extend([vx, vx, None])
            y_vert.extend([vy, vy, None])
            z_vert.extend([0.0, z, None])

        # Combine all lines
        x_all = x_floor + [None] + x_ceil + [None] + x_vert
        y_all = y_floor + [None] + y_ceil + [None] + y_vert
        z_all = z_floor + [None] + z_ceil + [None] + z_vert

        room_trace = go.Scatter3d(
            x=x_all,
            y=y_all,
            z=z_all,
            mode="lines",
            line=dict(color="#888888", width=2),
            name="Room",
            customdata=["room_outline"],
            showlegend=False,
        )

        # Only add if not already present
        traces = [trace.customdata[0] if trace.customdata else None for trace in fig.data]
        if "room_outline" not in traces:
            fig.add_trace(room_trace)

        return fig
