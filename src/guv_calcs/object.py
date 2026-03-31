"""Scene objects: physical objects placed in rooms that occlude and reflect light."""

import numpy as np
from .calc_zone import CalcPlane
from .reflectance import Surface
from .geometry import Polygon2D, SurfaceGrid
from .units import convert_length, convert_length_tuple



class Object:
    """A physical object composed of surfaces, placed and oriented in a room."""

    def __init__(
        self,
        *,
        object_id=None,
        name=None,
        position=(0, 0, 0),
        yaw=0,
        pitch=0,
        roll=0,
        R=0.0,
        T=0.0,
        enabled=True,
        num_points=5,
        _shape=None,
    ):
        self._object_id = object_id or "Object"
        self.name = name if name is not None else str(self._object_id)
        self.position = tuple(float(v) for v in position)
        self._yaw = float(yaw)
        self._pitch = float(pitch)
        self._roll = float(roll)
        self._rotation = _rotation_matrix(self._yaw, self._pitch, self._roll)
        self.R = float(R)
        self.T = float(T)
        self.enabled = bool(enabled)
        self._num_points = int(num_points)

        self._shape = _shape or {"type": "box", "width": 1, "length": 1, "height": 1}
        self._local_surfaces = {}
        self._build_local_surfaces()
        self._world_surfaces = {}
        self._rebuild_world_surfaces()

    def __repr__(self):
        shape = self._shape
        if shape["type"] == "box":
            dim_str = f"box({shape['width']}x{shape['length']}x{shape['height']})"
        else:
            poly = Polygon2D.from_dict(shape["polygon"]) if isinstance(shape["polygon"], dict) else shape["polygon"]
            dim_str = f"extrusion({poly.n_vertices} vertices, height={shape['height']})"
        return (
            f"Object(id='{self._object_id}', {dim_str}, "
            f"position={self.position}, R={self.R}, T={self.T})"
        )

    def __eq__(self, other):
        if not isinstance(other, Object):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    # ---- identity ----

    @property
    def id(self):
        return self._object_id

    def _assign_id(self, value):
        if self.name == self._object_id:
            self.name = value
        old_id = self._object_id
        self._object_id = value
        if old_id != value:
            rekeyed = {}
            for old_key, surface in self._world_surfaces.items():
                face_id = old_key.split(":", 1)[1]
                new_key = f"{self._object_id}:{face_id}"
                surface.plane._zone_id = new_key
                rekeyed[new_key] = surface
            self._world_surfaces = rekeyed

    # ---- dimensions ----

    @property
    def width(self):
        """Object width (X extent). For boxes, the native width; for extrusions, the bounding box width."""
        shape = self._shape
        if shape["type"] == "box":
            return shape["width"]
        poly = self._get_polygon()
        x_min, _, x_max, _ = poly.bounding_box
        return x_max - x_min

    @property
    def length(self):
        """Object length (Y extent). For boxes, the native length; for extrusions, the bounding box length."""
        shape = self._shape
        if shape["type"] == "box":
            return shape["length"]
        poly = self._get_polygon()
        _, y_min, _, y_max = poly.bounding_box
        return y_max - y_min

    @property
    def height(self):
        """Object height (Z extent)."""
        return self._shape["height"]

    def _get_polygon(self):
        """Return the base Polygon2D for this shape."""
        shape = self._shape
        if shape["type"] == "box":
            w, l = shape["width"], shape["length"]
            return Polygon2D.rectangle(w, l).translate(-w / 2, -l / 2)
        poly_data = shape["polygon"]
        return Polygon2D.from_dict(poly_data) if isinstance(poly_data, dict) else poly_data

    # ---- factories ----

    @classmethod
    def box(cls, width, length, height, **kwargs):
        """Create a box object."""
        if width <= 0 or length <= 0 or height <= 0:
            raise ValueError("Box dimensions must be positive")
        shape = {
            "type": "box",
            "width": float(width),
            "length": float(length),
            "height": float(height),
        }
        return cls(_shape=shape, **kwargs)

    @classmethod
    def extrusion(cls, polygon, height, **kwargs):
        """Create an extruded polygon object."""
        if height <= 0:
            raise ValueError("Height must be positive")
        if not isinstance(polygon, Polygon2D):
            polygon = Polygon2D(vertices=tuple(tuple(v) for v in polygon))
        shape = {
            "type": "extrusion",
            "polygon": polygon.to_dict(),
            "height": float(height),
        }
        return cls(_shape=shape, **kwargs)

    # ---- positioning ----

    def move(self, x=None, y=None, z=None):
        """Set world-space position."""
        cx, cy, cz = self.position
        self.position = (
            float(x) if x is not None else cx,
            float(y) if y is not None else cy,
            float(z) if z is not None else cz,
        )
        self._rebuild_world_surfaces()
        return self

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def z(self):
        return self.position[2]

    def rotate(self, yaw=None, pitch=None, roll=None):
        """Set rotation angles (degrees)."""
        if yaw is not None:
            self._yaw = float(yaw)
        if pitch is not None:
            self._pitch = float(pitch)
        if roll is not None:
            self._roll = float(roll)
        self._rotation = _rotation_matrix(self._yaw, self._pitch, self._roll)
        self._rebuild_world_surfaces()
        return self

    # ---- surface access ----

    @property
    def surfaces(self):
        """World-space surfaces with namespaced keys."""
        return self._world_surfaces

    @property
    def face_ids(self):
        """List of face identifiers."""
        return list(self._local_surfaces.keys())

    def get_face_properties(self, face=None):
        """Return optical properties for one face or all faces.

        If face is given, returns {"R": float, "T": float} for that face.
        If face is None, returns {face_id: {"R": float, "T": float}, ...} for all faces.
        """
        if face is not None:
            if face not in self._local_surfaces:
                raise KeyError(f"Unknown face: {face!r}. Available: {self.face_ids}")
            s = self._local_surfaces[face]
            return {"R": s.R, "T": s.T}
        return {fid: {"R": s.R, "T": s.T} for fid, s in self._local_surfaces.items()}

    def set_face_properties(self, R, T, face=None):
        """Atomically set both R and T for one or all faces.

        Validates R + T <= 1 before applying either value, avoiding
        ordering issues with separate set_reflectance/set_transmittance calls.
        """
        R, T = float(R), float(T)
        if not (0 <= R <= 1):
            raise ValueError("R must be in [0, 1]")
        if not (0 <= T <= 1):
            raise ValueError("T must be in [0, 1]")
        if R + T > 1:
            raise ValueError("R + T must be <= 1")

        if face is None:
            self.R = R
            self.T = T
            for s in self._local_surfaces.values():
                s.R = R
                s.T = T
        else:
            if face not in self._local_surfaces:
                raise KeyError(f"Unknown face: {face!r}. Available: {self.face_ids}")
            self._local_surfaces[face].R = R
            self._local_surfaces[face].T = T
        self._rebuild_world_surfaces()

    def set_reflectance(self, R, face=None):
        """Set reflectance for one or all faces."""
        if face is None:
            self.R = float(R)
            for s in self._local_surfaces.values():
                s.set_reflectance(R)
        else:
            if face not in self._local_surfaces:
                raise KeyError(f"Unknown face: {face!r}. Available: {self.face_ids}")
            self._local_surfaces[face].set_reflectance(R)
        self._rebuild_world_surfaces()

    def set_transmittance(self, T, face=None):
        """Set transmittance for one or all faces."""
        if face is None:
            self.T = float(T)
            for s in self._local_surfaces.values():
                s.set_transmittance(T)
        else:
            if face not in self._local_surfaces:
                raise KeyError(f"Unknown face: {face!r}. Available: {self.face_ids}")
            self._local_surfaces[face].set_transmittance(T)
        self._rebuild_world_surfaces()

    # ---- internal: build surfaces ----

    def _build_local_surfaces(self):
        """Build local-space surfaces from shape definition."""
        shape = self._shape
        shape_type = shape["type"]
        np_init = (self._num_points, self._num_points)

        if shape_type == "box":
            w, l, h = shape["width"], shape["length"], shape["height"]
            polygon = Polygon2D.rectangle(w, l).translate(-w / 2, -l / 2)
        elif shape_type == "extrusion":
            poly_data = shape["polygon"]
            polygon = Polygon2D.from_dict(poly_data) if isinstance(poly_data, dict) else poly_data
            h = shape["height"]
            cx, cy = polygon.centroid
            polygon = polygon.translate(-cx, -cy)
        else:
            raise ValueError(f"Unknown shape type: {shape_type!r}")

        surfaces = {}

        # bottom (normal points down, outward from object)
        bottom_geom = SurfaceGrid.from_polygon(
            polygon, height=0.0, direction=-1, num_points_init=np_init
        )
        surfaces["bottom"] = Surface(
            R=self.R, T=self.T,
            plane=CalcPlane(zone_id="bottom", geometry=bottom_geom, horiz=True),
        )

        # top (normal points up, outward from object)
        top_geom = SurfaceGrid.from_polygon(
            polygon, height=h, direction=1, num_points_init=np_init
        )
        surfaces["top"] = Surface(
            R=self.R, T=self.T,
            plane=CalcPlane(zone_id="top", geometry=top_geom, horiz=True),
        )

        # walls (normals point outward via CCW winding)
        for i, ((x1, y1), (x2, y2)) in enumerate(polygon.edges):
            wall_geom = SurfaceGrid.from_wall(
                (x1, y1), (x2, y2), z_height=h, num_points_init=np_init
            )
            face_id = f"wall_{i}"
            surfaces[face_id] = Surface(
                R=self.R, T=self.T,
                plane=CalcPlane(zone_id=face_id, geometry=wall_geom, horiz=True),
            )

        self._local_surfaces = surfaces

    def _rebuild_world_surfaces(self):
        """Transform local surfaces to world space."""
        R_mat = self._rotation
        pos = np.asarray(self.position, float)

        self._world_surfaces = {}
        for face_id, local_surface in self._local_surfaces.items():
            local_geom = local_surface.plane.geometry

            new_origin = tuple(R_mat @ np.asarray(local_geom.origin) + pos)
            new_u_vec = tuple(R_mat @ np.asarray(local_geom.u_vec))
            new_v_vec = tuple(R_mat @ np.asarray(local_geom.v_vec))

            world_geom = local_geom.update(
                origin=new_origin, u_vec=new_u_vec, v_vec=new_v_vec,
            )

            world_key = f"{self._object_id}:{face_id}"
            world_plane = CalcPlane(
                zone_id=world_key, geometry=world_geom, horiz=True,
            )
            self._world_surfaces[world_key] = Surface(
                R=local_surface.R, T=local_surface.T, plane=world_plane,
            )

    def nudge_into_bounds(self, room_dims, max_iterations: int = 3) -> bool:
        """Nudge object so all surface vertices stay within room bounds. Returns True if changed."""
        polygon = room_dims.polygon
        z_max = room_dims.z
        moved = False
        margin = 1e-4

        for _ in range(max_iterations):
            # Gather all world-space surface boundary vertices
            all_verts = []
            for surface in self._world_surfaces.values():
                all_verts.append(surface.plane.geometry.boundary_vertices)
            if not all_verts:
                return moved
            coords = np.vstack(all_verts)

            dx = 0.0
            dy = 0.0
            dz = 0.0

            for point in coords:
                cx, cy, cz = point
                if not polygon.contains_point_inclusive(cx, cy):
                    nearest = polygon.nearest_boundary_point(cx, cy)
                    nudge_x = nearest[0] - cx
                    nudge_y = nearest[1] - cy
                    if abs(nudge_x) > abs(dx):
                        dx = nudge_x
                    if abs(nudge_y) > abs(dy):
                        dy = nudge_y
                if cz > z_max:
                    shift = z_max - cz
                    if shift < dz:
                        dz = shift
                if cz < 0:
                    shift = -cz
                    if shift > dz:
                        dz = shift

            if abs(dx) < 1e-9 and abs(dy) < 1e-9 and abs(dz) < 1e-9:
                return moved

            if dx != 0:
                dx += margin if dx > 0 else -margin
            if dy != 0:
                dy += margin if dy > 0 else -margin
            if dz != 0:
                dz += margin if dz > 0 else -margin
            self.move(self.x + dx, self.y + dy, self.z + dz)
            moved = True

        return moved

    # ---- state ----

    @property
    def calc_state(self):
        return (
            tuple(self.position),
            self._yaw, self._pitch, self._roll,
            self.enabled,
            tuple(
                (k, s.R, s.T, s.calc_state)
                for k, s in sorted(self._local_surfaces.items())
            ),
        )

    @property
    def update_state(self):
        return ()

    # ---- serialization ----

    def to_dict(self):
        shape = dict(self._shape)
        face_props = {}
        for face_id, s in self._local_surfaces.items():
            if s.R != self.R or s.T != self.T:
                face_props[face_id] = {"R": s.R, "T": s.T}

        return {
            "object_id": self._object_id,
            "name": self.name,
            "position": list(self.position),
            "yaw": self._yaw,
            "pitch": self._pitch,
            "roll": self._roll,
            "R": self.R,
            "T": self.T,
            "enabled": self.enabled,
            "num_points": self._num_points,
            "shape": shape,
            "face_properties": face_props,
        }

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        shape = data.get("shape", {})
        face_props = data.pop("face_properties", {})

        obj = cls(
            object_id=data.get("object_id"),
            name=data.get("name"),
            position=tuple(data.get("position", (0, 0, 0))),
            yaw=data.get("yaw", 0),
            pitch=data.get("pitch", 0),
            roll=data.get("roll", 0),
            R=data.get("R", 0.0),
            T=data.get("T", 0.0),
            enabled=data.get("enabled", True),
            num_points=data.get("num_points", 5),
            _shape=shape,
        )

        for face_id, props in face_props.items():
            if "R" in props:
                obj.set_reflectance(props["R"], face=face_id)
            if "T" in props:
                obj.set_transmittance(props["T"], face=face_id)

        return obj

    # ---- units ----

    def convert_units(self, old_units, new_units):
        """Convert all spatial dimensions and position."""
        face_props = {k: (s.R, s.T) for k, s in self._local_surfaces.items()}

        factor = convert_length(old_units, new_units, 1.0)
        self.position = convert_length_tuple(old_units, new_units, *self.position)

        shape = dict(self._shape)
        if shape["type"] == "box":
            shape["width"] *= factor
            shape["length"] *= factor
            shape["height"] *= factor
        elif shape["type"] == "extrusion":
            poly = Polygon2D.from_dict(shape["polygon"])
            new_verts = tuple((x * factor, y * factor) for x, y in poly.vertices)
            shape["polygon"] = Polygon2D(vertices=new_verts).to_dict()
            shape["height"] *= factor
        self._shape = shape

        self._build_local_surfaces()

        # restore per-face properties
        for face_id, (r, t) in face_props.items():
            if face_id in self._local_surfaces:
                self._local_surfaces[face_id].set_reflectance(r)
                self._local_surfaces[face_id].set_transmittance(t)

        self._rebuild_world_surfaces()


def _rotation_matrix(yaw, pitch, roll):
    """3x3 rotation from yaw (Z), pitch (Y), roll (X) in degrees."""
    y, p, r = np.radians([yaw, pitch, roll])
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx
