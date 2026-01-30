from collections.abc import Iterable
import copy
import warnings
from matplotlib import colormaps
from .room_dims import RoomDimensions, PolygonRoomDimensions
from .lamp import Lamp
from .calc_zone import CalcZone, CalcPlane, CalcVol
from .reflectance import Surface, init_room_surfaces
from .lamp_placement import LampPlacer
from .safety import PhotStandard
from .scene_registry import LampRegistry, ZoneRegistry, SurfaceRegistry
from .units import LengthUnits, convert_length


class Scene:
    def __init__(
        self,
        dim: RoomDimensions,
        on_collision: str = "increment",  # error | increment | overwrite"
        colormap: str = "plasma",
    ):
        self.dim = dim
        self.colormap = colormap
        self.on_collision = on_collision

        # self.lamps: dict[str, Lamp] = {}
        self.lamps = LampRegistry(
            dims=lambda: self.dim,
            on_collision=on_collision,
        )
        # self.calc_zones: dict[str, CalcZone] = {}
        self.calc_zones = ZoneRegistry(
            dims=lambda: self.dim,
            on_collision=on_collision,
        )
        # self.surfaces: dict[str, Surface] = {}
        self.surfaces = SurfaceRegistry(
            dims=lambda: self.dim,
            on_collision=on_collision,
        )

    def __deepcopy__(self, memo):
        """Deep copy with proper rebinding of registry lambdas."""
        # Create a new instance without calling __init__
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Deep copy all attributes
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))

        # Rebind registry lambdas to point to the new instance's dim
        result.lamps.dims = lambda: result.dim
        result.calc_zones.dims = lambda: result.dim
        result.surfaces.dims = lambda: result.dim

        return result

    # ------------------------ Global --------------------------
    def add(self, *args, on_collision=None):
        """
        TODO: possibly the specific add_lamp etc functions should go away?

        Add objects to the Scene.
        - If an object is a Lamp, it is added as a lamp.
        - If an object is a CalcZone, CalcPlane, or CalcVol, it is added as a calculation zone.
        - If an object is iterable, it is recursively processed.
        - Otherwise, a warning is printed.
        """

        for obj in args:
            if isinstance(obj, Lamp):
                self.add_lamp(obj, on_collision=on_collision)
            elif isinstance(obj, CalcZone):
                self.add_calc_zone(obj, on_collision=on_collision)
            elif isinstance(obj, Surface):
                self.add_surface(obj, on_collision=on_collision)
            elif isinstance(obj, dict):
                self.add(*obj.values(), on_collision=on_collision)
            elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
                # Recursively process other iterables
                self.add(*obj, on_collision=on_collision)
            else:
                msg = f"Cannot add object of type {type(obj).__name__} to Room."
                warnings.warn(msg, stacklevel=3)

    def update_dimensions(self, x=None, y=None, z=None, polygon=None):
        """update the dimensions and any objects that depend on the dimensions"""
        if polygon is not None and self.dim.is_polygon:
            self.dim = self.dim.with_(z=z, polygon=polygon)
        elif not self.dim.is_polygon:
            self.dim = self.dim.with_(x=x, y=y, z=z)
        else:
            # Polygon room but only z changed
            self.dim = self.dim.with_(z=z)
        self.update_standard_surfaces()

    def update_units(self, units):
        self.dim = self.dim.with_(units=LengthUnits.from_any(units))
        self.lamps.validate()  # update lamp units...but this should really not be happening

    def set_colormap(self, colormap: str):
        """Set the scene's colormap"""
        if colormap not in list(colormaps):
            raise ValueError(f"{colormap} is not a valid colormap.")

        self.colormap = colormap
        for zone in self.calc_zones.values():
            zone.colormap = self.colormap
        for surface in self.surfaces.values():
            surface.plane.colormap = self.colormap

    def check_positions(self):
        """
        verify the positions of all objects in the scene and return any warning messages
        """
        msgs = {}
        msgs["lamps"] = self.lamps.get_position_warnings()
        msgs["calc_zones"] = self.calc_zones.get_position_warnings()
        msgs["surfaces"] = self.surfaces.get_position_warnings()
        return msgs

    # ------------------ Lamps -----------------------

    def add_lamp(self, lamp, on_collision=None):
        """Add a lamp to the room"""
        self.lamps.add(lamp, on_collision=on_collision)
        return self

    def place_lamp(
        self,
        *args,
        on_collision=None,
        mode="downlight",
        tilt: float | None = None,
        max_tilt: float | None = None,
    ):
        """
        Position lamps using the specified placement mode.

        Args:
            mode: Placement mode:
                - "downlight": places lamps inside the room pointing down
                - "corner": places lamps at corners, ranked by visibility
                - "edge": places lamps at edge centers, aiming perpendicular to wall
                - "horizontal": like edge but aimed straight horizontally
            tilt: Force exact tilt angle in degrees (0 = straight down, 90 = horizontal)
            max_tilt: Maximum allowed tilt angle in degrees (clamps auto-calculated tilt)
        """
        offset = convert_length(LengthUnits.METERS, self.dim.units, 0.1, sigfigs=2)
        existing = [(l.position[0], l.position[1]) for l in self.lamps.values()]
        placer = LampPlacer.for_dims(self.dim, existing=existing)

        for lamp in self._init_lamp(*args):
            placer.place_lamp(lamp, mode=mode, tilt=tilt, max_tilt=max_tilt, offset=offset)
            self.add_lamp(lamp, on_collision=on_collision)

    def _init_lamp(self, *args):
        lst = []
        for obj in args:
            if isinstance(obj, Lamp):
                lst.append(obj)
            elif isinstance(obj, str):
                lst.append(Lamp.from_keyword(obj))
            elif isinstance(obj, int):
                lst.append(Lamp.from_index(obj))
            elif isinstance(obj, dict):
                lst.append(self._init_lamp(*obj.values()))
            elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
                lst.append(self._init_lamp(*obj))
            else:
                raise TypeError(
                    f"{type(obj)} is not a valid Lamp or method of generating a Lamp"
                )
        for i, x in enumerate(lst):
            while i < len(lst) and isinstance(lst[i], list):
                lst[i : i + 1] = lst[i]
        return lst

    def remove_lamp(self, lamp_id):
        """Remove a lamp from the scene"""
        # self.lamps.pop(lamp_id, None)
        self.lamps.remove(lamp_id)

    def get_valid_lamps(self):
        """return all the lamps that can participate in a calculation"""
        return {
            k: v for k, v in self.lamps.items() if v.enabled and v.ies is not None
        }

    # -------------- Zones ------------------

    def plane_from_face(self, wall, **kwargs):
        """
        add a plane to the scene matching the room dimensions, referenced
        """
        plane = CalcPlane.from_face(wall=wall, dims=self.dim, **kwargs)
        self.add_calc_zone(plane)

    def add_calc_zone(self, zone, on_collision=None):
        """
        Add a calculation zone to the scene
        """
        zone.colormap = self.colormap  # this maybe should go away?
        self.calc_zones.add(zone, on_collision=on_collision)

    def remove_calc_zone(self, zone_id):
        """remove calculation zone from scene"""
        self.calc_zones.pop(zone_id, None)

    def add_standard_zones(self, standard: "PhotStandard", *, on_collision=None):
        """
        Add the special calculation zones SkinLimits, EyeLimits, and
        WholeRoomFluence to the scene
        """
        flags = standard.flags(self.dim.units)
        standard_zones = [
            CalcVol.from_dims(
                dims=self.dim,
                zone_id="WholeRoomFluence",
                name="Whole Room Fluence",
                show_values=False,
                num_points=(25, 25, 25),
            ),
            CalcPlane.from_face(
                dims=self.dim,
                wall="floor",
                normal_offset=flags["height"],
                zone_id="EyeLimits",
                name="Eye Dose (8 Hours)",
                dose=True,
                hours=8,
                use_normal=False,
                vert=flags["eye_vert"],
                fov_vert=flags["fov_vert"],
                spacing=(0.1, 0.1),
            ),
            CalcPlane.from_face(
                dims=self.dim,
                wall="floor",
                normal_offset=flags["height"],
                zone_id="SkinLimits",
                name="Skin Dose (8 Hours)",
                dose=True,
                hours=8,
                use_normal=False,
                horiz=flags["skin_horiz"],
                spacing=(0.1, 0.1),
            ),
        ]

        self.add(standard_zones, on_collision=on_collision)

    def update_standard_zones(self, standard: "PhotStandard", preserve_spacing: bool):
        """
        update the standard safety calculation zones based on the current
        standard, units, and room dimensions
        """
        flags = standard.flags(self.dim.units)
        if "SkinLimits" in self.calc_zones.keys():
            zone = self.calc_zones["SkinLimits"]
            zone.set_dimensions(
                x2=self.dim.x, y2=self.dim.y, preserve_spacing=preserve_spacing
            )
            zone.set_height(height=flags["height"])
            zone.horiz = flags["skin_horiz"]
        if "EyeLimits" in self.calc_zones.keys():
            zone = self.calc_zones["EyeLimits"]
            zone.set_dimensions(
                x2=self.dim.x, y2=self.dim.y, preserve_spacing=preserve_spacing
            )
            zone.set_height(height=flags["height"])
            zone.fov_vert = flags["fov_vert"]
            zone.vert = flags["eye_vert"]
        if "WholeRoomFluence" in self.calc_zones.keys():
            zone = self.calc_zones["WholeRoomFluence"]
            zone.set_dimensions(
                x2=self.dim.x,
                y2=self.dim.y,
                z2=self.dim.z,
                preserve_spacing=preserve_spacing,
            )

    # ------------------- Surfaces ----------------------

    def add_surface(self, surface, on_collision=None):
        """add a reflective surface to the room"""
        surface.plane.colormap = self.colormap
        self.surfaces.add(surface, on_collision=on_collision)

    def remove_surface(self, surface_id):
        """remove reflective surface from the room"""
        self.surfaces.remove(surface_id)

    def init_standard_surfaces(
        self,
        reflectances=None,
        x_spacings=None,
        y_spacings=None,
        num_x=None,
        num_y=None,
    ):
        """add surfaces to the scene corresponding to the faces of the room"""
        room_surfaces = init_room_surfaces(
            dims=self.dim,
            reflectances=reflectances,
            x_spacings=x_spacings,
            y_spacings=y_spacings,
            num_x=num_x,
            num_y=num_y,
        )
        for key, val in room_surfaces.items():
            self.add_surface(val)

    def update_standard_surfaces(self):
        """update the faces of the room with the new dimensions"""
        keys = self.dim.faces.keys()
        # Build dicts only for keys that exist in current surfaces
        existing_keys = [k for k in keys if k in self.surfaces]
        reflectances = {key: self.surfaces[key].R for key in existing_keys}
        x_spacings = {key: self.surfaces[key].plane.x_spacing for key in existing_keys}
        y_spacings = {key: self.surfaces[key].plane.y_spacing for key in existing_keys}
        num_x = {key: self.surfaces[key].plane.num_x for key in existing_keys}
        num_y = {key: self.surfaces[key].plane.num_y for key in existing_keys}

        room_surfaces = init_room_surfaces(
            dims=self.dim,
            reflectances=reflectances,
            x_spacings=x_spacings,
            y_spacings=y_spacings,
            num_x=num_x,
            num_y=num_y,
        )
        for key, val in room_surfaces.items():
            self.add_surface(val, on_collision="overwrite")

    def set_reflectance(self, R, wall_id=None):
        """set reflectance by wall_id or, if wall_if is None, to all walls"""
        keys = self.surfaces.keys()
        if wall_id is None:
            # set this value for all walls
            for wall in keys:
                self.surfaces.get(wall).set_reflectance(R)
        else:
            if wall_id not in keys:
                raise KeyError(f"wall_id must be in {keys}")
            else:
                self.surfaces.get(wall_id).set_reflectance(R)

    def set_spacing(self, x_spacing=None, y_spacing=None, wall_id=None):
        """set x and y spacing by wall_id or, if wall_if is None, to all walls"""
        keys = self.surfaces.keys()
        if wall_id is None:
            # set this value for all walls
            for wall in keys:
                self.surfaces.get(wall).set_spacing(
                    x_spacing=x_spacing, y_spacing=y_spacing
                )
        else:
            if wall_id not in keys:
                raise KeyError(f"wall_id must be in {keys}")
            else:
                self.surfaces.get(wall_id).set_spacing(
                    x_spacing=x_spacing, y_spacing=y_spacing
                )

    def set_num_points(self, num_x=None, num_y=None, wall_id=None):
        """set number of x and y points by wall_id or, if wall_if is None, to all walls"""
        keys = self.surfaces.keys()
        if wall_id is None:
            for wall in keys:
                # set for all walls
                self.surfaces.get(wall).set_num_points(num_x=num_x, num_y=num_y)
        else:
            if wall_id not in keys:
                raise KeyError(f"wall_id must be in {keys}")
            else:
                self.surfaces.get(wall_id).set_num_points(num_x=num_x, num_y=num_y)

    # --------------------------- internals ----------------------------

    def _check_lamp_position(self, lamp):
        return self._check_position(lamp.position, lamp.name)

    def _check_zone_position(self, calc_zone):
        x, y, z = calc_zone.coords.T
        dimensions = x.max(), y.max(), z.max()
        return self._check_position(dimensions, calc_zone.name)

    def _check_position(self, dimensions, obj_name):
        """
        Method to check if an object's dimensions exceed the room's boundaries.
        """
        msg = None
        for coord, roomcoord in zip(dimensions, self.dim.dimensions):
            if coord > roomcoord:
                msg = f"{obj_name} exceeds room boundaries!"
                warnings.warn(msg, stacklevel=2)
        return msg
