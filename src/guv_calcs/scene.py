from collections.abc import Iterable
import warnings
from matplotlib import colormaps
from .room_dims import RoomDimensions
from .lamp import Lamp
from .calc_zone import CalcZone, CalcPlane, CalcVol
from .filters import FilterBase
from .obstacles import BoxObstacle
from .reflectance import ReflectiveSurface, init_room_surfaces
from .lamp_helpers import new_lamp_position


class Scene:
    def __init__(
        self, dim: RoomDimensions, unit_mode: str, on_collision: str, colormap: str
    ):
        self.dim = dim
        self.unit_mode: str = unit_mode  # "strict" → raise; "auto" → convert in place
        self.on_collision: str = on_collision  # error | increment | overwrite"
        self.colormap: str = colormap

        self.lamps: dict[str, Lamp] = {}
        self.calc_zones: dict[str, CalcZone] = {}
        
        self.filters: dict[str, FilterBase] = {}
        self.obstacles: dict[str, BoxObstacle] = {}

        self.surfaces: dict[str, ReflectiveSurface] = {}

    # ------------------------ Global --------------------------
    def add(self, *args, on_collision=None, unit_mode=None):
        """
        Add objects to the Scene.
        - If an object is a Lamp, it is added as a lamp.
        - If an object is a CalcZone, CalcPlane, or CalcVol, it is added as a calculation zone.
        - If an object is iterable, it is recursively processed.
        - Otherwise, a warning is printed.
        """

        for obj in args:
            if isinstance(obj, Lamp):
                self.add_lamp(obj, on_collision=on_collision, unit_mode=unit_mode)
            elif isinstance(obj, CalcZone):
                self.add_calc_zone(obj, on_collision=on_collision)
            elif isinstance(obj, FilterBase):
                self.add_filter(obj, on_collision=on_collision)
            elif isinstance(obj, BoxObstacle):
                self.add_obstacle(obj, on_collision=on_collision)
            elif isinstance(obj, ReflectiveSurface):
                self.add_surface(obj, on_collision=on_collision)
            elif isinstance(obj, dict):
                self.add(*obj.values(), on_collision=on_collision)
            elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
                self.add(
                    *obj, on_collision=on_collision
                )  # Recursively process other iterables
            else:
                msg = f"Cannot add object of type {type(obj).__name__} to Room."
                warnings.warn(msg, stacklevel=3)

    def update_dimensions(self, x=None, y=None, z=None):
        """update the dimensions and any objects that depend on the dimensions"""
        self.dim = self.dim.with_(x=x, y=y, z=z)
        self.update_standard_surfaces()

    def update_units(self, units, unit_mode=None):
        self.dim = self.dim.with_(units=units)
        self.to_units(unit_mode)

    def set_colormap(self, colormap: str):
        """Set the scene's colormap"""
        if colormap not in list(colormaps):
            warnings.warn(f"{colormap} is not a valid colormap.")
        else:
            self.colormap = colormap
            for zone in self.calc_zones.values():
                zone.colormap = self.colormap
            for surface in self.surfaces.values():
                surface.plane.colormap = self.colormap

    def check_positions(self):
        """
        verify the positions of all objects in the scene and return any warning messages
        """
        lamps, zones, surfaces = {}, {}, {}
        for lamp_id, lamp in self.lamps.items():
            lamps[lamp_id] = self._check_lamp_position(lamp)
        for zone_id, zone in self.calc_zones.items():
            zones[zone_id] = self._check_zone_position(zone)
        for surface_id, surface in self.surfaces.items():
            surfaces[surface_id] = self._check_zone_position(surface.plane)

        msgs = {}
        msgs["lamps"] = lamps
        msgs["calc_zones"] = zones
        msgs["surfaces"] = surfaces
        return msgs

    # ------------------ Lamps -----------------------

    def add_lamp(self, lamp, base_id="Lamp", on_collision=None, unit_mode=None):
        """Add a lamp to the room"""

        lamp_id = self._get_id(
            mapping=self.lamps,
            obj_id=lamp.lamp_id,
            base_id=base_id,
            on_collision=on_collision,
        )
        lamp.lamp_id = lamp_id
        if lamp.name is None:
            lamp.name = lamp_id
        self.lamps[lamp_id] = self._check_lamp(lamp, unit_mode=unit_mode)
        return self

    def place_lamp(self, lamp_arg, on_collision=None, unit_mode=None):
        """
        Position a lamp as far from other lamps and the walls as possible
        """
        # ok I'm adding this as a convenience
        if isinstance(lamp_arg, Lamp):
            lamp = lamp_arg
        elif isinstance(lamp_arg, str):
            lamp = Lamp.from_keyword(lamp_arg)
        elif isinstance(lamp_arg, int):
            lamp = Lamp.from_index(lamp_arg)
        else:
            raise TypeError(
                f"{type(lamp_arg)} is not a valid Lamp or method of generating a Lamp"
            )

        idx = len(self.lamps) + 1
        x, y = new_lamp_position(idx, self.dim.x, self.dim.y)
        lamp.move(x, y, self.dim.z)
        self.add_lamp(lamp, on_collision=on_collision, unit_mode=unit_mode)

    def place_lamps(self, *args, on_collision=None, unit_mode=None):
        """place multiple lamps in the room, as far away from each other and the walls as possible"""
        for obj in args:
            if isinstance(obj, Lamp):
                self.place_lamp(obj, on_collision=on_collision, unit_mode=unit_mode)
            else:
                msg = f"Cannot add object of type {type(obj).__name__} to Room."
                warnings.warn(msg, stacklevel=3)

    def remove_lamp(self, lamp_id):
        """Remove a lamp from the scene"""
        self.lamps.pop(lamp_id, None)

    def get_valid_lamps(self):
        """return all the lamps that can participate in a calculation"""
        return {
            k: v for k, v in self.lamps.items() if v.enabled and v.filedata is not None
        }

    def to_units(self, unit_mode=None):
        """
        ensure that all lamps in the state have the correct units, or raise an error
        in strict mode
        """
        for lamp in self.lamps.values():
            self._check_lamp_units(lamp, unit_mode=unit_mode)

    # -------------- Zones ------------------
    def add_calc_zone(self, zone, base_id=None, on_collision=None):
        """
        Add a calculation zone to the scene
        """
        if base_id is None:
            base_id = "Calc" + zone.calctype

        zone_id = self._get_id(
            mapping=self.calc_zones,
            obj_id=zone.zone_id,
            base_id=base_id,
            on_collision=on_collision,
        )
        zone.zone_id = zone_id
        if zone.name is None:
            zone.name = zone_id
        zone.colormap = self.colormap
        self.calc_zones[zone_id] = self._check_zone(zone)

    def remove_calc_zone(self, zone_id):
        """remove calculation zone from scene"""
        self.calc_zones.pop(zone_id, None)

    def add_filter(self, filt, base_id="Filter", on_collision=None):
        """add a correction filter to the scene"""
        filter_id = self._get_id(
            mapping=self.filters,
            obj_id=filt.filter_id,
            base_id=base_id,
            counter=self._filter_counter,
            on_collision=on_collision,
        )
        filt.filter_id = filter_id
        if filt.name is None:
            filt.name = filter_id
        self.filters[filter_id] = filt

    def remove_filter(self, filter_id):
        """remove a correction filter from the scene"""
        self.filters.pop(filter_id, None)

    def add_obstacle(self, obs, base_id="Obstacle", on_collision=None):
        """add a 3d box obstacle to the scene"""
        obs_id = self._get_id(
            mapping=self.obstacles,
            obj_id=obs.obs_id,
            base_id=base_id,
            counter=self._obstacle_counter,
            on_collision=on_collision,
        )
        obs.obs_id = obs_id
        if obs.name is None:
            obs.name = obs_id
        self.obstacles[obs_id] = obs

    def remove_obstacle(self, obs_id):
        self.obstacles.pop(obs_id, None)

    def add_standard_zones(self, standard, *, on_collision=None):
        """
        Add the special calculation zones SkinLimits, EyeLimits, and
        WholeRoomFluence to the scene
        """
        standard_zones = [
            CalcVol(
                zone_id="WholeRoomFluence",
                name="Whole Room Fluence",
                show_values=False,
            ),
            CalcPlane(
                zone_id="EyeLimits",
                name="Eye Dose (8 Hours)",
                dose=True,
                hours=8,
                direction=0,
            ),
            CalcPlane(
                zone_id="SkinLimits",
                name="Skin Dose (8 Hours)",
                dose=True,
                hours=8,
                direction=0,
            ),
        ]

        self.add(standard_zones, on_collision=on_collision)
        # sets the height and field of view parameters
        self.update_standard_zones(standard, preserve_spacing=True)

    def update_standard_zones(self, standard: str, preserve_spacing: bool):
        """
        update the standard safety calculation zones based on the current
        standard, units, and room dimensions
        """
        if "UL8802" in standard:
            height = 1.9 if self.dim.units == "meters" else 6.25
            skin_horiz = False
            eye_vert = False
            fov_vert = 180
        else:
            height = 1.8 if self.dim.units == "meters" else 5.9
            skin_horiz = True
            eye_vert = True
            fov_vert = 80

        if "SkinLimits" in self.calc_zones.keys():
            zone = self.calc_zones["SkinLimits"]
            zone.set_dimensions(
                x2=self.dim.x, y2=self.dim.y, preserve_spacing=preserve_spacing
            )
            zone.update_from_legacy(height=height, ref_surface="xy", direction=1)
            zone.horiz = skin_horiz
        if "EyeLimits" in self.calc_zones.keys():
            zone = self.calc_zones["EyeLimits"]
            zone.set_dimensions(
                x2=self.dim.x, y2=self.dim.y, preserve_spacing=preserve_spacing
            )
            zone.update_from_legacy(height=height, ref_surface="xy", direction=1)
            zone.fov_vert = fov_vert
            zone.vert = eye_vert
        if "WholeRoomFluence" in self.calc_zones.keys():
            zone = self.calc_zones["WholeRoomFluence"]
            zone.set_dimensions(
                x2=self.dim.x,
                y2=self.dim.y,
                z2=self.dim.z,
                preserve_spacing=preserve_spacing,
            )

    # ------------------- Surfaces ----------------------

    def add_surface(self, surface, base_id="Surface", on_collision=None):
        """add a reflective surface to the room"""
        surface_id = self._get_id(
            mapping=self.surfaces,
            obj_id=surface.plane.zone_id,
            base_id=base_id,
            on_collision=on_collision,
        )
        surface.plane.zone_id = surface_id
        surface.plane.colormap = self.colormap
        self.surfaces[surface_id] = self._check_surface(surface)

    def remove_surface(self, surface_id):
        """remove reflective surface from the room"""
        self.surfaces.pop(surface_id, None)

    def init_standard_surfaces(
        self,
        reflectances=None,
        x_spacings=None,
        y_spacings=None,
        num_x=None,
        num_y=None,
    ):
        """add surfaces to the scene corresponding to the 6 faces of a rectangular room"""
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
        """update the 6 faces of the room with the new dimensions"""
        keys = self.dim.faces.keys()
        room_surfaces = init_room_surfaces(
            dims=self.dim,
            reflectances={key: self.surfaces[key].R for key in keys},
            x_spacings={key: self.surfaces[key].plane.x_spacing for key in keys},
            y_spacings={key: self.surfaces[key].plane.y_spacing for key in keys},
            num_x={key: self.surfaces[key].plane.num_x for key in keys},
            num_y={key: self.surfaces[key].plane.num_y for key in keys},
        )
        for key, val in room_surfaces.items():
            self.add_surface(val, on_collision="overwrite")

    def set_reflectance(self, R, wall_id=None):
        """set reflectance by wall_id or, if wall_if is None, to all walls"""
        keys = self.surfaces.keys()
        if wall_id is None:
            # set this value for all walls
            for wall in keys:
                self.surfaces[wall].set_reflectance(R)
        else:
            if wall_id not in keys:
                raise KeyError(f"wall_id must be in {keys}")
            else:
                self.surfaces[wall_id].set_reflectance(R)

    def set_spacing(self, x_spacing=None, y_spacing=None, wall_id=None):
        """set x and y spacing by wall_id or, if wall_if is None, to all walls"""
        keys = self.surfaces.keys()
        if wall_id is None:
            # set this value for all walls
            for wall in keys:
                self.surfaces[wall].set_spacing(
                    x_spacing=x_spacing, y_spacing=y_spacing
                )
        else:
            if wall_id not in keys:
                raise KeyError(f"wall_id must be in {keys}")
            else:
                self.surfaces[wall_id].set_spacing(
                    x_spacing=x_spacing, y_spacing=y_spacing
                )

    def set_num_points(self, num_x=None, num_y=None, wall_id=None):
        """set number of x and y points by wall_id or, if wall_if is None, to all walls"""
        keys = self.surfaces.keys()
        if wall_id is None:
            for wall in keys:
                # set for all walls
                self.surfaces[wall].set_num_points(num_x=num_x, num_y=num_y)
        else:
            if wall_id not in keys:
                raise KeyError(f"wall_id must be in {keys}")
            else:
                self.surfaces[wall_id].set_num_points(num_x=num_x, num_y=num_y)

    def set_colormap(self, colormap: str):
        """Set the scene's colormap"""
        if colormap not in list(colormaps):
            warnings.warn(f"{colormap} is not a valid colormap.")
        else:
            self.colormap = colormap
            for zone in self.calc_zones.values():
                zone.colormap = self.colormap

    # --------------------------- internals ----------------------------

    def _get_id(self, mapping, obj_id, base_id, on_collision=None):
        """Generate an ID for a lamp, zone, or surface object.

        - If obj_id is None: generate a unique ID based on base_id.
        - If obj_id is provided and not in mapping: use it as-is.
        - If obj_id is provided and in mapping:
            * 'error'      -> raise
            * 'overwrite'  -> reuse obj_id
            * 'increment'  -> generate a unique variant of obj_id
        """
        policy = on_collision or self.on_collision

        # No explicit id → generate from base_id
        if obj_id is None:
            return self._unique_id(str(base_id), mapping)

        obj_id = str(obj_id)

        # Existing id
        if obj_id in mapping:
            if policy == "error":
                raise ValueError(f"'{obj_id}' already exists")
            elif policy == "overwrite":
                # explicitly replace existing entry; no renaming
                return obj_id
            # default: increment policy → find a free variant of obj_id
            return self._unique_id(obj_id, mapping)

        # No collision → accept obj_id unchanged
        return obj_id

    def _unique_id(self, base: str, mapping: dict) -> str:
        """
        Return a unique id based on `base`, using existing keys in `mapping`.

        - If `base` is unused, return `base`.
        - Otherwise look for keys like `base`, `base-2`, `base-3`, ... and
          return the next free suffix (e.g. 'Lamp-5' if Lamp, Lamp-2..Lamp-4 exist).
        """
        if base not in mapping:
            return base

        prefix = base + "-"
        max_suffix = 1  # 1 corresponds to plain `base` being present

        for key in mapping.keys():
            if key == base:
                max_suffix = max(max_suffix, 1)
            elif key.startswith(prefix):
                rest = key[len(prefix) :]
                if rest.isdigit():
                    n = int(rest)
                    if n > max_suffix:
                        max_suffix = n

        # Next free number
        return f"{base}-{max_suffix + 1}"

    def _check_lamp(self, lamp, unit_mode=None):
        """check lamp position and units"""
        if not isinstance(lamp, Lamp):
            raise TypeError(f"Must be type Lamp, not {type(lamp)}")
        self._check_lamp_position(lamp)
        self._check_lamp_units(lamp, unit_mode)
        return lamp

    def _check_lamp_units(self, lamp, unit_mode=None):
        """convert lamp units, or raise error in strict mode"""
        policy = unit_mode or self.unit_mode
        if lamp.surface.units != self.dim.units:
            if policy == "strict":
                raise ValueError(
                    f"Lamp {lamp.lamp_id} is in {lamp.surface.units}, "
                    f"room is {self.dim.units}"
                )
            lamp.set_units(self.dim.units)

    def _check_zone(self, zone):
        if not isinstance(zone, (CalcZone, CalcPlane, CalcVol)):
            raise TypeError(f"Must be CalcZone, CalcPlane, or CalcVol not {type(zone)}")
        self._check_zone_position(zone)
        return zone

    def _check_surface(self, surface):
        if not isinstance(surface, ReflectiveSurface):
            raise TypeError(f"Must be ReflectiveSurface, not {type(surface)}")
        self._check_zone_position(surface.plane)
        return surface

    def _check_lamp_position(self, lamp):
        return self._check_position(lamp.position, lamp.name)

    def _check_zone_position(self, calc_zone):
        if isinstance(calc_zone, (CalcPlane, CalcVol)):
            x, y, z = calc_zone.coords.T
            dimensions = x.max(), y.max(), z.max()
        elif isinstance(calc_zone, CalcZone):
            # this is a hack; a generic CalcZone is just a placeholder
            dimensions = self.dim.dimensions
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
