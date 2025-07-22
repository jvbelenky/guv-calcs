from collections.abc import Iterable
from collections import defaultdict
import warnings
from matplotlib import colormaps
from .room_dims import RoomDimensions
from .lamp import Lamp
from .calc_zone import CalcZone, CalcPlane, CalcVol
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

        # for generating unique IDs
        self._lamp_counter = defaultdict(int)
        self._zone_counter = defaultdict(int)

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
            elif isinstance(obj, (CalcZone, CalcPlane, CalcVol)):
                self.add_calc_zone(obj, on_collision=on_collision)
            elif isinstance(obj, dict):
                self.add(*obj.values(), on_collision=on_collision)
            elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
                self.add(
                    *obj, on_collision=on_collision
                )  # Recursively process other iterables
            else:
                msg = f"Cannot add object of type {type(obj).__name__} to Room."
                warnings.warn(msg, stacklevel=3)

    def add_lamp(self, lamp, base_id="Lamp", on_collision=None, unit_mode=None):
        """Add a lamp to the room"""
        lamp_id = self._get_id(
            mapping=self.lamps,
            obj_id=lamp.lamp_id,
            base_id=base_id,
            counter=self._lamp_counter,
            on_collision=on_collision,
        )
        lamp.lamp_id = lamp_id
        self.lamps[lamp_id] = self._check_lamp(lamp, unit_mode=unit_mode)
        return self

    def place_lamp(self, lamp, on_collision=None, unit_mode=None):
        """
        Position a lamp as far from other lamps and the walls as possible
        """
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

    def set_colormap(self, colormap: str):
        """Set the scene's colormap"""
        if colormap not in list(colormaps):
            warnings.warn(f"{colormap} is not a valid colormap.")
        else:
            self.colormap = colormap
            for zone in self.calc_zones.values():
                zone.colormap = self.colormap

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
            counter=self._zone_counter,
            on_collision=on_collision,
        )
        zone.zone_id = zone_id
        zone.colormap = self.colormap
        self.calc_zones[zone_id] = self._check_zone(zone)

    def remove_calc_zone(self, zone_id):
        """remove calculation zone from scene"""
        self.calc_zones.pop(zone_id, None)

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
            ),
            CalcPlane(
                zone_id="SkinLimits",
                name="Skin Dose (8 Hours)",
                dose=True,
                hours=8,
            ),
        ]

        self.add(standard_zones, on_collision=on_collision)
        # sets the height and field of view parameters
        self.update_standard_zones(standard)

    def update_standard_zones(self, standard):
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
            zone.set_dimensions(x2=self.dim.x, y2=self.dim.y)
            zone.set_height(height)
            zone.horiz = skin_horiz
        if "EyeLimits" in self.calc_zones.keys():
            zone = self.calc_zones["EyeLimits"]
            zone.set_dimensions(x2=self.dim.x, y2=self.dim.y)
            zone.set_height(height)
            zone.fov_vert = fov_vert
            zone.vert = eye_vert
        if "WholeRoomFluence" in self.calc_zones.keys():
            zone = self.calc_zones["WholeRoomFluence"]
            zone.set_dimensions(x2=self.dim.x, y2=self.dim.y, z2=self.dim.z)

    def check_positions(self):
        """
        verify the positions of all objects in the scene and return any warning messages
        """
        msgs = []
        for lamp_id, lamp in self.lamps.items():
            msgs.append(self._check_lamp_position(lamp))
        for zone_id, zone in self.calc_zones.items():
            msgs.append(self._check_zone_position(zone))
        return msgs

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

    # --------------------------- internals ----------------------------

    def _get_id(self, mapping, obj_id, base_id, counter, on_collision=None):
        """generate an ID for a lamp or calc zone object"""
        policy = on_collision or self.on_collision
        if obj_id is None:
            return self._unique_id(base_id, counter)
        else:
            if obj_id not in mapping:
                return obj_id
            elif policy == "error":
                raise ValueError(f"'{obj_id}' already exists")
            elif policy == "overwrite":
                # msg = f"'{obj_id}' already exists - overwriting"
                # warnings.warn(msg, stacklevel=3)
                return obj_id
            # only remaining policy is to increment
            new_id = self._unique_id(obj_id, counter)
            return new_id

    def _unique_id(self, base: str, counter: defaultdict) -> str:
        counter[base] += 1
        return base if counter[base] == 1 else f"{base}-{counter[base]}"

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
        # self._check_zone_position(zone)
        return zone

    def _check_lamp_position(self, lamp):
        return self._check_position(lamp.position, lamp.name)

    def _check_zone_position(self, calc_zone):
        if isinstance(calc_zone, CalcPlane):
            dimensions = [calc_zone.x2, calc_zone.y2]
        elif isinstance(calc_zone, CalcVol):
            dimensions = [calc_zone.x2, calc_zone.y2, calc_zone.z2]
        elif isinstance(calc_zone, CalcZone):
            # this is a hack; a generic CalcZone is just a placeholder
            dimensions = self.dim.dimensions()
        return self._check_position(dimensions, calc_zone.name)

    def _check_position(self, dimensions, obj_name):
        """
        Method to check if an object's dimensions exceed the room's boundaries.
        """
        msg = None
        for coord, roomcoord in zip(dimensions, self.dim.dimensions()):
            if coord > roomcoord:
                msg = f"{obj_name} exceeds room boundaries!"
                warnings.warn(msg, stacklevel=2)
        return msg
