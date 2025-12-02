import warnings
import inspect
import copy
from .lamp import Lamp
from .calc_zone import CalcPlane, CalcVol
from .room_plotter import RoomPlotter
from .room_dims import RoomDimensions
from .disinfection_calculator import DisinfectionCalculator
from .reflectance import ReflectanceManager, ReflectiveSurface
from .scene import Scene
from .io import load_room, save_room, export_room_zip, generate_report

UNIT_DEFAULTS = {"meters": [6.0, 4.0, 2.7], "feet": [20.0, 13.0, 9.0]}
VALID_UNITS = UNIT_DEFAULTS.keys()


class Room:
    """
    Represents a room containing lamps and calculation zones.

    The room is defined by its dimensions and can contain multiple lamps and
    different calculation zones for evaluating lighting conditions and other
    characteristics.
    """

    def __init__(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        units: str = "meters",
        standard: str = "ANSI IES RP 27.1-22 (ACGIH Limits)",
        enable_reflectance: bool = True,
        reflectances: dict | None = None,
        reflectance_x_spacings: dict | None = None,
        reflectance_y_spacings: dict | None = None,
        reflectance_num_x: dict | None = None,
        reflectance_num_y: dict | None = None,
        reflectance_max_num_passes: int = 100,
        reflectance_threshold: float = 0.02,
        air_changes: float = 1.0,
        ozone_decay_constant: float = 2.7,
        precision: int = 1,
        colormap: str = "plasma",
        on_collision: str = "increment",  # error | increment | overwrite
        unit_mode: str = "auto",  # strict | auto
    ):

        ### Dimensions
        if units.lower() not in VALID_UNITS:
            raise KeyError(f"Invalid unit {units}")
        default = UNIT_DEFAULTS[units.lower()]

        self.dim = RoomDimensions(
            x if x is not None else default[0],
            y if y is not None else default[1],
            z if z is not None else default[2],
            "meters" if units is None else units.lower(),
        )

        ### Misc flags
        self.standard = standard
        self.air_changes = air_changes
        self.ozone_decay_constant = ozone_decay_constant
        self.precision = precision

        ### Scene - lamps, zones, and reflective surfaces
        self.scene = Scene(
            dim=self.dim,
            on_collision=on_collision,
            unit_mode=unit_mode,
            colormap=colormap,
        )
        # may be overriden by any registered surfaces
        self.scene.init_standard_surfaces(
            reflectances=reflectances,
            x_spacings=reflectance_x_spacings,
            y_spacings=reflectance_y_spacings,
            num_x=reflectance_num_x,
            num_y=reflectance_num_y,
        )
        self.lamps = self.scene.lamps
        self.calc_zones = self.scene.calc_zones
        self.surfaces = self.scene.surfaces

        ### Reflectance
        self.enable_reflectance = enable_reflectance

        self.ref_manager = ReflectanceManager(
            surfaces=self.surfaces,
            max_num_passes=reflectance_max_num_passes,
            threshold=reflectance_threshold,
        )

        ### Plotting and data extraction
        self._plotter = RoomPlotter(self)
        self._disinfection = DisinfectionCalculator(self)

        self.calc_state = {}
        self.update_state = {}

    def _eq_dict(self):
        data = self.to_dict()

        # Strip volatile / representation-only fields from lamps
        for lamp_data in data["lamps"].values():
            lamp_data.pop("filedata", None)
            lamp_data.pop("filename", None)

        return data

    def __eq__(self, other):
        if not isinstance(other, Room):
            return NotImplemented

        if self.lamps != other.lamps:
            return False

        for lamp_id in self.lamps.keys():
            if self.lamps[lamp_id] != other.lamps[lamp_id]:
                return False

        return self._eq_dict() == other._eq_dict()

    def __repr__(self):
        return (
            f"Room(x={self.dim.x}, y={self.dim.y}, z={self.dim.z}, "
            f"units='{self.units}', lamps={[k for k,v in self.lamps.items()]}, "
            f"calc_zones={[k for k,v in self.calc_zones.items()]}), "
            f"enable_reflectance={self.enable_reflectance}, "
            f"reflectances={self.ref_manager.reflectances}"
        )

    def copy(self):
        return copy.deepcopy(self)

    def to_dict(self):
        data = {}
        data["x"] = self.dim.x
        data["y"] = self.dim.y
        data["z"] = self.dim.z
        data["units"] = self.units

        data["enable_reflectance"] = self.enable_reflectance
        # data["reflectances"] = self.ref_manager.reflectances
        # data["reflectance_x_spacings"] = self.ref_manager.x_spacings
        # data["reflectance_y_spacings"] = self.ref_manager.y_spacings
        # data["reflectance_max_num_passes"] = self.ref_manager.max_num_passes
        # data["reflectance_threshold"] = self.ref_manager.threshold

        data["standard"] = self.standard
        data["air_changes"] = self.air_changes
        data["ozone_decay_constant"] = self.ozone_decay_constant
        data["precision"] = self.precision
        data["on_collision"] = self.scene.on_collision
        data["unit_mode"] = self.scene.unit_mode
        data["colormap"] = self.scene.colormap

        dct = self.__dict__.copy()
        data["surfaces"] = {k: v.to_dict() for k, v in dct["surfaces"].items()}
        data["lamps"] = {k: v.to_dict() for k, v in dct["lamps"].items()}
        data["calc_zones"] = {k: v.to_dict() for k, v in dct["calc_zones"].items()}
        return data

    @classmethod
    def from_dict(cls, data: dict):
        """recreate a room from a dict"""

        room_kwargs = list(inspect.signature(cls.__init__).parameters.keys())[1:]
        
        # room has been created with default surfaces, they can be overwritten
        room = cls(**{k: v for k, v in data.items() if k in room_kwargs})

        for surface_id, surface in data["surfaces"].items():
            room.add_surface(ReflectiveSurface.from_dict(surface), on_collision='overwrite')

        for lampid, lamp in data["lamps"].items():
            room.add_lamp(Lamp.from_dict(lamp))

        for zoneid, zone in data["calc_zones"].items():
            if zone["calctype"] == "Plane":
                room.add_calc_zone(CalcPlane.from_dict(zone))
            elif zone["calctype"] == "Volume":
                room.add_calc_zone(CalcVol.from_dict(zone))      

        return room

    def save(self, fname=None):
        """save all relevant parameters to a json file"""
        return save_room(self, fname)

    @classmethod
    def load(cls, filedata):
        """load a room from a json object"""
        return load_room(filedata)

    def export_zip(
        self,
        fname=None,
        include_plots=False,
        include_lamp_files=False,
        include_lamp_plots=False,
    ):
        """
        write the project file and all results files to a zip file. Optionally include
        extra files like lamp ies files, spectra files, and plots.
        """
        return export_room_zip(
            self,
            fname=fname,
            include_plots=include_plots,
            include_lamp_files=include_lamp_files,
            include_lamp_plots=include_lamp_plots,
        )

    def generate_report(self, fname=None):
        """generate a csv report of all the rooms contents and zone statistics"""
        return generate_report(self, fname)

    def get_calc_state(self):
        """
        Save all the features in the room that, if changed, will require re-calculation
        """

        lamp_state = {}
        for key, lamp in self.scene.lamps.items():
            if lamp.enabled:
                lamp_state[key] = lamp.calc_state

        zone_state = {}
        for key, zone in self.scene.calc_zones.items():
            if zone.calctype != "Zone" and zone.enabled:
                zone_state[key] = zone.calc_state

        ref_state = (self.enable_reflectance,) + self.ref_manager.calc_state

        filter_state = {}
        for key, filt in self.scene.filters.items():
            filter_state[key] = filt.get_calc_state()

        calc_state = {}
        calc_state["lamps"] = lamp_state
        calc_state["calc_zones"] = zone_state
<<<<<<< HEAD
        calc_state["filters"] = filter_state
=======
        calc_state["reflectance"] = ref_state
>>>>>>> c66aa8e (overhaul of how reflective surfaces are handled + update to scene unique id generation)

        return calc_state

    def get_update_state(self):
        """
        Save all the features in the room that should NOT trigger
        a recalculation, only an update
        """

        lamp_state = {}
        for key, lamp in self.scene.lamps.items():
            lamp_state[key] = lamp.update_state

        zone_state = {}
        for key, zone in self.scene.calc_zones.items():
            if zone.calctype != "Zone":
                zone_state[key] = zone.update_state

        ref_state = (
            tuple(self.ref_manager.reflectances.values()),
            self.units,
        )

        update_state = {}
        update_state["lamps"] = lamp_state
        update_state["calc_zones"] = zone_state
        update_state["reflectance"] = ref_state

        return update_state

    @property
    def recalculate_incidence(self) -> bool:
        """
        TODO: possibly should be off in the ref_manager
        true of incident reflections need recalculation
        """
        new_calc_state = self.get_calc_state()
        new_update_state = self.get_update_state()
        LAMP_RECALC = self.calc_state.get("lamps") != new_calc_state.get("lamps")
        REF_RECALC = self.calc_state.get("reflectance") != new_calc_state.get(
            "reflectance"
        )
        REF_UPDATE = self.update_state.get("reflectance") != new_update_state.get(
            "reflectance"
        )
        return LAMP_RECALC or REF_RECALC or REF_UPDATE

    # --------------------- Misc flags -----------------------

    def set_standard(self, standard, preserve_spacing=True):
        """update the photobiological safety standard the Room is subject to"""
        self.standard = standard
        self.scene.update(
            dim=self.dim, standard=self.standard, preserve_spacing=preserve_spacing
        )
        return self

    # --------------------- Reflectance ----------------------

    def set_reflectance(self, R, wall_id=None):
        """
        set the reflectance (a float between 0 and 1) for the reflective walls
        If wall_id is none, the value is set for all walls.
        """
        self.scene.set_reflectance(R=R, wall_id=wall_id)
        return self

    def set_reflectance_spacing(self, x_spacing=None, y_spacing=None, wall_id=None):
        """
        set the spacing of the calculation points for the reflective walls
        If wall_id is none, the value is set for all walls.
        """
        self.scene.set_spacing(
            x_spacing=x_spacing, y_spacing=y_spacing, wall_id=wall_id
        )
        return self

    def set_reflectance_num_points(self, num_x=None, num_y=None, wall_id=None):
        """
        set the number of calculation points for the reflective walls
        If wall_id is none, the value is set for all walls.
        """
        self.scene.set_num_points(num_x=num_x, num_y=num_y, wall_id=wall_id)
        return self

    def set_max_num_passes(self, max_num_passes):
        """set the maximum number of passes for the interreflection module"""
        self.ref_manager.max_num_passes = max_num_passes
        return self

    def set_reflectance_threshold(self, reflectance_threshold):
        """
        set the threshold percentage (a float between 0 and 1) for the
        interreflection module
        """
        self.ref_manager.threshold = reflectance_threshold
        return self

    # -------------- Dimensions and Units -----------------------

    def set_units(self, units, unit_mode="auto", preserve_spacing=True):
        """set room units"""
        if units.lower() not in VALID_UNITS:
            raise KeyError("Valid units are `meters` or `feet`")
        self.dim = self.dim.with_(units=units)
        self.scene.update(
            self.dim, preserve_spacing=preserve_spacing, unit_mode=unit_mode
        )
        return self

    def set_dimensions(self, x=None, y=None, z=None, preserve_spacing=True):
        """set room dimensions"""
        self.dim = self.dim.with_(x=x, y=y, z=z)
        self.scene.update(
            dim=self.dim, standard=self.standard, preserve_spacing=preserve_spacing
        )
        return self

    @property
    def units(self) -> str:
        return self.dim.units

    @property
    def x(self) -> float:
        return self.dim.x

    @property
    def y(self) -> float:
        return self.dim.y

    @property
    def z(self) -> float:
        return self.dim.z

    @units.setter
    def units(self, value: str):
        self.set_units(value)

    @x.setter
    def x(self, value: float):
        self.set_dimensions(x=value)

    @y.setter
    def y(self, value: float):
        self.set_dimensions(y=value)

    @z.setter
    def z(self, value: float):
        self.set_dimensions(z=value)

    @property
    def dimensions(self) -> tuple[float, float, float]:
        return (self.dim.x, self.dim.y, self.dim.z)

    @property
    def volume(self) -> float:
        return self.dim.volume

    # -------------------- Scene: lamps, zones, surfaces ---------------------

    def add(self, *args, on_collision=None, unit_mode=None):
        """
        Add objects to the Scene.
        - If an object is a Lamp, it is added as a lamp.
        - If an object is a CalcZone, CalcPlane, or CalcVol, it is added as a calculation zone.
        - If an object is iterable, it is recursively processed.
        - Otherwise, a warning is printed.
        """
        self.scene.add(*args, on_collision=on_collision, unit_mode=unit_mode)
        return self

    def add_lamp(self, lamp, on_collision=None, unit_mode=None):
        """
        Add a lamp to the room scene
        """
        self.scene.add_lamp(lamp, on_collision=on_collision, unit_mode=unit_mode)
        return self

    def place_lamp(self, lamp, on_collision=None, unit_mode=None):
        """
        Position a lamp as far from other lamps and the walls as possible
        """
        self.scene.place_lamp(lamp, on_collision=on_collision, unit_mode=unit_mode)
        return self

    def place_lamps(self, *args, on_collision=None, unit_mode=None):
        """
        Place multiple lamps in the room, as far away from each other and the walls as possible
        """
        self.scene.place_lamps(*args, on_collision=on_collision, unit_mode=unit_mode)
        return self

    def remove_lamp(self, lamp_id):
        """Remove a lamp from the room scene"""
        self.scene.remove_lamp(lamp_id)
        return self

    def add_calc_zone(self, calc_zone, on_collision=None):
        """
        Add a calculation zone to the room
        """
        self.scene.add_calc_zone(calc_zone, on_collision=on_collision)
        return self

    def add_standard_zones(self, on_collision=None):
        """
        Add the special calculation zones SkinLimits, EyeLimits, and
        WholeRoomFluence to the room scene.
        If not overridden by user, standard zones are overwritten, not incremented
        """
        policy = on_collision or "overwrite"
        self.scene.add_standard_zones(self.standard, on_collision=policy)
        return self

    def remove_calc_zone(self, zone_id):
        """Remove a calculation zone from the room"""
        self.scene.remove_calc_zone(zone_id)
        return self

<<<<<<< HEAD
    def add_filter(self, filt, on_collision=None):
        """Add a measured correction filter to the room"""
        self.scene.add_filter(filt=filt, on_collision=on_collision)
        return self

    def remove_filter(self, filt_id):
        """remove a measured correction filter from the room"""
        self.scene.remove_filter(filt_id)
=======
    def add_surface(self, surface, on_collision=None):
        self.scene.add_surface(surface, on_collision=on_collision)
        return self

    def remove_surface(self, surface_id):
        self.scene.remove_surface(surface_id)
>>>>>>> c66aa8e (overhaul of how reflective surfaces are handled + update to scene unique id generation)
        return self

    def set_colormap(self, colormap):
        """
        Set the room colormap
        """
        self.scene.set_colormap(colormap)
        return self

    def check_positions(self):
        """
        Verify the positions of all objects in the scene and return any warning messages
        """
        msgs = self.scene.check_positions()
        return msgs

    # -------------------------- Calculation ---------------------------

    def calculate(self, hard=False):
        """
        Triggers the calculation of lighting values in each calculation zone
        based on the current lamps in the room.

        If no updates have been made since the last calculate call that would
        require a full recalculation, either only an update will be performed
        or no recalculation will occur.

        If `hard` is True, this behavior is overriden and the full
        recalculation will be performed
        """

        valid_lamps = self.scene.get_valid_lamps()
        # calculate incidence on the surfaces if the reflectances or lamps have changed
        if (self.recalculate_incidence or hard) and self.enable_reflectance:
            self.ref_manager.calculate_incidence(valid_lamps, hard=hard)

        ref_manager = self.ref_manager if self.enable_reflectance else None

        # calculate incidence on any correction filters
        for filt in self.filters.values():
            filt.calculate_values(valid_lamps, ref_manager=ref_manager, hard=hard)

        # main calculation loop
        for name, zone in self.calc_zones.items():
            if zone.enabled:
                zone.calculate_values(
                    lamps=valid_lamps,
                    ref_manager=ref_manager,
                    filters=self.filters,
                    obstacles=self.obstacles,
                    hard=hard,
                )
        # update calc states.
        self.calc_state = self.get_calc_state()
        self.update_state = self.get_update_state()

        if len(valid_lamps) == 0:
            msg = "No valid lamps are present in the room--either lamps have been disabled, or filedata has not been provided."
            if len(self.lamps) == 0:
                msg = "No lamps are present in the room."
            warnings.warn(msg, stacklevel=3)

        return self

    def calculate_by_id(self, zone_id, hard=False):
        """calculate just the calc zone selected"""
        valid_lamps = self.scene.get_valid_lamps()

        if len(valid_lamps) > 0:

            # calculate incidence on the surfaces if the reflectances or lamps have changed
            if (self.recalculate_incidence or hard) and self.enable_reflectance:
                self.ref_manager.calculate_incidence(valid_lamps, hard=hard)
            ref_manager = self.ref_manager if self.enable_reflectance else None
            self.calc_zones[zone_id].calculate_values(
                lamps=valid_lamps, ref_manager=ref_manager, hard=hard
            )
            self.calc_state = self.get_calc_state()
            self.update_state = self.get_update_state()
        return self

    # ------------------- Data and Plotting ----------------------

    def get_disinfection_data(self, zone_id="WholeRoomFluence"):
        """return the fluence_dict, dataframe, and violin plot"""
        return self._disinfection.get_disinfection_data(zone_id=zone_id)

    def plotly(self, fig=None, select_id=None, title=""):
        """return a plotly figure of all the room's components"""
        return self._plotter.plotly(fig=fig, select_id=select_id, title=title)

    def plot(self, fig=None, select_id=None, title=""):
        """alias for plotly"""
        return self._plotter.plotly(fig=fig, select_id=select_id, title=title)
