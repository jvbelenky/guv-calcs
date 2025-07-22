import warnings
import inspect
from .lamp import Lamp
from .calc_zone import CalcPlane, CalcVol
from .room_plotter import RoomPlotter
from .room_dims import RoomDimensions
from .disinfection_calculator import DisinfectionCalculator
from .reflectance import ReflectanceManager
from .scene import Scene
from .io import load_room, save_room, export_room_zip, generate_report

VALID_UNITS = ["meters", "feet"]


class Room:
    """
    Represents a room containing lamps and calculation zones.

    The room is defined by its dimensions and can contain multiple lamps and
    different calculation zones for evaluating lighting conditions and other
    characteristics.
    """

    def __init__(
        self,
        x=None,
        y=None,
        z=None,
        units=None,
        standard=None,
        enable_reflectance=None,
        reflectances=None,
        reflectance_x_spacings=None,
        reflectance_y_spacings=None,
        reflectance_max_num_passes=None,
        reflectance_threshold=None,
        air_changes=None,
        ozone_decay_constant=None,
        precision=1,
        colormap="plasma",
        on_collision="increment",  # error | increment | overwrite
        unit_mode="auto",  # strict | auto
    ):

        ### Dimensions
        if units is not None:
            if units.lower() not in VALID_UNITS:
                raise KeyError(f"Invalid unit {units}")
            default = (
                [6.0, 4.0, 2.7] if units.lower() == "meters" else [20.0, 13.0, 9.0]
            )
        else:
            default = [6.0, 4.0, 2.7]

        self.dim = RoomDimensions(
            x if x is not None else default[0],
            y if y is not None else default[1],
            z if z is not None else default[2],
            "meters" if units is None else units.lower(),
        )

        ### Misc flags
        self.standard = standard or "ANSI IES RP 27.1-22 (ACGIH Limits)"
        self.air_changes = air_changes or 1.0
        self.ozone_decay_constant = ozone_decay_constant or 2.7
        self.precision = precision

        ### Scene - lamps and zones
        self.scene = Scene(
            dim=self.dim,
            on_collision=on_collision,
            unit_mode=unit_mode,
            colormap=colormap,
        )
        self.lamps = self.scene.lamps
        self.calc_zones = self.scene.calc_zones

        ### Reflectance
        self.enable_reflectance = (
            True if enable_reflectance is None else enable_reflectance
        )
        self.ref_manager = ReflectanceManager(
            x=self.dim.x,
            y=self.dim.y,
            z=self.dim.z,
            reflectances=reflectances,
            x_spacings=reflectance_x_spacings,
            y_spacings=reflectance_y_spacings,
            max_num_passes=reflectance_max_num_passes,
            threshold=reflectance_threshold,
        )

        ### Plotting and data extraction
        self._plotter = RoomPlotter(self)
        self._disinfection = DisinfectionCalculator(self)

        self.calc_state = {}
        self.update_state = {}

    def to_dict(self):
        data = {}
        data["x"] = self.dim.x
        data["y"] = self.dim.y
        data["z"] = self.dim.z
        data["units"] = self.units
        data["enable_reflectance"] = self.enable_reflectance
        data["reflectances"] = self.ref_manager.reflectances
        data["reflectance_x_spacings"] = self.ref_manager.x_spacings
        data["reflectance_y_spacings"] = self.ref_manager.y_spacings
        data["reflectance_max_num_passes"] = self.ref_manager.max_num_passes
        data["reflectance_threshold"] = self.ref_manager.threshold
        data["standard"] = self.standard
        data["air_changes"] = self.air_changes
        data["ozone_decay_constant"] = self.ozone_decay_constant
        data["precision"] = self.precision
        data["on_collision"] = self.scene.on_collision
        data["unit_mode"] = self.scene.unit_mode
        data["colormap"] = self.scene.colormap

        dct = self.__dict__.copy()
        data["lamps"] = {k: v.to_dict() for k, v in dct["lamps"].items()}
        data["calc_zones"] = {k: v.to_dict() for k, v in dct["calc_zones"].items()}
        return data

    @classmethod
    def from_dict(cls, data: dict):
        """recreate a room from a dict"""

        room_kwargs = list(inspect.signature(cls.__init__).parameters.keys())[1:]
        room = cls(**{k: v for k, v in data.items() if k in room_kwargs})

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

        room_state = [  # temp..this should be put with the ref_manager eventually
            self.enable_reflectance,
            tuple(self.ref_manager.x_spacings),
            tuple(self.ref_manager.y_spacings),
            self.ref_manager.max_num_passes,
            self.ref_manager.threshold,
        ]

        lamp_state = {}
        for key, lamp in self.scene.lamps.items():
            if lamp.enabled:
                lamp_state[key] = lamp.get_calc_state()

        zone_state = {}
        for key, zone in self.scene.calc_zones.items():
            if zone.calctype != "Zone" and zone.enabled:
                zone_state[key] = zone.get_calc_state()

        calc_state = {}
        calc_state["room"] = room_state
        calc_state["lamps"] = lamp_state
        calc_state["calc_zones"] = zone_state

        return calc_state

    def get_update_state(self):
        """
        Save all the features in the room that should NOT trigger
        a recalculation, only an update
        """

        room_state = [
            tuple(self.ref_manager.reflectances),
            self.units,
        ]

        lamp_state = {}
        for key, lamp in self.scene.lamps.items():
            lamp_state[key] = lamp.get_update_state()

        zone_state = {}
        for key, zone in self.scene.calc_zones.items():
            if zone.calctype != "Zone":
                zone_state[key] = zone.get_update_state()

        update_state = {}
        update_state["room"] = room_state
        update_state["lamps"] = lamp_state
        update_state["calc_zones"] = zone_state

        return update_state

    # --------------------- Misc flags -----------------------

    def set_standard(self, standard):
        """update the photobiological safety standard the Room is subject to"""
        self.standard = standard
        self.scene.update_standard_zones(standard)
        return self

    # --------------------- Reflectance ----------------------

    def set_reflectance(self, R, wall_id=None):
        """
        set the reflectance (a float between 0 and 1) for the reflective walls
        If wall_id is none, the value is set for all walls.
        """
        self.ref_manager.set_reflectance(R=R, wall_id=wall_id)
        return self

    def set_reflectance_spacing(self, x_spacing=None, y_spacing=None, wall_id=None):
        """
        set the spacing of the calculation points for the reflective walls
        If wall_id is none, the value is set for all walls.
        """
        self.ref_manager.set_spacing(
            x_spacing=x_spacing, y_spacing=y_spacing, wall_id=wall_id
        )
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

    def set_units(self, units, unit_mode="auto"):
        """set room units"""
        if units.lower() not in ["meters", "feet"]:
            raise KeyError("Valid units are `meters` or `feet`")
        self.dim = self.dim.with_(units=units)

        self.scene.dim = self.dim
        self.scene.update_standard_zones(self.standard)
        self.scene.to_units(unit_mode=unit_mode)
        return self

    def set_dimensions(self, x=None, y=None, z=None):
        """set room dimensions"""
        self.dim = self.dim.with_(x=x, y=y, z=z)
        self.ref_manager.update_dimensions(self.dim.x, self.dim.y, self.dim.z)
        self.scene.dim = self.dim
        self.scene.update_standard_zones(self.standard)
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
        return self.dim.volume()

    # -------------------- Scene: lamps and zones ---------------------

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
        """
        Remove a calculation zone from the room
        """
        self.scene.remove_calc_zone(zone_id)
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

        new_calc_state = self.get_calc_state()
        new_update_state = self.get_update_state()

        LAMP_RECALC = self.calc_state.get("lamps") != new_calc_state.get("lamps")
        REF_RECALC = self.calc_state.get("room") != new_calc_state.get("room")
        REF_UPDATE = self.update_state.get("room") != new_update_state.get("room")

        # calculate incidence on the surfaces if the reflectances or lamps have changed
        if (
            LAMP_RECALC or REF_RECALC or REF_UPDATE or hard
        ) and self.enable_reflectance:
            self.ref_manager.calculate_incidence(valid_lamps, hard=hard)

        ref_manager = self.ref_manager if self.enable_reflectance else None
        for name, zone in self.calc_zones.items():
            if zone.enabled:
                zone.calculate_values(
                    lamps=valid_lamps, ref_manager=ref_manager, hard=hard
                )
        # update calc states.
        self.calc_state = new_calc_state
        self.update_state = new_update_state

        # possibly this should be per-calc zone? idk maybe it's fine
        for lamp_id in valid_lamps.keys():
            new_calc_state = self.lamps[lamp_id].get_calc_state()
            self.lamps[lamp_id].calc_state = new_calc_state

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
            new_calc_state = self.get_calc_state()
            new_update_state = self.get_update_state()

            LAMP_RECALC = self.calc_state.get("lamps") != new_calc_state.get("lamps")
            REF_RECALC = self.calc_state.get("room") != new_calc_state.get("room")
            REF_UPDATE = self.update_state.get("room") != new_update_state.get("room")

            # calculate incidence on the surfaces if the reflectances or lamps have changed
            if (
                LAMP_RECALC or REF_RECALC or REF_UPDATE or hard
            ) and self.enable_reflectance:
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
