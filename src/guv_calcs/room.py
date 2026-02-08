import warnings
import inspect
import copy
from collections.abc import Iterable
from matplotlib import colormaps
from .lamp import Lamp, LampPlacer
from .calc_zone import CalcPlane, CalcVol, CalcZone
from .room_plotter import RoomPlotter
from .room_dims import RoomDimensions
from .polygon import Polygon2D
from .reflectance import ReflectanceManager, Surface, init_room_surfaces
from .io import parse_guv_file, save_room_data, export_room_zip, generate_report, get_version
from packaging.version import Version
from pathlib import Path
from .safety import PhotStandard, check_lamps, SafetyCheckResult
from .units import LengthUnits, convert_length
from .efficacy import InactivationData
from .scene_registry import LampRegistry, ZoneRegistry, SurfaceRegistry

DEFAULT_DIMS = {}
for member in list(LengthUnits):
    base = (6.0, 4.0, 2.7)  # meters
    DEFAULT_DIMS[member] = convert_length("meters", member, *base, sigfigs=0)


class Room:
    """
    Represents a room containing lamps and calculation zones.

    The room is defined by its dimensions and can contain multiple lamps and
    different calculation zones for evaluating lighting conditions and other
    characteristics.
    """

    def __init__(
        self,
        room_id: str = None,
        name: str = None,
        x: "float | tuple[float, float] | list[float]" = None,
        y: "float | tuple[float, float] | list[float]" = None,
        z: "float | tuple[float, float] | list[float]" = None,
        polygon: "Polygon2D | list[tuple[float, float]]" = None,
        units: str = "meters",
        standard: str = "ANSI IES RP 27.1-22 (ACGIH Limits)",
        enable_reflectance: bool = True,
        reflectance_max_num_passes: int = 100,
        reflectance_threshold: float = 0.02,
        air_changes: float = 1.0,
        ozone_decay_constant: float = 2.7,
        precision: int = 1,
        colormap: str = "plasma",
        on_collision: str = "increment",  # error | increment | overwrite
    ):

        ### Identity
        self._room_id = room_id or "Room"
        self.name = str(self._room_id) if name is None else str(name)

        ### Dimensions — always polygon-based internally
        units = LengthUnits.from_any(units)

        if polygon is not None:
            if not isinstance(polygon, Polygon2D):
                polygon = Polygon2D(vertices=tuple(tuple(v) for v in polygon))
            self._explicit_polygon = True
        else:
            # Normalize x/y: scalar → (0, val), tuple/list → (min, max)
            def _parse_range(val, default):
                if val is None:
                    return (0.0, float(default))
                if isinstance(val, (tuple, list)):
                    return (float(val[0]), float(val[1]))
                return (0.0, float(val))

            x1, x2 = _parse_range(x, DEFAULT_DIMS[units][0])
            y1, y2 = _parse_range(y, DEFAULT_DIMS[units][1])
            polygon = Polygon2D(vertices=(
                (x1, y1), (x2, y1), (x2, y2), (x1, y2)
            ))
            self._explicit_polygon = False

        # Normalize z: scalar → height, tuple/list → (z_min, z_max) → height = max - min
        if z is None:
            z_val = DEFAULT_DIMS[units][2]
        elif isinstance(z, (tuple, list)):
            z_val = float(z[1]) - float(z[0])
        else:
            z_val = float(z)

        self.dim = RoomDimensions(
            polygon=polygon,
            z=z_val,
            units=units,
        )

        ### Registries — owned directly, no Scene intermediary
        self.colormap = colormap
        self.on_collision = on_collision

        dims_getter = lambda: self.dim
        self.lamps = LampRegistry(dims=dims_getter, on_collision=on_collision)
        self.calc_zones = ZoneRegistry(dims=dims_getter, on_collision=on_collision)
        self.surfaces = SurfaceRegistry(dims=dims_getter, on_collision=on_collision)

        # May be overridden if loaded serially
        self._init_standard_surfaces()

        ### Misc flags
        self._standard = PhotStandard.from_any(standard)
        self.air_changes = air_changes
        self.ozone_decay_constant = ozone_decay_constant
        self.precision = precision

        ### Reflectance
        self.ref_manager = ReflectanceManager(
            surfaces=self.surfaces,
            max_num_passes=reflectance_max_num_passes,
            threshold=reflectance_threshold,
            enabled=enable_reflectance,
        )

        ### Plotting
        self._plotter = RoomPlotter(self)

        self.calc_state = {}
        self.update_state = {}

    def __deepcopy__(self, memo):
        """Deep copy with proper rebinding of registry lambdas."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))

        # Rebind registry lambdas to point to the new instance's dim
        result.lamps.dims = lambda: result.dim
        result.calc_zones.dims = lambda: result.dim
        result.surfaces.dims = lambda: result.dim

        return result

    def __eq__(self, other):
        if not isinstance(other, Room):
            return NotImplemented

        if self.lamps != other.lamps:
            return False

        for lamp_id in self.lamps.keys():
            if self.lamps[lamp_id] != other.lamps[lamp_id]:
                return False

        return self.to_dict() == other.to_dict()

    def __repr__(self):
        if self.is_polygon:
            dim_str = f"polygon={self.polygon.n_vertices} vertices, z={self.dim.z}"
        else:
            dim_str = f"x={self.dim.x}, y={self.dim.y}, z={self.dim.z}"
        return (
            f"Room(room_id='{self._room_id}', {dim_str}, "
            f"units='{self.units}', lamps={[k for k,v in self.lamps.items()]}, "
            f"calc_zones={[k for k,v in self.calc_zones.items()]}), "
            f"enable_reflectance={self.ref_manager.enabled}, "
            f"reflectances={self.ref_manager.reflectances}"
        )

    @property
    def room_id(self) -> str:
        return self._room_id

    @property
    def id(self) -> str:
        return self._room_id

    def _assign_id(self, value: str) -> None:
        """Used by RoomRegistry to set the resolved ID."""
        self._room_id = value

    def copy(self):
        return copy.deepcopy(self)

    def save(self, fname=None):
        """Save all relevant parameters to a json file."""
        return save_room_data(self, fname)

    def to_dict(self):
        data = {}
        data["room_id"] = self._room_id
        data["name"] = self.name
        if self.is_polygon:
            data["polygon"] = self.polygon.to_dict()
        else:
            bb = self.dim.polygon.bounding_box
            if bb[0] == 0.0 and bb[1] == 0.0:
                data["x"] = self.dim.x
                data["y"] = self.dim.y
            else:
                data["x"] = [bb[0], bb[2]]
                data["y"] = [bb[1], bb[3]]
        data["z"] = self.dim.z
        data["units"] = self.units

        data["enable_reflectance"] = self.ref_manager.enabled
        data["reflectance_max_num_passes"] = self.ref_manager.max_num_passes
        data["reflectance_threshold"] = self.ref_manager.threshold

        data["standard"] = self.standard
        data["air_changes"] = self.air_changes
        data["ozone_decay_constant"] = self.ozone_decay_constant
        data["precision"] = self.precision
        data["on_collision"] = self.on_collision
        data["colormap"] = self.colormap

        dct = self.__dict__.copy()
        data["surfaces"] = {k: v.to_dict() for k, v in dct["surfaces"].items()}
        data["lamps"] = {k: v.to_dict() for k, v in dct["lamps"].items()}
        data["calc_zones"] = {k: v.to_dict() for k, v in dct["calc_zones"].items()}
        return data

    @classmethod
    def load(cls, filedata):
        """Load a room from a .guv file, JSON string, or dict."""
        load_data = parse_guv_file(filedata)

        saved_version = load_data.get("guv-calcs_version", "0.0.0")
        current_version = get_version(Path(__file__).parent / "_version.py")
        if saved_version != current_version:
            warnings.warn(
                f"File was saved with guv-calcs {saved_version}, "
                f"current version is {current_version}"
            )

        room_dict = load_data.get("data", load_data)

        # Migrate legacy schema (< 0.4.33)
        if Version(saved_version) < Version("0.4.33"):
            from .reflectance import init_room_surfaces
            dims = RoomDimensions(
                polygon=Polygon2D.rectangle(
                    room_dict.get("x", 6.0),
                    room_dict.get("y", 4.0),
                ),
                z=room_dict.get("z", 2.7),
            )
            surfaces = init_room_surfaces(
                dims=dims,
                reflectances=room_dict.get("reflectances"),
                x_spacings=room_dict.get("x_spacings"),
                y_spacings=room_dict.get("y_spacings"),
            )
            room_dict["surfaces"] = {k: v.to_dict() for k, v in surfaces.items()}

        return cls.from_dict(room_dict)

    @classmethod
    def from_dict(cls, data: dict):
        """Recreate a room from a dict."""

        room_kwargs = list(inspect.signature(cls.__init__).parameters.keys())[1:]

        # Handle polygon deserialization
        init_data = dict(data)
        if "polygon" in init_data and init_data["polygon"] is not None:
            init_data["polygon"] = Polygon2D.from_dict(init_data["polygon"])

        room = cls(**{k: v for k, v in init_data.items() if k in room_kwargs})

        for surface_id, surface in data.get("surfaces", {}).items():
            surf = Surface.from_dict(surface)
            room.add_surface(surf, on_collision="overwrite")

        for lampid, lamp in data.get("lamps", {}).items():
            room.add_lamp(Lamp.from_dict(lamp))

        for zoneid, zone in data.get("calc_zones", {}).items():
            if zone["calctype"] == "Plane":
                room.add_calc_zone(CalcPlane.from_dict(zone))
            elif zone["calctype"] == "Volume":
                room.add_calc_zone(CalcVol.from_dict(zone))

        return room

    def export_zip(
        self,
        fname=None,
        include_plots=False,
        include_lamp_files=False,
        include_lamp_plots=False,
    ):
        """Write the project file and all results files to a zip file."""
        return export_room_zip(
            self,
            fname=fname,
            include_plots=include_plots,
            include_lamp_files=include_lamp_files,
            include_lamp_plots=include_lamp_plots,
        )

    def generate_report(self, fname=None):
        """Generate a csv report of all the room's contents and zone statistics."""
        return generate_report(self, fname)

    def get_calc_state(self):
        """Save all features that, if changed, require re-calculation."""

        lamp_state = {}
        for key, lamp in self.lamps.items():
            if lamp.enabled:
                lamp_state[key] = lamp.calc_state

        zone_state = {}
        for key, zone in self.calc_zones.items():
            if zone.enabled:
                zone_state[key] = zone.calc_state

        calc_state = {}
        calc_state["lamps"] = lamp_state
        calc_state["calc_zones"] = zone_state
        calc_state["reflectance"] = self.ref_manager.calc_state

        return calc_state

    def get_update_state(self):
        """Save all features that should NOT trigger a recalculation, only an update."""

        lamp_state = {}
        for key, lamp in self.lamps.items():
            lamp_state[key] = lamp.update_state

        zone_state = {}
        for key, zone in self.calc_zones.items():
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
        """True if incident reflections need recalculation."""
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

    @property
    def standard(self):
        return self._standard

    @standard.setter
    def standard(self, value):
        self._standard = PhotStandard.from_any(value)

    def set_standard(self, standard, preserve_spacing=True):
        """Update the photobiological safety standard the Room is subject to."""
        self.standard = standard
        self._update_standard_zones(
            standard=self.standard, preserve_spacing=preserve_spacing
        )
        return self

    # --------------------- Reflectance ----------------------

    def enable_reflectance(self, val):
        """Enable/disable reflectance."""
        self.ref_manager.enabled = val
        return self

    def set_reflectance(self, R, wall_id=None):
        """Set the reflectance for the reflective walls.

        If wall_id is None, the value is set for all walls.
        """
        keys = self.surfaces.keys()
        if wall_id is None:
            for wall in keys:
                self.surfaces.get(wall).set_reflectance(R)
        else:
            if wall_id not in keys:
                raise KeyError(f"wall_id must be in {keys}")
            self.surfaces.get(wall_id).set_reflectance(R)
        return self

    def set_reflectance_spacing(self, x_spacing=None, y_spacing=None, wall_id=None):
        """Set the spacing of the calculation points for the reflective walls."""
        keys = self.surfaces.keys()
        if wall_id is None:
            for wall in keys:
                self.surfaces.get(wall).set_spacing(
                    x_spacing=x_spacing, y_spacing=y_spacing
                )
        else:
            if wall_id not in keys:
                raise KeyError(f"wall_id must be in {keys}")
            self.surfaces.get(wall_id).set_spacing(
                x_spacing=x_spacing, y_spacing=y_spacing
            )
        return self

    def set_reflectance_num_points(self, num_x=None, num_y=None, wall_id=None):
        """Set the number of calculation points for the reflective walls."""
        keys = self.surfaces.keys()
        if wall_id is None:
            for wall in keys:
                self.surfaces.get(wall).set_num_points(num_x=num_x, num_y=num_y)
        else:
            if wall_id not in keys:
                raise KeyError(f"wall_id must be in {keys}")
            self.surfaces.get(wall_id).set_num_points(num_x=num_x, num_y=num_y)
        return self

    def set_max_num_passes(self, max_num_passes):
        """Set the maximum number of passes for the interreflection module."""
        self.ref_manager.max_num_passes = max_num_passes
        return self

    def set_reflectance_threshold(self, reflectance_threshold):
        """Set the threshold percentage for the interreflection module."""
        self.ref_manager.threshold = reflectance_threshold
        return self

    # -------------- Dimensions and Units -----------------------

    def set_units(self, units, preserve_spacing=True):
        """Set room units."""
        self._update_units(units)
        self._update_standard_zones(
            standard=self.standard, preserve_spacing=preserve_spacing
        )
        return self

    def set_dimensions(self, x=None, y=None, z=None, preserve_spacing=True):
        """Set room dimensions."""
        self._update_dimensions(x=x, y=y, z=z)
        self._update_standard_zones(
            standard=self.standard, preserve_spacing=preserve_spacing
        )
        return self

    @property
    def is_polygon(self) -> bool:
        """True if this is a polygon-based room."""
        return self._explicit_polygon or self.dim.is_polygon

    @property
    def polygon(self) -> "Polygon2D | None":
        """The floor polygon if this is a polygon room, else None."""
        if self.is_polygon:
            return self.dim.polygon
        return None

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

    @property
    def dimensions(self) -> tuple[float, float, float]:
        return (self.dim.x, self.dim.y, self.dim.z)

    @property
    def volume(self) -> float:
        return self.dim.volume

    # -------------------- Lamps, zones, surfaces ---------------------

    def lamp(self, lamp_id):
        return self.lamps[lamp_id]

    def zone(self, zone_id):
        return self.calc_zones[zone_id]

    def add(self, *args, on_collision=None):
        """Add objects to the Room (Lamp, CalcZone, Surface, or iterables thereof)."""
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
                self.add(*obj, on_collision=on_collision)
            else:
                msg = f"Cannot add object of type {type(obj).__name__} to Room."
                warnings.warn(msg, stacklevel=3)
        return self

    def add_lamp(self, lamp, **kwargs):
        self.lamps.add(lamp, **kwargs)
        return self

    def place_lamp(
        self,
        *args,
        on_collision=None,
        mode: str | None = None,
        tilt: float | None = None,
        max_tilt: float | None = None,
    ):
        """Position lamps using the specified placement mode."""
        offset = convert_length(LengthUnits.METERS, self.dim.units, 0.1, sigfigs=2)
        existing = [(l.position[0], l.position[1]) for l in self.lamps.values()]
        placer = LampPlacer.for_dims(self.dim, existing=existing)

        for lamp in self._resolve_lamps(*args):
            # Convert units before placement so bbox nudge sees correct dimensions
            if lamp.surface.units != self.dim.units:
                lamp.set_units(self.dim.units)
            placer.place_lamp(lamp, mode=mode, tilt=tilt, max_tilt=max_tilt, offset=offset)
            self.add_lamp(lamp, on_collision=on_collision)
        return self

    def remove_lamp(self, lamp_id):
        self.lamps.remove(lamp_id)
        return self

    def plane_from_face(self, wall, **kwargs):
        plane = CalcPlane.from_face(wall=wall, dims=self.dim, **kwargs)
        self.add_calc_zone(plane)
        return self

    def add_calc_zone(self, calc_zone, **kwargs):
        calc_zone.colormap = self.colormap
        self.calc_zones.add(calc_zone, **kwargs)
        return self

    def add_standard_zones(self, on_collision=None):
        """Add SkinLimits, EyeLimits, and WholeRoomFluence to the room."""
        policy = on_collision or "overwrite"
        flags = self._standard.flags(self.dim.units)
        spacing = convert_length("meters", self.dim.units, 0.1, 0.1)
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
                spacing=spacing,
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
                spacing=spacing,
            ),
        ]
        self.add(standard_zones, on_collision=policy)
        return self

    def remove_calc_zone(self, zone_id):
        """Remove a calculation zone from the room."""
        self.calc_zones.pop(zone_id, None)
        return self

    def add_surface(self, surface, **kwargs):
        surface.plane.colormap = self.colormap
        self.surfaces.add(surface, **kwargs)
        return self

    def remove_surface(self, surface_id):
        self.surfaces.remove(surface_id)
        return self

    def set_colormap(self, colormap):
        """Set the room colormap."""
        if colormap not in list(colormaps):
            raise ValueError(f"{colormap} is not a valid colormap.")
        self.colormap = colormap
        for zone in self.calc_zones.values():
            zone.colormap = self.colormap
        for surface in self.surfaces.values():
            surface.plane.colormap = self.colormap
        return self

    def check_positions(self):
        """Verify the positions of all objects and return any warning messages."""
        msgs = {}
        msgs["lamps"] = self.lamps.get_position_warnings()
        msgs["calc_zones"] = self.calc_zones.get_position_warnings()
        msgs["surfaces"] = self.surfaces.get_position_warnings()
        return msgs

    # -------------------------- Calculation ---------------------------

    def calculate(self, hard=False):
        """Triggers calculation of lighting values in each calculation zone."""

        valid_lamps = self._get_valid_lamps()
        # calculate incidence on the surfaces if the reflectances or lamps have changed
        if self.recalculate_incidence or hard:
            self.ref_manager.calculate_incidence(valid_lamps, hard=hard)

        for name, zone in self.calc_zones.items():
            zone.calculate_values(
                lamps=valid_lamps, ref_manager=self.ref_manager, hard=hard
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
        """Calculate just the calc zone selected."""
        valid_lamps = self._get_valid_lamps()
        if len(valid_lamps) > 0:
            if self.recalculate_incidence or hard:
                self.ref_manager.calculate_incidence(valid_lamps, hard=hard)
            self.calc_zones[zone_id].calculate_values(
                lamps=valid_lamps, ref_manager=self.ref_manager, hard=hard
            )
            self.calc_state = self.get_calc_state()
            self.update_state = self.get_update_state()
        return self

    # ------------------- Data and Plotting ----------------------

    def get_efficacy_data(self, zone_id: str = "WholeRoomFluence", **kwargs) -> InactivationData:
        """Create Data instance from this room's fluence values."""
        zone = self.zone(zone_id)
        fluence_dict = {wv: 0.0 for wv in self.lamps.wavelengths.values()}
        for k, v in self.lamps.wavelengths.items():
            if k in zone.lamp_cache.keys():
                fluence_dict[v] = fluence_dict[v] + zone.lamp_cache[k].values.mean()
        if len(fluence_dict) == 0:
            warnings.warn("Fluence not available; returning base table.", stacklevel=2)
            data = InactivationData()
        else:
            data = InactivationData(fluence=fluence_dict, volume_m3=self.dim.cubic_meters)
        use_metric = self.dim.units in [LengthUnits.METERS, LengthUnits.CENTIMETERS]

        if zone.calctype == "Plane" and zone.horiz:
            medium = "Surface"
        else:
            medium = "Aerosol"
        return data.subset(medium=medium, use_metric=use_metric, **kwargs)

    def disinfection_table(self, zone_id="WholeRoomFluence", which="default", **kwargs):
        """Return a table of expected disinfection rates."""
        data = self.get_efficacy_data(zone_id, **kwargs)
        if which == "default":
            return data.display_df
        elif which == "full":
            return data.full_df
        elif which == "combined":
            return data.combined_df
        elif which == "combined_full":
            return data.combined_full_df
        raise ValueError(
            "Valid table arguments are default, full, combined, and combined_full"
        )

    def average_value(self, zone_id: str = "WholeRoomFluence", **kwargs):
        """Compute a derived efficacy value from averaged k parameters."""
        data = self.get_efficacy_data(zone_id)
        return data.average_value(**kwargs)

    def disinfection_plot(self, zone_id="WholeRoomFluence", category=None, **kwargs):
        """A violin plot of expected disinfection rates for all available species."""
        data = self.get_efficacy_data(zone_id, category=category)
        return data.plot(air_changes=self.air_changes, **kwargs)

    def survival_plot(self, zone_id="WholeRoomFluence", species=None, **kwargs):
        """Plot survival fraction over time for pathogens in a calculation zone."""
        data = self.get_efficacy_data(zone_id)
        return data.plot_survival(species=species, **kwargs)

    def plot(self, fig=None, select_id=None, title=""):
        """Return a plotly figure of all the room's components."""
        return self._plotter.plotly(fig=fig, select_id=select_id, title=title)

    def check_lamps(self) -> "SafetyCheckResult":
        """Check all lamps for safety compliance with skin and eye TLVs."""
        return check_lamps(self)

    # -------------------- Private helpers (absorbed from Scene) --------------------

    def _get_valid_lamps(self):
        """Return all lamps that can participate in a calculation."""
        return {
            k: v for k, v in self.lamps.items() if v.enabled and v.ies is not None
        }

    def _resolve_lamps(self, *args):
        """Convert various inputs (str keywords, Lamp objects, etc.) to Lamp instances."""
        lst = []
        for obj in args:
            if isinstance(obj, Lamp):
                lst.append(obj)
            elif isinstance(obj, str):
                lst.append(Lamp.from_keyword(obj))
            elif isinstance(obj, int):
                lst.append(Lamp.from_index(obj))
            elif isinstance(obj, dict):
                lst.append(self._resolve_lamps(*obj.values()))
            elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
                lst.append(self._resolve_lamps(*obj))
            else:
                raise TypeError(
                    f"{type(obj)} is not a valid Lamp or method of generating a Lamp"
                )
        for i, x in enumerate(lst):
            while i < len(lst) and isinstance(lst[i], list):
                lst[i : i + 1] = lst[i]
        return lst

    def _update_dimensions(self, x=None, y=None, z=None, polygon=None):
        """Update dimensions and rebuild dependent objects."""
        if polygon is not None and self.dim.is_polygon:
            self.dim = self.dim.with_(z=z, polygon=polygon)
        elif not self.dim.is_polygon:
            self.dim = self.dim.with_(x=x, y=y, z=z)
        else:
            self.dim = self.dim.with_(z=z)
        self._update_standard_surfaces()

    def _update_units(self, units):
        """Update units and convert zone spacings."""
        old_units = self.dim.units
        new_units = LengthUnits.from_any(units)
        self.dim = self.dim.with_(units=new_units)
        for zone in self.calc_zones.values():
            if zone.geometry is not None:
                old_spacing = zone.geometry.spacing
                new_spacing = convert_length(old_units, new_units, *old_spacing)
                if not isinstance(new_spacing, tuple):
                    new_spacing = (new_spacing,)
                zone.set_spacing(*new_spacing)
        self.lamps.validate()

    def _update_standard_zones(self, standard: "PhotStandard", preserve_spacing: bool):
        """Update the standard safety calculation zones."""
        flags = standard.flags(self.dim.units)
        x_min, y_min, x_max, y_max = self.dim.polygon.bounding_box
        if "SkinLimits" in self.calc_zones.keys():
            zone = self.calc_zones["SkinLimits"]
            zone.set_dimensions(
                x1=x_min, x2=x_max, y1=y_min, y2=y_max,
                preserve_spacing=preserve_spacing,
            )
            zone.set_height(height=flags["height"])
            zone.horiz = flags["skin_horiz"]
        if "EyeLimits" in self.calc_zones.keys():
            zone = self.calc_zones["EyeLimits"]
            zone.set_dimensions(
                x1=x_min, x2=x_max, y1=y_min, y2=y_max,
                preserve_spacing=preserve_spacing,
            )
            zone.set_height(height=flags["height"])
            zone.fov_vert = flags["fov_vert"]
            zone.vert = flags["eye_vert"]
        if "WholeRoomFluence" in self.calc_zones.keys():
            zone = self.calc_zones["WholeRoomFluence"]
            zone.set_dimensions(
                x1=x_min, x2=x_max,
                y1=y_min, y2=y_max,
                z2=self.dim.z,
                preserve_spacing=preserve_spacing,
            )

    def _init_standard_surfaces(self, reflectances=None, x_spacings=None,
                                 y_spacings=None, num_x=None, num_y=None):
        """Add surfaces to the room corresponding to the faces of the room."""
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

    def _update_standard_surfaces(self):
        """Update surfaces to match current dimensions."""
        keys = self.dim.faces.keys()
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
