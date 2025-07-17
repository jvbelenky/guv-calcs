from importlib import resources
import inspect
import json
import pathlib
import warnings
import copy
import numpy as np
from photompy import Photometry, IESFile
from .spectrum import Spectrum
from .lamp_surface import LampSurface
from .lamp_plotter import LampPlotter
from .lamp_orientation import LampOrientation
from .trigonometry import to_polar
from ._data import get_tlvs

VALID_LAMPS = [
    "aerolamp",
    "beacon",
    "lumenizer_zone",
    "nukit_lantern",
    "nukit_torch",
    "sterilray",
    "ushio_b1",
    "ushio_b1.5",
    "uvpro222_b1",
    "uvpro222_b2",
    "visium",
]

KRCL_KEYS = [
    "krypton chloride",
    "krypton-chloride",
    "krypton_chloride",
    "krcl",
    "kr-cl",
    "kr cl",
]

LPHG_KEYS = [
    "low pressure mercury",
    "low-pressure mercury",
    "mercury",
    "lphg",
    "lp-hg",
    "lp hg",
]


class Lamp:
    """
    Represents a lamp with properties defined by a photometric data file.
    This class handles the loading of IES photometric data, orienting the lamp in 3D space,
    and provides methods for moving, rotating, and aiming the lamp.

    Arguments
    -------------------
    lamp_id: str
        A unique identifier for the lamp object.
    name: str, default=None
        Non-unique display name for the lamp. If None set by lamp_id
    filename: Path, str
        If None or not pathlike, `filedata` must not be None
    filedata: Path or bytes, default=None
        Set by `filename` if filename is pathlike.
    x, y, z: floats, default=[0,0,0]
        Sets initial position of lamp in cartesian space
    angle: float, default=0
        Sets lamps initial rotation on its own axis.
    aimx, aimy, aimz: floats, default=[0,0,z-1]
        Sets initial aim point of lamp in cartesian space.
    guv_type: str
        Optional label for type of GUV source. Presently available:
        ["Krypton chloride (222 nm)", "Low-pressure mercury (254 nm)", "Other"]
    wavelength: float
        Optional label for principle GUV wavelength. Set from guv_type if guv_type
        is not "Other".
    spectra_source: Path or bytes, default=None
        Optional. Data source for spectra. May be a filepath, a binary stream,
        or a dict where the first value contains values of wavelengths, and
        the second value contains values of relative intensity.
    width, length, depth: floats, default=[None, None]
        x-axis and y-axis source extent, plus fixture depth in the units
        provided. If not provided, will be read from the .ies file. Note that
        the ies file interface labels `depth` as `height` instead.
    units: str or int in [1, 2] or None
        `feet` or `meters`. 1 corresponds to feet, 2 to `meters`. If not
        provided, will be read from .ies file, and lengt and width parameters
        will be ignored.
    source_density: int or float, default=1
        parameter that determines the fineness of the source discretization.
        Grid size follows fibonacci sequence. For an approximately square
        source, SD=1 => 1x1 grid, SD=2 => 3x3 grid, SD=3 => 5x5 grid. This is
        to ensure that a center point is always present while ensuring evenness
        of grid size.
    intensity_map: arraylike
        A relative intensity map for non-uniform sources. Must be of the same
        size as the grid generated
    enabled: bool, defualt=True
        Determines if lamp participates in calculations. A lamp may be created
        and added to a room, but disabled.

    """

    def __init__(
        self,
        lamp_id=None,
        name=None,
        filedata=None,
        filename=None,
        x=None,
        y=None,
        z=None,
        angle=None,
        aimx=None,
        aimy=None,
        aimz=None,
        intensity_units=None,
        guv_type=None,
        wavelength=None,
        spectra_source=None,
        width=None,
        length=None,
        depth=None,
        units=None,
        source_density=None,
        intensity_map=None,
        enabled=None,
        scaling_factor=None,
    ):

        self.lamp_id = lamp_id
        self.name = lamp_id if name is None else str(name)
        self.enabled = True if enabled is None else enabled

        # Position / orientation
        x = 0.0 if x is None else x
        y = 0.0 if y is None else y
        z = 0.0 if z is None else z
        self.pose = LampOrientation(
            x=x,
            y=y,
            z=z,
            angle=0.0 if angle is None else angle,
            aimx=x if aimx is None else aimx,
            aimy=y if aimy is None else aimy,
            aimz=z - 1.0 if aimz is None else aimz,
        )

        # Surface data
        self.surface = LampSurface(
            width=width,
            length=length,
            depth=depth,
            units=units,
            source_density=source_density,
            intensity_map=intensity_map,
            pose=self.pose,
        )

        # Photometric data
        if filename is not None:
            warnings.warn(
                "`filename` is deprecated and will be removed in v1.4; "
                "pass the bytes or Path in `filedata` instead.",
                DeprecationWarning,  # maybe should be FutureWarning
                stacklevel=2,
            )
            # honour old behaviour if the new arg wasn’t supplied
            if filedata is None:
                filedata = filename
        self.filedata = filedata  # temp - property eventually to be removed
        self.ies = None
        self._base_ies = None
        self._scaling_factor = scaling_factor or 1.0
        self._scale_mode = "factor"
        self.load_ies(filedata)
        self.filename = None  # VERY temp - just for illluminate compatibility

        # Spectral data
        self.spectra_source = spectra_source
        self.spectra = self._load_spectra(spectra_source)

        # source type & wavelength - just labels if Spectra is provided
        self.guv_type = guv_type
        if guv_type is not None:
            if any([key in guv_type.lower() for key in KRCL_KEYS]):
                self.wavelength = 222
            elif any([key in guv_type.lower() for key in LPHG_KEYS]):
                self.wavelength = 254
            else:  # set from wavelength if guv_type not recognized
                self.wavelength = wavelength
        else:
            self.wavelength = wavelength
        if self.wavelength is not None:
            if not isinstance(self.wavelength, (int, float)):
                msg = f"Wavelength must be int or float, not {type(self.wavelength)}"
                raise TypeError(msg)

        # mW/sr or uW/cm2 typically; not directly specified in .ies file and they can vary for GUV fixtures
        self.intensity_units = self._set_intensity_units(intensity_units)

        # plotting
        self.plotter = LampPlotter(self)

        # state
        self.calc_state = None
        self.update_state = None

    # ------------------------ Basics ------------------------------

    def to_dict(self):
        """
        save just the minimum number of parameters required to re-instantiate the lamp
        Returns dict. If filename is not None, saves dict as json.
        """

        data = {}
        data["lamp_id"] = self.lamp_id
        data["name"] = self.name
        data["x"] = float(self.pose.x)
        data["y"] = float(self.pose.y)
        data["z"] = float(self.pose.z)
        data["angle"] = float(self.pose.angle)
        data["aimx"] = float(self.pose.aimx)
        data["aimy"] = float(self.pose.aimy)
        data["aimz"] = float(self.pose.aimz)
        data["intensity_units"] = self.intensity_units
        data["guv_type"] = self.guv_type
        data["wavelength"] = self.wavelength
        data["width"] = self.surface.width
        data["length"] = self.surface.length
        data["depth"] = self.surface.depth
        data["units"] = self.surface.units
        data["source_density"] = self.surface.source_density
        data["scaling_factor"] = float(self.scaling_factor)

        if self.surface.intensity_map_orig is None:
            data["intensity_map"] = None
        else:
            data["intensity_map"] = self.surface.intensity_map_orig.tolist()

        data["enabled"] = True

        data["filename"] = str(self.filename)
        filedata = self.save_ies(original=True)
        data["filedata"] = filedata.decode() if filedata is not None else None

        if self.spectra is not None:
            spectra_dict = self.spectra.to_dict(as_string=True)
            keys = list(spectra_dict.keys())[0:2]  # keep the first two keys only
            data["spectra"] = {key: spectra_dict[key] for key in keys}
        else:
            data["spectra"] = None
        return data

    @classmethod
    def from_dict(cls, data):
        """initialize class from dict"""
        keys = list(inspect.signature(cls.__init__).parameters.keys())[1:]
        if data["spectra"] is not None:
            data["spectra_source"] = {}
            for k, v in data["spectra"].items():
                if isinstance(v, str):
                    lst = list(map(float, v.split(", ")))
                elif isinstance(v, list):
                    lst = v
                data["spectra_source"][k] = np.array(lst)
        return cls(**{k: v for k, v in data.items() if k in keys})

    @property
    def keywords(self):
        return VALID_LAMPS
        
    @classmethod
    def from_keyword(cls, key, lamp_id=None, **kwargs):
        """define a Lamp object from a predefined keyword"""
        if not isinstance(key, str):
            raise TypeError(f"Keyword must be str, not {type(key)}")
        if key.lower() in VALID_LAMPS:
            path = "guv_calcs.data.lamp_data"
            fn = resources.files(path).joinpath(key.lower() + ".ies")
            sn = resources.files(path).joinpath(key.lower() + ".csv")
            kwargs.setdefault("filedata", fn)
            kwargs.setdefault("spectra_source", sn)
        else:
            raise KeyError(
                f"{key} is not a valid lamp key. Valid keys are {VALID_LAMPS}"
            )

        kwargs.setdefault("lamp_id", key)

        return cls(**kwargs)

    @classmethod
    def from_index(cls, key_index=0, lamp_id=None, **kwargs):
        """define a Lamp object from an index value"""

        if not isinstance(key_index, int):
            raise TypeError(f"Keyword index must be int, not {type(key_index)}")

        if key_index < len(VALID_LAMPS):
            key = VALID_LAMPS[key_index]
            path = "guv_calcs.data.lamp_data"
            fn = resources.files(path).joinpath(key.lower() + ".ies")
            sn = resources.files(path).joinpath(key.lower() + ".csv")
            kwargs.setdefault("filedata", fn)
            kwargs.setdefault("spectra_source", sn)
        else:
            raise IndexError(
                f"Only {len(VALID_LAMPS)} lamps are available. Available lamps: {VALID_LAMPS}"
            )

        kwargs.setdefault("lamp_id", key)

        return cls(**kwargs)

    @classmethod
    def from_index(cls, key_index=0, lamp_id=None, **kwargs):
        """define a Lamp object from an index value"""
        if not isinstance(key_index, int):
            raise TypeError(f"Keyword index must be int, not {type(key_index)}")
        if key_index < len(VALID_LAMPS):
            key = VALID_LAMPS[key_index]
            path = "guv_calcs.data.lamp_data"
            fname = key.lower() + ".ies"
            spec_fname = key.lower() + ".csv"
            filename = resources.files(path).joinpath(fname)
            spectra_source = resources.files(path).joinpath(spec_fname)
            kwargs.setdefault("filedata", filename)
            kwargs.setdefault("spectra_source", spectra_source)
        else:
            raise IndexError(
                f"Only {len(VALID_LAMPS)} lamps are available. Available lamps: {VALID_LAMPS}"
            )
            
        kwargs.setdefault("lamp_id", key)
        
        return cls(**kwargs)

    def get_calc_state(self):
        """
        return a set of paramters that, if changed, indicate that
        this lamp must be recalculated
        """
        # this needs summed to make comparison not fail, might need to investigate later
        intensity_map_orig = (
            self.surface.intensity_map_orig.sum()
            if self.surface.intensity_map_orig is not None
            else None
        )
        return [
            self.filedata,
            self.x,
            self.y,
            self.z,
            self.angle,
            self.aimx,
            self.aimy,
            self.aimz,
            self.surface.length,  # only for nearfield
            self.surface.width,  # ""
            self.surface.depth,
            self.surface.units,  # ""
            self.surface.source_density,  # ""
            intensity_map_orig,
            self.scaling_factor,
        ]

    def get_update_state(self):
        return [self.intensity_units]

    # ----------------------- IO ------------------------------------

    def load_ies(self, filedata, override=True):
        """load an ies file"""

        self.filedata = filedata  # tmp
        if filedata is None:
            self._base_ies = None
        elif isinstance(filedata, IESFile):
            self._base_ies = filedata
        elif isinstance(filedata, Photometry):
            self._base_ies = IESFile.from_photometry(filedata)
        else:  # all other datasource cases covered here
            self._base_ies = IESFile.read(filedata)

        # in case the base object is mutated
        self.ies = copy.deepcopy(self._base_ies)

        if self.ies is not None:
            self.ies.scale(self.scaling_factor)

        # update length/width/units
        if override:
            self.surface.set_ies(self.ies)

        return self.ies

    def load_spectra(self, spectra_source):
        """external method for loading spectra after lamp object has been instantiated"""
        self.spectra = self._load_spectra(spectra_source)

    def load_intensity_map(self, intensity_map):
        """external method for loading relative intensity map after lamp object has been instantiated"""
        self.surface.load_intensity_map(intensity_map)

    def save_ies(self, fname=None, original=False):
        """
        Save the current lamp paramters as an .ies file; alternatively, save the original ies file.
        """
        if self.ies is not None:
            if original:
                iesbytes = self._base_ies.write(which="orig")
            else:
                iesbytes = self.ies.write(which="orig")

            # write to file if provided, otherwise
            if fname is not None:
                with open(fname, "wb") as file:
                    file.write(iesbytes)
            else:
                return iesbytes
        else:
            return None

    def save(self, filename):
        """save lamp information as json"""
        data = self.to_dict()
        with open(filename, "w") as json_file:
            json.dump(data, json_file, indent=4)
        return data

    def copy(self, lamp_id):
        """copy the lamp object with a new ID"""
        lamp = copy.deepcopy(self)
        lamp.lamp_id = lamp_id
        return lamp

    # ------------------- Position / Orientation ---------------------

    # temp properties...
    @property
    def x(self):
        return self.pose.x

    @property
    def y(self):
        return self.pose.y

    @property
    def z(self):
        return self.pose.z

    @property
    def position(self):
        return self.pose.position

    @property
    def aimx(self):
        return self.pose.aimx

    @property
    def aimy(self):
        return self.pose.aimy

    @property
    def aimz(self):
        return self.pose.aimz

    @property
    def aim_point(self):
        return self.pose.aim_point

    @property
    def angle(self):
        return self.pose.angle

    @property
    def heading(self):
        return self.pose.heading

    @property
    def bank(self):
        return self.pose.bank
        
    def move(self, x=None, y=None, z=None):
        """Designate lamp position in cartesian space"""
        self.pose = self.pose.move(x=x, y=y, z=z)
        self.surface.set_pose(self.pose)
        return self

    def rotate(self, angle):
        """designate lamp orientation with respect to its z axis"""
        self.pose = self.pose.rotate(angle)
        self.surface.set_pose(self.pose)
        return self

    def aim(self, x=None, y=None, z=None):
        """aim lamp at a point in cartesian space"""
        self.pose = self.pose.aim(x=x, y=y, z=z)
        self.surface.set_pose(self.pose)
        return self

    def transform_to_world(self, coords, scale=1, which="cartesian"):
        """
        transform coordinates from the lamp frame of reference to the world
        Scale parameter should generally only be used for photometric_coords
        """
        return self.pose.transform_to_world(coords, scale=scale, which=which)

    def transform_to_lamp(self, coords, which="cartesian"):
        """
        transform coordinates to align with the lamp's coordinates
        """
        return self.pose.transform_to_lamp(coords, which=which)

    def set_orientation(self, orientation, dimensions=None, distance=None):
        """
        set orientation/heading.
        alternative to setting aim point with `aim`
        distinct from rotation; applies to a tilted lamp. to rotate a lamp along its axis,
        use the `rotate` method
        """
        self.pose = self.pose.recalculate_aim_point(
            heading=orientation, dimensions=dimensions, distance=distance
        )
        self.surface.set_pose(self.pose)
        return self

    def set_tilt(self, tilt, dimensions=None, distance=None):
        """
        set tilt/bank
        alternative to setting aim point with `aim`
        """
        self.pose = self.pose.recalculate_aim_point(
            bank=tilt, dimensions=dimensions, distance=distance
        )
        self.surface.set_pose(self.pose)
        return self

    # ---------------------- Photometry --------------------------------

    @property
    def photometry(self) -> Photometry | None:
        """Active Photometry block (or None if lamp has no photometry)."""
        if self.ies is None:
            return None
        return self.ies.photometry

    @property
    def thetas(self):
        if self.ies is None:
            raise AttributeError("Lamp has no photometry")
        return self.ies.photometry.expanded().thetas

    @property
    def phis(self):
        if self.ies is None:
            raise AttributeError("Lamp has no photometry")
        return self.ies.photometry.expanded().phis

    @property
    def values(self):
        if self.ies is None:
            raise AttributeError("Lamp has no photometry")
        return self.ies.photometry.expanded().values

    @property
    def coords(self):
        if self.ies is None:
            raise AttributeError("Lamp has no photometry")
        return self.ies.photometry.coords

    @property
    def photometric_coords(self):
        if self.ies is None:
            raise AttributeError("Lamp has no photometry")
        return self.ies.photometry.photometric_coords

    def max(self):
        """maximum irradiance value"""
        if self.ies is None:
            raise AttributeError("Lamp has no photometry")
        if self.intensity_units == "mW/sr":
            return self.ies.photometry.max() / 10
        else:
            return self.ies.photometry.max()
            
    def center(self):
        """center irradiance value"""
        if self.ies is None:
            raise AttributeError("Lamp has no photometry")
        if self.intensity_units == "mW/sr":
            return self.ies.photometry.center() / 10
        else:
            return self.ies.photometry.center()
            
    def total(self):
        """just an alias for get_total_power for now"""
        return self.get_total_power()

    def get_total_power(self):
        """return the lamp's total optical power"""
        if self.ies is None:
            raise AttributeError("Lamp has no photometry")
        if self.intensity_units == "mW/sr":
            return self.ies.photometry.total_optical_power()
        else:
            return self.ies.photometry.total_optical_power() * 10

    def get_tlvs(self, standard=0):
        """
        get the threshold limit values for this lamp. Returns tuple
        (skin_limit, eye_limit) Will use the lamp spectrum if provided;
        if not provided will use wavelength; if neither is defined, returns
        (None, None). Standard may be a string in:
            [`ANSI IES RP 27.1-22`, `IEC 62471-6:2022`]
        Or an integer corresponding to the index of the desired standard.
        """
        if self.spectra is not None:
            skin_tlv, eye_tlv = get_tlvs(self.spectra, standard)
        elif self.wavelength is not None:
            skin_tlv, eye_tlv = get_tlvs(self.wavelength, standard)
        else:
            skin_tlv, eye_tlv = None, None
        return skin_tlv, eye_tlv

    def get_limits(self, standard=0):
        """compatibility alias for `get_tlvs()`"""
        return self.get_tlvs(standard=standard)

    def get_cartesian(self, scale=1, sigfigs=9):
        """Return lamp's true position coordinates in cartesian space"""
        return self.transform(self.coords, scale=scale).round(sigfigs)

    def get_polar(self, sigfigs=9):
        """Return lamp's true position coordinates in polar space"""
        cartesian = self.transform(self.coords) - self.position
        return np.array(to_polar(*cartesian.T)).round(sigfigs)

    # ---- scaling / dimming features -----

    @property
    def scaling_factor(self) -> float:
        """Current multiplier relative to the loaded photometry."""
        return self._scaling_factor

    @scaling_factor.setter  # block direct writes
    def scaling_factor(self, _):
        raise AttributeError("scaling_factor is read-only")
        
    def scale(self, scale_val):
        """scale the photometry by the given value"""
        if self.ies is None:
            msg = "No .ies file provided; scaling not applied"
            warnings.warn(msg, stacklevel=3)
        else:
            self.photometry.scale(scale_val / self.scaling_factor)
            self._update_scaling_factor()
        return self

    def scale_to_max(self, max_val):
        """scale the photometry to a maximum value [in uW/cm2]"""
        if self.ies is None:
            msg = "No .ies file provided; scaling not applied"
            warnings.warn(msg, stacklevel=3)
        else:
            if self.intensity_units == "mW/sr":        
                self.ies.photometry.scale_to_max(max_val * 10)
            else:
                self.ies.photometry.scale_to_max(max_val)
            self._update_scaling_factor()
        return self

    def scale_to_total(self, total_power):
        """scale the photometry to a total optical power [in mW]"""
        if self.ies is None:
            msg = "No .ies file provided; scaling not applied"
            warnings.warn(msg, stacklevel=3)
        else:
            self.ies.photometry.scale_to_total(total_power)
            self._update_scaling_factor()
        return self

    def scale_to_center(self, center_val):
        """scale the photometry to a center irradiance value [in uW/cm2]"""
        if self.ies is None:
            msg = "No .ies file provided; scaling not applied"
            warnings.warn(msg, stacklevel=3)
        else:
            if self.intensity_units == "mW/sr":                
                self.photometry.scale_to_center(center_val * 10)
            else:
                self.photometry.scale_to_center(center_val)
            self._update_scaling_factor()
            # self._scale_mode = "center"
        return self

    def _update_scaling_factor(self):
        """update scaling factor based on the last scaling operation"""
        self._scaling_factor = self.ies.center() / self._base_ies.center()

    # ---------------------- Surface ---------------------------

    @property
    def units(self):
        return self.surface.units

    @property
    def length(self):
        return self.surface.length

    @property
    def width(self):
        return self.surface.width

    @property
    def depth(self):
        return self.surface.depth

    def set_source_density(self, source_density):
        """change source discretization"""
        self.surface.set_source_density(source_density)

    def set_units(self, units):
        """set units"""
        if self.ies is not None:
            self.ies.update(units=1 if units == "feet" else 2)
        self.surface.set_units(units)
        return self

    def set_width(self, width):
        """change x-axis extent of lamp emissive surface"""
        self.ies.update(width=width)
        self.surface.set_width(width)
        return self

    def set_length(self, length):
        """change y-axis extent of lamp emissive surface"""
        self.ies.update(length=length)
        self.surface.set_length(length)
        return self

    def set_depth(self, depth):
        """
        TODO: this should actually be decoupled from the ies file, that
        property is rightly height not depth.
        change the z-axis offset of where the lamp's emissive surface is
        """
        self.ies.update(depth=depth)
        self.surface.set_depth(depth)

    # ------------------------ Plotting ------------------------------

    def plot_ies(self, **kwargs):
        """see LampPlotter.plot_ies"""
        return self.plotter.plot_ies(**kwargs)

    def plot_web(self, **kwargs):
        """see LampPlotter.plot_web"""
        return self.plotter.plot_web(**kwargs)

    def plot_3d(self, **kwargs):
        """see LampPlotter.plot_3d"""
        return self.plotter.plot_3d(**kwargs)

    def plot_spectra(self, **kwargs):
        """see LampPlotter.plot_spectra and Spectrum.plot"""
        return self.plotter.plot_spectra(**kwargs)

    def plot_surface(self, **kwargs):
        """see LampSurface.plot_surface"""
        return self.surface.plot_surface(**kwargs)

    # --------------------------- Internals -----------------------------

    def _set_intensity_units(self, arg):
        """
        TODO: this should probably just be an enum?  
        determine the units of the radiant intensity
        """
        if arg is not None:
            msg = f"Intensity unit {arg} not recognized. Using default value `mW/sr`"
            if isinstance(arg, int):
                if arg == 0:
                    intensity_units = "mW/sr"
                elif arg == 1:
                    intensity_units = "uW/cm²"
                else:
                    warnings.warn(msg, stacklevel=3)
                    intensity_units = "mW/sr"
            elif isinstance(arg, str):
                if arg.lower() in ["mw/sr", "uw/cm2", "uw/cm²"]:
                    intensity_units = arg
                else:
                    warnings.warn(msg, stacklevel=3)
                    intensity_units = "mW/sr"
            else:
                msg = f"Datatype {type(arg)} for intensity units not recognized. Using default value `mW/Sr`"
                intensity_units = "mW/sr"
        else:
            intensity_units = "mW/sr"
        return intensity_units

    def _load_spectra(self, spectra_source):
        """initialize a Spectrum object from the source"""
        if isinstance(spectra_source, dict):
            spectra = Spectrum.from_dict(spectra_source)
        elif isinstance(spectra_source, (str, pathlib.Path, bytes)):
            spectra = Spectrum.from_file(spectra_source)
        elif isinstance(spectra_source, tuple):
            spectra = Spectrum(spectra_source[0], spectra_source[1])
        elif spectra_source is None:
            spectra = None
        else:
            spectra = None
            msg = f"Datatype {type(spectra_source)} not recognized spectral data source"
            warnings.warn(msg, stacklevel=3)
        return spectra
