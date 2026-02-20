from importlib import resources
from collections.abc import Iterable
from pathlib import Path
import json
import warnings
import copy
import numpy as np
import hashlib
from photompy import Photometry, IESFile
from .spectrum import Spectrum
from .lamp_surface import LampSurface
from .lamp_plotter import LampPlotter
from .lamp_orientation import LampOrientation
from .lamp_geometry import LampGeometry
from .fixture import Fixture
from ..geometry import to_polar
from .._serialization import init_from_dict, migrate_lamp_dict
from ..safety import get_tlvs, PhotStandard
from .lamp_type import GUVType, LampUnitType, LampType
from ..units import LengthUnits, convert_length
from .lamp_configs import resolve_keyword, get_valid_keys


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
    filedata: Path or bytes, default=None
        Photometric data file (IES format).
    x, y, z: floats, default=[0,0,0]
        Sets initial position of lamp in cartesian space. This is the photometric
        center / luminous surface center.
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
    spectrum_source: Path or bytes, default=None
        Optional. Data source for spectrum.
    width, length, height: floats, default=None
        x, y, and z axes of luminous opening. If not provided, read from IES file.
    units: str or int in [1, 2] or None
        `feet` or `meters`. 1=feet, 2=meters. Defaults to IES file value.
    housing_width, housing_length, housing_height: floats, default=None
        Physical fixture housing dimensions. Defaults to luminous surface size.
    housing_units: str or LengthUnits, default=None
        Units for housing dimensions. If different from `units`, dimensions are converted.
        Accepts: "meters", "feet", "inches", "centimeters", "yards" (or aliases).
        Defaults to same as `units`.
    source_density: int, default=1
        Parameter that determines the fineness of the source discretization.
    intensity_map: arraylike
        A relative intensity map for non-uniform sources.
    enabled: bool, default=True
        Determines if lamp participates in calculations.
    scaling_factor: float, default=1.0
        Scaling factor for photometry.
    """

    def __init__(
        self,
        lamp_id: str | None = None,
        name: str | None = None,
        filedata=None,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        angle: float = 0.0,
        aimx: float | None = None,
        aimy: float | None = None,
        aimz: float | None = None,
        intensity_units=None,
        guv_type=None,
        wavelength: int | float | None = None,
        spectrum_source=None,
        width: float | None = None,
        length: float | None = None,
        height: float | None = None,
        units=LengthUnits.METERS,
        housing_width: float | None = None,
        housing_length: float | None = None,
        housing_height: float | None = None,
        housing_units=None,
        source_density: int = 1,
        intensity_map=None,
        enabled: bool = True,
        scaling_factor: float = 1.0,
    ):
        self._lamp_id = lamp_id or "Lamp"
        self.name = str(self.lamp_id) if name is None else str(name)
        self.enabled = True if enabled is None else enabled

        # Create pose (orientation)
        pose = LampOrientation(
            x=x,
            y=y,
            z=z,
            angle=angle,
            aimx=x if aimx is None else aimx,
            aimy=y if aimy is None else aimy,
            aimz=z - 1.0 if aimz is None else aimz,
        )

        # Create surface (luminous area)
        surface = LampSurface(
            width=width,
            length=length,
            height=height,
            units=units,
            source_density=source_density,
            intensity_map=intensity_map,
        )

        # Convert housing dimensions if specified in different units
        h_units = LengthUnits.from_any(housing_units) if housing_units else LengthUnits.from_any(units)
        target_units = LengthUnits.from_any(units)
        if h_units != target_units:
            if housing_width is not None:
                housing_width = convert_length(h_units, target_units, housing_width)
            if housing_length is not None:
                housing_length = convert_length(h_units, target_units, housing_length)
            if housing_height is not None:
                housing_height = convert_length(h_units, target_units, housing_height)

        # Create initial fixture with placeholder dimensions (will be finalized after IES load)
        fixture = Fixture(
            housing_width=housing_width if housing_width is not None else (width or 0.0),
            housing_length=housing_length if housing_length is not None else (length or 0.0),
            housing_height=housing_height or 0.0,
        )

        # Create geometry container (owns pose, surface, fixture)
        self.geometry = LampGeometry(pose, surface, fixture)

        # Photometric data
        self.ies = None
        self._base_ies = None
        self._scaling_factor = scaling_factor  # used in load_ies
        self.load_ies(filedata)

        # Finalize fixture with surface dimensions if user didn't specify housing
        if housing_width is None and housing_length is None:
            self.geometry._fixture = Fixture(
                housing_width=self.surface.width,
                housing_length=self.surface.length,
                housing_height=housing_height or 0.0,
            )

        # Spectral data
        self.spectrum_source = spectrum_source
        self.lamp_type = LampType(
            spectrum=Spectrum.from_source(spectrum_source),
            _guv_type=GUVType.from_any(guv_type),
            _wavelength=wavelength,
        )

        # mW/sr or uW/cm2 typically; not directly specified in .ies file
        self.intensity_units = LampUnitType.from_any(intensity_units)

        # plotting
        self.plotter = LampPlotter(self)

    # ------------------------ Basics ------------------------------

    def __eq__(self, other):
        if not isinstance(other, Lamp):
            return NotImplemented

        if self.ies.header != other.ies.header:
            return False

        if self.ies.photometry != other.ies.photometry:
            return False

        return self.to_dict() == other.to_dict()

    def __repr__(self):
        # compact photometry tag
        if self.ies is None:
            phot = "None"
        else:
            p = self.ies.photometry
            maxval = p.values.max().round(2)
            phot = f"Photometry(thetas={p.thetas.size}, phis={p.phis.size}, maxval={maxval})"

        return (
            f"Lamp(id={self.lamp_id!r}, name={self.name!r}, "
            f"pos=({self.pose.x:.3g}, {self.pose.y:.3g}, {self.pose.z:.3g}), "
            f"aim=({self.pose.aimx:.3g}, {self.pose.aimy:.3g}, {self.pose.aimz:.3g}), "
            f"phot={phot}, "
            f"units={self.intensity_units.label}, "
            f"scaling_factor={self.scaling_factor}, "
            f"{self.lamp_type.__repr__()}, "
            f"surface=({self.surface.width}×{self.surface.length} {self.surface.units}), "
            f"fixture=(housing_h={self.fixture.housing_height}), "
            f"source_density={self.surface.source_density}, "
            f"enabled={self.enabled})"
        )

    # --- Geometry access (backward compatible) ---

    @property
    def pose(self):
        """Lamp orientation/position (via geometry)."""
        return self.geometry.pose

    @property
    def surface(self):
        """Luminous surface (via geometry)."""
        return self.geometry.surface

    @property
    def fixture(self):
        """Physical fixture housing."""
        return self.geometry.fixture

    @property
    def lamp_id(self) -> str:
        return self._lamp_id

    @property
    def id(self) -> str:
        return self._lamp_id

    def _assign_id(self, value: str) -> None:
        """should typically only be used by Scene"""
        self._lamp_id = value

    def to_dict(self):
        """
        Save the minimum parameters required to re-instantiate the lamp.
        Returns dict.
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
        data["intensity_units"] = self.intensity_units.value
        data["guv_type"] = self.guv_type.value
        data["wavelength"] = self.wavelength

        # Store user-provided values (None if not explicitly set)
        data["width"] = self.surface._user_width
        data["length"] = self.surface._user_length
        data["height"] = self.surface._user_height
        data["units"] = self.surface._user_units
        data["source_density"] = self.surface.source_density
        data["scaling_factor"] = float(self.scaling_factor)

        if self.surface.intensity_map_orig is None:
            data["intensity_map"] = None
        else:
            data["intensity_map"] = self.surface.intensity_map_orig.tolist()

        data["enabled"] = True

        # Include surface dict for new format
        data["surface"] = self.surface.to_dict()

        # Include fixture dict
        data["fixture"] = self.fixture.to_dict()

        filedata = self.save_ies(original=True)
        data["filedata"] = filedata.decode() if filedata is not None else None

        if self.spectrum is not None:
            spectrum_dict = self.spectrum.to_dict(as_string=True)
            keys = list(spectrum_dict.keys())[0:2]  # keep the first two keys only
            data["spectrum"] = {key: spectrum_dict[key] for key in keys}
        else:
            data["spectrum"] = None
        return data

    @classmethod
    def from_dict(cls, data):
        """Initialize lamp from dict."""
        data = migrate_lamp_dict(data)

        # Convert serialized spectrum dict -> spectrum_source for __init__
        spectrum_data = data.get("spectrum")
        if spectrum_data is not None:
            data["spectrum_source"] = {}
            for k, v in spectrum_data.items():
                if isinstance(v, str):
                    lst = list(map(float, v.split(", ")))
                elif isinstance(v, list):
                    lst = v
                data["spectrum_source"][k] = np.array(lst)

        # Flatten nested fixture dict -> flat housing_* params
        if "fixture" in data:
            fixture_data = data["fixture"]
            data["housing_width"] = fixture_data.get("housing_width", 0.0)
            data["housing_length"] = fixture_data.get("housing_length", 0.0)
            data["housing_height"] = fixture_data.get("housing_height", 0.0)

        # Flatten nested surface dict -> user intent params
        if "surface" in data:
            surface_data = data["surface"]
            data["width"] = surface_data.get("_user_width")
            data["length"] = surface_data.get("_user_length")
            data["height"] = surface_data.get("_user_height")
        else:
            data.pop("width", None)
            data.pop("length", None)
            data.pop("height", None)

        return init_from_dict(cls, data)

    @property
    def keywords(self):
        return get_valid_keys()

    @classmethod
    def _prepare_from_key(cls, key: str, kwargs: dict) -> dict:
        """Common logic for from_keyword / from_index."""
        if not isinstance(key, str):
            raise TypeError(f"Keyword must be str, not {type(key)}")

        canonical, config = resolve_keyword(key)

        path = resources.files("guv_calcs.data.lamp_data")
        kwargs.setdefault("filedata", path.joinpath(config["ies_file"]))

        # Check if user-provided wavelength/guv_type conflicts with config's spectrum
        config_peak = config.get("peak_wavelength")
        should_load_spectrum = config.get("spectrum_file") is not None

        if config_peak is not None and should_load_spectrum:
            user_wv = kwargs.get("wavelength")
            user_type = kwargs.get("guv_type")

            # Convert guv_type to wavelength for comparison
            if user_type is not None and user_wv is None:
                user_type_obj = GUVType.from_any(user_type)
                user_wv = user_type_obj.default_wavelength if user_type_obj else None

            # If user's explicit wavelength doesn't match config, skip spectrum
            if user_wv is not None and round(user_wv) != round(config_peak):
                should_load_spectrum = False

        if should_load_spectrum:
            kwargs.setdefault("spectrum_source", path.joinpath(config["spectrum_file"]))

        # Apply guv_type default from config only if no spectrum file
        # (spectrum will determine guv_type when present)
        if config.get("guv_type") and not config.get("spectrum_file"):
            kwargs.setdefault("guv_type", config["guv_type"])

        # Apply fixture defaults from config (user kwargs override)
        fixture_cfg = config.get("fixture", {})
        for k in ("housing_width", "housing_length", "housing_height"):
            if fixture_cfg.get(k) is not None:
                kwargs.setdefault(k, fixture_cfg[k])

        kwargs.setdefault("lamp_id", canonical)
        return kwargs

    @classmethod
    def from_keyword(cls, key, **kwargs):
        """define a Lamp object from a predefined keyword"""
        kwargs = cls._prepare_from_key(key, kwargs)
        return cls(**kwargs)

    @classmethod
    def from_index(cls, key_index=0, **kwargs):
        """define a Lamp object from an index value"""
        if not isinstance(key_index, int):
            raise TypeError(f"Keyword index must be int, not {type(key_index)}")
        valid_keys = get_valid_keys()
        if key_index >= len(valid_keys):
            raise IndexError(
                f"Only {len(valid_keys)} lamps are available. "
                f"Available lamps: {valid_keys}"
            )
        key = valid_keys[key_index]
        kwargs = cls._prepare_from_key(key, kwargs)
        return cls(**kwargs)

    @classmethod
    def resolve(cls, *args):
        """Convert various inputs (str keywords, Lamp objects, filepaths, etc.) to Lamp instances."""
        lst = []
        for obj in args:
            if isinstance(obj, cls):
                lst.append(obj)
            elif isinstance(obj, Path):
                lst.append(cls(filedata=obj))
            elif isinstance(obj, str):
                try:
                    lst.append(cls.from_keyword(obj))
                except KeyError:
                    path = Path(obj)
                    if path.is_file():
                        lst.append(cls(filedata=path))
                    else:
                        raise FileNotFoundError(
                            f"{obj!r} is not a recognized lamp keyword or existing file path"
                        )
            elif isinstance(obj, int):
                lst.append(cls.from_index(obj))
            elif isinstance(obj, dict):
                lst.append(cls.resolve(*obj.values()))
            elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
                lst.append(cls.resolve(*obj))
            else:
                raise TypeError(
                    f"{type(obj)} is not a valid Lamp or method of generating a Lamp"
                )
        for i, x in enumerate(lst):
            while i < len(lst) and isinstance(lst[i], list):
                lst[i : i + 1] = lst[i]
        return lst

    @property
    def calc_state(self):
        """
        return a set of paramters that, if changed, indicate that
        this lamp must be recalculated
        """
        # this needs summed to make comparison not fail, might need to investigate later
        if self.surface.intensity_map_orig is not None:
            arr = self.surface.intensity_map_orig
            map_fingerprint = hashlib.sha1(arr.tobytes()).digest()
        else:
            map_fingerprint = None

        if self.photometry is not None:
            # Use to_fingerprint if available, otherwise compute hash from values
            if hasattr(self.photometry, "to_fingerprint"):
                photometry_fingerprint = self.photometry.to_fingerprint()
            else:
                # Fallback: create fingerprint from photometry values
                phot_data = (
                    self.photometry.values.tobytes()
                    + self.photometry.thetas.tobytes()
                    + self.photometry.phis.tobytes()
                )
                photometry_fingerprint = hashlib.sha1(phot_data).digest()
        else:
            photometry_fingerprint = None

        return (
            photometry_fingerprint,
            self.x,
            self.y,
            self.z,
            self.angle,
            self.aimx,
            self.aimy,
            self.aimz,
            self.surface.length,  # only for nearfield
            self.surface.width,  # ""
            self.surface.height,  # luminous z-extent
            self.surface.units,  # ""
            self.surface.source_density,  # ""
            map_fingerprint,
            self.scaling_factor,
        )

    @property
    def update_state(self):
        return (self.intensity_units,)

    # ----------------------- IO ------------------------------------

    def load_ies(self, filedata, override=True):
        """load an ies file"""

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

    def load_intensity_map(self, intensity_map):
        """external method for loading relative intensity map after lamp object has been instantiated"""
        self.surface.load_intensity_map(intensity_map)

    def save_ies(self, fname=None, original=False, precision=None):
        """
        Save the current lamp paramters as an .ies file; alternatively, save the original ies file.
        """
        if self.ies is not None:
            if original:
                iesbytes = self._base_ies.write(which="orig", precision=precision)
            else:
                iesbytes = self.ies.write(which="orig", precision=precision)

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

    def copy(self, **kwargs):
        """Create a fresh copy of this lamp."""
        dct = self.to_dict()
        for key, val in dct.items():
            new_val = kwargs.get(key, None)
            if new_val is not None:
                dct[key] = new_val
        return type(self).from_dict(dct)

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
        """Designate lamp position in cartesian space."""
        self.geometry.move(x=x, y=y, z=z)
        return self

    def rotate(self, angle):
        """Designate lamp orientation with respect to its z axis."""
        self.geometry.rotate(angle)
        return self

    def aim(self, x=None, y=None, z=None):
        """Aim lamp at a point in cartesian space."""
        self.geometry.aim(x=x, y=y, z=z)
        return self

    def transform_to_world(self, coords, scale=1, which="cartesian"):
        """
        Transform coordinates from the lamp frame of reference to the world.
        Scale parameter should generally only be used for photometric_coords.
        """
        return self.pose.transform_to_world(coords, scale=scale, which=which)

    def transform_to_lamp(self, coords, which="cartesian"):
        """Transform coordinates to align with the lamp's coordinates."""
        return self.pose.transform_to_lamp(coords, which=which)

    def set_orientation(self, orientation, dimensions=None, distance=None):
        """
        Set orientation/heading.
        Alternative to setting aim point with `aim`.
        Distinct from rotation; applies to a tilted lamp.
        """
        self.geometry.recalculate_aim_point(
            heading=orientation, dimensions=dimensions, distance=distance
        )
        return self

    def set_tilt(self, tilt, dimensions=None, distance=None):
        """
        Set tilt/bank.
        Alternative to setting aim point with `aim`.
        """
        self.geometry.recalculate_aim_point(
            bank=tilt, dimensions=dimensions, distance=distance
        )
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
        return self.photometry.max() * self.intensity_units.factor
            
    def center(self):
        """center irradiance value"""
        if self.ies is None:
            raise AttributeError("Lamp has no photometry")
        return self.photometry.center() * self.intensity_units.factor

    def total(self):
        """just an alias for get_total_power for now"""
        return self.get_total_power()

    def get_total_power(self):
        """return the lamp's total optical power"""
        if self.ies is None:
            raise AttributeError("Lamp has no photometry")
        return self.photometry.total_optical_power() * self.intensity_units.factor * 10

    def get_tlvs(self, standard=0):
        """
        get the threshold limit values for this lamp. Returns tuple
        (skin_limit, eye_limit) Will use the lamp spectrum if provided;
        if not provided will use wavelength; if neither is defined, returns
        (None, None). Standard may be a string in:
            [`ANSI IES RP 27.1-22`, `IEC 62471-6:2022`]
        Or an integer corresponding to the index of the desired standard.
        """
        standard = PhotStandard.from_any(standard)
        if self.spectrum is not None:
            skin_tlv, eye_tlv = get_tlvs(self.spectrum, standard)
        elif self.wavelength is not None:
            skin_tlv, eye_tlv = get_tlvs(self.wavelength, standard)
        else:
            skin_tlv, eye_tlv = None, None
        return skin_tlv, eye_tlv

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
            self.photometry.scale_to_max(max_val / self.intensity_units.factor)
            self._update_scaling_factor()
        return self

    def scale_to_total(self, total_power):
        """scale the photometry to a total optical power [in mW]"""
        if self.ies is None:
            msg = "No .ies file provided; scaling not applied"
            warnings.warn(msg, stacklevel=3)
        else:
            self.photometry.scale_to_total(total_power / self.intensity_units.factor / 10)
            self._update_scaling_factor()
        return self

    def scale_to_center(self, center_val):
        """scale the photometry to a center irradiance value [in uW/cm2]"""
        if self.ies is None:
            msg = "No .ies file provided; scaling not applied"
            warnings.warn(msg, stacklevel=3)
        else:
            self.photometry.scale_to_center(center_val / self.intensity_units.factor)
            self._update_scaling_factor()
        return self

    def _update_scaling_factor(self):
        """update scaling factor based on the last scaling operation"""
        self._scaling_factor = self.ies.center() / self._base_ies.center()

    # ---------------------- Spectrum / Lamp type ---------------

    def load_spectrum(self, spectrum_source):
        """Load spectrum after lamp object has been instantiated."""
        new_spectrum = Spectrum.from_source(spectrum_source)
        self.lamp_type = self.lamp_type.update(spectrum=new_spectrum)
        return self

    def set_guv_type(self, guv_type):
        guv_type = GUVType.from_any(guv_type)
        if guv_type is not None and self.guv_type == guv_type:
            return self  # no changes, don't accidentally override spectra
        self.lamp_type = LampType(_guv_type=guv_type)
        return self

    def set_wavelength(self, wavelength):
        if self.wavelength == wavelength:
            return self  # no changes, don't accidentally override spectra
        self.lamp_type = LampType(_wavelength=wavelength)
        return self

    @property
    def spectrum(self):
        return self.lamp_type.spectrum

    @property
    def guv_type(self):
        return self.lamp_type.guv_type

    @property
    def wavelength(self):
        return self.lamp_type.wavelength

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
    def height(self):
        """Z-extent of luminous opening (for 3D sources). From IES file."""
        return self.surface.height

    @property
    def depth(self):
        """Backward-compatible alias for housing_height."""
        return self.fixture.housing_height

    def set_source_density(self, source_density):
        """Change source discretization."""
        self.surface.set_source_density(source_density)

    def set_units(self, units):
        """Set units for lamp surface and fixture dimensions."""
        new_units = LengthUnits.from_any(units)
        old_units = self.surface.units

        if self.ies is not None:
            self.ies.update(units=1 if units == "feet" else 2)
        self.surface.set_units(units)

        # Convert fixture dimensions if units changed
        if old_units != new_units and self.fixture.has_dimensions:
            hw, hl, hh = convert_length(
                old_units, new_units,
                self.fixture.housing_width,
                self.fixture.housing_length,
                self.fixture.housing_height,
            )
            self.geometry._fixture = Fixture(
                housing_width=hw,
                housing_length=hl,
                housing_height=hh,
                shape=self.fixture.shape,
            )
        return self

    def set_width(self, width):
        """Change x-axis extent of lamp emissive surface."""
        if self.ies is not None:
            self.ies.update(width=width)
        self.surface.set_width(width)
        return self

    def set_length(self, length):
        """Change y-axis extent of lamp emissive surface."""
        if self.ies is not None:
            self.ies.update(length=length)
        self.surface.set_length(length)
        return self

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

    def plot_spectrum(self, **kwargs):
        """See LampPlotter.plot_spectrum and Spectrum.plot."""
        return self.plotter.plot_spectrum(**kwargs)

    def plot_surface(self, **kwargs):
        """see LampSurface.plot_surface"""
        return self.surface.plot_surface(**kwargs)
