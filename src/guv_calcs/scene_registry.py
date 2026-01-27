from enum import Enum
from dataclasses import dataclass, field
from collections.abc import MutableMapping, Iterator
from typing import Generic, TypeVar, Optional, Dict, Callable
from .room_dims import RoomDimensions
from .lamp import Lamp
from .calc_zone import CalcZone
from .reflectance import Surface
import warnings


# todo - integrate this
class OnCollision(str, Enum):
    ERROR = "error"
    OVERWRITE = "overwrite"
    INCREMENT = "increment"


T = TypeVar("T")


@dataclass
class Registry(Generic[T], MutableMapping[str, T]):
    """A thin wrapper around dict[str, T] with consistent ID/collision behavior."""

    base_id = "item"
    expected_type: type | None = None
    use_bounding_box: bool = False
    on_collision: str = "increment"  # "error" | "overwrite" | "increment"
    dims: Optional[Callable[[], "RoomDimensions"]] = None
    _items: Dict[str, T] = field(default_factory=dict)

    # ---- MutableMapping protocol ----
    def __getitem__(self, key: str) -> T:
        return self._items[key]

    def __setitem__(self, key: str, value: T) -> None:
        self._items[key] = value

    def __delitem__(self, key: str) -> None:
        del self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def require(self, key: str):
        try:
            return self._items[key]
        except KeyError as e:
            raise KeyError(
                f"Unknown id: {key!r}. Available: {list(self._items.keys())}"
            ) from e

    def remove(self, key: str):
        try:
            return self._items.pop(key)
        except KeyError as e:
            raise KeyError(f"Cannot remove {key!r}; not found.") from e

    def clear(self) -> None:
        self._items.clear()

    def _unique_id(self, base: str) -> str:
        if base not in self._items:
            return base
        prefix = base + "-"
        max_suffix = 1  # 1 corresponds to plain `base` being present
        for key in self.keys():
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

    def _check_position(self, dims, obj, use_bounding_box=False):
        """
        Check if an object's dimensions exceed the room's boundaries.

        For polygon rooms:
        - use_bounding_box=False: check if point is inside the polygon (for lamps)
        - use_bounding_box=True: check against bounding box (for zones/surfaces)
        """
        msg = None
        room_dims = self.dims()

        if room_dims.is_polygon:
            x, y, z = dims
            if use_bounding_box:
                # Check against bounding box (for zones/surfaces that are rectangular grids)
                x_min, y_min, x_max, y_max = room_dims.polygon.bounding_box
                if x > x_max or x < x_min or y > y_max or y < y_min:
                    msg = f"{obj.name} exceeds room boundaries!"
                    warnings.warn(msg, stacklevel=2)
                elif z < 0 or z > room_dims.z:
                    msg = f"{obj.name} exceeds room boundaries!"
                    warnings.warn(msg, stacklevel=2)
            else:
                # Check if point is inside the polygon (for lamps)
                if not room_dims.polygon.contains_point(x, y):
                    msg = f"{obj.name} exceeds room boundaries!"
                    warnings.warn(msg, stacklevel=2)
                elif z < 0 or z > room_dims.z:
                    msg = f"{obj.name} exceeds room boundaries!"
                    warnings.warn(msg, stacklevel=2)
        else:
            # Rectangular room - use bounding box check
            origin, roomdims = room_dims.origin, room_dims.dimensions
            for coord, pt1, pt2 in zip(dims, origin, roomdims):
                if coord > pt2 or coord < pt1:
                    msg = f"{obj.name} exceeds room boundaries!"
                    warnings.warn(msg, stacklevel=2)
        return msg

    def _extract_dimensions(self, obj: T) -> tuple:
        """Override to extract position/dimensions from object."""
        raise NotImplementedError

    def check_position(self, obj: T) -> str | None:
        """Check object position against room dimensions."""
        try:
            dimensions = self._extract_dimensions(obj)
            return self._check_position(dimensions, obj, use_bounding_box=self.use_bounding_box)
        except NotImplementedError:
            return None

    def get_position_warnings(self):
        dct = {}
        for obj_id, obj in self.items():
            dct[obj_id] = self.check_position(obj)
        return dct

    def _validate(self, obj: T) -> T:
        """Type check + position check. Override for additional validation."""
        if self.expected_type and not isinstance(obj, self.expected_type):
            raise TypeError(f"Must be {self.expected_type.__name__}, not {type(obj).__name__}")
        self.check_position(obj)
        return obj

    def validate(self):
        for obj_id, obj in self.items():
            self._validate(obj)

    def add(self, obj, on_collision=None):
        policy = on_collision or self.on_collision
        key = getattr(obj, "id") or self.base_id
        if key in self._items:
            if policy == "error":
                raise KeyError(f"ID {key!r} already exists.")
            if policy == "increment":
                key = self._unique_id(key)
            # OVERWRITE keeps key as-is
        obj._assign_id(key)
        self._items[key] = self._validate(obj)
        # self.on_added(key, obj)  # hook
        return key

    @property
    def calc_state(self) -> tuple:
        tpl = ()
        for val in self.values():
            if hasattr(val, "calc_state"):
                tpl += val.calc_state
        return tpl

    @property
    def update_state(self) -> tuple:
        tpl = ()
        for val in self.values():
            if hasattr(val, "update_state"):
                tpl += val.update_state
        return tpl


@dataclass
class LampRegistry(Registry["Lamp"]):
    base_id = "Lamp"
    expected_type: type | None = Lamp
    use_bounding_box: bool = False

    def _extract_dimensions(self, lamp):
        return lamp.position

    def _validate(self, lamp):
        lamp = super()._validate(lamp)
        if lamp.surface.units != self.dims().units:
            lamp.set_units(self.dims().units)
        return lamp

    @property
    def wavelengths(self) -> list:
        return {k: v.wavelength for k, v in self.items()}


@dataclass
class ZoneRegistry(Registry["CalcZone"]):
    base_id = "CalcZone"
    expected_type: type | None = CalcZone
    use_bounding_box: bool = True

    def _extract_dimensions(self, zone):
        x, y, z = zone.coords.T
        return x.max(), y.max(), z.max()


@dataclass
class SurfaceRegistry(Registry["Surface"]):
    base_id = "Surface"
    expected_type: type | None = Surface
    use_bounding_box: bool = True

    def _extract_dimensions(self, surface):
        x, y, z = surface.plane.coords.T
        return x.max(), y.max(), z.max()
