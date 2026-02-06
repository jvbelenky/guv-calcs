from enum import Enum
from dataclasses import dataclass, field
from collections.abc import MutableMapping, Iterator
from typing import Generic, TypeVar, Optional, Dict, Callable
import warnings

import numpy as np

from .room_dims import RoomDimensions
from .lamp import Lamp
from .calc_zone import CalcZone
from .reflectance import Surface


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

    def _check_position(self, corners, obj):
        """Check if all bounding box corners are within room boundaries."""
        room_dims = self.dims()
        for x, y, z in corners:
            if room_dims.is_polygon:
                # Use inclusive check to accept points on the boundary
                if not room_dims.polygon.contains_point_inclusive(x, y):
                    msg = f"{obj.name} exceeds room boundaries!"
                    warnings.warn(msg, stacklevel=2)
                    return msg
                if z < 0 or z > room_dims.z:
                    msg = f"{obj.name} exceeds room boundaries!"
                    warnings.warn(msg, stacklevel=2)
                    return msg
            else:
                origin, dims = room_dims.origin, room_dims.dimensions
                if not (origin[0] <= x <= dims[0] and
                        origin[1] <= y <= dims[1] and
                        origin[2] <= z <= dims[2]):
                    msg = f"{obj.name} exceeds room boundaries!"
                    warnings.warn(msg, stacklevel=2)
                    return msg
        return None

    def _extract_dimensions(self, obj: T) -> np.ndarray:
        """Override to extract bounding box corners (N, 3) array from object."""
        raise NotImplementedError

    def check_position(self, obj: T) -> str | None:
        """Check object position against room dimensions."""
        try:
            corners = self._extract_dimensions(obj)
            return self._check_position(corners, obj)
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

    def _extract_dimensions(self, lamp):
        return lamp.geometry.get_bounding_box_corners()

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

    def _extract_dimensions(self, zone):
        x, y, z = zone.coords.T
        return np.array([
            [x.min(), y.min(), z.min()], [x.max(), y.min(), z.min()],
            [x.max(), y.max(), z.min()], [x.min(), y.max(), z.min()],
            [x.min(), y.min(), z.max()], [x.max(), y.min(), z.max()],
            [x.max(), y.max(), z.max()], [x.min(), y.max(), z.max()],
        ])


@dataclass
class SurfaceRegistry(Registry["Surface"]):
    base_id = "Surface"
    expected_type: type | None = Surface

    def _extract_dimensions(self, surface):
        return surface.plane.coords


@dataclass
class RoomRegistry(Registry["Room"]):
    """Registry for Room objects within a Project."""
    base_id = "Room"
    expected_type: type | None = None  # set lazily to avoid circular import

    def _extract_dimensions(self, obj):
        raise NotImplementedError
