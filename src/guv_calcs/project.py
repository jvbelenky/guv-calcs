import copy
import warnings

from .room import Room
from .scene_registry import RoomRegistry
from .safety import PhotStandard


# Keys that map to Room.__init__ parameters for default propagation
_ROOM_DEFAULT_KEYS = (
    "standard",
    "units",
    "precision",
    "colormap",
    "on_collision",
    "enable_reflectance",
    "reflectance_max_num_passes",
    "reflectance_threshold",
)


class Project:
    """A collection of Rooms with shared default settings."""

    def __init__(
        self,
        standard="ANSI IES RP 27.1-22 (ACGIH Limits)",
        units="meters",
        precision=1,
        colormap="plasma",
        on_collision="increment",
        enable_reflectance=True,
        reflectance_max_num_passes=100,
        reflectance_threshold=0.02,
    ):
        self._standard = PhotStandard.from_any(standard)
        self.units = units
        self.precision = precision
        self.colormap = colormap
        self.on_collision = on_collision
        self.enable_reflectance = enable_reflectance
        self.reflectance_max_num_passes = reflectance_max_num_passes
        self.reflectance_threshold = reflectance_threshold

        self.rooms = RoomRegistry(on_collision=on_collision)

    def __repr__(self):
        room_ids = list(self.rooms.keys())
        return f"Project(rooms={room_ids}, standard='{self.standard}', units='{self.units}')"

    def __eq__(self, other):
        if not isinstance(other, Project):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def copy(self):
        return copy.deepcopy(self)

    # -------------------- Standard property --------------------

    @property
    def standard(self):
        return self._standard

    @standard.setter
    def standard(self, value):
        self._standard = PhotStandard.from_any(value)

    # -------------------- Room management --------------------

    def add_room(self, room, on_collision=None):
        """Add an existing Room to the project."""
        self.rooms.add(room, on_collision=on_collision)
        return self

    def create_room(self, **kwargs):
        """Create a new Room, using project defaults as fallbacks."""
        defaults = {k: getattr(self, k) for k in _ROOM_DEFAULT_KEYS}
        merged = {**defaults, **kwargs}
        room = Room(**merged)
        self.rooms.add(room, on_collision=kwargs.get("on_collision"))
        return room

    def room(self, room_id):
        """Get a room by ID."""
        return self.rooms.require(room_id)

    def remove_room(self, room_id):
        """Remove a room by ID."""
        self.rooms.remove(room_id)
        return self

    # -------------------- Bulk operations --------------------

    def calculate(self, hard=False):
        """Calculate all rooms."""
        for room in self.rooms.values():
            room.calculate(hard=hard)
        return self

    def check_lamps(self):
        """Check lamps in all rooms. Returns dict of room_id -> SafetyCheckResult."""
        return {rid: room.check_lamps() for rid, room in self.rooms.items()}

    # -------------------- Bulk setters --------------------

    def set_standard(self, standard):
        """Update standard on project and all rooms."""
        self.standard = standard
        for room in self.rooms.values():
            room.set_standard(self.standard)
        return self

    def set_precision(self, precision):
        """Update precision on project and all rooms."""
        self.precision = precision
        for room in self.rooms.values():
            room.precision = precision
        return self

    def set_colormap(self, colormap):
        """Update colormap on project and all rooms."""
        self.colormap = colormap
        for room in self.rooms.values():
            room.set_colormap(colormap)
        return self

    def set_reflectance(self, R, wall_id=None):
        """Set reflectance on all rooms."""
        for room in self.rooms.values():
            room.set_reflectance(R, wall_id=wall_id)
        return self

    def set_enable_reflectance(self, val):
        """Enable/disable reflectance on project and all rooms."""
        self.enable_reflectance = val
        for room in self.rooms.values():
            room.enable_reflectance(val)
        return self

    # -------------------- Serialization --------------------

    def to_dict(self):
        data = {}
        data["standard"] = str(self.standard)
        data["units"] = self.units
        data["precision"] = self.precision
        data["colormap"] = self.colormap
        data["on_collision"] = self.on_collision
        data["enable_reflectance"] = self.enable_reflectance
        data["reflectance_max_num_passes"] = self.reflectance_max_num_passes
        data["reflectance_threshold"] = self.reflectance_threshold
        data["rooms"] = {rid: room.to_dict() for rid, room in self.rooms.items()}
        return data

    @classmethod
    def from_dict(cls, data):
        """Recreate a Project from a dict."""
        rooms_data = data.pop("rooms", {})
        project = cls(**{k: v for k, v in data.items() if k in _ROOM_DEFAULT_KEYS})
        for room_id, room_dict in rooms_data.items():
            room = Room.from_dict(room_dict)
            project.rooms.add(room, on_collision="overwrite")
        return project

    def save(self, fname=None):
        """Save project to a .guv file."""
        from .io import save_project_data
        return save_project_data(self, fname)

    @classmethod
    def load(cls, filedata):
        """Load a project from a .guv file. Handles both project and legacy room formats."""
        from .io import load_project
        return load_project(cls, filedata)

    def export_zip(self, fname=None, **kwargs):
        """Export project as a zip file with per-room subdirectories."""
        from .io import export_project_zip
        return export_project_zip(self, fname, **kwargs)

    def generate_report(self, fname=None):
        """Generate a combined CSV report for all rooms."""
        from .io import generate_project_report
        return generate_project_report(self, fname)
