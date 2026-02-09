"""GUV file read/write, version, and save envelope utilities."""

import warnings
import pathlib
from pathlib import Path
import datetime
import json

from packaging.version import Version


def parse_guv_file(filedata):
    """Parse a .guv file from various input types (path, string, bytes, dict)."""
    if isinstance(filedata, dict):
        return filedata
    if isinstance(filedata, (str, bytes, bytearray)):
        try:
            return json.loads(filedata)
        except json.JSONDecodeError:
            # Not JSON string, try as file path
            return _load_guv_file(Path(filedata))
    if isinstance(filedata, pathlib.PurePath):
        return _load_guv_file(filedata)
    raise TypeError(f"Cannot load room from {type(filedata).__name__}")


def _load_guv_file(path):
    """Load JSON from a .guv file path."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    if path.suffix.lower() != ".guv":
        raise ValueError(f"Please provide a valid .guv file (got {path.suffix}): {path}")
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in {path}: {e}") from e


def _version_path():
    """Resolve the path to _version.py from this subpackage."""
    return Path(__file__).parent.parent / "_version.py"


def _make_envelope(format_key=None):
    """Build the common save envelope (version + timestamp)."""
    savedata = {}
    savedata["guv-calcs_version"] = get_version(_version_path())
    now = datetime.datetime.now()
    now_local = datetime.datetime.now(now.astimezone().tzinfo)
    savedata["timestamp"] = now_local.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    if format_key:
        savedata["format"] = format_key
    return savedata


def _check_savefile(filename, ext):
    """Enforce that a savefile has the correct extension."""
    if not ext.startswith("."):
        ext = "." + ext

    if isinstance(filename, str):
        if not filename.lower().endswith(ext):
            filename += ext
    elif isinstance(filename, pathlib.PurePath):
        if not filename.suffix == ext:
            filename = filename.parent / (filename.name + ext)
    return filename


def save_room_data(room, fname):
    """Save all relevant parameters to a json file."""
    savedata = _make_envelope()
    savedata["data"] = room.to_dict()
    if fname is not None:
        filename = _check_savefile(fname, ".guv")
        with open(filename, "w") as json_file:
            json.dump(savedata, json_file, indent=4)
    else:
        return json.dumps(savedata, indent=4)


def save_project_data(project, fname):
    """Save a Project to a .guv file (or return JSON string if fname is None)."""
    savedata = _make_envelope("project")
    savedata["data"] = project.to_dict()
    if fname is not None:
        filename = _check_savefile(fname, ".guv")
        with open(filename, "w") as f:
            json.dump(savedata, f, indent=4)
    else:
        return json.dumps(savedata, indent=4)


def load_project(cls, filedata):
    """Load a Project from a .guv file, JSON string, bytes, or dict.

    If the file is a legacy single-room format (no "format" key),
    wraps it in a one-room Project.
    """
    from ..room import Room

    load_data = parse_guv_file(filedata)

    saved_version = load_data.get("guv-calcs_version", "0.0.0")
    current_version = get_version(_version_path())
    if saved_version != current_version:
        warnings.warn(
            f"File was saved with guv-calcs {saved_version}, "
            f"current version is {current_version}"
        )

    fmt = load_data.get("format")
    data = load_data.get("data", load_data)

    if fmt == "project":
        return cls.from_dict(dict(data))

    # Legacy single-room format
    room = Room.load(filedata)
    project = cls(
        standard=room.standard,
        units=room.units,
        precision=room.precision,
        colormap=room.colormap,
        enable_reflectance=room.ref_manager.enabled,
        reflectance_max_num_passes=room.ref_manager.max_num_passes,
        reflectance_threshold=room.ref_manager.threshold,
    )
    project.rooms.add(room, on_collision="overwrite")
    return project


def get_version(path) -> dict:
    version = {}
    with open(path) as f:
        exec(f.read(), version)
    return version["__version__"]
