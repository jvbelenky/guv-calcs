from .lamp import Lamp
from .lamp_type import LampType, GUVType, LampUnitType
from .lamp_geometry import LampGeometry
from .lamp_orientation import LampOrientation
from .lamp_surface import LampSurface
from .lamp_plotter import LampPlotter
from .lamp_surface_plotter import LampSurfacePlotter
from .lamp_placement import (
    LampPlacer,
    PlacementResult,
    new_lamp_position,
    get_lamp_positions,
    new_lamp_position_polygon,
    new_lamp_position_corner,
    new_lamp_position_edge,
    get_corners,
    get_edge_centers,
    farthest_visible_point,
    calculate_tilt,
    clamp_aim_to_max_tilt,
    apply_tilt,
)
from .lamp_configs import (
    LAMP_CONFIGS,
    resolve_keyword,
    get_valid_keys,
    get_config,
)
from .spectrum import Spectrum, sum_spectrum, log_interp
from .fixture import Fixture, FixtureShape
from .intensity_map import IntensityMap

__all__ = [
    # Main class
    "Lamp",
    # Type classes
    "LampType",
    "GUVType",
    "LampUnitType",
    # Geometry classes
    "LampGeometry",
    "LampOrientation",
    "LampSurface",
    # Plotting
    "LampPlotter",
    "LampSurfacePlotter",
    # Placement
    "LampPlacer",
    "PlacementResult",
    "new_lamp_position",
    "get_lamp_positions",
    "new_lamp_position_polygon",
    "new_lamp_position_corner",
    "new_lamp_position_edge",
    "get_corners",
    "get_edge_centers",
    "farthest_visible_point",
    "calculate_tilt",
    "clamp_aim_to_max_tilt",
    "apply_tilt",
    # Config
    "LAMP_CONFIGS",
    "resolve_keyword",
    "get_valid_keys",
    "get_config",
    # Spectrum
    "Spectrum",
    "sum_spectrum",
    "log_interp",
    # Fixture
    "Fixture",
    "FixtureShape",
    # Intensity Map
    "IntensityMap",
]
