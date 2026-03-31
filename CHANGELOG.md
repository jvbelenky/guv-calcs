# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Zenodo metadata and script
- width/length/height properties on Object class
- getters and setters on per-face properties for Object class
- copy method on Object class
- set_dimensions method on Object class
- test coverage for Object class
- set_spacing / set_num_points handles for Object class 

## [0.7.0] - 2026-03-31

### Added

- Method on lamp to clear spectrum but preserve wavelength/guv_type
- place_at_index and place_lamp_at_index API in lamp_placement module
- Major occlusion/shadowing module
- Object class
- irradiance_at method for Lamp
- nearfield convenience property for Lamp
- Ozone calculation method
- PlaneCalcType fully implemented
- CalcPoint class (with GridPoint geometry class)
- performance.py file with both calc time and memory requirements estimations
- nudge_into_bounds() methods on registry objects
- Mass lamp operations: batch placement, aiming, and height. Lamp placement now uses lazy cacheing to speed up operations.  Room-level wrappers: Room.place_lamps(), Room.aim_lamps(), Room.set_lamp_height(). Lamps can be aimed with various programmatic methods: "down", "point", "direction", "centroid", "furthest_edge", "furthest_corner". 
- xlrd dependency: now supports .xls files

### Fixed

- room.set_units() early return is now self, not None
- lamp_types with no default wavelength will now return an explicitly set wavelength
- subset() in InactivationData now non-mutating
- LightingCalculator will now subsample in grid points that are too close to light sources
- Bug in loading an intensity map while width/length are 0
- Bug in update_managers call preventing interreflections from being calculated
- x/y coordinate swap in photometry
- lamp.nearfield parameter returning float and not bool
- lamp.irradiance_at() function not using lamp.intensity_factor properly
- break ties in downlight placement by preferring centroid
- properly check for None values in spacing/num_points tuples
- proper handling for 0D arrays in to_polar()

### Changed

- PolyGrids and RectGrids consolidated; PlaneGrid and VolGrid classes replaced with SurfaceGrid and VolumeGrid classes
- CalcPlane/CalcVol can no longer be constructed with legacy constructors--use from_legacy instead
- Lamp now stores preset_id if loaded from keyword
- Old CalcZone API no longer accepts x1/x2 etc constructors--migration logic added to _serialization
- Cleanup of how SceneRegistry handles object position checking
- CalcZone's show_values is now display_mode + zones no longer have individual colormaps
- CalcZone/LightingCalculator restructured to accept surfaces instead of the ref_manager
- Reflectance calculation moved to LightingCalculator
- calculate_nearfield method in LightingCalculator extracted to _irradiance_at method
- LampSurface position computation simplified
- ReflectanceManager no longer mirrors room surfaces
- More accurate calculation time estimates
- Optimized downlight placement, rename new_lamp_position_polygon->new_lamp_position_downlight to match corner and edge functions
- Simplify LampPlace API--separate computation and mutation paths
- Significant simplification of calc zone spacing/num_points flow--whichever value was last set is what is kept. If spacing is set, that's the spacing forever--if span is then set to smaller than the spacing, the spacing is temporarily clamped, but reverts to its original value when the span is restored.
- Package management fully migrated from pip to uv

## [0.6.5] - 2026-03-06

### Added

- Rooms saved as 'Project' files can now be loaded with Room.load()
- Calc zone dosages can now be specified in hours/minutes/seconds, not just hours
- Reports are now included in the "Export all" option for rooms
- __repr__ added to various classes missing them

### Changed

- Wavelength now part of lamp update state
- Rename CalcZone units->value_units to avoid confusion/collision with dimension units
- Make units an override point in lamp.surface (when ies changed after a lamp is already in a room)
- Calculation status now part of calc zone calc state
- Changing units now mutates dimensions / positions / spacings of all room objects 

## [0.6.4] - 2026-02-25

### Fixed

- Bug in UnitEnum default
- Bugged file_to_zone round trip, added test
- Removed unnecessary np.unique() call in zone construction
- Fixed function definition in Lamp.get_polar() and Lamp.get_cartesian()
- Lamp equality handled even if they have no photometry
- Fix in lamp placement algorithm for corners
- Fixed zero-value handling in calc zones by replacing truthy fallback logic with explicit None coalescing
- Made Project.from_dict() non-mutating
- Fix floating point comparisons in rect_grid.py
- Fix div by zero in reflectance form factor calculations
- Fix div by zero in survival fraction calculation
- EyeLimits fov_horiz not flowing through to standard zones
- Fix height/direction ignored in PlaneGrid if 0

### Changed

- Serialization/migration logic souped up, moved from Room.load() to _serialization.py 
- Simplified standard zones flow

### Added

-  New estimate_ozone_increase() method in Room

## [0.6.3] - 2026-02-21

### Added

- Calculation time estimate method added to Room
- Exposed lamp placement, zone stat collection, and standard zone definitions to external API

## [0.6.2] - 2026-02-20

### Fixed

- Handle all (int, float) pairs with numbers.Real instead 
- Workflow fix to catch future dependency issues
- Added missing openpyxl dependency
- Dropped Python 3.9/3.10 support (not working since photompy 0.2.0 dep update)

## [0.6.1] - 2026-02-20

### Added

- from_int parsing for ParseableEnums (like PhotStandard, etc)
- more flexible parsing of spectrum files--header data handling and allowing xls and xlsx files

### Changed

- Bumped photompy requirement to 0.2.0
- Simplified Room: standard_zones in its own file, lamp validity check on registry, lamp resolve on lamp
- More robust point-scaling for standard zones

### Fixed

- Bug in lamp copy method - lamp_id being improperly assigned
- Bug in room equality due to Registries comparing dim lambdas
- Simplified and corrected Lamp scaling factor errors (to do with intensity_units enum)
- Bug in lamp.get_tlvs()--standard not parsing correctly.

## [0.6.0] - 2026-02-09

### Added
- Polygonal room support (polygon rooms, volgrids, calcplanes)
- Multi-room interface
- Millimeter unit support
- Lamp fixture plotting
- Wavelength and survival plots
- Strain/subspecies filtering with permissive aliasing
- Renamed Data -> InactivationData

### Changed
- Beefed up lamp placement algorithm with fixture-aware bounding boxes
- Refactored lamp geometry into separate components
- Consolidated polygrid and Registry classes
- Broke out calc_zone, geometry, and io into separate submodules
- Removed dead code, extracted shared from_dict utility
- Replaced face_data tuples with NamedTuples
- Unified enum from_any boilerplate
- Spectrum is now immutable dataclass
- Renamed spectra -> spectrum
- Simplified efficacy module
- All rooms are polygons internally; scene dummied out

### Fixed
- Serialization bugfix for polygon rooms
- Plotting concave polygon plane values
- Lamp placement in edge mode and correct orientation
- Lamp placement logic with bounding boxes outside of rooms
- Type mixing bugfix

## [0.5.0.1] - 2026-01-28

Last release compatible with illuminate v1.

### Added
- Average derived value function for species/strains
- Safety checks ported from illuminate v1

### Fixed
- Photompy error and XZ plane display error
- Source density parameter not being set
- IES file override handling

## [0.5.0] - 2026-01-24

### Added
- Test suite
- eq/repr for lamps, zones, rooms, and reflectance manager
- Registries instead of dicts for managing objects
- Inactivation data overhaul with log4/log5 functions
- Survival plot function
- Composable filters for calc planes
- Convenience save function

### Changed
- Major calc zone refactor
- Calc state cleanup with convenience id handles
- Tightened string parameters to enums
- Harmonized lamp unit type
- Refactored lamp type encapsulation
- Renamed ReflectiveSurface -> Surface, extended to include transmittance
- Removed filters/obstacles (to be added back later)
- Python 3.13 compatibility

### Fixed
- Total power bugfix
- Roundtrip error bugfix
- ACGIH and ICNIRP always returning ACGIH values
- Bytearray never checked when reading
- Calc zone geometry fixes
- Ref surface intake logic

## [0.4.x] - 2025-04-21 to 2025-12-02

### Added
- Dimming support
- Windows support
- Air changes to disinfection data
- Addressable colormap
- Nukit and aerolamp fixture support
- Precision display for room
- Report generation function
- Expedited calculations mode
- eq/repr for lamps, zones, rooms
- 3D obstacles and soft shadows
- Measured correction filter module

### Changed
- Room API cleanup: surfaces treated like lamps and calc zones
- Version-dependent compatibility savefile loading
- Refactored lamp position/aimpoint to separate class
- Restructured calc zone specification
- Major refactor: calculations faster through calc state management
- Interreflections working properly
- Big refactor of Room class
- Reflectance can be disabled without setting values to 0
- Disinfection calculator available directly in Room object

### Fixed
- Lamp depth overwrite, spurious dimensional warnings
- Spacing for very small rooms/chambers
- _write_rows when number of points is small
- Div by zero for small volumes/planes
- ID generation bugfix
- Photompy scaling and cache bugfixes
- Point generation when changing plane dimensions
- Surface bug with improperly sized intensity map
- Major reflection bugfix
- Lamp unit conversion was backwards
- Source dimension interactions with IES files

## [0.3.x] - 2025-02-11 to 2025-04-21

### Added
- Copy function for rooms
- Add function for calc zones
- Disinfection calculator in Room object

### Changed
- Lamp surface and plotting refactors
- Faster calculations through calc state management
- Interreflections working properly

### Fixed
- Major reflection bugfix
- Export function bug handling
- Handling for malformed intensity_map files

## [0.2.x] - 2024-11-14 to 2024-12-21

### Added
- Lamp position generation function
- room.calc_state object for recalculation tracking
- Intensity unit setting for lamps
- fov_horiz parameter for more accurate eye dose
- IES file saving parameters
- Relative map interface
- Human coronavirus disinfection data

### Changed
- Cleaner logic for grid point generation and plotting
- Package name updated for PyPI compliance
- Cleaner statefulness management

### Fixed
- Reload function bugfix
- Spacing value after num_points set
- Edge case bugs in calc_zone
- JSON serialization bugfix

## [0.1.x] - 2024-08-26 to 2024-10-14

### Added
- Spectrum class integration
- Savefile stores filename as well as filedata

### Changed
- Room class updates
- Integration of spectrum class and functions from illuminate

### Fixed
- CSV encoding bugfix
- Export bugfix

## [0.0.x] - 2024-08-07 to 2024-08-12

### Added
- Initial release
- Save/load support for .guv files
- Export functions for calc planes and volumes
- Version tracking in separate file

## Notes

All notes to 2026-02-09 have been backfilled from commit history.
