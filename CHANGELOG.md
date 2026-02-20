# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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
