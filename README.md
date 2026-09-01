GUV Calcs
======================

A library for fluence and irradiance calculations for germicidal UV (GUV) applications. Simulate UV light propagation in rooms, calculate disinfection outcomes, and verify safety compliance.

## Installation

Install with pip:

```bash
pip install guv-calcs
```

Or install from source:

```bash
git clone https://github.com/jvbelenky/guv-calcs.git
cd guv-calcs
uv sync
```

## Quick Start

```python
from guv_calcs import Room, Lamp

# Create a room (6m x 4m x 2.7m)
room = Room(x=6, y=4, z=2.7, units="meters")

# Add a lamp at the ceiling, aimed downward
lamp = Lamp(filedata="my_lamp.ies").move(3, 2, 2.7).aim(3, 2, 0)
room.add_lamp(lamp)

# Add standard calculation zones (fluence volume + safety planes)
room.add_standard_zones()

# Run the calculation
room.calculate()

# Access results
fluence = room.calc_zones["WholeRoomFluence"]
print(f"Mean fluence rate: {fluence.values.mean():.2f} µW/cm²")
```

## Examples

### Method Chaining

```python
room = (
    Room(x=6, y=4, z=2.7)
    .place_lamp("my_lamp.ies")      # Auto-positions lamp
    .add_standard_zones()
    .calculate()
)
```

### Multiple Lamps

```python
room = Room(x=6, y=4, z=2.7)

lamp1 = Lamp(filedata="my_lamp.ies").move(2, 2, 2.7).aim(2, 2, 0)
lamp2 = Lamp(filedata="my_other_lamp.ies").move(4, 2, 2.7).aim(4, 2, 0)

room.add_lamp(lamp1).add_lamp(lamp2)
room.add_standard_zones()
room.calculate()
```

### Custom Calculation Planes

```python
from guv_calcs import Room, Lamp, CalcPlane

room = Room(x=6, y=4, z=2.7)
lamp = Lamp(filedata="my_lamp.ies").move(3, 2, 2.7).aim(3, 2, 0)
room.add_lamp(lamp)

# Add a plane at desk height (0.75m)
workplane = CalcPlane(
    zone_id="WorkPlane",
    x1=0, x2=6,
    y1=0, y2=4,
    height=0.75,
    x_spacing=0.25,
    y_spacing=0.25,
)
room.add_calc_zone(workplane)
room.calculate()
```

### Non-Rectangular Rooms

```python
from guv_calcs import Room, Polygon2D

# L-shaped room
floor = Polygon2D(vertices=[
    (0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)
])
room = Room(floor_polygon=floor, z=2.7)
```

### Safety Compliance Check

```python
room = (
    Room(x=6, y=4, z=2.7)
    .place_lamp("aerolamp")
    .add_standard_zones()
    .calculate()
)

result = room.check_lamps()
print(f"Compliant: {result.compliant}")
```

### Save and Load

```python
# Save
room.save("my_room.guv")

# Load
loaded = Room.load("my_room.guv")
loaded.calculate()
```

## Lamp Keywords

```
lamp = Lamp.from_keyword('ushio_b1')
```

Currently, only 222nm lamps are available, with data downloaded from reports.osluv.org. We welcome collaboration to expand the availability of lamp data.

- `aerolamp` - Aerolamp DevKit
- `beacon` - Beacon
- `lumenizer_zone` - LumenLabs Lumenizer Zone
- `nukit_lantern` - NuKit Lantern
- `nukit_torch` - NuKit Torch
- `sterilray` - SterilRay Germbuster Sabre
- `ushio_b1` - Ushio B1
- `ushio_b15` - Ushio B1.5
- `uvpro222_b1` - Bioabundance UVPro222 B1
- `uvpro222_b2` - Bioabundance UVPro222 B2 

## Cutting a Release

Releases are automated via `make release`:

```bash
make release VERSION=patch   # 0.7.1 -> 0.7.2
make release VERSION=minor   # 0.7.1 -> 0.8.0
make release VERSION=major   # 0.7.1 -> 1.0.0
make release VERSION=1.0.0   # explicit version
```

Before running it, make sure:

- you are on `main` with a clean working tree (the script refuses otherwise),
- the `[Unreleased]` section of `CHANGELOG.md` has content — this becomes the release notes,
- tests pass (`make test`),
- you have PyPI publish credentials configured for `uv publish`.

The `release` target runs `scripts/release.sh`, which:

1. Bumps `__version__` in `src/guv_calcs/_version.py` (the single source of version truth — `pyproject.toml` reads it via hatch).
2. Converts `[Unreleased]` in `CHANGELOG.md` into a dated `[X.Y.Z]` section, leaving a fresh `[Unreleased]` header on top.
3. Stamps `.zenodo.json` with a versioned title and this release's notes (restored to the base description in a follow-up commit).
4. Commits `Release vX.Y.Z`, tags `vX.Y.Z`, and pushes `main` and the tag.

It then runs `make publish` (`uv build` + `uv publish`) to upload to PyPI.

Pushing the tag triggers the `Release` GitHub Actions workflow (`.github/workflows/release.yml`), which creates a GitHub Release using that version's `CHANGELOG.md` section as the body.

If `uv publish` fails after the tag was pushed (e.g. missing credentials), fix the credentials and run `make publish` again — the git side is already done and doesn't need to be repeated.

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Vivian Belenky - jvb@osluv.org

Project Link: [https://github.com/jvbelenky/guv-calcs/](https://github.com/jvbelenky/guv-calcs/)

