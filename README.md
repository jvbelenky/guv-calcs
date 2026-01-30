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
pip install .
```

## Quick Start

```python
from guv_calcs import Room, Lamp

# Create a room (6m x 4m x 2.7m)
room = Room(x=6, y=4, z=2.7, units="meters")

# Add a lamp at the ceiling, aimed downward
lamp = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
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
    .place_lamp("aerolamp")      # Auto-positions lamp
    .add_standard_zones()
    .calculate()
)
```

### Multiple Lamps

```python
room = Room(x=6, y=4, z=2.7)

lamp1 = Lamp.from_keyword("aerolamp").move(2, 2, 2.7).aim(2, 2, 0)
lamp2 = Lamp.from_keyword("ushio_b1").move(4, 2, 2.7).aim(4, 2, 0)

room.add_lamp(lamp1).add_lamp(lamp2)
room.add_standard_zones()
room.calculate()
```

### Custom Calculation Planes

```python
from guv_calcs import Room, Lamp, CalcPlane

room = Room(x=6, y=4, z=2.7)
lamp = Lamp.from_keyword("aerolamp").move(3, 2, 2.7).aim(3, 2, 0)
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

## Available Lamp Keywords

- `aerolamp` - Aerolamp DevKit (222nm KrCl)
- `beacon` - Beacon (222nm KrCl)
- `lumenizer_zone` - Lumenizer Zone
- `nukit_lantern` - NuKit Lantern
- `nukit_torch` - NuKit Torch
- `sterilray` - SterilRay
- `ushio_b1` - Ushio B1

## Key Concepts

- **Fluence rate**: UV power per unit area at a point (µW/cm²)
- **Irradiance**: UV power incident on a surface (µW/cm²)
- **CalcVol**: 3D volume grid for fluence calculations
- **CalcPlane**: 2D surface for irradiance calculations
- **GUV types**: 222nm (Far-UVC/KrCl), 254nm (conventional), other wavelengths

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Vivian Belenky - jvb@osluv.org

Project Link: [https://github.com/jvbelenky/guv-calcs/](https://github.com/jvbelenky/guv-calcs/)

