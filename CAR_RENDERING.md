# Car Texture Rendering

This document explains how the car texture rendering system works.

## Overview

The car rendering system supports both basic geometric rendering and textured rendering using the `Car_skin` class. The car is composed of multiple layered components that can be customized.

## Car Components

The car is rendered in layers from back to front:

1. **Car_Shape** - Base car shape (required)
2. **Hood_Shape** - Hood shape
3. **Posterior_Wide_Body** - Rear wide body kit
4. **Anterior_Wide_Body** - Front wide body kit
5. **Side_Bumper** - Side bumpers
6. **Back_Bumper** - Back bumper
7. **Front_Bumper** - Front bumper
8. **Hood** - Hood details/stripes
9. **Car_Cabin** - Cabin/windshield
10. **Spoiler** - Rear spoiler
11. **Roof_Scoops** - Roof scoops/vents
12. **Anterior_Lights** - Front lights
13. **Posterior_Lights** - Rear lights

## Directory Structure

To use textured cars, organize your assets as follows:

```
assets/
└── Car/
    ├── Anterior_Lights/
    │   ├── lights_001.png
    │   └── ...
    ├── Anterior_Wide_Body/
    ├── Back_Bumper/
    ├── Car_Cabin/
    ├── Car_Shape/
    ├── Front_Bumper/
    ├── Hood/
    ├── Hood_Shape/
    ├── Posterior_Lights/
    ├── Posterior_Wide_Body/
    ├── Roof_Scoops/
    ├── Side_Bumper/
    └── Spoiler/
```

## Usage

### Automatic Random Skin

The environment will automatically generate a random car skin if the `assets/Car` directory exists:

```python
from src.carRacingEnv import CarRacingEnv

env = CarRacingEnv(
    map_path="assets/maps/track1.tmx",
    tileset_path="assets/tilesets/flat_race.tsx",
    render_mode="human"
)
```

### Custom Skin

Create a custom car skin with specific components:

```python
from src.car_skin import Car_skin
from src.carRacingEnv import CarRacingEnv

# Create custom skin
car_skin = Car_skin(
    car_shape="my_car_base.png",
    anterior_lights="custom_lights.png",
    spoiler="racing_spoiler.png"
)

# Use it in the environment
env = CarRacingEnv(
    map_path="assets/maps/track1.tmx",
    tileset_path="assets/tilesets/flat_race.tsx",
    render_mode="human",
    car_skin=car_skin
)
```

### Random Skin Generation

Generate a random skin from available assets:

```python
from src.car_skin import Car_skin

# Generate random skin
random_skin = Car_skin.generate_random_skin()

# Use it
env = CarRacingEnv(..., car_skin=random_skin)
```

### Save and Load Skins

Save a car skin configuration to JSON:

```python
# Save skin
skin_json = car_skin.to_json()
with open("my_car.json", "w") as f:
    f.write(skin_json)

# Load skin
with open("my_car.json", "r") as f:
    skin_json = f.read()
car_skin = Car_skin.from_json(skin_json)
```

## Rendering Modes

The renderer supports three view modes (toggle with buttons in the UI):

1. **Normal View** - Textured map and car rendering
2. **Mask View** - Shows track mask and collision detection
3. **Flood View** - Shows distance field for debugging

## Fallback Behavior

If no car skin is provided or the assets cannot be loaded:
- The car will be rendered as a simple red polygon with a blue circle indicating the front
- This ensures the game remains playable even without texture assets

## Performance Notes

- Car textures are cached after first load
- All components are composited once per frame
- The final car sprite is rotated to match the car's angle
- Scaling is handled automatically based on window size
