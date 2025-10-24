"""
Example of how to manually create and use a car skin.
"""
from src.car_skin import Car_skin
from src.carRacingEnv import CarRacingEnv

def example_with_random_skin():
    """Example using a randomly generated car skin."""
    # Generate a random car skin
    car_skin = Car_skin.generate_random_skin()
    print("Generated random car skin:")
    print(car_skin)
    
    # Create environment with the car skin
    env = CarRacingEnv(
        map_path="assets/maps/track1.tmx",
        tileset_path="assets/tilesets/flat_race.tsx",
        render_mode="human",
        car_skin=car_skin
    )
    
    return env

def example_with_custom_skin():
    """Example using a custom car skin with specific components."""
    # Create a custom car skin (you need to have the asset files)
    car_skin = Car_skin(
        car_shape="car_shape_001.png",
        anterior_lights="lights_001.png",
        posterior_lights="lights_002.png",
        spoiler="spoiler_001.png",
        # Add more components as needed
    )
    
    # Create environment with the car skin
    env = CarRacingEnv(
        map_path="assets/maps/track1.tmx",
        tileset_path="assets/tilesets/flat_race.tsx",
        render_mode="human",
        car_skin=car_skin
    )
    
    return env

def example_without_skin():
    """Example without a car skin (will use basic rendering)."""
    # Create environment without specifying a skin
    env = CarRacingEnv(
        map_path="assets/maps/track1.tmx",
        tileset_path="assets/tilesets/flat_race.tsx",
        render_mode="human"
    )
    
    return env

if __name__ == "__main__":
    # Try to use random skin if assets exist, otherwise use basic rendering
    import os
    
    if os.path.exists("assets/Car"):
        print("Car assets found! Using random skin...")
        env = example_with_random_skin()
    else:
        print("No car assets found. Using basic rendering...")
        env = example_without_skin()
    
    # Run the environment
    observation, info = env.reset()
    
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        if not env.render():
            break
        
        if terminated or truncated:
            observation, info = env.reset()
    
    env.close()
