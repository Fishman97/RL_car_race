import numpy as np
from typing import Tuple, List, Optional
from src.map import Map
from src.car_skin import Car_skin


class Car:
    
    def __init__(self, x: float, y: float, angle: float = 0.0, skin: Optional['Car_skin'] = None):
        self.x = x
        self.y = y
        
        self.angle = angle
        
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        
        self.max_speed = 15.0
        self.acceleration_force = 6.0
        self.brake_force = 5.0
        self.steering_sensitivity = 1.6
        self.friction = 0.995

        self.width = 1.0
        self.height = 1.5
        
        self.start_x = x
        self.start_y = y
        self.start_angle = angle
        
        if skin is None:
            self.skin = Car_skin.generate_random_skin()
        else:
            self.skin = skin

    def reset(self, x: float = 0.0, y: float = 0.0, angle: float = 0.0):
        self.x = x if x is not None else self.start_x
        self.y = y if y is not None else self.start_y
        self.angle = angle if angle is not None else self.start_angle
        
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.skin = Car_skin.generate_random_skin()

    
    def update(self, steering: float, acceleration: float, brake: float, dt: float, map: Map):
        self.angle += steering * self.steering_sensitivity * dt
        
        self.angle = np.arctan2(np.sin(self.angle), np.cos(self.angle))
        
        speed = self.get_speed()
        
        speed *= self.friction

        if brake > 0:
            speed -= brake * self.brake_force * dt
        
        if acceleration > 0:
            speed += acceleration * self.acceleration_force * dt
                
        speed = np.clip(speed, 0, self.max_speed)
        
        self.velocity_x = speed * np.cos(self.angle - np.pi/2)
        self.velocity_y = speed * np.sin(self.angle - np.pi/2)
        
        self.x += self.velocity_x * dt
        self.y += self.velocity_y * dt
    
    def get_speed(self) -> float:
        return np.sqrt(self.velocity_x**2 + self.velocity_y**2)
    
    def get_collision_box(self) -> List[Tuple[float, float]]:
        half_width = self.width / 2
        half_height = self.height / 2
        
        angle_rad = self.angle 
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        corners = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height)
        ]
        
        rotated_corners = []
        for cx, cy in corners:
            rotated_x = self.x + (cx * cos_angle - cy * sin_angle)
            rotated_y = self.y + (cx * sin_angle + cy * cos_angle)
            rotated_corners.append((rotated_x, rotated_y))
        
        return rotated_corners

    def get_observation(self, map: Map) -> np.ndarray:
        sensors_data = self.get_sensors_data(map)

        observation = np.zeros(8)
        for i, (sensor_x, sensor_y, distance, value) in enumerate(sensors_data):
            observation[i] = value

        return observation

    # first 3 for collision sensors, the 4 for flood value detection [x, y, distance, value]
    def get_sensors_data(self, map: Map) -> List[Tuple[float, float, float, float]]:
        sensor_angles = [-np.pi/4, 0, np.pi/4]
        # subtracting pi/2 to align with car's forward direction
        sensor_angles = [angle - np.pi/2 for angle in sensor_angles]
        collision_sensor_max_distance = 5.0
        flood_sensor_distance = 2.0
        sensor_data = [] 

        # first 3 sensors for collision detection
        for angle_offset in sensor_angles:
            angle = self.angle + angle_offset

            distance, flood_value = self._cast_ray_for_collision(map, angle, collision_sensor_max_distance)
            sensor_x = self.x + distance * np.cos(angle)
            sensor_y = self.y + distance * np.sin(angle)

            sensor_data.append((sensor_x, sensor_y, distance, distance))

        # flood value sensor
        for angle_offset in sensor_angles:
            angle = self.angle + angle_offset

            distance, flood_value = self._cast_ray_for_collision(map, angle, flood_sensor_distance)
            sensor_x = self.x + distance * np.cos(angle)
            sensor_y = self.y + distance * np.sin(angle)

            sensor_data.append((sensor_x, sensor_y, flood_sensor_distance, flood_value))

        # car flood sensor
        distance, flood_value = self._cast_ray_for_collision(map, 0, 0.0)
        sensor_data.append((self.x, self.y, 0.0, flood_value))

        return sensor_data

    def _cast_ray_for_collision(self, map: Map, angle: float , max_distance: float):
        # angle parameter already includes self.angle offset from caller
        step_size = 0.02
        flood_value = map.get_normalized_distance_to_finish(self.x, self.y)

        for distance in np.arange(0, max_distance, step_size):
            ray_x = self.x + distance * np.cos(angle)
            ray_y = self.y + distance * np.sin(angle)

            grid_x = int(ray_x)
            grid_y = int(ray_y)

            if not (0 <= grid_x < map.grid_width and 0 <= grid_y < map.grid_height):
                return (distance, flood_value)

            if not map.track_mask[grid_y, grid_x]:
                return (distance, flood_value)

            flood_value = map.get_normalized_distance_to_finish(ray_x, ray_y)

        return (max_distance, flood_value)