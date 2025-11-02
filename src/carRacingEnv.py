import gymnasium as gym
import numpy as np
import os
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from src.car import Car
from src.map import Map, load_tileset
from src.car_skin import Car_skin

class CarRacingEnv(gym.Env):
    
    metadata = {"render_modes": ["human", None], "render_fps": 60}
    
    def __init__(self, map_path: str, tileset_path: str, render_mode: Optional[str] = None, car_skin: Optional[Car_skin] = None):
        super().__init__()
        
        self.render_mode = render_mode
        
        self.tileset = load_tileset(tileset_path)
        
        self.map = Map(map_path, tileset=self.tileset)
        
        # Try to generate a random car skin if none provided and assets exist
        if car_skin is None and os.path.exists("assets/Car"):
            try:
                car_skin = Car_skin.generate_random_skin()
                print("Generated random car skin!")
            except Exception as e:
                print(f"Could not generate car skin: {e}")
        
        start_pos = self.map.get_start_position()
        self.car = Car(x=start_pos[0], y=start_pos[1], skin=car_skin)
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )
        
        self.action_space = spaces.Dict({
            'steering': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            'acceleration': spaces.Discrete(2),  # 0 or 1
            'brake': spaces.Discrete(2)  # 0 or 1
        })
        
        self.renderer = None
        if self.render_mode == "human":
            from src.renderer import Renderer
            self.renderer = Renderer(self.map, self.map.grid_width, self.map.grid_height)
        
        self.steps = 0
        self.max_steps = 100000
        self.previous_distance = None
        self.attempts = 0
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        
        start_pos = self.map.get_start_position()
        self.car.reset(x=start_pos[0], y=start_pos[1])
        
        self.steps = 0
        self.previous_distance = self.map.get_distance_to_finish(self.car.x, self.car.y)

        observation = self.car.get_normalized_observation(self.map)
        observation = np.append(observation, self.car.get_speed() / self.car.max_speed)
        info = {}
        
        self.attempts += 1
        self.last_action = np.array([0.0, 0.0, 0.0])

        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        steering, acceleration, brake = action
        self.last_action = action
        
        dt = 1.0 / self.metadata["render_fps"]
        self.car.update(steering, acceleration, brake, dt, self.map)
        
        collision = self.map.check_collision(self.car)
        
        finished = self.map.check_finish_line(self.car)
        observation = self.car.get_normalized_observation(self.map)

        observation = np.append(observation, self.car.get_speed() / self.car.max_speed)
        
        reward = self._calculate_reward(collision, finished)
        
        terminated = collision or finished
        truncated = self.steps >= self.max_steps
        
        self.steps += 1
        
        info = {
            "position": (self.car.x, self.car.y),
            "velocity": self.car.get_speed(),
            "collision": collision,
            "finished": finished,
        }
        
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, collision: bool, finished: bool) -> float:
        reward = 0.0
        
        if finished:
            return 1000.0
        
        if collision:
            return -100.0
        
        # Reward for getting closer to finish
        current_dist = self.map.get_distance_to_finish(self.car.x, self.car.y)
        if self.previous_distance is not None:
            progress = self.previous_distance - current_dist
            reward += progress * 10.0
        self.previous_distance = current_dist
        
        # Reward for speed (encourage going fast)
        reward += self.car.get_speed() * 0.1
        
        # Small time penalty (encourage efficiency)
        reward -= 0.01
        
        return reward
    
    def render(self) -> bool:
        if self.render_mode == "human":
            if self.renderer is not None:
                return self.renderer.render(self.car, self.car.get_normalized_observation(self.map), self.last_action, self.attempts)
        return True
    
    def close(self):
        if self.renderer is not None:
            self.renderer.close()