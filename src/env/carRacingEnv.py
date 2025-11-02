import gymnasium as gym
import numpy as np
import os
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from src.env.car import Car
from src.env.map import Map, load_tileset
from src.env.car_skin import Car_skin

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

        # Discrete action space: 0=accelerate, 1=brake, 2=steer left, 3=steer right
        self.action_space = spaces.Discrete(4)
        self._action_mapping = {
            0: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            1: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            2: np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            3: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }
        
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
        info = {}
        
        self.attempts += 1
        self.last_action = np.zeros(4, dtype=np.float32)

        return observation, info
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        mapped_action = self._map_action(action)
        acceleration, brake, steer_left, steer_right = mapped_action
        self.last_action = mapped_action
        
        dt = 1.0 / self.metadata["render_fps"]
        self.car.update(acceleration, brake, steer_left, steer_right, dt, self.map)
        
        collision = self.map.check_collision(self.car)
        
        finished = self.map.check_finish_line(self.car)
        observation = self.car.get_normalized_observation(self.map)
        
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

    def _map_action(self, action: Any) -> np.ndarray:
        if isinstance(action, (np.integer, int)):
            idx = int(action)
            if idx in self._action_mapping:
                return self._action_mapping[idx]
        elif action is None:
            return np.zeros(4, dtype=np.float32)

        return np.zeros(4, dtype=np.float32)
    
    def _calculate_reward(self, collision: bool, finished: bool) -> float:
        reward = 0.0
        
        if finished:
            return 1000.0
        
        if collision:
            return -100.0
        
        current_dist = self.map.get_normalized_distance_to_finish(self.car.x, self.car.y)
        if self.previous_distance is not None:
            progress = self.previous_distance - current_dist
            reward += progress * 10.0
        self.previous_distance = current_dist

        reward -= 0.1

        return reward
    
    def render(self) -> bool:
        if self.render_mode == "human":
            if self.renderer is not None:
                return self.renderer.render(self.car, self.car.get_normalized_observation(self.map), self.last_action, self.attempts)
        return True
    
    def close(self):
        if self.renderer is not None:
            self.renderer.close()