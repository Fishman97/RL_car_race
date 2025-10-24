import numpy as np
import pygame
from typing import Any


class SimpleAgent:
    
    def __init__(self, steering: float = 0.0, acceleration: float = 1.0, brake: float = 0.0):

        self.steering = np.clip(steering, -1.0, 1.0)
        self.acceleration = np.clip(acceleration, 0.0, 1.0)
        self.brake = int(np.clip(brake, 0, 1))
        
        # Store the fixed action
        self.action = np.array([
            self.steering,
            self.acceleration,
            self.brake
        ], dtype=np.float32)
    
    def act(self, observation: np.ndarray) -> np.ndarray:

        return self.action.copy()
    
    def reset(self):

        pass


class ManualAgent:

    def __init__(self):
        self.steering = 0.0
        self.acceleration = 0.0
        self.brake = 0.0
    
    def act(self, observation: np.ndarray) -> np.ndarray:
        keys = pygame.key.get_pressed()
        
        self.steering = 0.0
        self.acceleration = 0.0
        self.brake = 0.0
        
        if keys[pygame.K_LEFT]:
            self.steering = -1.0
        elif keys[pygame.K_RIGHT]:
            self.steering = 1.0
        
        if keys[pygame.K_UP]:
            self.acceleration = 1.0
        
        if keys[pygame.K_DOWN]:
            self.brake = 1.0
        
        action = np.array([
            self.steering,
            self.acceleration,
            self.brake
        ], dtype=np.float32)
        
        return action
    
    def reset(self):
        self.steering = 0.0
        self.acceleration = 0.0
        self.brake = 0.0
