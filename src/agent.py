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
    """
    Agent for manual control using arrow keys:
    - Left/Right arrows: Steering
    - Up arrow: Acceleration
    - Down arrow: Brake
    """
    
    def __init__(self):
        self.steering = 0.0
        self.acceleration = 0.0
        self.brake = 0.0
    
    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Get action based on current keyboard input.
        """
        # Get pressed keys
        keys = pygame.key.get_pressed()
        
        # Reset values
        self.steering = 0.0
        self.acceleration = 0.0
        self.brake = 0.0
        
        # Steering: Left/Right arrows
        if keys[pygame.K_LEFT]:
            self.steering = -1.0
        elif keys[pygame.K_RIGHT]:
            self.steering = 1.0
        
        # Acceleration: Up arrow
        if keys[pygame.K_UP]:
            self.acceleration = 1.0
        
        # Brake: Down arrow
        if keys[pygame.K_DOWN]:
            self.brake = 1.0
        
        # Return action as numpy array
        action = np.array([
            self.steering,
            self.acceleration,
            self.brake
        ], dtype=np.float32)
        
        return action
    
    def reset(self):
        """Reset agent state."""
        self.steering = 0.0
        self.acceleration = 0.0
        self.brake = 0.0
