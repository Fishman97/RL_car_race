from typing import Optional

import pygame
import numpy as np

class ManualAgent:

    def __init__(self):
        self.current_action = None
    
    def act(self, observation: np.ndarray) -> Optional[int]:
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            self.current_action = 0
        elif keys[pygame.K_DOWN]:
            self.current_action = 1
        elif keys[pygame.K_LEFT]:
            self.current_action = 2
        elif keys[pygame.K_RIGHT]:
            self.current_action = 3
        else:
            self.current_action = None

        return self.current_action
    
    def reset(self):
        self.current_action = None
