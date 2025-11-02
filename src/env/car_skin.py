import os
import json
import random
from typing import Optional, Dict, Any, List
import pygame


class Car_skin:
    
    def __init__(self, 
                 anterior_lights: Optional[str] = None,
                 anterior_wide_body: Optional[str] = None,
                 back_bumper: Optional[str] = None,
                 car_cabin: Optional[str] = None,
                 car_shape: Optional[str] = None,
                 front_bumper: Optional[str] = None,
                 hood: Optional[str] = None,
                 hood_shape: Optional[str] = None,
                 posterior_lights: Optional[str] = None,
                 posterior_wide_body: Optional[str] = None,
                 roof_scoops: Optional[str] = None,
                 side_bumper: Optional[str] = None,
                 spoiler: Optional[str] = None):

        self.anterior_lights = anterior_lights
        self.anterior_wide_body = anterior_wide_body
        self.back_bumper = back_bumper
        self.car_cabin = car_cabin
        self.car_shape = car_shape
        self.front_bumper = front_bumper
        self.hood = hood
        self.hood_shape = hood_shape
        self.posterior_lights = posterior_lights
        self.posterior_wide_body = posterior_wide_body
        self.roof_scoops = roof_scoops
        self.side_bumper = side_bumper
        self.spoiler = spoiler
        
        self._surfaces: Dict[str, pygame.Surface] = {}
        self._assets_path = "assets/Car"
    
    def _load_surface(self, component: str, filename: str) -> Optional[pygame.Surface]:

        if not filename:
            return None
            
        cache_key = f"{component}/{filename}"
        if cache_key in self._surfaces:
            return self._surfaces[cache_key]
        
        file_path = os.path.join(self._assets_path, component, filename)
        if os.path.exists(file_path):
            try:
                surface = pygame.image.load(file_path).convert_alpha()
                self._surfaces[cache_key] = surface
                return surface
            except pygame.error as e:
                print(f"Error loading {file_path}: {e}")
                return None
        return None
    
    def get_surfaces(self) -> Dict[str, pygame.Surface]:

        surfaces = {}
        
        component_map = {
            "Car_Shape": self.car_shape,
            "Anterior_Wide_Body": self.anterior_wide_body,
            "Posterior_Wide_Body": self.posterior_wide_body,
            "Hood_Shape": self.hood_shape,
            "Hood": self.hood,
            "Back_Bumper": self.back_bumper,
            "Front_Bumper": self.front_bumper,
            "Side_Bumper": self.side_bumper,
            "Posterior_Lights": self.posterior_lights,
            "Anterior_Lights": self.anterior_lights,
            "Car_Cabin": self.car_cabin,
            "Roof_Scoops": self.roof_scoops,
            "Spoiler": self.spoiler
        }
        
        for component, filename in component_map.items():
            if filename:
                surface = self._load_surface(component, filename)
                if surface:
                    surfaces[component] = surface
        
        return surfaces
    
    def to_json(self) -> str:

        data = {
            "anterior_lights": self.anterior_lights,
            "anterior_wide_body": self.anterior_wide_body,
            "back_bumper": self.back_bumper,
            "car_cabin": self.car_cabin,
            "car_shape": self.car_shape,
            "front_bumper": self.front_bumper,
            "hood": self.hood,
            "hood_shape": self.hood_shape,
            "posterior_lights": self.posterior_lights,
            "posterior_wide_body": self.posterior_wide_body,
            "roof_scoops": self.roof_scoops,
            "side_bumper": self.side_bumper,
            "spoiler": self.spoiler
        }
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Car_skin':

        data = json.loads(json_str)
        return cls(**data)
    
    @classmethod
    def generate_random_skin(cls, assets_path: str = "assets/Car") -> 'Car_skin':

        def get_random_file_from_folder(folder_path: str) -> Optional[str]:
            """Get a random file from a folder, or None if folder doesn't exist or is empty."""
            if not os.path.exists(folder_path):
                return None
            
            files = [f for f in os.listdir(folder_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            
            if not files:
                return None
            
            return random.choice(files)
        
        # Component folder mapping
        component_folders = {
            "anterior_lights": "Anterior_Lights",
            "anterior_wide_body": "Anterior_Wide_Body", 
            "back_bumper": "Back_Bumper",
            "car_cabin": "Car_Cabin",
            "car_shape": "Car_Shape",
            "front_bumper": "Front_Bumper",
            "hood": "Hood",
            "hood_shape": "Hood_Shape",
            "posterior_lights": "Posterior_Lights",
            "posterior_wide_body": "Posterior_Wide_Body",
            "roof_scoops": "Roof_Scoops",
            "side_bumper": "Side_Bumper",
            "spoiler": "Spoiler"
        }
        
        # Generate random components
        random_components = {}
        
        # Define probability for each component type
        component_probabilities = {
            "anterior_wide_body": 0.5,   # 50% chance
            "posterior_wide_body": 0.5,  # 50% chance  
            "side_bumper": 0.5,          # 50% chance
            "front_bumper": 0.5,         # 50% chance
            # All other components have 90% chance (default)
        }
        
        for component_name, folder_name in component_folders.items():
            folder_path = os.path.join(assets_path, folder_name)
            random_file = get_random_file_from_folder(folder_path)
            
            if random_file:
                # Get the probability for this component (default 100% if not specified)
                probability = component_probabilities.get(component_name, 1)
                
                # Include component based on its probability
                if random.random() < probability:
                    random_components[component_name] = random_file
        
        return cls(**random_components)
    
    def __repr__(self) -> str:
        """String representation of the car skin."""
        components = []
        if self.car_shape:
            components.append(f"shape={self.car_shape}")
        if self.anterior_lights:
            components.append(f"anterior_lights={self.anterior_lights}")
        if self.spoiler:
            components.append(f"spoiler={self.spoiler}")
        
        return f"Car_skin({', '.join(components)})"
    