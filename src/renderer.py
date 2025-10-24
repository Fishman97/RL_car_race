import pygame
import numpy as np
import os
from typing import Optional, Dict, Any
from src.car import Car
from src.map import Map
from src.car_skin import Car_skin

class Renderer:
    
    def __init__(self, map_obj: Map, x_tiles: int, y_tiles: int):
        pygame.init()
        
        self.map = map_obj
        self.tileset = map_obj.tileset
        self.x_tiles = x_tiles
        self.y_tiles = y_tiles

        self.window_width = 600
        self.window_height = 800

        self._update_scale()

        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
        pygame.display.set_caption("RL Car Racing")
        
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (200, 200, 200)
        self.DARK_GRAY = (64, 64, 64)
        
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
        
        self.clock = pygame.time.Clock()
        
        self.view_mode = "normal"  # "normal", "mask", "flood"
        
        self.buttons = {}
        
        self._load_tileset_image()
    
    def _load_tileset_image(self):
        tileset_dir = os.path.dirname(os.path.dirname(__file__))
        tileset_path = os.path.join(tileset_dir, "assets", "tilesets", "flat_race.png")
        
        try:
            self.tileset_image = pygame.image.load(tileset_path).convert_alpha()
        except pygame.error as e:
            print(f"Could not load tileset image: {e}")
            self.tileset_image = None
        
        self.tile_width = self.tileset.get("tilewidth", 32)
        self.tile_height = self.tileset.get("tileheight", 32)
        self.tileset_columns = self.tileset.get("columns", 39)
    
    def _get_tile_surface(self, tile_id: int) -> Optional[pygame.Surface]:
        if self.tileset_image is None or tile_id < 0:
            return None
        
        col = tile_id % self.tileset_columns
        row = tile_id // self.tileset_columns
        
        tile_rect = pygame.Rect(
            col * self.tile_width,
            row * self.tile_height,
            self.tile_width,
            self.tile_height
        )
        
        try:
            tile_surface = self.tileset_image.subsurface(tile_rect)
            return tile_surface
        except ValueError:
            return None
                    
    def _update_scale(self):
        self.tile_size = self.calculate_tile_size()

        self.map_width = self.x_tiles * self.tile_size
        self.map_height = self.y_tiles * self.tile_size

    def calculate_tile_size(self):
        self.panel_width = self.window_width / 4
        map_area_width = self.window_width - self.panel_width
        map_area_height = self.window_height

        tile_size_x = map_area_width / self.x_tiles
        tile_size_y = map_area_height / self.y_tiles
        self.tile_size = min(tile_size_x, tile_size_y)

        map_width = self.x_tiles * self.tile_size
        if map_width + self.panel_width < self.window_width:
            new_panel_width = self.window_width - map_width
            self.panel_width = new_panel_width
        return self.tile_size

    def world_to_screen(self, x: float, y: float) -> tuple:
        screen_x = int(x * self.tile_size) + self.panel_width
        screen_y = int(y * self.tile_size)
        return screen_x, screen_y

    def render(self, car: Car, observation: Optional[np.ndarray] = None, attempts: int = -1) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_button_click(event.pos)
            elif event.type == pygame.VIDEORESIZE:
                self.window_width = event.w
                self.window_height = event.h
                self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
                self._update_scale()
        
        self.screen.fill(self.WHITE)
        
        self._draw_panel(car, observation, attempts)
        
        if self.view_mode == "normal":
            self._draw_map()
            self._draw_texture_car(car)
        elif self.view_mode == "mask":
            self._draw_mask_view()
            self._draw_sensors(car)
            self._draw_car(car)
        elif self.view_mode == "flood":
            self._draw_flood_view()
            self._draw_sensors(car)
            self._draw_car(car)
        
        pygame.display.flip()
        self.clock.tick(60)
        return True
    
    def _handle_button_click(self, pos):
        for button_key, button_data in self.buttons.items():
            if button_data["rect"].collidepoint(pos):
                self.view_mode = button_key
                break

    def _draw_map(self):
        if self.tileset_image is None:
            self._draw_mask_view()
            return
        
        layers = self.map.map_data.get("layers", [])
        
        for layer in layers:
            layer_data = layer.get("data", [])
            layer_width = layer.get("width", self.x_tiles)
            layer_height = layer.get("height", self.y_tiles)
            
            for y in range(min(layer_height, self.y_tiles)):
                for x in range(min(layer_width, self.x_tiles)):
                    idx = y * layer_width + x
                    
                    if idx >= len(layer_data):
                        continue
                    
                    tile_gid = layer_data[idx]
                    tile_id = tile_gid - 1  # Convert from GID to tile ID (0-based)
                    
                    if tile_id < 0:
                        continue
                    
                    tile_surface = self._get_tile_surface(tile_id)
                    
                    if tile_surface is None:
                        continue
                    
                    screen_x = int(x * self.tile_size + self.panel_width)
                    screen_y = int(y * self.tile_size)
                    next_screen_x = int((x + 1) * self.tile_size + self.panel_width)
                    next_screen_y = int((y + 1) * self.tile_size)
                    
                    tile_width = next_screen_x - screen_x
                    tile_height = next_screen_y - screen_y
                    
                    scaled_tile = pygame.transform.scale(tile_surface, (tile_width, tile_height))
                    
                    self.screen.blit(scaled_tile, (screen_x, screen_y))
    
    def _draw_mask_view(self):
        for y in range(self.map.grid_height):
            for x in range(self.map.grid_width):
                # Calculate pixel-perfect boundaries for each tile
                screen_x = int(x * self.tile_size + self.panel_width)
                screen_y = int(y * self.tile_size)
                next_screen_x = int((x + 1) * self.tile_size + self.panel_width)
                next_screen_y = int((y + 1) * self.tile_size)
                
                tile_width = next_screen_x - screen_x
                tile_height = next_screen_y - screen_y
                
                if not self.map.track_mask[y, x]:
                    pygame.draw.rect(self.screen, self.BLACK, 
                                   (screen_x, screen_y, tile_width, tile_height))
                else:
                    pygame.draw.rect(self.screen, self.LIGHT_GRAY, 
                                   (screen_x, screen_y, tile_width, tile_height))

        # Draw finish line in green
        if hasattr(self.map, 'finish_line'):
            finish = self.map.finish_line
            for finish_point in finish:
                x, y = self.world_to_screen(finish_point[0] + 0.5, finish_point[1] + 0.5)
                pygame.draw.circle(self.screen, self.GREEN, (x, y), self.tile_size / 3)

    def _draw_texture_car(self, car: Car):
        #TODO: Implement textured car drawing
        self._draw_car(car)

    def _draw_car(self, car: Car):
        collision_box = car.get_collision_box()
        
        screen_points = [self.world_to_screen(x, y) for x, y in collision_box]
        
        pygame.draw.polygon(self.screen, self.RED, screen_points)
        
        front_x = car.x + (car.height / 2) * np.cos(car.angle - np.pi / 2)
        front_y = car.y + (car.height / 2) * np.sin(car.angle - np.pi / 2)
        front_screen_x, front_screen_y = self.world_to_screen(front_x, front_y)
        
        pygame.draw.circle(self.screen, self.BLUE, (front_screen_x, front_screen_y), car.width * self.tile_size/2)

    def _draw_sensors(self, car: Car):
        sensors = car.get_sensors_data(self.map)
        
        for i, angle_offset in enumerate(sensors):
            sensor_x, sensor_y, distance, value = angle_offset
            
            start_screen_x, start_screen_y = self.world_to_screen(car.x, car.y)
            end_line_x, end_line_y = self.world_to_screen(sensor_x, sensor_y)

            pygame.draw.line(self.screen, self.GRAY, (start_screen_x, start_screen_y), (end_line_x, end_line_y), 1)
            color = self.RED if i < 3 else self.BLUE
            pygame.draw.circle(self.screen, color, (end_line_x, end_line_y), self.tile_size // 3)


    def _draw_panel(self, car: Car, observation: Optional[np.ndarray] = None, attempts: int = -1):
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, (0, 0, self.panel_width, self.window_height))
        
        scale = self.window_width//4 / 200.0

        speed = car.get_speed()
        angle_deg = np.degrees(car.angle) % 360
        
        info_text = [
            f"Attempts: {attempts}",
            "",
            f"Speed: {speed:.1f}",
            f"Angle: {angle_deg:.1f}°",
            f"Position:",
            f"  X: {car.x:.0f}",
            f"  Y: {car.y:.0f}"
        ]
    
        font_size = max(12, int(20 * scale))
        font = pygame.font.Font(None, font_size)
        
        y_offset = int(10 * scale)
        for text in info_text:
            text_surface = font.render(text, True, self.BLACK)
            self.screen.blit(text_surface, (int(10 * scale), y_offset))
            y_offset += int(22 * scale)

        button_y_start = max(int(350 * scale), y_offset + int(20 * scale))
        button_width = int(180 * scale)
        button_height = int(40 * scale)
        button_spacing = int(50 * scale)
        button_margin = int(10 * scale)

        self.buttons = {
            "mask": {"rect": pygame.Rect(button_margin, button_y_start, button_width, button_height), "label": "Mask View"},
            "flood": {"rect": pygame.Rect(button_margin, button_y_start + button_spacing, button_width, button_height), "label": "Flood View"},
            "normal": {"rect": pygame.Rect(button_margin, button_y_start + button_spacing * 2, button_width, button_height), "label": "Normal View"}
        }
        
        for button_key, button_data in self.buttons.items():
            rect = button_data["rect"]
            label = button_data["label"]
            
            # Highlight active view mode
            if button_key == self.view_mode:
                pygame.draw.rect(self.screen, self.BLUE, rect)
                text_color = self.WHITE
            else:
                pygame.draw.rect(self.screen, self.DARK_GRAY, rect)
                text_color = self.WHITE
            
            pygame.draw.rect(self.screen, self.BLACK, rect, max(1, int(2 * scale)))
            
            text_surface = font.render(label, True, text_color)
            text_rect = text_surface.get_rect(center=rect.center)
            self.screen.blit(text_surface, text_rect)
    
    def _draw_flood_view(self):
        max_dist = self.map.max_distance
        
        for y in range(self.map.grid_height):
            for x in range(self.map.grid_width):
                # Calculate pixel-perfect boundaries for each tile
                screen_x = int(x * self.tile_size + self.panel_width)
                screen_y = int(y * self.tile_size)
                next_screen_x = int((x + 1) * self.tile_size + self.panel_width)
                next_screen_y = int((y + 1) * self.tile_size)
                
                tile_width = next_screen_x - screen_x
                tile_height = next_screen_y - screen_y
                
                distance = self.map.distance_field[y, x]
                
                # Color based on distance (gradient from green to red)
                if distance == np.inf:
                    color = self.BLACK
                else:
                    norm_dist = distance / max_dist
                    red = int(255 * norm_dist)
                    green = int(255 * (1 - norm_dist))
                    color = (red, green, 0)
                
                pygame.draw.rect(self.screen, color, (screen_x, screen_y, tile_width, tile_height))

                if distance != np.inf and self.tile_size > 10:
                    dist_text = f"{int(distance)}"
                    font_size = max(1, int( self.tile_size))
                    font_dist = pygame.font.Font(None, font_size)
                    text_surface = font_dist.render(dist_text, True, self.WHITE)
                    text_rect = text_surface.get_rect(center=(screen_x + tile_width // 2, screen_y + tile_height // 2))
                    self.screen.blit(text_surface, text_rect)
    
    def close(self):
        pygame.quit()