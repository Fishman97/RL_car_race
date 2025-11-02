import pygame
import numpy as np
import os
from typing import Optional, Dict, Any
from src.car import Car
from src.map import Map
from src.car_skin import Car_skin

PANEL_WIDTH = 300
DEFAULT_WINDOW_HEIGHT = 800
DEFAULT_WINDOW_WIDTH = 834
class Renderer:
    
    def __init__(self, map_obj: Map, x_tiles: int, y_tiles: int):
        pygame.init()
        
        self.map = map_obj
        self.tileset = map_obj.tileset
        self.x_tiles = x_tiles
        self.y_tiles = y_tiles

        self._resize_window(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

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
        self.YELLOW = (255, 255, 0)

        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
        self._font_cache: Dict[int, pygame.font.Font] = {}
        
        self.clock = pygame.time.Clock()
        
        self.view_mode = "normal"  # "normal", "mask", "flood"
        
        self.buttons = {}
        
        self._load_tileset_image()
    
    def _resize_window(self, width: int, height: int):
        tile_size_by_width = (width - PANEL_WIDTH) / self.x_tiles
        tile_size_by_height = height / self.y_tiles
        self.tile_size = min(tile_size_by_width, tile_size_by_height)
        self.window_width = width
        self.window_height = height

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

    def world_to_screen(self, x: float, y: float) -> tuple:
        screen_x = int(x * self.tile_size) + PANEL_WIDTH
        screen_y = int(y * self.tile_size)
        return screen_x, screen_y

    def render(self, car: Car, observation: np.ndarray, action: np.ndarray, attempts: int = -1) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_button_click(event.pos)
            elif event.type == pygame.VIDEORESIZE:
                self._resize_window(event.w, event.h)
                self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
        
        self.screen.fill(self.LIGHT_GRAY)

        self._draw_panel(car, observation, action, attempts)

        if self.view_mode == "normal":
            self._draw_map()
            self._draw_texture_car(car)
        elif self.view_mode == "mask":
            self._draw_mask_view()
            self._draw_car(car)
            self._draw_sensors(car)
        elif self.view_mode == "flood":
            self._draw_flood_view()
            self._draw_car(car)
            self._draw_sensors(car)

        
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
                    
                    screen_x = int(x * self.tile_size + PANEL_WIDTH)
                    screen_y = int(y * self.tile_size)
                    next_screen_x = int((x + 1) * self.tile_size + PANEL_WIDTH)
                    next_screen_y = int((y + 1) * self.tile_size)
                    
                    tile_width = next_screen_x - screen_x
                    tile_height = next_screen_y - screen_y
                    
                    scaled_tile = pygame.transform.scale(tile_surface, (tile_width, tile_height))
                    
                    self.screen.blit(scaled_tile, (screen_x, screen_y))
    
    def _draw_mask_view(self):
        for y in range(self.map.grid_height):
            for x in range(self.map.grid_width):
                # Calculate pixel-perfect boundaries for each tile
                screen_x = int(x * self.tile_size + PANEL_WIDTH)
                screen_y = int(y * self.tile_size)
                next_screen_x = int((x + 1) * self.tile_size + PANEL_WIDTH)
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
        if car.skin is None:
            self._draw_car(car)
            return

        surfaces = car.skin.get_surfaces()
        if not surfaces:
            self._draw_car(car)
            return

        base_surface = surfaces.get("Car_Shape", next(iter(surfaces.values())))
        composite = pygame.Surface(base_surface.get_size(), pygame.SRCALPHA)
        for surface in surfaces.values():
            composite.blit(surface, (0, 0))

        angle_degrees = -np.degrees(car.angle)
        rotated_composite = pygame.transform.rotate(composite, angle_degrees)

        screen_points = [self.world_to_screen(x, y) for x, y in car.get_collision_box()]
        min_x = min(p[0] for p in screen_points)
        max_x = max(p[0] for p in screen_points)
        min_y = min(p[1] for p in screen_points)
        max_y = max(p[1] for p in screen_points)

        bbox_width = int(max_x - min_x)
        bbox_height = int(max_y - min_y)
        if bbox_width <= 0 or bbox_height <= 0:
            return

        zoom_factor = 1.6
        scaled_width = max(1, int(bbox_width * zoom_factor))
        scaled_height = max(1, int(bbox_height * zoom_factor))
        scaled_composite = pygame.transform.scale(rotated_composite, (scaled_width, scaled_height))

        offset_x = (scaled_width - bbox_width) // 2
        offset_y = (scaled_height - bbox_height) // 2
        adjusted_points = [(x - min_x + offset_x, y - min_y + offset_y) for x, y in screen_points]

        mask_surface = pygame.Surface((scaled_width, scaled_height), pygame.SRCALPHA)
        pygame.draw.polygon(mask_surface, (255, 255, 255, 255), adjusted_points)

        textured_surface = scaled_composite.copy()
        textured_surface.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

        blit_x = min_x - offset_x
        blit_y = min_y - offset_y
        self.screen.blit(textured_surface, (blit_x, blit_y))
    

    def _draw_car(self, car: Car):
        collision_box = car.get_collision_box()
        
        screen_points = [self.world_to_screen(x, y) for x, y in collision_box]
        
        pygame.draw.polygon(self.screen, self.YELLOW, screen_points)
        
        front_x = car.x + (car.height / 2) * np.cos(car.angle - np.pi / 2)
        front_y = car.y + (car.height / 2) * np.sin(car.angle - np.pi / 2)
        front_screen_x, front_screen_y = self.world_to_screen(front_x, front_y)
        

    def _draw_sensors(self, car: Car):
        sensors = car.get_sensors_data(self.map)
        
        for i, angle_offset in enumerate(sensors):
            sensor_x, sensor_y, distance, value = angle_offset
            
            start_screen_x, start_screen_y = self.world_to_screen(car.x, car.y)
            end_line_x, end_line_y = self.world_to_screen(sensor_x, sensor_y)

            pygame.draw.line(self.screen, self.GRAY, (start_screen_x, start_screen_y), (end_line_x, end_line_y), 1)
            color = self.RED if i < 3 else self.BLUE
            pygame.draw.circle(self.screen, color, (end_line_x, end_line_y), self.tile_size // 3)


    def _draw_panel(self, car: Car, observation: np.ndarray, action: np.ndarray, attempts: int = -1):
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, (0, 0, PANEL_WIDTH, self.window_height))
        scale = PANEL_WIDTH / 300.0
        font_size = max(12, int(20 * scale))
        font = self._get_font(font_size)
        base_x = int(10 * scale)
        y_offset = int(10 * scale)

        runs_surface = font.render(f"Runs: {attempts}", True, self.BLACK)
        self.screen.blit(runs_surface, (base_x, y_offset))
        y_offset += max(int(font_size * 1.2), 1)

        headers = ["Speed", "Angle", "X", "Y"]
        values = [f"{car.get_speed():.1f}", f"{np.degrees(car.angle) % 360:.1f}", f"{car.x:.0f}", f"{car.y:.0f}"]
        col_width = max(int((PANEL_WIDTH - 2 * base_x) / max(len(headers), 1)), 40)
        value_y = y_offset + max(int(font_size * 1.05), 1)

        for idx, header in enumerate(headers):
            x_pos = base_x + idx * col_width
            self.screen.blit(font.render(header, True, self.BLACK), (x_pos, y_offset))
            self.screen.blit(font.render(values[idx], True, self.BLACK), (x_pos, value_y))

        y_offset = value_y + max(int(font_size * 1.2), 1)

        def _draw_value_table(title: str, table_headers, table_values, y_start: int) -> int:
            table_values = table_values.flatten().tolist()
            if not table_headers or not table_values:
                return y_start
            title_surface = font.render(title, True, self.BLACK)
            self.screen.blit(title_surface, (base_x, y_start))
            y_start += max(int(font_size * 1.1), 1)
            cell_height = max(int(font_size * 1.2), 18)
            table_width = max(int(PANEL_WIDTH - 2 * base_x), 10)
            col_count = len(table_headers)
            base_width = max(int(table_width / max(col_count, 1)), 1)
            widths = [base_width] * max(col_count - 1, 0)
            last_width = table_width - sum(widths)
            widths.append(max(last_width, 1))
            x_cursor = base_x
            for idx, header in enumerate(table_headers):
                rect = pygame.Rect(x_cursor, y_start, widths[idx], cell_height)
                pygame.draw.rect(self.screen, self.DARK_GRAY, rect)
                pygame.draw.rect(self.screen, self.BLACK, rect, 1)
                header_surface = font.render(str(header), True, self.WHITE)
                header_rect = header_surface.get_rect(center=rect.center)
                self.screen.blit(header_surface, header_rect)
                x_cursor += widths[idx]
            y_start += cell_height
            x_cursor = base_x
            for idx, raw_value in enumerate(table_values[:len(widths)]):
                shade = abs(int(raw_value * 255))
                rect = pygame.Rect(x_cursor, y_start, widths[idx], cell_height)
                color = (shade, shade, shade)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.BLACK, rect, 1)
                text_color = self.BLACK if shade > 160 else self.WHITE
                value_surface = font.render(f"{raw_value:.2f}", True, text_color)
                value_rect = value_surface.get_rect(center=rect.center)
                self.screen.blit(value_surface, value_rect)
                x_cursor += widths[idx]
            return y_start + cell_height + max(int(font_size * 0.8), 4)

        y_offset = _draw_value_table("Observation", ["c1", "c2", "c3", "f1", "f2","f3","f4","v"], observation, y_offset)
        y_offset = _draw_value_table("Action", ["Steering", "Acceleration", "Brake"], action, y_offset)


        button_margin = max(int(10 * scale), 4)
        button_height = max(int(40 * scale), 24)
        button_spacing = max(int(10 * scale), 6)
        button_definitions = [
            ("normal", "Texture"),
            ("mask", "Collision"),
            ("flood", "Flood"),
        ]
        available_width = PANEL_WIDTH - 2 * button_margin
        button_width = max(int((available_width - button_spacing * (len(button_definitions) - 1)) / len(button_definitions)), 20)
        button_y = self.window_height - button_height - button_margin
        self.buttons = {}
        for index, (key, label) in enumerate(button_definitions):
            rect_x = button_margin + index * (button_width + button_spacing)
            rect = pygame.Rect(rect_x, button_y, button_width, button_height)
            self.buttons[key] = {"rect": rect, "label": label}
        for button_key, button_data in self.buttons.items():
            rect = button_data["rect"]
            label = button_data["label"]
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
        show_text = self.tile_size > 10
        font_dist = self._get_font(int(self.tile_size))

        for y in range(self.map.grid_height):
            for x in range(self.map.grid_width):
                # Calculate pixel-perfect boundaries for each tile
                screen_x = int(x * self.tile_size + PANEL_WIDTH)
                screen_y = int(y * self.tile_size)
                next_screen_x = int((x + 1) * self.tile_size + PANEL_WIDTH)
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

                if show_text and distance != np.inf:
                    dist_text = f"{int(distance)}"
                    text_surface = font_dist.render(dist_text, True, self.WHITE)
                    text_rect = text_surface.get_rect(center=(screen_x + tile_width // 2, screen_y + tile_height // 2))
                    self.screen.blit(text_surface, text_rect)
    
    def _get_font(self, size: int) -> pygame.font.Font:
        size = max(1, int(size))
        font = self._font_cache.get(size)
        if font is None:
            font = pygame.font.Font(None, size)
            self._font_cache[size] = font
        return font

    def close(self):
        pygame.quit()