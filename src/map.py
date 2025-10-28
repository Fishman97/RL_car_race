import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
import xml.etree.ElementTree as ET

if TYPE_CHECKING:
    from src.car import Car

class Map:
    
    def __init__(self, map_path: str, tileset: Dict[str, Any]):
        self.map_path = map_path
        self.tileset = tileset
        
        self.map_data = self._load_map()
        
        self.grid_width = self.map_data.get("map_width", 20)
        self.grid_height = self.map_data.get("map_height", 30)
        
        self.start_position = self.map_data.get("start_position", [100, 100])
        self.finish_line = self.map_data.get("finish_line", [(400, 300)])
        
        self.track_mask = self._create_track_mask()
        
        self.distance_field = self._compute_distance_field(self.track_mask)
        self.max_distance = np.max(self.distance_field[self.distance_field != np.inf])
    
    def _create_track_mask(self) -> np.ndarray:
        mask = np.ones((self.grid_height, self.grid_width), dtype=bool)
        
        tile_layers = self.map_data.get("layers", [])
        
        for layer in tile_layers:
            tiles = layer.get("data", [])
            layer_width = layer.get("width", self.grid_width)
            layer_height = layer.get("height", self.grid_height)
            
            for y in range(layer_height):
                for x in range(layer_width):
                    tile_id = tiles[y * layer_width + x] - 1
                    
                    if self._is_solid_tile(tile_id):
                        if x < self.grid_width and y < self.grid_height:
                            mask[y, x] = False
        
        return mask
    
    def _is_solid_tile(self, tile_id: int) -> bool:
        if tile_id == 0:
            return False
        tile_properties = self.tileset.get("tiles", {}).get(str(tile_id), {})
        
        return tile_properties.get("solid", False)
    
    def _compute_distance_field(self, track_mask: np.ndarray) -> np.ndarray:
        distance_field = np.full((self.grid_height, self.grid_width), np.inf, dtype=float)
        
        queue = deque()
        
        for finish_x, finish_y in self.finish_line:
            grid_x = int(finish_x)
            grid_y = int(finish_y)
            
            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                distance_field[grid_y, grid_x] = 0.0
                queue.append((grid_x, grid_y, 0.0))
        
        visited = set()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            curr_x, curr_y, curr_dist = queue.popleft()
            
            if (curr_x, curr_y) in visited:
                continue
            visited.add((curr_x, curr_y))
            
            for dx, dy in directions:
                next_x = curr_x + dx
                next_y = curr_y + dy
                
                if not (0 <= next_x < self.grid_width and 0 <= next_y < self.grid_height):
                    continue
                
                if (next_x, next_y) in visited:
                    continue
                
                if not track_mask[next_y, next_x]:
                    continue
                
                new_dist = curr_dist + 1.0
                
                if new_dist < distance_field[next_y, next_x]:
                    distance_field[next_y, next_x] = new_dist
                    queue.append((next_x, next_y, new_dist))
        
        return distance_field
    
    def get_start_position(self) -> Tuple[float, float]:
        return tuple(self.start_position)
    
    def get_distance_to_finish(self, x: float, y: float) -> float:
        grid_x = int(x)
        grid_y = int(y)
        
        if not (0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height):
            return np.inf
        
        return self.distance_field[grid_y, grid_x]
    
    def get_normalized_distance_to_finish(self, x: float, y: float) -> float:
        distance = self.get_distance_to_finish(x, y)
        
        if distance == np.inf:
            return 1.0
        
        return np.clip(distance / self.max_distance, 0.0, 1.0)
    
    def check_collision(self, car: 'Car') -> bool:
        collision_box = car.get_collision_box()
        
        for corner_x, corner_y in collision_box:
            grid_x = int(corner_x)
            grid_y = int(corner_y)
            
            if not (0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height):
                return True
            
            if not self.track_mask[grid_y, grid_x]:
                return True
        
        return False

    def check_finish_line(self, car: 'Car') -> bool:
        collision_box = car.get_collision_box()
        
        for corner_x, corner_y in collision_box:
            grid_x = int(corner_x)
            grid_y = int(corner_y)
            
            if (grid_x, grid_y) in self.finish_line:
                return True
        return False
    
    def _load_map(self) -> Dict[str, Any]:
        tree = ET.parse(self.map_path)
        root = tree.getroot()
        
        map_width = int(root.get("width", 20))
        map_height = int(root.get("height", 30))

        
        map_data = {
            "map_width": map_width,
            "map_height": map_height,
            "layers": [],
            "obstacles": [],
            "start_position": None,
            "finish_line": []
        }
        
        for layer in root.findall("layer"):
            layer_name = layer.get("name", "")
            layer_width = int(layer.get("width", map_width))
            layer_height = int(layer.get("height", map_height))
            
            data_element = layer.find("data")
            if data_element is not None:
                encoding = data_element.get("encoding", "")
                
                if encoding == "csv":
                    csv_data = (data_element.text or "").strip()
                    tile_ids = [int(tid) for tid in csv_data.replace('\n', ',').split(',') if tid.strip()]
                    
                    map_data["layers"].append({
                        "name": layer_name,
                        "width": layer_width,
                        "height": layer_height,
                        "data": tile_ids
                    })
                    
                    for idx, tile_id in enumerate(tile_ids):
                        tile_id = tile_id - 1
                        
                        tile_props = self.tileset.get("tiles", {}).get(str(tile_id), {})
                        
                        tile_x = (idx % layer_width)
                        tile_y = (idx // layer_width)
                        
                        if tile_props.get("start", False) and map_data["start_position"] is None:
                            map_data["start_position"] = [tile_x, tile_y]
                        
                        if tile_props.get("finish", False):
                            map_data["finish_line"].append((tile_x, tile_y))

        if map_data["start_position"] is None or not map_data["finish_line"]:
            raise ValueError("Map must have a start position and at least one finish line point.")
            
        return map_data

def load_tileset(tileset_path: str) -> Dict[str, Any]:
    tree = ET.parse(tileset_path)
    root = tree.getroot()
    
    tileset_data = {
        "name": root.get("name", ""),
        "tilewidth": int(root.get("tilewidth", 32)),
        "tileheight": int(root.get("tileheight", 32)),
        "tilecount": int(root.get("tilecount", 0)),
        "columns": int(root.get("columns", 1)),
        "tiles": {}
    }
    
    for tile in root.findall("tile"):
        tile_id = tile.get("id")
        tile_properties = {}
        
        properties = tile.find("properties")
        if properties is not None:
            for prop in properties.findall("property"):
                prop_name = prop.get("name")
                prop_value = prop.get("value")
                prop_type = prop.get("type", "string")
                
                if prop_type == "bool":
                    tile_properties[prop_name] = prop_value is not None and prop_value.lower() == "true"
                elif prop_type == "int":
                    tile_properties[prop_name] = int(prop_value) if prop_value is not None else 0
                elif prop_type == "float":
                    tile_properties[prop_name] = float(prop_value) if prop_value is not None else 0.0
                else:
                    tile_properties[prop_name] = prop_value
        
        tileset_data["tiles"][tile_id] = tile_properties
    
    return tileset_data

