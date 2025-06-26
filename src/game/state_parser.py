"""
Pokemon Leaf Green game state parser.
Extracts game state information from memory or screen analysis.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import struct
import logging
from dataclasses import dataclass
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger(__name__)


class GameRegion(Enum):
    """Pokemon Leaf Green map regions."""
    PALLET_TOWN = "pallet_town"
    VIRIDIAN_CITY = "viridian_city"
    PEWTER_CITY = "pewter_city"
    CERULEAN_CITY = "cerulean_city"
    VERMILLION_CITY = "vermillion_city"
    LAVENDER_TOWN = "lavender_town"
    CELADON_CITY = "celadon_city"
    FUCHSIA_CITY = "fuchsia_city"
    SAFFRON_CITY = "saffron_city"
    CINNABAR_ISLAND = "cinnabar_island"
    INDIGO_PLATEAU = "indigo_plateau"
    ROUTE = "route"
    UNKNOWN = "unknown"


@dataclass
class Pokemon:
    """Pokemon data structure."""
    species_id: int
    level: int
    hp: int
    max_hp: int
    attack: int
    defense: int
    speed: int
    special_attack: int
    special_defense: int
    experience: int
    status: str = "normal"
    
    @property
    def hp_percentage(self) -> float:
        return self.hp / self.max_hp if self.max_hp > 0 else 0.0


@dataclass
class PlayerState:
    """Player character state."""
    x: int
    y: int
    map_id: int
    direction: int  # 0=down, 1=up, 2=left, 3=right
    money: int
    badges: List[bool]  # 8 gym badges
    pokemon_count: int
    current_pokemon: Optional[Pokemon]


@dataclass
class GameState:
    """Complete game state representation."""
    player: PlayerState
    party_pokemon: List[Pokemon]
    current_region: GameRegion
    in_battle: bool
    in_menu: bool
    text_displayed: bool
    game_time: int  # in frames
    
    # Battle-specific state
    enemy_pokemon: Optional[Pokemon] = None
    battle_menu_open: bool = False
    
    # Additional metadata
    frame_count: int = 0
    last_action: int = 0


class MemoryParser:
    """Parses Pokemon Leaf Green memory addresses for game state."""
    
    # Memory addresses for Pokemon Leaf Green (USA v1.1)
    MEMORY_ADDRESSES = {
        'player_x': 0x02024284,
        'player_y': 0x02024286,
        'map_id': 0x020241E0,
        'player_direction': 0x02024288,
        'money': 0x020244EC,
        'badges': 0x02024572,
        'party_count': 0x02024029,
        'party_pokemon': 0x0202402C,
        'battle_state': 0x02022B4C,
        'text_state': 0x0202223C,
        'menu_state': 0x02022B50,
        'game_time': 0x020244F8,
    }
    
    BADGE_NAMES = [
        "Boulder", "Cascade", "Thunder", "Rainbow",
        "Soul", "Marsh", "Volcano", "Earth"
    ]
    
    def __init__(self):
        """Initialize memory parser."""
        self.last_state = None
        logger.info("Memory parser initialized")
    
    def parse_memory(self, memory_data: bytes) -> Optional[GameState]:
        """
        Parse memory dump to extract game state.
        
        Args:
            memory_data: Raw memory dump from emulator
            
        Returns:
            GameState: Parsed game state or None if failed
        """
        try:
            # Parse player state
            player = self._parse_player_state(memory_data)
            
            # Parse Pokemon party
            party_pokemon = self._parse_party_pokemon(memory_data)
            
            # Parse battle state
            in_battle = self._is_in_battle(memory_data)
            enemy_pokemon = self._parse_enemy_pokemon(memory_data) if in_battle else None
            
            # Parse UI state
            in_menu = self._is_in_menu(memory_data)
            text_displayed = self._is_text_displayed(memory_data)
            battle_menu_open = self._is_battle_menu_open(memory_data)
            
            # Determine current region
            current_region = self._determine_region(player.map_id, player.x, player.y)
            
            # Parse game time
            game_time = self._read_u32(memory_data, self.MEMORY_ADDRESSES['game_time'])
            
            state = GameState(
                player=player,
                party_pokemon=party_pokemon,
                current_region=current_region,
                in_battle=in_battle,
                in_menu=in_menu,
                text_displayed=text_displayed,
                game_time=game_time,
                enemy_pokemon=enemy_pokemon,
                battle_menu_open=battle_menu_open
            )
            
            self.last_state = state
            return state
            
        except Exception as e:
            logger.error(f"Failed to parse memory: {e}")
            return None
    
    def _parse_player_state(self, memory_data: bytes) -> PlayerState:
        """Parse player character state from memory."""
        x = self._read_u16(memory_data, self.MEMORY_ADDRESSES['player_x'])
        y = self._read_u16(memory_data, self.MEMORY_ADDRESSES['player_y'])
        map_id = self._read_u8(memory_data, self.MEMORY_ADDRESSES['map_id'])
        direction = self._read_u8(memory_data, self.MEMORY_ADDRESSES['player_direction'])
        money = self._read_u32(memory_data, self.MEMORY_ADDRESSES['money'])
        
        # Parse badges (8 bits in one byte)
        badge_byte = self._read_u8(memory_data, self.MEMORY_ADDRESSES['badges'])
        badges = [(badge_byte >> i) & 1 == 1 for i in range(8)]
        
        # Get party count and first Pokemon
        pokemon_count = self._read_u8(memory_data, self.MEMORY_ADDRESSES['party_count'])
        current_pokemon = None
        if pokemon_count > 0:
            party_pokemon = self._parse_party_pokemon(memory_data)
            if party_pokemon:
                current_pokemon = party_pokemon[0]
        
        return PlayerState(
            x=x, y=y, map_id=map_id, direction=direction,
            money=money, badges=badges, pokemon_count=pokemon_count,
            current_pokemon=current_pokemon
        )
    
    def _parse_party_pokemon(self, memory_data: bytes) -> List[Pokemon]:
        """Parse party Pokemon from memory."""
        party_pokemon = []
        party_count = self._read_u8(memory_data, self.MEMORY_ADDRESSES['party_count'])
        
        if party_count == 0:
            return party_pokemon
        
        # Each Pokemon structure is 100 bytes
        pokemon_size = 100
        base_addr = self.MEMORY_ADDRESSES['party_pokemon']
        
        for i in range(min(party_count, 6)):  # Max 6 Pokemon in party
            offset = base_addr + (i * pokemon_size)
            
            try:
                pokemon = self._parse_single_pokemon(memory_data, offset)
                if pokemon:
                    party_pokemon.append(pokemon)
            except Exception as e:
                logger.warning(f"Failed to parse Pokemon {i}: {e}")
                continue
        
        return party_pokemon
    
    def _parse_single_pokemon(self, memory_data: bytes, offset: int) -> Optional[Pokemon]:
        """Parse a single Pokemon structure from memory."""
        try:
            # Pokemon data structure offsets
            species_id = self._read_u16(memory_data, offset + 0x20)
            level = self._read_u8(memory_data, offset + 0x54)
            hp = self._read_u16(memory_data, offset + 0x56)
            max_hp = self._read_u16(memory_data, offset + 0x58)
            attack = self._read_u16(memory_data, offset + 0x5A)
            defense = self._read_u16(memory_data, offset + 0x5C)
            speed = self._read_u16(memory_data, offset + 0x5E)
            special_attack = self._read_u16(memory_data, offset + 0x60)
            special_defense = self._read_u16(memory_data, offset + 0x62)
            experience = self._read_u32(memory_data, offset + 0x50)
            
            # Status condition
            status_byte = self._read_u8(memory_data, offset + 0x53)
            status = self._decode_status(status_byte)
            
            if species_id == 0:  # Empty slot
                return None
            
            return Pokemon(
                species_id=species_id,
                level=level,
                hp=hp,
                max_hp=max_hp,
                attack=attack,
                defense=defense,
                speed=speed,
                special_attack=special_attack,
                special_defense=special_defense,
                experience=experience,
                status=status
            )
            
        except Exception as e:
            logger.error(f"Error parsing Pokemon at offset {offset:08x}: {e}")
            return None
    
    def _parse_enemy_pokemon(self, memory_data: bytes) -> Optional[Pokemon]:
        """Parse enemy Pokemon data during battle."""
        # Enemy Pokemon data is at a different memory location
        # This would need the specific memory addresses for battle state
        # For now, return a placeholder
        return None
    
    def _is_in_battle(self, memory_data: bytes) -> bool:
        """Check if currently in battle."""
        battle_state = self._read_u8(memory_data, self.MEMORY_ADDRESSES['battle_state'])
        return battle_state != 0
    
    def _is_in_menu(self, memory_data: bytes) -> bool:
        """Check if a menu is currently open."""
        menu_state = self._read_u8(memory_data, self.MEMORY_ADDRESSES['menu_state'])
        return menu_state != 0
    
    def _is_text_displayed(self, memory_data: bytes) -> bool:
        """Check if text dialog is currently displayed."""
        text_state = self._read_u8(memory_data, self.MEMORY_ADDRESSES['text_state'])
        return text_state != 0
    
    def _is_battle_menu_open(self, memory_data: bytes) -> bool:
        """Check if battle menu is open."""
        # This would need specific battle menu state address
        return False
    
    def _determine_region(self, map_id: int, x: int, y: int) -> GameRegion:
        """Determine current game region based on map ID and coordinates."""
        # Map ID to region mapping (simplified)
        region_mapping = {
            0: GameRegion.PALLET_TOWN,
            1: GameRegion.VIRIDIAN_CITY,
            2: GameRegion.PEWTER_CITY,
            3: GameRegion.CERULEAN_CITY,
            4: GameRegion.VERMILLION_CITY,
            5: GameRegion.LAVENDER_TOWN,
            6: GameRegion.CELADON_CITY,
            7: GameRegion.FUCHSIA_CITY,
            8: GameRegion.SAFFRON_CITY,
            9: GameRegion.CINNABAR_ISLAND,
            10: GameRegion.INDIGO_PLATEAU,
        }
        
        if 20 <= map_id <= 50:
            return GameRegion.ROUTE
        
        return region_mapping.get(map_id, GameRegion.UNKNOWN)
    
    def _decode_status(self, status_byte: int) -> str:
        """Decode Pokemon status condition."""
        if status_byte == 0:
            return "normal"
        elif status_byte & 0x07:
            return "sleep"
        elif status_byte & 0x08:
            return "poison"
        elif status_byte & 0x10:
            return "burn"
        elif status_byte & 0x20:
            return "freeze"
        elif status_byte & 0x40:
            return "paralysis"
        else:
            return "unknown"
    
    def _read_u8(self, data: bytes, address: int) -> int:
        """Read unsigned 8-bit value from memory."""
        if address >= len(data):
            return 0
        return data[address]
    
    def _read_u16(self, data: bytes, address: int) -> int:
        """Read unsigned 16-bit value from memory (little endian)."""
        if address + 1 >= len(data):
            return 0
        return struct.unpack('<H', data[address:address+2])[0]
    
    def _read_u32(self, data: bytes, address: int) -> int:
        """Read unsigned 32-bit value from memory (little endian)."""
        if address + 3 >= len(data):
            return 0
        return struct.unpack('<I', data[address:address+4])[0]


class ScreenStateParser:
    """Parses game state from screen analysis when memory access isn't available."""
    
    def __init__(self):
        """Initialize screen-based state parser."""
        self.last_state = None
        logger.info("Screen state parser initialized")
    
    def parse_screen(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Parse game state from screen analysis.
        
        Args:
            frame: Game screen frame (84x84 grayscale)
            
        Returns:
            dict: Extracted game state information
        """
        try:
            state = {
                'in_battle': self._detect_battle_screen(frame),
                'in_menu': self._detect_menu_screen(frame),
                'text_displayed': self._detect_text_box(frame),
                'hp_bars': self._detect_hp_bars(frame),
                'overworld': self._detect_overworld(frame)
            }
            
            self.last_state = state
            return state
            
        except Exception as e:
            logger.error(f"Screen parsing failed: {e}")
            return {}
    
    def _detect_battle_screen(self, frame: np.ndarray) -> bool:
        """Detect if currently in a Pokemon battle."""
        # Look for battle UI elements (HP bars, Pokemon sprites)
        # This is a simplified detection based on common UI patterns
        
        # Check for HP bar regions (top portion of screen)
        top_region = frame[:20, :]
        hp_bar_pixels = np.sum(top_region > 0.8)  # White/bright pixels
        
        # Battle screens typically have more UI elements
        return hp_bar_pixels > 50
    
    def _detect_menu_screen(self, frame: np.ndarray) -> bool:
        """Detect if a menu is currently displayed."""
        # Menus typically have rectangular borders and text
        edges = cv2.Canny((frame * 255).astype(np.uint8), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for rectangular contours (menu boxes)
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:  # Rectangular shape
                area = cv2.contourArea(contour)
                if area > 200:  # Large enough to be a menu
                    return True
        
        return False
    
    def _detect_text_box(self, frame: np.ndarray) -> bool:
        """Detect if text dialog is displayed."""
        # Text boxes are usually at the bottom of the screen
        bottom_region = frame[60:, :]
        
        # Look for horizontal lines (text box borders)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        horizontal_lines = cv2.morphologyEx(
            (bottom_region * 255).astype(np.uint8),
            cv2.MORPH_OPEN,
            horizontal_kernel
        )
        
        return np.sum(horizontal_lines > 0) > 20
    
    def _detect_hp_bars(self, frame: np.ndarray) -> Dict[str, float]:
        """Detect HP bars and estimate HP percentages."""
        hp_info = {'player_hp': 1.0, 'enemy_hp': 1.0}
        
        # HP bars are typically green/yellow/red horizontal bars
        # This would require more sophisticated color detection
        # For now, return default values
        
        return hp_info
    
    def _detect_overworld(self, frame: np.ndarray) -> bool:
        """Detect if currently in the overworld (not in battle/menu)."""
        return not (self._detect_battle_screen(frame) or self._detect_menu_screen(frame))


class StateParser:
    """Main state parser that combines memory and screen analysis."""
    
    def __init__(self, use_memory: bool = True, use_screen: bool = True):
        """
        Initialize state parser.
        
        Args:
            use_memory: Whether to use memory parsing
            use_screen: Whether to use screen analysis
        """
        self.use_memory = use_memory
        self.use_screen = use_screen
        
        self.memory_parser = MemoryParser() if use_memory else None
        self.screen_parser = ScreenStateParser() if use_screen else None
        
        self.frame_count = 0
        logger.info(f"State parser initialized (memory: {use_memory}, screen: {use_screen})")
    
    def parse_state(
        self,
        frame: np.ndarray,
        memory_data: Optional[bytes] = None
    ) -> Optional[GameState]:
        """
        Parse complete game state from available data.
        
        Args:
            frame: Current game frame
            memory_data: Emulator memory dump (optional)
            
        Returns:
            GameState: Complete game state or None if parsing failed
        """
        self.frame_count += 1
        
        try:
            # Try memory parsing first (more accurate)
            if self.memory_parser and memory_data:
                state = self.memory_parser.parse_memory(memory_data)
                if state:
                    state.frame_count = self.frame_count
                    return state
            
            # Fall back to screen analysis
            if self.screen_parser:
                screen_state = self.screen_parser.parse_screen(frame)
                
                # Create basic state from screen analysis
                # This is limited compared to memory parsing
                state = GameState(
                    player=PlayerState(
                        x=0, y=0, map_id=0, direction=0,
                        money=0, badges=[False]*8, pokemon_count=1,
                        current_pokemon=None
                    ),
                    party_pokemon=[],
                    current_region=GameRegion.UNKNOWN,
                    in_battle=screen_state.get('in_battle', False),
                    in_menu=screen_state.get('in_menu', False),
                    text_displayed=screen_state.get('text_displayed', False),
                    game_time=self.frame_count,
                    frame_count=self.frame_count
                )
                
                return state
            
            logger.warning("No parsing method available")
            return None
            
        except Exception as e:
            logger.error(f"State parsing failed: {e}")
            return None
    
    def get_state_features(self, state: GameState) -> np.ndarray:
        """
        Extract numerical features from game state for RL agent.
        
        Args:
            state: Parsed game state
            
        Returns:
            np.ndarray: Feature vector
        """
        features = []
        
        # Player position (normalized)
        features.extend([state.player.x / 1000.0, state.player.y / 1000.0])
        
        # Player direction (one-hot encoded)
        direction_onehot = [0, 0, 0, 0]
        if 0 <= state.player.direction < 4:
            direction_onehot[state.player.direction] = 1
        features.extend(direction_onehot)
        
        # Badges (8 binary features)
        features.extend([int(badge) for badge in state.player.badges])
        
        # Pokemon party info
        features.append(state.player.pokemon_count / 6.0)  # Normalized party size
        
        # Current Pokemon HP (if available)
        if state.player.current_pokemon:
            features.append(state.player.current_pokemon.hp_percentage)
            features.append(state.player.current_pokemon.level / 100.0)
        else:
            features.extend([0.0, 0.0])
        
        # Game state flags
        features.extend([
            int(state.in_battle),
            int(state.in_menu),
            int(state.text_displayed)
        ])
        
        # Money (log-normalized)
        features.append(np.log10(max(1, state.player.money)) / 6.0)
        
        return np.array(features, dtype=np.float32)


if __name__ == "__main__":
    # Test the state parser
    parser = StateParser(use_memory=False, use_screen=True)
    
    # Create a dummy frame
    test_frame = np.random.rand(84, 84).astype(np.float32)
    
    print("Testing state parser...")
    state = parser.parse_state(test_frame)
    
    if state:
        print(f"Parsed state: in_battle={state.in_battle}, in_menu={state.in_menu}")
        features = parser.get_state_features(state)
        print(f"Feature vector shape: {features.shape}")
        print(f"Features: {features}")
    else:
        print("Failed to parse state")
