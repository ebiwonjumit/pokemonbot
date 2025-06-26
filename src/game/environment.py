"""
OpenAI Gym environment wrapper for Pokemon Leaf Green.
Provides standardized RL interface for training agents.
"""

import gym
from gym import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import cv2
from collections import deque
import time

from .emulator import VBAEmulator, EmulatorAction
from .state_parser import StateParser, GameState
from .reward_calculator import RewardCalculator, RewardConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PokemonEnv(gym.Env):
    """
    OpenAI Gym environment for Pokemon Leaf Green.
    
    Observation Space: 4 stacked 84x84 grayscale frames
    Action Space: 9 discrete actions (D-pad + A,B,Start,Select + No-op)
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        rom_path: str = "roms/pokemon_leaf_green.gba",
        emulator_path: str = "/usr/local/bin/vbam",
        save_path: str = "saves/",
        reward_config: Optional[RewardConfig] = None,
        frame_stack: int = 4,
        frame_skip: int = 4,
        max_episode_steps: int = 10000,
        headless: bool = False,
        auto_save_interval: int = 1000,
        enable_sound: bool = False
    ):
        """
        Initialize Pokemon environment.
        
        Args:
            rom_path: Path to Pokemon ROM file
            emulator_path: Path to VBA-M emulator
            save_path: Directory for save states
            reward_config: Reward calculation configuration
            frame_stack: Number of frames to stack for observation
            frame_skip: Number of frames to skip between actions
            max_episode_steps: Maximum steps per episode
            headless: Run emulator in headless mode
            auto_save_interval: Interval for automatic save states
            enable_sound: Enable emulator sound
        """
        super(PokemonEnv, self).__init__()
        
        # Environment configuration
        self.rom_path = rom_path
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.auto_save_interval = auto_save_interval
        
        # Initialize emulator
        self.emulator = VBAEmulator(
            emulator_path=emulator_path,
            rom_path=rom_path,
            save_path=save_path,
            headless=headless
        )
        
        # Initialize state parser and reward calculator
        self.state_parser = StateParser(use_memory=False, use_screen=True)
        self.reward_calculator = RewardCalculator(reward_config or RewardConfig())
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(9)  # 9 possible actions
        
        # Observation space: 4 stacked 84x84 grayscale frames
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(frame_stack, 84, 84),
            dtype=np.float32
        )
        
        # Frame buffer for stacking
        self.frame_buffer = deque(maxlen=frame_stack)
        
        # Episode tracking
        self.episode_count = 0
        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_start_time = 0.0
        
        # Game state tracking
        self.current_state: Optional[GameState] = None
        self.last_state: Optional[GameState] = None
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        # Action mapping
        self.action_map = {
            0: EmulatorAction.NO_OP,
            1: EmulatorAction.UP,
            2: EmulatorAction.DOWN,
            3: EmulatorAction.LEFT,
            4: EmulatorAction.RIGHT,
            5: EmulatorAction.A,
            6: EmulatorAction.B,
            7: EmulatorAction.START,
            8: EmulatorAction.SELECT,
        }
        
        logger.info(f"Pokemon environment initialized with {self.action_space.n} actions")
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment and return initial observation.
        
        Returns:
            np.ndarray: Initial stacked frames observation
        """
        try:
            # Start emulator if not running
            if not self.emulator.is_running:
                if not self.emulator.start():
                    raise RuntimeError("Failed to start emulator")
                
                # Wait for emulator to stabilize
                time.sleep(2)
            
            # Load save state or reset game
            if self.episode_count == 0:
                # First episode - start from beginning
                self.emulator.reset()
                time.sleep(1)
            else:
                # Load from save state for consistency
                if self.episode_count % 10 == 0:  # Reset every 10 episodes
                    self.emulator.reset()
                    time.sleep(1)
                else:
                    self.emulator.load_state(1)  # Load from slot 1
                    time.sleep(0.5)
            
            # Reset tracking variables
            self.step_count = 0
            self.episode_reward = 0.0
            self.episode_start_time = time.time()
            self.reward_calculator.reset()
            
            # Clear frame buffer
            self.frame_buffer.clear()
            
            # Capture initial frames
            for _ in range(self.frame_stack):
                frame = self._capture_frame()
                if frame is not None:
                    self.frame_buffer.append(frame)
                else:
                    # Fallback: create black frame
                    self.frame_buffer.append(np.zeros((84, 84), dtype=np.float32))
            
            # Get initial game state
            observation = self._get_observation()
            self.current_state = self._parse_game_state(observation)
            
            self.episode_count += 1
            
            logger.info(f"Episode {self.episode_count} started")
            
            return observation
            
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            # Return empty observation as fallback
            return np.zeros((self.frame_stack, 84, 84), dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute action and return step results.
        
        Args:
            action: Action to execute (0-8)
            
        Returns:
            Tuple containing:
            - observation: New observation (stacked frames)
            - reward: Reward for this step
            - done: Whether episode is terminated
            - info: Additional information
        """
        try:
            if action not in self.action_map:
                logger.warning(f"Invalid action {action}, using NO_OP")
                action = 0
            
            # Store previous state
            self.last_state = self.current_state
            
            # Execute action with frame skipping
            emulator_action = self.action_map[action]
            
            for _ in range(self.frame_skip):
                self.emulator.send_action(emulator_action, duration=0.1)
                time.sleep(0.01)  # Small delay between repeated actions
            
            # Capture new frame
            frame = self._capture_frame()
            if frame is not None:
                self.frame_buffer.append(frame)
            
            # Get new observation and state
            observation = self._get_observation()
            self.current_state = self._parse_game_state(observation)
            
            # Calculate reward
            reward, reward_breakdown = self.reward_calculator.calculate_reward(
                self.current_state, action
            )
            
            # Check if episode is done
            done = self._is_episode_done()
            
            # Update counters
            self.step_count += 1
            self.episode_reward += reward
            
            # Auto-save periodically
            if self.step_count % self.auto_save_interval == 0:
                self.emulator.save_state(1)
            
            # Create info dictionary
            info = self._create_info_dict(reward_breakdown, action)
            
            # Update FPS counter
            self._update_fps_counter()
            
            return observation, reward, done, info
            
        except Exception as e:
            logger.error(f"Step failed: {e}")
            # Return safe fallback values
            observation = np.zeros((self.frame_stack, 84, 84), dtype=np.float32)
            return observation, 0.0, True, {'error': str(e)}
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Render mode ('human' or 'rgb_array')
            
        Returns:
            Optional[np.ndarray]: RGB image if mode='rgb_array'
        """
        try:
            frame = self.emulator.capture_screen()
            
            if frame is None:
                return None
            
            if mode == 'rgb_array':
                # Convert grayscale to RGB and resize for viewing
                rgb_frame = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                rgb_frame = cv2.resize(rgb_frame, (240, 160))  # Original GBA resolution
                return rgb_frame
            
            elif mode == 'human':
                # Display using OpenCV (for debugging)
                display_frame = cv2.resize((frame * 255).astype(np.uint8), (480, 320))
                cv2.imshow('Pokemon RL', display_frame)
                cv2.waitKey(1)
            
        except Exception as e:
            logger.error(f"Render failed: {e}")
        
        return None
    
    def close(self):
        """Clean up environment resources."""
        try:
            if hasattr(self, 'emulator'):
                self.emulator.stop()
            
            # Close any OpenCV windows
            cv2.destroyAllWindows()
            
            logger.info("Environment closed")
            
        except Exception as e:
            logger.error(f"Close failed: {e}")
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed
            
        Returns:
            List[int]: Seeds used
        """
        np.random.seed(seed)
        return [seed]
    
    def get_action_meanings(self) -> List[str]:
        """Get human-readable action names."""
        return [
            "NO_OP",
            "UP", 
            "DOWN",
            "LEFT",
            "RIGHT",
            "A",
            "B", 
            "START",
            "SELECT"
        ]
    
    def get_game_stats(self) -> Dict[str, Any]:
        """Get current game statistics."""
        stats = self.reward_calculator.get_reward_stats()
        
        # Add environment stats
        stats.update({
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'episode_reward': self.episode_reward,
            'fps': self.current_fps,
            'emulator_running': self.emulator.is_running,
        })
        
        # Add emulator stats
        emulator_stats = self.emulator.get_emulator_stats()
        stats.update(emulator_stats)
        
        return stats
    
    def save_state(self, slot: int = 1) -> bool:
        """Save current game state."""
        return self.emulator.save_state(slot)
    
    def load_state(self, slot: int = 1) -> bool:
        """Load game state."""
        return self.emulator.load_state(slot)
    
    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture and preprocess a single frame."""
        try:
            frame = self.emulator.capture_screen()
            
            if frame is None:
                logger.warning("Failed to capture frame")
                return None
            
            # Frame is already preprocessed by emulator (84x84 grayscale, normalized)
            return frame
            
        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            return None
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (stacked frames)."""
        if len(self.frame_buffer) < self.frame_stack:
            # Pad with zeros if not enough frames
            while len(self.frame_buffer) < self.frame_stack:
                self.frame_buffer.append(np.zeros((84, 84), dtype=np.float32))
        
        # Stack frames along first dimension
        stacked_frames = np.stack(list(self.frame_buffer), axis=0)
        return stacked_frames
    
    def _parse_game_state(self, observation: np.ndarray) -> Optional[GameState]:
        """Parse game state from observation."""
        try:
            # Use the latest frame for state parsing
            latest_frame = observation[-1] if len(observation.shape) == 3 else observation
            
            # Parse state using screen analysis
            state = self.state_parser.parse_state(latest_frame)
            
            return state
            
        except Exception as e:
            logger.error(f"State parsing failed: {e}")
            return None
    
    def _is_episode_done(self) -> bool:
        """Check if episode should terminate."""
        # Episode ends if:
        # 1. Maximum steps reached
        if self.step_count >= self.max_episode_steps:
            logger.info(f"Episode ended: max steps ({self.max_episode_steps}) reached")
            return True
        
        # 2. Emulator stopped working
        if not self.emulator.is_running:
            logger.warning("Episode ended: emulator stopped")
            return True
        
        # 3. Game state indicates episode should end (optional)
        if self.current_state:
            # Example: End if all Pokemon fainted (placeholder logic)
            # This would need more sophisticated state detection
            pass
        
        return False
    
    def _create_info_dict(self, reward_breakdown: Dict[str, float], action: int) -> Dict[str, Any]:
        """Create info dictionary for step return."""
        info = {
            'reward_breakdown': reward_breakdown,
            'action_taken': action,
            'action_name': self.get_action_meanings()[action],
            'step_count': self.step_count,
            'episode_reward': self.episode_reward,
            'fps': self.current_fps,
        }
        
        # Add game state info if available
        if self.current_state:
            info.update({
                'badges_earned': sum(self.current_state.player.badges),
                'pokemon_count': self.current_state.player.pokemon_count,
                'in_battle': self.current_state.in_battle,
                'in_menu': self.current_state.in_menu,
                'current_region': self.current_state.current_region.value,
            })
        
        # Add reward calculator stats
        reward_stats = self.reward_calculator.get_reward_stats()
        info['game_stats'] = reward_stats
        
        return info
    
    def _update_fps_counter(self):
        """Update FPS counter for performance monitoring."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time


class PokemonEnvWrapper(gym.Wrapper):
    """Additional wrapper for Pokemon environment with extra features."""
    
    def __init__(self, env: PokemonEnv):
        """Initialize wrapper."""
        super(PokemonEnvWrapper, self).__init__(env)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def reset(self, **kwargs):
        """Reset with additional tracking."""
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Step with additional tracking."""
        obs, reward, done, info = self.env.step(action)
        
        if done:
            self.episode_rewards.append(self.env.episode_reward)
            self.episode_lengths.append(self.env.step_count)
        
        return obs, reward, done, info
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Get episode statistics."""
        if not self.episode_rewards:
            return {}
        
        return {
            'mean_reward': np.mean(self.episode_rewards[-100:]),
            'mean_length': np.mean(self.episode_lengths[-100:]),
            'total_episodes': len(self.episode_rewards),
        }


def make_pokemon_env(
    rom_path: str = "roms/pokemon_leaf_green.gba",
    **kwargs
) -> PokemonEnv:
    """
    Factory function to create Pokemon environment.
    
    Args:
        rom_path: Path to Pokemon ROM
        **kwargs: Additional environment parameters
        
    Returns:
        PokemonEnv: Configured Pokemon environment
    """
    env = PokemonEnv(rom_path=rom_path, **kwargs)
    return env


if __name__ == "__main__":
    # Test the environment
    print("Testing Pokemon environment...")
    
    try:
        env = make_pokemon_env()
        
        print("Environment created successfully")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action meanings: {env.get_action_meanings()}")
        
        # Test reset
        obs = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            print(f"Step {i+1}: action={action}, reward={reward:.3f}, done={done}")
            print(f"  Info keys: {list(info.keys())}")
            
            if done:
                break
        
        # Test render
        frame = env.render(mode='rgb_array')
        if frame is not None:
            print(f"Rendered frame shape: {frame.shape}")
        
        # Get stats
        stats = env.get_game_stats()
        print(f"Game stats: {stats}")
        
        env.close()
        print("Environment test completed successfully!")
        
    except Exception as e:
        print(f"Environment test failed: {e}")
        import traceback
        traceback.print_exc()
