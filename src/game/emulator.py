"""
VBA-M Emulator Wrapper for Pokemon Leaf Green
Handles emulator process management, screen capture, and input injection.
"""

import subprocess
import time
import signal
import os
import numpy as np
import cv2
import pyautogui
from PIL import Image, ImageGrab
import psutil
from typing import Optional, Tuple, List
import logging
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


class EmulatorAction:
    """Emulator action constants matching GBA button layout."""
    NO_OP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    A = 5
    B = 6
    START = 7
    SELECT = 8
    L = 9
    R = 10

    ACTION_NAMES = {
        NO_OP: "NO_OP",
        UP: "UP",
        DOWN: "DOWN", 
        LEFT: "LEFT",
        RIGHT: "RIGHT",
        A: "A",
        B: "B",
        START: "START",
        SELECT: "SELECT",
        L: "L",
        R: "R"
    }

    # VBA-M key mappings
    KEY_MAPPINGS = {
        UP: "Up",
        DOWN: "Down",
        LEFT: "Left", 
        RIGHT: "Right",
        A: "z",
        B: "x",
        START: "Return",
        SELECT: "Backslash",
        L: "a",
        R: "s"
    }


class VBAEmulator:
    """VBA-M emulator wrapper with screen capture and input injection."""
    
    def __init__(
        self,
        emulator_path: str = "/usr/local/bin/vbam",
        rom_path: str = "roms/pokemon_leaf_green.gba",
        save_path: str = "saves/",
        window_title: str = "VisualBoyAdvance-M",
        capture_region: Optional[Tuple[int, int, int, int]] = None,
        headless: bool = False
    ):
        """
        Initialize VBA emulator wrapper.
        
        Args:
            emulator_path: Path to VBA-M executable
            rom_path: Path to Pokemon ROM file
            save_path: Directory for save states
            window_title: Emulator window title for screen capture
            capture_region: Screen region to capture (x, y, width, height)
            headless: Run in headless mode (for cloud deployment)
        """
        self.emulator_path = emulator_path
        self.rom_path = rom_path
        self.save_path = Path(save_path)
        self.window_title = window_title
        self.capture_region = capture_region
        self.headless = headless
        
        self.process: Optional[subprocess.Popen] = None
        self.window_handle = None
        self.is_running = False
        
        # Create save directory
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Screen capture settings
        self.frame_width = 240
        self.frame_height = 160
        self.output_width = 84
        self.output_height = 84
        
        # Performance monitoring
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        logger.info(f"VBA Emulator initialized with ROM: {rom_path}")
    
    def start(self) -> bool:
        """
        Start the VBA-M emulator process.
        
        Returns:
            bool: True if started successfully
        """
        try:
            if self.is_running:
                logger.warning("Emulator already running")
                return True
            
            if not os.path.exists(self.emulator_path):
                raise FileNotFoundError(f"Emulator not found: {self.emulator_path}")
            
            if not os.path.exists(self.rom_path):
                raise FileNotFoundError(f"ROM not found: {self.rom_path}")
            
            # VBA-M command line arguments
            cmd = [
                self.emulator_path,
                self.rom_path,
                "--frame-skip=0",
                "--throttle=100",
                "--save-dir=" + str(self.save_path)
            ]
            
            if self.headless:
                cmd.extend(["--no-gui", "--disable-audio"])
            
            logger.info(f"Starting emulator: {' '.join(cmd)}")
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            # Wait for emulator to initialize
            time.sleep(3)
            
            if self.process.poll() is None:
                self.is_running = True
                logger.info("Emulator started successfully")
                
                # Find window for screen capture
                if not self.headless:
                    self._find_window()
                
                return True
            else:
                stdout, stderr = self.process.communicate()
                logger.error(f"Emulator failed to start: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start emulator: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the emulator process gracefully.
        
        Returns:
            bool: True if stopped successfully
        """
        try:
            if not self.is_running or not self.process:
                return True
            
            logger.info("Stopping emulator...")
            
            # Try graceful shutdown first
            self.process.terminate()
            
            # Wait for process to terminate
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Force killing emulator process")
                if os.name != 'nt':
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                else:
                    self.process.kill()
            
            self.process = None
            self.is_running = False
            logger.info("Emulator stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping emulator: {e}")
            return False
    
    def capture_screen(self) -> Optional[np.ndarray]:
        """
        Capture current game screen and preprocess for RL agent.
        
        Returns:
            np.ndarray: Preprocessed 84x84 grayscale frame or None if failed
        """
        try:
            if self.headless:
                # For headless mode, implement framebuffer capture
                return self._capture_framebuffer()
            
            # Capture screen using PIL
            if self.capture_region:
                screenshot = ImageGrab.grab(bbox=self.capture_region)
            else:
                screenshot = ImageGrab.grab()
            
            # Convert to numpy array
            frame = np.array(screenshot)
            
            # Convert BGR to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Preprocess frame
            processed_frame = self._preprocess_frame(frame)
            
            # Update FPS counter
            self._update_fps_counter()
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return None
    
    def send_action(self, action: int, duration: float = 0.1) -> bool:
        """
        Send input action to the emulator.
        
        Args:
            action: Action constant from EmulatorAction
            duration: How long to hold the key (seconds)
            
        Returns:
            bool: True if action sent successfully
        """
        try:
            if not self.is_running:
                return False
            
            if action == EmulatorAction.NO_OP:
                time.sleep(duration)
                return True
            
            if action not in EmulatorAction.KEY_MAPPINGS:
                logger.error(f"Invalid action: {action}")
                return False
            
            key = EmulatorAction.KEY_MAPPINGS[action]
            
            # Send key press using pyautogui
            pyautogui.keyDown(key)
            time.sleep(duration)
            pyautogui.keyUp(key)
            
            logger.debug(f"Sent action: {EmulatorAction.ACTION_NAMES.get(action, action)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send action {action}: {e}")
            return False
    
    def save_state(self, slot: int = 1) -> bool:
        """
        Save current game state to specified slot.
        
        Args:
            slot: Save slot number (1-10)
            
        Returns:
            bool: True if saved successfully
        """
        try:
            if not (1 <= slot <= 10):
                raise ValueError("Save slot must be between 1-10")
            
            # VBA-M save state hotkey: Shift + F{slot}
            pyautogui.hotkey('shift', f'f{slot}')
            time.sleep(0.5)
            
            logger.info(f"Saved state to slot {slot}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save state to slot {slot}: {e}")
            return False
    
    def load_state(self, slot: int = 1) -> bool:
        """
        Load game state from specified slot.
        
        Args:
            slot: Save slot number (1-10)
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            if not (1 <= slot <= 10):
                raise ValueError("Save slot must be between 1-10")
            
            # VBA-M load state hotkey: F{slot}
            pyautogui.press(f'f{slot}')
            time.sleep(0.5)
            
            logger.info(f"Loaded state from slot {slot}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state from slot {slot}: {e}")
            return False
    
    def reset(self) -> bool:
        """
        Reset the game (Ctrl+R in VBA-M).
        
        Returns:
            bool: True if reset successfully
        """
        try:
            pyautogui.hotkey('ctrl', 'r')
            time.sleep(1)
            logger.info("Game reset")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset game: {e}")
            return False
    
    def get_emulator_stats(self) -> dict:
        """
        Get emulator performance statistics.
        
        Returns:
            dict: Performance stats including FPS, memory usage, etc.
        """
        stats = {
            'is_running': self.is_running,
            'fps': self.current_fps,
            'process_id': self.process.pid if self.process else None,
            'memory_usage': 0,
            'cpu_usage': 0
        }
        
        if self.process and self.is_running:
            try:
                process = psutil.Process(self.process.pid)
                stats['memory_usage'] = process.memory_info().rss / 1024 / 1024  # MB
                stats['cpu_usage'] = process.cpu_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return stats
    
    def _find_window(self) -> bool:
        """Find the emulator window for screen capture."""
        # Implementation depends on OS and GUI framework
        # For now, use the full screen capture
        logger.info("Using full screen capture mode")
        return True
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess captured frame for RL agent.
        
        Args:
            frame: Raw captured frame
            
        Returns:
            np.ndarray: Processed 84x84 grayscale frame
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        # Resize to 84x84 for DQN/PPO
        resized = cv2.resize(gray, (self.output_width, self.output_height))
        
        # Normalize pixel values to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def _capture_framebuffer(self) -> Optional[np.ndarray]:
        """Capture framebuffer in headless mode."""
        # This would require VBA-M framebuffer access or Xvfb
        logger.warning("Framebuffer capture not implemented")
        return None
    
    def _update_fps_counter(self):
        """Update FPS counter for performance monitoring."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


if __name__ == "__main__":
    # Test the emulator wrapper
    emulator = VBAEmulator()
    
    try:
        print("Starting emulator...")
        if emulator.start():
            print("Emulator started successfully")
            
            # Test screen capture
            for i in range(10):
                frame = emulator.capture_screen()
                if frame is not None:
                    print(f"Captured frame {i+1}: {frame.shape}")
                else:
                    print(f"Failed to capture frame {i+1}")
                time.sleep(1)
            
            # Test actions
            print("Testing actions...")
            actions = [EmulatorAction.A, EmulatorAction.B, EmulatorAction.UP]
            for action in actions:
                print(f"Sending action: {EmulatorAction.ACTION_NAMES[action]}")
                emulator.send_action(action)
                time.sleep(1)
        
    finally:
        print("Stopping emulator...")
        emulator.stop()
