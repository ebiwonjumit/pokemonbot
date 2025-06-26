"""
Pokemon Leaf Green Reinforcement Learning Bot

A comprehensive RL system for training AI agents to play Pokemon Leaf Green
using PPO with VBA-M emulator integration and real-time web monitoring.
"""

__version__ = "1.0.0"
__author__ = "Pokemon RL Team"
__description__ = "Pokemon Leaf Green RL Bot with PPO and Web Dashboard"

# Import main components for easy access
from .game.emulator import VBAEmulator, EmulatorAction
from .game.environment import PokemonEnv
from .game.state_parser import StateParser, GameState
from .game.reward_calculator import RewardCalculator

from .agent.ppo_agent import PPOAgent
from .agent.neural_network import PokemonCNN
from .agent.trainer import RLTrainer

from .web.app import create_app
from .utils.logger import setup_logging, get_logger
from .utils.metrics import MetricsCollector, MetricsVisualizer
from .utils.cloud_storage import ModelManager, create_storage_provider

# Package metadata
__all__ = [
    # Game components
    "VBAEmulator",
    "EmulatorAction", 
    "PokemonEnv",
    "StateParser",
    "GameState",
    "RewardCalculator",
    
    # Agent components
    "PPOAgent",
    "PokemonCNN",
    "RLTrainer",
    
    # Web components
    "create_app",
    
    # Utilities
    "setup_logging",
    "get_logger",
    "MetricsCollector",
    "MetricsVisualizer", 
    "ModelManager",
    "create_storage_provider",
]
