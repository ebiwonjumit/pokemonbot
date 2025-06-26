#!/usr/bin/env python3
"""
Pokemon RL Bot Training Script

This script trains the PPO agent to play Pokemon Leaf Green using reinforcement learning.
Supports resuming from checkpoints, distributed training, and real-time monitoring.
"""

import os
import sys
import json
import argparse
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from game.environment import PokemonEnvironment
from agent.ppo_agent import PPOAgent
from agent.trainer import Trainer
from utils.logger import get_logger
from utils.metrics import MetricsTracker
from utils.cloud_storage import CloudStorageManager

logger = get_logger(__name__)


class TrainingCallback(BaseCallback):
    """Custom callback for training monitoring and checkpointing."""
    
    def __init__(
        self,
        save_freq: int = 10000,
        save_path: str = "./models/",
        metrics_tracker: Optional[MetricsTracker] = None,
        cloud_manager: Optional[CloudStorageManager] = None,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.metrics_tracker = metrics_tracker
        self.cloud_manager = cloud_manager
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = float('-inf')
        self.last_save_step = 0
        
    def _on_step(self) -> bool:
        """Called at each training step."""
        # Save model periodically
        if self.n_calls % self.save_freq == 0:
            self._save_checkpoint()
        
        # Track episode completion
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    episode_reward = info['episode']['r']
                    episode_length = info['episode']['l']
                    
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    
                    # Update best reward
                    if episode_reward > self.best_reward:
                        self.best_reward = episode_reward
                        self._save_best_model()
                    
                    # Log episode
                    logger.info(
                        f"Episode completed - Reward: {episode_reward:.2f}, "
                        f"Length: {episode_length}, Best: {self.best_reward:.2f}"
                    )
                    
                    # Update metrics tracker
                    if self.metrics_tracker:
                        self.metrics_tracker.log_episode(episode_reward, episode_length)
        
        return True
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = self.save_path / f"checkpoint_{timestamp}_step_{self.n_calls}.zip"
            
            self.model.save(str(checkpoint_path))
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Upload to cloud if configured
            if self.cloud_manager:
                try:
                    self.cloud_manager.upload_model(str(checkpoint_path))
                except Exception as e:
                    logger.warning(f"Failed to upload checkpoint to cloud: {e}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _save_best_model(self):
        """Save the best performing model."""
        try:
            best_path = self.save_path / "best_model.zip"
            self.model.save(str(best_path))
            logger.info(f"Best model saved: {best_path} (reward: {self.best_reward:.2f})")
            
            # Upload to cloud
            if self.cloud_manager:
                try:
                    self.cloud_manager.upload_model(str(best_path), "best_model.zip")
                except Exception as e:
                    logger.warning(f"Failed to upload best model to cloud: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to save best model: {e}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def create_environment(config: Dict[str, Any], monitor_path: str = None) -> PokemonEnvironment:
    """Create and configure the Pokemon environment.
    
    Args:
        config: Environment configuration
        monitor_path: Path for monitoring wrapper
        
    Returns:
        Configured environment
    """
    env_config = config.get('emulator', {})
    env_config.update(config.get('environment', {}))
    
    env = PokemonEnvironment(env_config)
    
    # Wrap with monitor for statistics
    if monitor_path:
        env = Monitor(env, monitor_path, allow_early_resets=True)
    
    logger.info("Environment created and configured")
    return env


def create_vectorized_env(config: Dict[str, Any], n_envs: int = 1) -> DummyVecEnv:
    """Create vectorized environment for parallel training.
    
    Args:
        config: Environment configuration
        n_envs: Number of parallel environments
        
    Returns:
        Vectorized environment
    """
    def make_env(rank: int):
        def _init():
            env_config = config.get('emulator', {}).copy()
            env_config.update(config.get('environment', {}))
            
            # Use different ROM for each environment if available
            base_rom = env_config.get('rom_path', 'roms/pokemon_leafgreen.gba')
            rom_dir = os.path.dirname(base_rom)
            rom_name = os.path.basename(base_rom)
            
            # Try rank-specific ROM first
            rank_rom = os.path.join(rom_dir, f"{rank}_{rom_name}")
            if os.path.exists(rank_rom):
                env_config['rom_path'] = rank_rom
            
            # Create unique save path for each environment
            base_save = env_config.get('save_path', 'saves/pokemon_leafgreen.sav')
            save_dir = os.path.dirname(base_save)
            save_name = os.path.basename(base_save)
            env_config['save_path'] = os.path.join(save_dir, f"{rank}_{save_name}")
            
            env = PokemonEnvironment(env_config)
            monitor_path = f"logs/env_{rank}_monitor"
            env = Monitor(env, monitor_path, allow_early_resets=True)
            
            return env
        return _init
    
    if n_envs == 1:
        env = DummyVecEnv([make_env(0)])
    else:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    
    logger.info(f"Vectorized environment created with {n_envs} processes")
    return env


def setup_agent(env, config: Dict[str, Any], model_path: str = None) -> PPOAgent:
    """Setup the PPO agent.
    
    Args:
        env: Training environment
        config: Agent configuration
        model_path: Path to load existing model
        
    Returns:
        Configured PPO agent
    """
    training_config = config.get('training', {})
    
    agent = PPOAgent(
        env=env,
        learning_rate=training_config.get('learning_rate', 3e-4),
        batch_size=training_config.get('batch_size', 64),
        n_epochs=training_config.get('n_epochs', 10),
        gamma=training_config.get('gamma', 0.99),
        clip_range=training_config.get('clip_range', 0.2),
        tensorboard_log="logs/tensorboard",
        verbose=1
    )
    
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
        logger.info(f"Agent loaded from {model_path}")
    else:
        logger.info("New agent created")
    
    return agent


def train_agent(
    agent: PPOAgent,
    total_timesteps: int,
    callbacks: list,
    config: Dict[str, Any]
) -> PPOAgent:
    """Train the agent.
    
    Args:
        agent: PPO agent to train
        total_timesteps: Total training timesteps
        callbacks: Training callbacks
        config: Training configuration
        
    Returns:
        Trained agent
    """
    training_config = config.get('training', {})
    
    logger.info(f"Starting training for {total_timesteps} timesteps")
    
    # Train the agent
    agent.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=training_config.get('log_frequency', 1000),
        progress_bar=True
    )
    
    logger.info("Training completed")
    return agent


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Pokemon RL Bot")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Path to existing model to resume training'
    )
    parser.add_argument(
        '--timesteps', '-t',
        type=int,
        help='Override total timesteps'
    )
    parser.add_argument(
        '--envs', '-e',
        type=int,
        default=1,
        help='Number of parallel environments'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run in headless mode'
    )
    parser.add_argument(
        '--no-cloud',
        action='store_true',
        help='Disable cloud storage uploads'
    )
    parser.add_argument(
        '--gpu',
        type=str,
        choices=['auto', 'cpu', 'cuda', 'mps'],
        default='auto',
        help='Device to use for training'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger.info("=" * 60)
    logger.info("🎮 Pokemon RL Bot Training Started")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override config with command line arguments
        if args.timesteps:
            config.setdefault('training', {})['total_timesteps'] = args.timesteps
        
        if args.headless:
            config.setdefault('emulator', {})['headless'] = True
        
        # Setup device
        if args.gpu == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        else:
            device = args.gpu
        
        logger.info(f"Using device: {device}")
        
        # Create environment
        if args.envs > 1:
            env = create_vectorized_env(config, args.envs)
        else:
            env = create_environment(config, "logs/monitor")
        
        # Setup metrics tracking
        metrics_tracker = MetricsTracker()
        
        # Setup cloud storage
        cloud_manager = None
        if not args.no_cloud:
            try:
                cloud_config = config.get('cloud_storage', {})
                if cloud_config.get('enabled', False):
                    cloud_manager = CloudStorageManager(cloud_config)
                    logger.info("Cloud storage enabled")
            except Exception as e:
                logger.warning(f"Failed to setup cloud storage: {e}")
        
        # Setup agent
        agent = setup_agent(env, config, args.model)
        
        # Setup callbacks
        training_callback = TrainingCallback(
            save_freq=config.get('training', {}).get('save_frequency', 10000),
            save_path="models/",
            metrics_tracker=metrics_tracker,
            cloud_manager=cloud_manager
        )
        
        callbacks = [training_callback]
        
        # Setup signal handler for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("Received interrupt signal, saving model...")
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                emergency_path = f"models/emergency_save_{timestamp}.zip"
                agent.save(emergency_path)
                logger.info(f"Emergency save completed: {emergency_path}")
            except Exception as e:
                logger.error(f"Emergency save failed: {e}")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Train the agent
        total_timesteps = config.get('training', {}).get('total_timesteps', 1000000)
        trained_agent = train_agent(agent, total_timesteps, callbacks, config)
        
        # Final save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = f"models/final_model_{timestamp}.zip"
        trained_agent.save(final_path)
        logger.info(f"Final model saved: {final_path}")
        
        # Upload final model to cloud
        if cloud_manager:
            try:
                cloud_manager.upload_model(final_path)
                logger.info("Final model uploaded to cloud")
            except Exception as e:
                logger.warning(f"Failed to upload final model: {e}")
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Cleanup
        try:
            if 'env' in locals():
                env.close()
                logger.info("Environment closed")
        except Exception as e:
            logger.error(f"Error closing environment: {e}")


if __name__ == "__main__":
    main()
