#!/usr/bin/env python3
"""
Pokemon RL Bot Monitoring and Web Dashboard Script

This script starts the web dashboard for monitoring and controlling the Pokemon RL bot.
Supports real-time game streaming, training metrics, and manual control.
"""

import os
import sys
import json
import argparse
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from web.app import create_app
from game.environment import PokemonEnvironment
from agent.ppo_agent import PPOAgent
from utils.logger import get_logger
from utils.metrics import MetricsTracker

logger = get_logger(__name__)


class MonitoringServer:
    """Pokemon RL Bot monitoring server with web dashboard."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the monitoring server.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.web_app = None
        self.environment = None
        self.agent = None
        self.metrics_tracker = MetricsTracker()
        self.running = False
        
        # Training state
        self.training_thread = None
        self.training_active = False
        
    def setup_environment(self, rom_path: str, save_path: str = None):
        """Setup the Pokemon environment.
        
        Args:
            rom_path: Path to Pokemon ROM file
            save_path: Path to save state file
        """
        try:
            env_config = self.config.get('emulator', {}).copy()
            env_config.update(self.config.get('environment', {}))
            env_config['rom_path'] = rom_path
            
            if save_path:
                env_config['save_path'] = save_path
            
            # Web mode should not be headless by default
            env_config['headless'] = False
            
            self.environment = PokemonEnvironment(env_config)
            logger.info(f"Environment setup with ROM: {rom_path}")
            
        except Exception as e:
            logger.error(f"Failed to setup environment: {e}")
            raise
    
    def setup_agent(self, model_path: str = None):
        """Setup the PPO agent.
        
        Args:
            model_path: Path to saved model file
        """
        try:
            if not self.environment:
                raise ValueError("Environment must be setup before agent")
            
            self.agent = PPOAgent(
                env=self.environment,
                tensorboard_log=self.config.get('logging', {}).get('tensorboard_log', './logs/tensorboard')
            )
            
            if model_path and os.path.exists(model_path):
                self.agent.load(model_path)
                logger.info(f"Agent loaded from: {model_path}")
            else:
                logger.info("New agent created")
                
        except Exception as e:
            logger.error(f"Failed to setup agent: {e}")
            raise
    
    def setup_web_app(self):
        """Setup the Flask web application."""
        try:
            web_config = self.config.get('web', {})
            web_config.update({
                'secret_key': web_config.get('secret_key', 'pokemon-rl-monitoring'),
                'headless': False,  # Web mode shows display
                'frame_skip': self.config.get('emulator', {}).get('frame_skip', 4)
            })
            
            self.web_app = create_app(web_config)
            
            # Load environment and agent into web app
            if self.environment:
                rom_path = self.config.get('emulator', {}).get('rom_path', 'roms/pokemon_leafgreen.gba')
                save_path = self.config.get('emulator', {}).get('save_path')
                self.web_app.load_environment(rom_path, save_path)
            
            if self.agent:
                self.web_app.load_agent()
            
            logger.info("Web application setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup web application: {e}")
            raise
    
    def start_training_session(self, timesteps: int = 100000):
        """Start a training session in a separate thread.
        
        Args:
            timesteps: Number of timesteps to train
        """
        if self.training_active:
            logger.warning("Training session already active")
            return False
        
        if not self.agent:
            logger.error("Agent not loaded")
            return False
        
        def training_worker():
            """Training worker function."""
            try:
                self.training_active = True
                logger.info(f"Starting training session: {timesteps} timesteps")
                
                # Create callback for web updates
                class WebCallback:
                    def __init__(self, web_app, metrics_tracker):
                        self.web_app = web_app
                        self.metrics_tracker = metrics_tracker
                        self.step_count = 0
                    
                    def __call__(self, locals_dict, globals_dict):
                        self.step_count += 1
                        
                        # Update metrics every 100 steps
                        if self.step_count % 100 == 0:
                            metrics = self.metrics_tracker.get_summary()
                            self.web_app.broadcast_metrics(metrics)
                        
                        return True
                
                callback = WebCallback(self.web_app, self.metrics_tracker)
                
                # Train the agent
                self.agent.learn(
                    total_timesteps=timesteps,
                    callback=callback,
                    progress_bar=False  # Disable progress bar for web mode
                )
                
                logger.info("Training session completed")
                
            except Exception as e:
                logger.error(f"Training session failed: {e}")
            finally:
                self.training_active = False
        
        self.training_thread = threading.Thread(target=training_worker, daemon=True)
        self.training_thread.start()
        return True
    
    def stop_training_session(self):
        """Stop the current training session."""
        if not self.training_active:
            logger.warning("No training session active")
            return False
        
        # Note: Stable-Baselines3 doesn't have built-in training interruption
        # In a production environment, you would implement a more sophisticated
        # training loop with proper interruption handling
        self.training_active = False
        logger.info("Training session stop requested")
        return True
    
    def start_game_streaming(self):
        """Start streaming game frames to web clients."""
        if not self.environment:
            logger.error("Environment not loaded")
            return False
        
        def streaming_worker():
            """Streaming worker function."""
            try:
                while self.running:
                    if hasattr(self.environment, 'emulator'):
                        frame = self.environment.emulator.get_screen()
                        if frame is not None and self.web_app:
                            self.web_app.broadcast_frame(frame)
                    
                    time.sleep(1/30)  # 30 FPS streaming
                    
            except Exception as e:
                logger.error(f"Streaming worker error: {e}")
        
        streaming_thread = threading.Thread(target=streaming_worker, daemon=True)
        streaming_thread.start()
        logger.info("Game streaming started")
        return True
    
    def start_metrics_broadcasting(self):
        """Start broadcasting training metrics to web clients."""
        def metrics_worker():
            """Metrics broadcasting worker."""
            try:
                while self.running:
                    if self.web_app:
                        metrics = self.metrics_tracker.get_summary()
                        self.web_app.broadcast_metrics(metrics)
                    
                    time.sleep(5)  # Update every 5 seconds
                    
            except Exception as e:
                logger.error(f"Metrics worker error: {e}")
        
        metrics_thread = threading.Thread(target=metrics_worker, daemon=True)
        metrics_thread.start()
        logger.info("Metrics broadcasting started")
        return True
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Run the monitoring server.
        
        Args:
            host: Host address to bind to
            port: Port number to listen on
            debug: Enable debug mode
        """
        if not self.web_app:
            raise RuntimeError("Web application not setup")
        
        self.running = True
        
        # Start background services
        self.start_game_streaming()
        self.start_metrics_broadcasting()
        
        logger.info(f"Starting Pokemon RL monitoring server on {host}:{port}")
        logger.info(f"Dashboard URL: http://{host}:{port}")
        
        try:
            self.web_app.run(host=host, port=port, debug=debug)
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        finally:
            self.running = False
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up monitoring server...")
        
        # Stop training if active
        if self.training_active:
            self.stop_training_session()
        
        # Close environment
        if self.environment:
            try:
                self.environment.close()
                logger.info("Environment closed")
            except Exception as e:
                logger.error(f"Error closing environment: {e}")
        
        logger.info("Cleanup completed")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
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


def main():
    """Main monitoring function."""
    parser = argparse.ArgumentParser(description="Pokemon RL Bot Monitoring Dashboard")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--rom', '-r',
        type=str,
        help='Path to Pokemon ROM file'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Path to trained model file'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host address to bind to'
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=5000,
        help='Port number to listen on'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--no-env',
        action='store_true',
        help='Start without loading environment (ROM required later)'
    )
    parser.add_argument(
        '--no-agent',
        action='store_true',
        help='Start without loading agent'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger.info("=" * 60)
    logger.info("🎮 Pokemon RL Bot Monitoring Dashboard")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Create monitoring server
        server = MonitoringServer(config)
        
        # Setup environment if ROM provided
        if not args.no_env:
            rom_path = args.rom or config.get('emulator', {}).get('rom_path', 'roms/pokemon_leafgreen.gba')
            
            if os.path.exists(rom_path):
                save_path = config.get('emulator', {}).get('save_path')
                server.setup_environment(rom_path, save_path)
            else:
                logger.warning(f"ROM file not found: {rom_path}")
                if not args.rom:
                    logger.info("You can load a ROM later through the web interface")
        
        # Setup agent if model provided
        if not args.no_agent and server.environment:
            model_path = args.model
            if model_path and os.path.exists(model_path):
                server.setup_agent(model_path)
            else:
                logger.info("No model loaded - you can train a new one through the web interface")
        
        # Setup web application
        server.setup_web_app()
        
        # Override config with command line arguments
        host = args.host
        port = args.port
        debug = args.debug
        
        # Run the server
        server.run(host=host, port=port, debug=debug)
        
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        raise


if __name__ == "__main__":
    main()
