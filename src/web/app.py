"""
Flask Web Application for Pokemon RL Bot Dashboard

This module provides a web interface for monitoring and controlling the Pokemon RL bot.
Features include real-time game streaming, training metrics, and manual controls.
"""

import os
import json
import base64
from datetime import datetime
from typing import Dict, Any, Optional

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np

from game.environment import PokemonEnvironment
from agent.ppo_agent import PPOAgent
from utils.logger import get_logger
from utils.metrics import MetricsTracker

logger = get_logger(__name__)

class PokemonWebApp:
    """Flask application for Pokemon RL bot dashboard."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the web application.
        
        Args:
            config: Configuration dictionary containing app settings
        """
        self.config = config
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = config.get('secret_key', 'pokemon-rl-secret')
        
        # Initialize SocketIO
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            ping_timeout=60,
            ping_interval=25
        )
        
        # Initialize components
        self.environment: Optional[PokemonEnvironment] = None
        self.agent: Optional[PPOAgent] = None
        self.metrics_tracker = MetricsTracker()
        
        # Training state
        self.training_active = False
        self.current_episode = 0
        self.total_reward = 0
        
        # Setup routes and socket handlers
        self._setup_routes()
        self._setup_socket_handlers()
        
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return render_template('index.html')
        
        @self.app.route('/api/status')
        def status():
            """Get current application status."""
            return jsonify({
                'training_active': self.training_active,
                'episode': self.current_episode,
                'total_reward': self.total_reward,
                'environment_loaded': self.environment is not None,
                'agent_loaded': self.agent is not None
            })
        
        @self.app.route('/api/config')
        def get_config():
            """Get current configuration."""
            return jsonify({
                'rom_path': self.config.get('rom_path', ''),
                'save_path': self.config.get('save_path', ''),
                'headless': self.config.get('headless', True),
                'frame_skip': self.config.get('frame_skip', 4)
            })
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """Get training metrics."""
            metrics = self.metrics_tracker.get_summary()
            return jsonify(metrics)
        
        @self.app.route('/api/start_training', methods=['POST'])
        def start_training():
            """Start training session."""
            try:
                if not self.environment or not self.agent:
                    return jsonify({'error': 'Environment or agent not loaded'}), 400
                
                self.training_active = True
                logger.info("Training started via web interface")
                return jsonify({'message': 'Training started'})
            
            except Exception as e:
                logger.error(f"Error starting training: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/stop_training', methods=['POST'])
        def stop_training():
            """Stop training session."""
            self.training_active = False
            logger.info("Training stopped via web interface")
            return jsonify({'message': 'Training stopped'})
        
        @self.app.route('/api/reset_environment', methods=['POST'])
        def reset_environment():
            """Reset the game environment."""
            try:
                if self.environment:
                    observation = self.environment.reset()
                    self.total_reward = 0
                    logger.info("Environment reset via web interface")
                    return jsonify({'message': 'Environment reset'})
                else:
                    return jsonify({'error': 'Environment not loaded'}), 400
            
            except Exception as e:
                logger.error(f"Error resetting environment: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/manual_action', methods=['POST'])
        def manual_action():
            """Execute manual action in the environment."""
            try:
                data = request.get_json()
                action = data.get('action')
                
                if not self.environment:
                    return jsonify({'error': 'Environment not loaded'}), 400
                
                if action is None or not (0 <= action <= 8):
                    return jsonify({'error': 'Invalid action'}), 400
                
                observation, reward, done, info = self.environment.step(action)
                self.total_reward += reward
                
                if done:
                    observation = self.environment.reset()
                    self.current_episode += 1
                    self.total_reward = 0
                
                return jsonify({
                    'reward': reward,
                    'done': done,
                    'total_reward': self.total_reward,
                    'info': info
                })
            
            except Exception as e:
                logger.error(f"Error executing manual action: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _setup_socket_handlers(self):
        """Setup Socket.IO event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            logger.info(f"Client connected: {request.sid}")
            emit('status', {
                'training_active': self.training_active,
                'episode': self.current_episode,
                'total_reward': self.total_reward
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('request_frame')
        def handle_frame_request():
            """Send current game frame to client."""
            if self.environment and hasattr(self.environment, 'emulator'):
                try:
                    frame = self.environment.emulator.get_screen()
                    if frame is not None:
                        # Convert frame to base64 for transmission
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_b64 = base64.b64encode(buffer).decode('utf-8')
                        emit('frame', {'data': frame_b64})
                except Exception as e:
                    logger.error(f"Error getting frame: {e}")
        
        @self.socketio.on('action')
        def handle_action(data):
            """Handle action from client."""
            action = data.get('action')
            if self.environment and action is not None:
                try:
                    observation, reward, done, info = self.environment.step(action)
                    self.total_reward += reward
                    
                    emit('action_result', {
                        'reward': reward,
                        'done': done,
                        'total_reward': self.total_reward,
                        'info': info
                    })
                    
                    if done:
                        self.environment.reset()
                        self.current_episode += 1
                        self.total_reward = 0
                        emit('episode_complete', {
                            'episode': self.current_episode,
                            'final_reward': self.total_reward
                        })
                
                except Exception as e:
                    logger.error(f"Error handling action: {e}")
                    emit('error', {'message': str(e)})
    
    def load_environment(self, rom_path: str, save_path: str = None):
        """Load the Pokemon environment.
        
        Args:
            rom_path: Path to the Pokemon ROM file
            save_path: Path to save state file (optional)
        """
        try:
            config = {
                'rom_path': rom_path,
                'save_path': save_path,
                'headless': self.config.get('headless', False),  # Web mode should show display
                'frame_skip': self.config.get('frame_skip', 4)
            }
            
            self.environment = PokemonEnvironment(config)
            logger.info(f"Environment loaded with ROM: {rom_path}")
            
        except Exception as e:
            logger.error(f"Error loading environment: {e}")
            raise
    
    def load_agent(self, model_path: str = None):
        """Load the PPO agent.
        
        Args:
            model_path: Path to saved model (optional)
        """
        try:
            if not self.environment:
                raise ValueError("Environment must be loaded before agent")
            
            self.agent = PPOAgent(
                env=self.environment,
                model_path=model_path,
                tensorboard_log=self.config.get('tensorboard_log', './logs/tensorboard')
            )
            
            if model_path and os.path.exists(model_path):
                self.agent.load(model_path)
                logger.info(f"Agent loaded from: {model_path}")
            else:
                logger.info("New agent created")
                
        except Exception as e:
            logger.error(f"Error loading agent: {e}")
            raise
    
    def broadcast_metrics(self, metrics: Dict[str, Any]):
        """Broadcast training metrics to all connected clients.
        
        Args:
            metrics: Training metrics dictionary
        """
        self.socketio.emit('metrics', metrics)
    
    def broadcast_frame(self, frame: np.ndarray):
        """Broadcast game frame to all connected clients.
        
        Args:
            frame: Game frame as numpy array
        """
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            self.socketio.emit('frame', {'data': frame_b64})
        except Exception as e:
            logger.error(f"Error broadcasting frame: {e}")
    
    def update_training_status(self, episode: int, reward: float, done: bool = False):
        """Update training status and broadcast to clients.
        
        Args:
            episode: Current episode number
            reward: Current total reward
            done: Whether episode is complete
        """
        self.current_episode = episode
        self.total_reward = reward
        
        status_data = {
            'episode': episode,
            'total_reward': reward,
            'done': done,
            'timestamp': datetime.now().isoformat()
        }
        
        self.socketio.emit('training_status', status_data)
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Run the Flask application.
        
        Args:
            host: Host address to bind to
            port: Port number to listen on
            debug: Enable debug mode
        """
        logger.info(f"Starting Pokemon RL Web Dashboard on {host}:{port}")
        self.socketio.run(
            self.app,
            host=host,
            port=port,
            debug=debug,
            use_reloader=False  # Disable reloader in production
        )


def create_app(config: Dict[str, Any]) -> PokemonWebApp:
    """Create and configure the Flask application.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured PokemonWebApp instance
    """
    app = PokemonWebApp(config)
    return app


if __name__ == '__main__':
    # Default configuration for standalone execution
    default_config = {
        'secret_key': 'pokemon-rl-development-key',
        'headless': False,
        'frame_skip': 4,
        'tensorboard_log': './logs/tensorboard'
    }
    
    app = create_app(default_config)
    app.run(debug=True)
