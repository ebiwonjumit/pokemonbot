"""
Metrics tracking and analysis utilities for Pokemon RL Bot.
Provides comprehensive metrics collection, analysis, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class EpisodeMetrics:
    """Metrics for a single training episode."""
    episode: int
    total_reward: float
    episode_length: int
    badges_earned: int
    pokemon_caught: int
    areas_explored: int
    battles_won: int
    battles_lost: int
    damage_taken: float
    money_earned: int
    time_in_menu: int
    time_in_battle: int
    exploration_efficiency: float
    
    # Training metrics
    actor_loss: Optional[float] = None
    critic_loss: Optional[float] = None
    policy_entropy: Optional[float] = None
    value_estimate: Optional[float] = None
    
    # Performance metrics
    fps: Optional[float] = None
    memory_usage: Optional[float] = None
    gpu_usage: Optional[float] = None
    
    # Timestamps
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Episode duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def win_rate(self) -> float:
        """Battle win rate."""
        total_battles = self.battles_won + self.battles_lost
        return self.battles_won / total_battles if total_battles > 0 else 0.0


@dataclass
class TrainingMetrics:
    """Aggregated training metrics across episodes."""
    total_episodes: int
    total_timesteps: int
    training_time: timedelta
    
    # Performance statistics
    avg_reward: float
    max_reward: float
    min_reward: float
    reward_std: float
    
    # Progress statistics
    total_badges: int
    total_pokemon: int
    total_areas: int
    avg_episode_length: float
    
    # Learning statistics
    avg_actor_loss: Optional[float] = None
    avg_critic_loss: Optional[float] = None
    avg_policy_entropy: Optional[float] = None
    convergence_episode: Optional[int] = None
    
    # System performance
    avg_fps: Optional[float] = None
    peak_memory: Optional[float] = None
    avg_gpu_usage: Optional[float] = None


class MetricsCollector:
    """Collects and manages training metrics."""
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize metrics collector.
        
        Args:
            save_dir: Directory to save metrics data
        """
        self.save_dir = Path(save_dir) if save_dir else Path("logs/metrics")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.episodes: List[EpisodeMetrics] = []
        self.current_episode: Optional[EpisodeMetrics] = None
        
        # Real-time metrics
        self.recent_rewards = deque(maxlen=100)
        self.recent_lengths = deque(maxlen=100)
        self.recent_fps = deque(maxlen=50)
        
        # Training session info
        self.session_start = datetime.now()
        self.total_timesteps = 0
        
        logger.info(f"Metrics collector initialized, saving to {self.save_dir}")
    
    def start_episode(self, episode: int) -> None:
        """Start tracking a new episode."""
        self.current_episode = EpisodeMetrics(
            episode=episode,
            total_reward=0.0,
            episode_length=0,
            badges_earned=0,
            pokemon_caught=0,
            areas_explored=0,
            battles_won=0,
            battles_lost=0,
            damage_taken=0.0,
            money_earned=0,
            time_in_menu=0,
            time_in_battle=0,
            exploration_efficiency=0.0,
            start_time=datetime.now()
        )
    
    def update_episode(self, metrics: Dict[str, Any]) -> None:
        """Update current episode metrics."""
        if not self.current_episode:
            return
        
        # Update episode metrics from dictionary
        for key, value in metrics.items():
            if hasattr(self.current_episode, key):
                setattr(self.current_episode, key, value)
    
    def end_episode(
        self,
        total_reward: float,
        episode_length: int,
        game_stats: Optional[Dict[str, Any]] = None,
        training_stats: Optional[Dict[str, Any]] = None,
        performance_stats: Optional[Dict[str, Any]] = None
    ) -> EpisodeMetrics:
        """
        End current episode and save metrics.
        
        Args:
            total_reward: Total episode reward
            episode_length: Number of steps in episode
            game_stats: Game-specific statistics
            training_stats: Training algorithm statistics
            performance_stats: System performance statistics
            
        Returns:
            EpisodeMetrics: Completed episode metrics
        """
        if not self.current_episode:
            raise ValueError("No active episode to end")
        
        # Update basic metrics
        self.current_episode.total_reward = total_reward
        self.current_episode.episode_length = episode_length
        self.current_episode.end_time = datetime.now()
        
        # Update game statistics
        if game_stats:
            self.current_episode.badges_earned = game_stats.get('badges_earned', 0)
            self.current_episode.pokemon_caught = game_stats.get('pokemon_caught', 0)
            self.current_episode.areas_explored = game_stats.get('areas_explored', 0)
            self.current_episode.battles_won = game_stats.get('battles_won', 0)
            self.current_episode.battles_lost = game_stats.get('battles_lost', 0)
            self.current_episode.damage_taken = game_stats.get('damage_taken', 0.0)
            self.current_episode.money_earned = game_stats.get('money_earned', 0)
            self.current_episode.time_in_menu = game_stats.get('time_in_menu', 0)
            self.current_episode.time_in_battle = game_stats.get('time_in_battle', 0)
            
            # Calculate exploration efficiency
            if episode_length > 0:
                self.current_episode.exploration_efficiency = (
                    self.current_episode.areas_explored / episode_length * 1000
                )
        
        # Update training statistics
        if training_stats:
            self.current_episode.actor_loss = training_stats.get('actor_loss')
            self.current_episode.critic_loss = training_stats.get('critic_loss')
            self.current_episode.policy_entropy = training_stats.get('policy_entropy')
            self.current_episode.value_estimate = training_stats.get('value_estimate')
        
        # Update performance statistics
        if performance_stats:
            self.current_episode.fps = performance_stats.get('fps')
            self.current_episode.memory_usage = performance_stats.get('memory_usage')
            self.current_episode.gpu_usage = performance_stats.get('gpu_usage')
        
        # Store episode
        self.episodes.append(self.current_episode)
        
        # Update real-time metrics
        self.recent_rewards.append(total_reward)
        self.recent_lengths.append(episode_length)
        if self.current_episode.fps:
            self.recent_fps.append(self.current_episode.fps)
        
        # Update total timesteps
        self.total_timesteps += episode_length
        
        # Save episode data
        self._save_episode(self.current_episode)
        
        completed_episode = self.current_episode
        self.current_episode = None
        
        return completed_episode
    
    def get_recent_stats(self, window: int = 100) -> Dict[str, float]:
        """Get statistics for recent episodes."""
        if not self.episodes:
            return {}
        
        recent_episodes = self.episodes[-window:]
        
        rewards = [ep.total_reward for ep in recent_episodes]
        lengths = [ep.episode_length for ep in recent_episodes]
        badges = [ep.badges_earned for ep in recent_episodes]
        pokemon = [ep.pokemon_caught for ep in recent_episodes]
        
        stats = {
            'avg_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'std_reward': np.std(rewards),
            'avg_length': np.mean(lengths),
            'total_badges': sum(badges),
            'total_pokemon': sum(pokemon),
            'avg_badges_per_episode': np.mean(badges),
            'avg_pokemon_per_episode': np.mean(pokemon),
        }
        
        # Add performance metrics if available
        fps_values = [ep.fps for ep in recent_episodes if ep.fps is not None]
        if fps_values:
            stats['avg_fps'] = np.mean(fps_values)
        
        memory_values = [ep.memory_usage for ep in recent_episodes if ep.memory_usage is not None]
        if memory_values:
            stats['avg_memory'] = np.mean(memory_values)
        
        return stats
    
    def get_training_summary(self) -> TrainingMetrics:
        """Get complete training summary."""
        if not self.episodes:
            return TrainingMetrics(
                total_episodes=0,
                total_timesteps=0,
                training_time=timedelta(0),
                avg_reward=0.0,
                max_reward=0.0,
                min_reward=0.0,
                reward_std=0.0,
                total_badges=0,
                total_pokemon=0,
                total_areas=0,
                avg_episode_length=0.0
            )
        
        rewards = [ep.total_reward for ep in self.episodes]
        lengths = [ep.episode_length for ep in self.episodes]
        
        # Calculate training time
        training_time = datetime.now() - self.session_start
        
        # Aggregate statistics
        summary = TrainingMetrics(
            total_episodes=len(self.episodes),
            total_timesteps=self.total_timesteps,
            training_time=training_time,
            avg_reward=np.mean(rewards),
            max_reward=np.max(rewards),
            min_reward=np.min(rewards),
            reward_std=np.std(rewards),
            total_badges=sum(ep.badges_earned for ep in self.episodes),
            total_pokemon=sum(ep.pokemon_caught for ep in self.episodes),
            total_areas=len(set(ep.areas_explored for ep in self.episodes)),
            avg_episode_length=np.mean(lengths)
        )
        
        # Training-specific metrics
        actor_losses = [ep.actor_loss for ep in self.episodes if ep.actor_loss is not None]
        if actor_losses:
            summary.avg_actor_loss = np.mean(actor_losses)
        
        critic_losses = [ep.critic_loss for ep in self.episodes if ep.critic_loss is not None]
        if critic_losses:
            summary.avg_critic_loss = np.mean(critic_losses)
        
        entropies = [ep.policy_entropy for ep in self.episodes if ep.policy_entropy is not None]
        if entropies:
            summary.avg_policy_entropy = np.mean(entropies)
        
        # Performance metrics
        fps_values = [ep.fps for ep in self.episodes if ep.fps is not None]
        if fps_values:
            summary.avg_fps = np.mean(fps_values)
        
        memory_values = [ep.memory_usage for ep in self.episodes if ep.memory_usage is not None]
        if memory_values:
            summary.peak_memory = np.max(memory_values)
        
        gpu_values = [ep.gpu_usage for ep in self.episodes if ep.gpu_usage is not None]
        if gpu_values:
            summary.avg_gpu_usage = np.mean(gpu_values)
        
        return summary
    
    def save_metrics(self, filename: Optional[str] = None) -> Path:
        """Save all metrics to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_metrics_{timestamp}.json"
        
        filepath = self.save_dir / filename
        
        # Convert episodes to dictionaries
        episodes_data = [asdict(ep) for ep in self.episodes]
        
        # Convert datetime objects to strings
        for episode_data in episodes_data:
            if episode_data['start_time']:
                episode_data['start_time'] = episode_data['start_time'].isoformat()
            if episode_data['end_time']:
                episode_data['end_time'] = episode_data['end_time'].isoformat()
        
        # Create complete data structure
        metrics_data = {
            'session_info': {
                'start_time': self.session_start.isoformat(),
                'total_episodes': len(self.episodes),
                'total_timesteps': self.total_timesteps
            },
            'episodes': episodes_data,
            'summary': asdict(self.get_training_summary())
        }
        
        # Convert timedelta to string
        summary = metrics_data['summary']
        if 'training_time' in summary and summary['training_time']:
            summary['training_time'] = str(summary['training_time'])
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        logger.info(f"Metrics saved to {filepath}")
        return filepath
    
    def load_metrics(self, filepath: str) -> None:
        """Load metrics from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load episodes
        self.episodes = []
        for episode_data in data['episodes']:
            # Convert datetime strings back to objects
            if episode_data['start_time']:
                episode_data['start_time'] = datetime.fromisoformat(episode_data['start_time'])
            if episode_data['end_time']:
                episode_data['end_time'] = datetime.fromisoformat(episode_data['end_time'])
            
            self.episodes.append(EpisodeMetrics(**episode_data))
        
        # Update counters
        if 'session_info' in data:
            self.total_timesteps = data['session_info']['total_timesteps']
        
        logger.info(f"Loaded {len(self.episodes)} episodes from {filepath}")
    
    def _save_episode(self, episode: EpisodeMetrics) -> None:
        """Save individual episode data."""
        # Save to CSV for easy analysis
        csv_path = self.save_dir / "episodes.csv"
        
        # Convert episode to dictionary and handle datetime
        episode_dict = asdict(episode)
        if episode_dict['start_time']:
            episode_dict['start_time'] = episode_dict['start_time'].isoformat()
        if episode_dict['end_time']:
            episode_dict['end_time'] = episode_dict['end_time'].isoformat()
        
        # Create DataFrame
        df = pd.DataFrame([episode_dict])
        
        # Append to CSV (create if doesn't exist)
        if csv_path.exists():
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)


class MetricsVisualizer:
    """Creates visualizations for training metrics."""
    
    def __init__(self, save_dir: Optional[str] = None):
        """Initialize metrics visualizer."""
        self.save_dir = Path(save_dir) if save_dir else Path("logs/plots")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_progress(
        self,
        episodes: List[EpisodeMetrics],
        save_path: Optional[str] = None
    ) -> str:
        """Plot training progress overview."""
        if not episodes:
            return ""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Progress Overview', fontsize=16)
        
        episode_numbers = [ep.episode for ep in episodes]
        rewards = [ep.total_reward for ep in episodes]
        lengths = [ep.episode_length for ep in episodes]
        badges = [ep.badges_earned for ep in episodes]
        pokemon = [ep.pokemon_caught for ep in episodes]
        
        # Reward progress
        axes[0, 0].plot(episode_numbers, rewards, alpha=0.6)
        axes[0, 0].plot(episode_numbers, pd.Series(rewards).rolling(20).mean(), 'r-', linewidth=2)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        
        # Episode length
        axes[0, 1].plot(episode_numbers, lengths, alpha=0.6)
        axes[0, 1].plot(episode_numbers, pd.Series(lengths).rolling(20).mean(), 'g-', linewidth=2)
        axes[0, 1].set_title('Episode Length')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)
        
        # Badges earned
        cumulative_badges = np.cumsum(badges)
        axes[0, 2].plot(episode_numbers, cumulative_badges, 'b-', linewidth=2)
        axes[0, 2].set_title('Cumulative Badges Earned')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Total Badges')
        axes[0, 2].grid(True)
        
        # Pokemon caught
        cumulative_pokemon = np.cumsum(pokemon)
        axes[1, 0].plot(episode_numbers, cumulative_pokemon, 'orange', linewidth=2)
        axes[1, 0].set_title('Cumulative Pokemon Caught')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Total Pokemon')
        axes[1, 0].grid(True)
        
        # Reward distribution
        axes[1, 1].hist(rewards, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)
        
        # Performance metrics
        fps_values = [ep.fps for ep in episodes if ep.fps is not None]
        if fps_values:
            fps_episodes = [ep.episode for ep in episodes if ep.fps is not None]
            axes[1, 2].plot(fps_episodes, fps_values, 'purple', alpha=0.6)
            axes[1, 2].set_title('FPS Performance')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('FPS')
            axes[1, 2].grid(True)
        else:
            axes[1, 2].text(0.5, 0.5, 'No FPS data', ha='center', va='center', transform=axes[1, 2].transAxes)
        
        plt.tight_layout()
        
        if not save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.save_dir / f"training_progress_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_learning_curves(
        self,
        episodes: List[EpisodeMetrics],
        save_path: Optional[str] = None
    ) -> str:
        """Plot learning algorithm specific curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Learning Curves', fontsize=16)
        
        # Filter episodes with training metrics
        training_episodes = [ep for ep in episodes if ep.actor_loss is not None]
        
        if not training_episodes:
            fig.text(0.5, 0.5, 'No training metrics available', ha='center', va='center', fontsize=14)
        else:
            episode_numbers = [ep.episode for ep in training_episodes]
            actor_losses = [ep.actor_loss for ep in training_episodes]
            critic_losses = [ep.critic_loss for ep in training_episodes if ep.critic_loss is not None]
            entropies = [ep.policy_entropy for ep in training_episodes if ep.policy_entropy is not None]
            values = [ep.value_estimate for ep in training_episodes if ep.value_estimate is not None]
            
            # Actor loss
            axes[0, 0].plot(episode_numbers, actor_losses, 'r-', alpha=0.6)
            axes[0, 0].plot(episode_numbers, pd.Series(actor_losses).rolling(10).mean(), 'r-', linewidth=2)
            axes[0, 0].set_title('Actor Loss')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
            
            # Critic loss
            if critic_losses:
                critic_episodes = [ep.episode for ep in training_episodes if ep.critic_loss is not None]
                axes[0, 1].plot(critic_episodes, critic_losses, 'b-', alpha=0.6)
                axes[0, 1].plot(critic_episodes, pd.Series(critic_losses).rolling(10).mean(), 'b-', linewidth=2)
                axes[0, 1].set_title('Critic Loss')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].grid(True)
            
            # Policy entropy
            if entropies:
                entropy_episodes = [ep.episode for ep in training_episodes if ep.policy_entropy is not None]
                axes[1, 0].plot(entropy_episodes, entropies, 'g-', alpha=0.6)
                axes[1, 0].plot(entropy_episodes, pd.Series(entropies).rolling(10).mean(), 'g-', linewidth=2)
                axes[1, 0].set_title('Policy Entropy')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Entropy')
                axes[1, 0].grid(True)
            
            # Value estimates
            if values:
                value_episodes = [ep.episode for ep in training_episodes if ep.value_estimate is not None]
                axes[1, 1].plot(value_episodes, values, 'orange', alpha=0.6)
                axes[1, 1].plot(value_episodes, pd.Series(values).rolling(10).mean(), 'orange', linewidth=2)
                axes[1, 1].set_title('Value Estimates')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Value')
                axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if not save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.save_dir / f"learning_curves_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_dashboard(self, episodes: List[EpisodeMetrics]) -> str:
        """Create comprehensive training dashboard."""
        if not episodes:
            return ""
        
        # Create plots
        progress_plot = self.plot_training_progress(episodes)
        learning_plot = self.plot_learning_curves(episodes)
        
        # Create HTML dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_path = self.save_dir / f"dashboard_{timestamp}.html"
        
        # Calculate statistics
        recent_stats = self._calculate_recent_stats(episodes[-100:] if len(episodes) > 100 else episodes)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pokemon RL Training Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
                .stat-card {{ background: #f5f5f5; padding: 15px; border-radius: 8px; text-align: center; }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .stat-label {{ font-size: 14px; color: #7f8c8d; }}
                .plot-container {{ margin: 20px 0; text-align: center; }}
                .plot-container img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Pokemon RL Training Dashboard</h1>
                <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{len(episodes)}</div>
                    <div class="stat-label">Total Episodes</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{recent_stats['avg_reward']:.1f}</div>
                    <div class="stat-label">Avg Reward (Recent)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{recent_stats['total_badges']}</div>
                    <div class="stat-label">Total Badges</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{recent_stats['total_pokemon']}</div>
                    <div class="stat-label">Total Pokemon</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{recent_stats['avg_length']:.0f}</div>
                    <div class="stat-label">Avg Episode Length</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{recent_stats.get('avg_fps', 0):.1f}</div>
                    <div class="stat-label">Avg FPS</div>
                </div>
            </div>
            
            <div class="plot-container">
                <h2>Training Progress</h2>
                <img src="{Path(progress_plot).name}" alt="Training Progress">
            </div>
            
            <div class="plot-container">
                <h2>Learning Curves</h2>
                <img src="{Path(learning_plot).name}" alt="Learning Curves">
            </div>
        </body>
        </html>
        """
        
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        return str(dashboard_path)
    
    def _calculate_recent_stats(self, episodes: List[EpisodeMetrics]) -> Dict[str, float]:
        """Calculate statistics for recent episodes."""
        if not episodes:
            return {}
        
        rewards = [ep.total_reward for ep in episodes]
        lengths = [ep.episode_length for ep in episodes]
        badges = [ep.badges_earned for ep in episodes]
        pokemon = [ep.pokemon_caught for ep in episodes]
        fps_values = [ep.fps for ep in episodes if ep.fps is not None]
        
        return {
            'avg_reward': np.mean(rewards),
            'avg_length': np.mean(lengths),
            'total_badges': sum(badges),
            'total_pokemon': sum(pokemon),
            'avg_fps': np.mean(fps_values) if fps_values else 0.0
        }


if __name__ == "__main__":
    # Test metrics system
    collector = MetricsCollector("test_metrics")
    visualizer = MetricsVisualizer("test_plots")
    
    # Simulate training episodes
    for i in range(50):
        collector.start_episode(i)
        
        # Simulate episode data
        reward = np.random.normal(100 + i, 20)
        length = np.random.randint(500, 2000)
        
        game_stats = {
            'badges_earned': np.random.randint(0, 2),
            'pokemon_caught': np.random.randint(0, 3),
            'areas_explored': np.random.randint(1, 5),
            'battles_won': np.random.randint(0, 5),
            'battles_lost': np.random.randint(0, 2),
        }
        
        training_stats = {
            'actor_loss': np.random.uniform(0.01, 0.1),
            'critic_loss': np.random.uniform(0.01, 0.1),
            'policy_entropy': np.random.uniform(0.5, 2.0),
        }
        
        performance_stats = {
            'fps': np.random.uniform(25, 35),
            'memory_usage': np.random.uniform(1000, 2000),
        }
        
        collector.end_episode(reward, length, game_stats, training_stats, performance_stats)
    
    # Generate visualizations
    progress_plot = visualizer.plot_training_progress(collector.episodes)
    learning_plot = visualizer.plot_learning_curves(collector.episodes)
    dashboard = visualizer.create_dashboard(collector.episodes)
    
    # Save metrics
    collector.save_metrics()
    
    print(f"Generated plots: {progress_plot}, {learning_plot}")
    print(f"Dashboard: {dashboard}")
    print("Metrics testing completed!")
