"""
PPO (Proximal Policy Optimization) agent for Pokemon Leaf Green.
Implements custom PPO with Pokemon-specific optimizations.
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from pathlib import Path
import pickle

try:
    import torch
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.evaluation import evaluate_policy
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .neural_network import PokemonActorCriticPolicy, create_pokemon_cnn
from ..game.environment import PokemonEnv
from ..utils.logger import get_logger
from ..utils.metrics import MetricsCollector, EpisodeMetrics

logger = get_logger(__name__)


class PokemonPPOCallback(BaseCallback):
    """Custom callback for Pokemon PPO training with enhanced monitoring."""
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        save_freq: int = 10000,
        save_path: str = "models/",
        verbose: int = 1
    ):
        """
        Initialize Pokemon PPO callback.
        
        Args:
            metrics_collector: Metrics collector for tracking training
            save_freq: Frequency of model saving (timesteps)
            save_path: Directory to save models
            verbose: Verbosity level
        """
        super(PokemonPPOCallback, self).__init__(verbose)
        
        self.metrics_collector = metrics_collector
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Training state tracking
        self.episode_count = 0
        self.best_mean_reward = -np.inf
        self.last_save_timestep = 0
        
        # Episode data collection
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_infos = []
    
    def _on_training_start(self) -> None:
        """Called at the beginning of training."""
        logger.info("Pokemon PPO training started")
    
    def _on_rollout_start(self) -> None:
        """Called at the beginning of each rollout."""
        pass
    
    def _on_step(self) -> bool:
        """
        Called after each environment step.
        
        Returns:
            bool: True to continue training, False to stop
        """
        # Check if episode ended
        if len(self.locals.get('dones', [])) > 0 and any(self.locals['dones']):
            # Collect episode data
            infos = self.locals.get('infos', [])
            
            for i, done in enumerate(self.locals['dones']):
                if done and i < len(infos):
                    info = infos[i]
                    
                    # Extract episode data
                    episode_reward = info.get('episode', {}).get('r', 0)
                    episode_length = info.get('episode', {}).get('l', 0)
                    
                    if episode_reward != 0:  # Valid episode
                        self.episode_rewards.append(episode_reward)
                        self.episode_lengths.append(episode_length)
                        self.episode_infos.append(info)
                        self.episode_count += 1
                        
                        # Log episode completion
                        if self.verbose >= 1:
                            logger.info(f"Episode {self.episode_count} completed: "
                                      f"reward={episode_reward:.2f}, length={episode_length}")
                        
                        # Update metrics collector
                        if self.metrics_collector and hasattr(self.metrics_collector, 'current_episode'):
                            game_stats = info.get('game_stats', {})
                            
                            # Get training statistics if available
                            training_stats = {}
                            if hasattr(self.model, 'logger') and self.model.logger:
                                try:
                                    training_stats = {
                                        'actor_loss': self.model.logger.name_to_value.get('train/policy_loss', None),
                                        'critic_loss': self.model.logger.name_to_value.get('train/value_loss', None),
                                        'policy_entropy': self.model.logger.name_to_value.get('train/entropy_loss', None),
                                    }
                                except:
                                    pass
                            
                            # End episode in metrics collector
                            try:
                                self.metrics_collector.end_episode(
                                    episode_reward,
                                    episode_length,
                                    game_stats,
                                    training_stats
                                )
                            except:
                                pass
        
        # Save model periodically
        if self.num_timesteps - self.last_save_timestep >= self.save_freq:
            self._save_model()
            self.last_save_timestep = self.num_timesteps
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        # Calculate recent performance
        if len(self.episode_rewards) >= 10:
            recent_rewards = self.episode_rewards[-10:]
            mean_reward = np.mean(recent_rewards)
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self._save_model(suffix="best")
                logger.info(f"New best model saved! Mean reward: {mean_reward:.2f}")
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        logger.info(f"Pokemon PPO training completed after {self.num_timesteps} timesteps")
        
        # Final model save
        self._save_model(suffix="final")
        
        # Training summary
        if self.episode_rewards:
            logger.info(f"Training summary:")
            logger.info(f"  Total episodes: {len(self.episode_rewards)}")
            logger.info(f"  Mean reward: {np.mean(self.episode_rewards):.2f}")
            logger.info(f"  Best mean reward: {self.best_mean_reward:.2f}")
            logger.info(f"  Mean episode length: {np.mean(self.episode_lengths):.2f}")
    
    def _save_model(self, suffix: str = "") -> None:
        """Save model checkpoint."""
        try:
            if suffix:
                filename = f"pokemon_ppo_{suffix}_{self.num_timesteps}.zip"
            else:
                filename = f"pokemon_ppo_checkpoint_{self.num_timesteps}.zip"
            
            filepath = self.save_path / filename
            self.model.save(filepath)
            
            if self.verbose >= 1:
                logger.info(f"Model saved: {filepath}")
                
        except Exception as e:
            logger.error(f"Failed to save model: {e}")


class PPOAgent:
    """
    Enhanced PPO agent for Pokemon Leaf Green with custom optimizations.
    """
    
    def __init__(
        self,
        env: Union[PokemonEnv, VecEnv],
        policy: str = "PokemonCNN",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        device: str = "auto",
        verbose: int = 1,
        seed: Optional[int] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize PPO agent for Pokemon.
        
        Args:
            env: Pokemon environment or vectorized environment
            policy: Policy architecture ("PokemonCNN", "MlpPolicy", etc.)
            learning_rate: Learning rate for optimizer
            n_steps: Number of steps per rollout
            batch_size: Batch size for training
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm for clipping
            use_sde: Whether to use State Dependent Exploration
            sde_sample_freq: SDE sampling frequency
            target_kl: Target KL divergence threshold
            device: Device to use ("auto", "cpu", "cuda")
            verbose: Verbosity level
            seed: Random seed
            tensorboard_log: Directory for tensorboard logs
            policy_kwargs: Additional policy arguments
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and Stable-Baselines3 required for PPO agent")
        
        self.env = env
        self.verbose = verbose
        
        # Set up policy kwargs
        if policy_kwargs is None:
            policy_kwargs = {}
        
        # Use custom Pokemon policy if specified
        if policy == "PokemonCNN":
            policy = PokemonActorCriticPolicy
            policy_kwargs.update({
                "features_extractor_class": create_pokemon_cnn,
                "features_extractor_kwargs": {"architecture": "standard"},
                "net_arch": dict(pi=[256, 256], vf=[256, 256]),
            })
        
        # Initialize PPO model
        self.model = PPO(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            device=device,
            verbose=verbose,
            seed=seed,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            **kwargs
        )
        
        # Training state
        self.is_training = False
        self.training_start_time = None
        self.total_timesteps = 0
        
        logger.info(f"PPO agent initialized with policy: {policy}")
        logger.info(f"Device: {self.model.device}")
    
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Union[BaseCallback, List[BaseCallback]]] = None,
        log_interval: int = 100,
        eval_env: Optional[Union[PokemonEnv, VecEnv]] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        tb_log_name: str = "pokemon_ppo",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        **kwargs
    ) -> "PPOAgent":
        """
        Train the PPO agent.
        
        Args:
            total_timesteps: Total training timesteps
            callback: Training callbacks
            log_interval: Logging interval
            eval_env: Evaluation environment
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            tb_log_name: Tensorboard log name
            reset_num_timesteps: Reset timestep counter
            progress_bar: Show progress bar
            
        Returns:
            PPOAgent: Self for chaining
        """
        self.is_training = True
        self.training_start_time = time.time()
        
        try:
            # Set up evaluation if provided
            if eval_env is not None and eval_freq > 0:
                from stable_baselines3.common.callbacks import EvalCallback
                
                eval_callback = EvalCallback(
                    eval_env,
                    best_model_save_path="models/best_model",
                    log_path="logs/eval",
                    eval_freq=eval_freq,
                    n_eval_episodes=n_eval_episodes,
                    deterministic=True,
                    render=False
                )
                
                # Combine with existing callbacks
                if callback is None:
                    callback = eval_callback
                elif isinstance(callback, list):
                    callback.append(eval_callback)
                else:
                    callback = [callback, eval_callback]
            
            # Start training
            logger.info(f"Starting PPO training for {total_timesteps} timesteps")
            
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                log_interval=log_interval,
                tb_log_name=tb_log_name,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=progress_bar,
                **kwargs
            )
            
            self.total_timesteps += total_timesteps
            
            training_time = time.time() - self.training_start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            self.is_training = False
        
        return self
    
    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple] = None,
        episode_start: Optional[bool] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[Tuple]]:
        """
        Predict action for given observation.
        
        Args:
            observation: Environment observation
            state: RNN state (if applicable)
            episode_start: Whether this is the start of episode
            deterministic: Use deterministic policy
            
        Returns:
            Tuple of (action, state)
        """
        try:
            action, state = self.model.predict(
                observation,
                state=state,
                episode_start=episode_start,
                deterministic=deterministic
            )
            return action, state
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return random action as fallback
            return self.env.action_space.sample(), state
    
    def save(self, path: str) -> None:
        """Save the agent model."""
        try:
            self.model.save(path)
            logger.info(f"Agent model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load(self, path: str, env: Optional[Union[PokemonEnv, VecEnv]] = None) -> None:
        """
        Load agent model from file.
        
        Args:
            path: Path to model file
            env: Environment (optional, uses current if not provided)
        """
        try:
            if env is not None:
                self.env = env
            
            self.model = PPO.load(path, env=self.env)
            logger.info(f"Agent model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def evaluate(
        self,
        env: Optional[Union[PokemonEnv, VecEnv]] = None,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        return_episode_rewards: bool = False
    ) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
        """
        Evaluate the agent performance.
        
        Args:
            env: Evaluation environment
            n_eval_episodes: Number of episodes to evaluate
            deterministic: Use deterministic policy
            render: Render during evaluation
            return_episode_rewards: Return individual episode rewards
            
        Returns:
            Mean reward and std, or episode rewards and lengths
        """
        if env is None:
            env = self.env
        
        try:
            result = evaluate_policy(
                self.model,
                env,
                n_eval_episodes=n_eval_episodes,
                deterministic=deterministic,
                render=render,
                return_episode_rewards=return_episode_rewards
            )
            
            if return_episode_rewards:
                episode_rewards, episode_lengths = result
                logger.info(f"Evaluation completed: {n_eval_episodes} episodes")
                logger.info(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
                return episode_rewards, episode_lengths
            else:
                mean_reward, std_reward = result
                logger.info(f"Evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
                return mean_reward, std_reward
                
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = {
            'total_timesteps': self.total_timesteps,
            'is_training': self.is_training,
            'device': str(self.model.device),
            'policy_class': str(type(self.model.policy)),
        }
        
        # Add model parameters count
        if hasattr(self.model.policy, 'parameters'):
            total_params = sum(p.numel() for p in self.model.policy.parameters())
            trainable_params = sum(p.numel() for p in self.model.policy.parameters() if p.requires_grad)
            stats.update({
                'total_parameters': total_params,
                'trainable_parameters': trainable_params
            })
        
        # Add logger information if available
        if hasattr(self.model, 'logger') and self.model.logger:
            try:
                stats.update({
                    'current_lr': self.model.logger.name_to_value.get('train/learning_rate', None),
                    'policy_loss': self.model.logger.name_to_value.get('train/policy_loss', None),
                    'value_loss': self.model.logger.name_to_value.get('train/value_loss', None),
                    'entropy_loss': self.model.logger.name_to_value.get('train/entropy_loss', None),
                })
            except:
                pass
        
        return stats
    
    def set_learning_rate(self, learning_rate: float) -> None:
        """Update learning rate during training."""
        try:
            self.model.learning_rate = learning_rate
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group['lr'] = learning_rate
            logger.info(f"Learning rate updated to {learning_rate}")
        except Exception as e:
            logger.error(f"Failed to update learning rate: {e}")
    
    def reset_hidden_state(self) -> None:
        """Reset hidden state for recurrent policies."""
        if hasattr(self.model.policy, 'reset_hidden_state'):
            self.model.policy.reset_hidden_state()


def create_ppo_agent(
    env: Union[PokemonEnv, VecEnv],
    architecture: str = "standard",
    **kwargs
) -> PPOAgent:
    """
    Factory function to create PPO agent with specific architecture.
    
    Args:
        env: Pokemon environment
        architecture: CNN architecture type
        **kwargs: Additional PPO parameters
        
    Returns:
        PPOAgent: Configured PPO agent
    """
    # Set up policy kwargs based on architecture
    policy_kwargs = kwargs.pop('policy_kwargs', {})
    policy_kwargs.update({
        "features_extractor_class": create_pokemon_cnn,
        "features_extractor_kwargs": {"architecture": architecture},
    })
    
    agent = PPOAgent(
        env=env,
        policy="PokemonCNN",
        policy_kwargs=policy_kwargs,
        **kwargs
    )
    
    return agent


if __name__ == "__main__":
    # Test PPO agent creation (requires environment)
    print("Testing PPO agent creation...")
    
    try:
        from ..game.environment import make_pokemon_env
        
        # Create test environment
        env = make_pokemon_env(headless=True)
        
        # Create agent
        agent = create_ppo_agent(
            env,
            architecture="standard",
            learning_rate=3e-4,
            verbose=1
        )
        
        print(f"Agent created successfully")
        print(f"Policy: {type(agent.model.policy)}")
        print(f"Device: {agent.model.device}")
        
        # Test prediction
        obs = env.reset()
        action, _ = agent.predict(obs, deterministic=True)
        print(f"Prediction test: action={action}")
        
        # Get stats
        stats = agent.get_training_stats()
        print(f"Training stats: {stats}")
        
        env.close()
        print("PPO agent test completed!")
        
    except Exception as e:
        print(f"PPO agent test failed: {e}")
        import traceback
        traceback.print_exc()
