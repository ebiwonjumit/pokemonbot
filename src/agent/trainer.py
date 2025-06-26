"""
RL Training coordinator for Pokemon Leaf Green bot.
Orchestrates the complete training pipeline with monitoring and checkpointing.
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
import numpy as np

try:
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.utils import set_random_seed
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from .ppo_agent import PPOAgent, PokemonPPOCallback, create_ppo_agent
from ..game.environment import PokemonEnv, make_pokemon_env
from ..utils.logger import get_logger
from ..utils.metrics import MetricsCollector, MetricsVisualizer
from ..utils.cloud_storage import ModelManager, create_storage_provider

logger = get_logger(__name__)


class TrainingConfig:
    """Configuration for Pokemon RL training."""
    
    def __init__(
        self,
        # Environment settings
        rom_path: str = "roms/pokemon_leaf_green.gba",
        emulator_path: str = "/usr/local/bin/vbam",
        save_path: str = "saves/",
        headless: bool = False,
        
        # Training settings
        total_timesteps: int = 1000000,
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
        
        # Architecture settings
        architecture: str = "standard",
        policy_kwargs: Optional[Dict] = None,
        
        # Multiprocessing
        n_envs: int = 1,
        multiprocessing: bool = False,
        
        # Checkpointing and evaluation
        save_freq: int = 10000,
        eval_freq: int = 25000,
        n_eval_episodes: int = 5,
        checkpoint_freq: int = 50000,
        
        # Logging and monitoring
        log_interval: int = 100,
        tensorboard_log: str = "logs/tensorboard",
        verbose: int = 1,
        
        # Cloud storage
        use_cloud_storage: bool = False,
        cloud_provider: str = "aws",
        cloud_bucket: str = "pokemon-rl-models",
        
        # Experiment settings
        experiment_name: str = "pokemon_rl_experiment",
        description: str = "Pokemon Leaf Green RL training",
        tags: Optional[List[str]] = None,
        
        # Advanced settings
        use_wandb: bool = False,
        wandb_project: str = "pokemon-rl",
        seed: Optional[int] = None,
        device: str = "auto"
    ):
        """Initialize training configuration."""
        # Environment
        self.rom_path = rom_path
        self.emulator_path = emulator_path
        self.save_path = save_path
        self.headless = headless
        
        # Training
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
        # Architecture
        self.architecture = architecture
        self.policy_kwargs = policy_kwargs or {}
        
        # Multiprocessing
        self.n_envs = n_envs
        self.multiprocessing = multiprocessing
        
        # Checkpointing
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.checkpoint_freq = checkpoint_freq
        
        # Logging
        self.log_interval = log_interval
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        
        # Cloud
        self.use_cloud_storage = use_cloud_storage
        self.cloud_provider = cloud_provider
        self.cloud_bucket = cloud_bucket
        
        # Experiment
        self.experiment_name = experiment_name
        self.description = description
        self.tags = tags or []
        
        # Advanced
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.seed = seed
        self.device = device


class RLTrainer:
    """Main RL training coordinator for Pokemon Leaf Green."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize RL trainer.
        
        Args:
            config: Training configuration
        """
        if not SB3_AVAILABLE:
            raise ImportError("Stable-Baselines3 required for training")
        
        self.config = config
        
        # Create directories
        self.experiment_dir = Path(f"experiments/{config.experiment_name}")
        self.models_dir = self.experiment_dir / "models"
        self.logs_dir = self.experiment_dir / "logs"
        self.metrics_dir = self.experiment_dir / "metrics"
        
        for directory in [self.experiment_dir, self.models_dir, self.logs_dir, self.metrics_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.env = None
        self.eval_env = None
        self.agent = None
        self.metrics_collector = MetricsCollector(str(self.metrics_dir))
        self.metrics_visualizer = MetricsVisualizer(str(self.metrics_dir / "plots"))
        
        # Cloud storage setup
        self.model_manager = None
        if config.use_cloud_storage:
            try:
                storage = create_storage_provider(
                    config.cloud_provider,
                    bucket_name=config.cloud_bucket
                )
                self.model_manager = ModelManager(storage, str(self.models_dir))
                logger.info(f"Cloud storage initialized: {config.cloud_provider}")
            except Exception as e:
                logger.warning(f"Cloud storage setup failed: {e}")
        
        # Weights & Biases setup
        if config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.wandb_project,
                    name=config.experiment_name,
                    config=vars(config),
                    sync_tensorboard=True
                )
                logger.info("Weights & Biases initialized")
            except ImportError:
                logger.warning("wandb not available, skipping W&B integration")
            except Exception as e:
                logger.warning(f"W&B setup failed: {e}")
        
        # Training state
        self.training_start_time = None
        self.is_training = False
        
        # Save configuration
        self._save_config()
        
        logger.info(f"RL Trainer initialized: {config.experiment_name}")
    
    def setup_environment(self) -> None:
        """Set up training and evaluation environments."""
        logger.info("Setting up environments...")
        
        # Set random seed
        if self.config.seed is not None:
            set_random_seed(self.config.seed)
        
        def make_env(rank: int = 0):
            """Factory function for creating environments."""
            def _init():
                env = make_pokemon_env(
                    rom_path=self.config.rom_path,
                    emulator_path=self.config.emulator_path,
                    save_path=self.config.save_path,
                    headless=self.config.headless,
                    max_episode_steps=10000 + rank * 100  # Slight variation per env
                )
                
                # Wrap with Monitor for episode tracking
                monitor_path = self.logs_dir / f"monitor_env_{rank}.log"
                env = Monitor(env, str(monitor_path))
                
                return env
            
            return _init
        
        # Create training environment(s)
        if self.config.n_envs == 1:
            self.env = DummyVecEnv([make_env(0)])
        else:
            if self.config.multiprocessing:
                self.env = SubprocVecEnv([make_env(i) for i in range(self.config.n_envs)])
            else:
                self.env = DummyVecEnv([make_env(i) for i in range(self.config.n_envs)])
        
        # Create evaluation environment
        eval_env_fn = make_env(rank=999)  # Special rank for eval
        self.eval_env = DummyVecEnv([eval_env_fn])
        
        logger.info(f"Environments created: {self.config.n_envs} training, 1 evaluation")
    
    def setup_agent(self) -> None:
        """Set up the PPO agent."""
        logger.info("Setting up PPO agent...")
        
        if self.env is None:
            raise ValueError("Environment must be set up before agent")
        
        # Create agent
        self.agent = create_ppo_agent(
            env=self.env,
            architecture=self.config.architecture,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            device=self.config.device,
            verbose=self.config.verbose,
            seed=self.config.seed,
            tensorboard_log=str(self.logs_dir / "tensorboard"),
            policy_kwargs=self.config.policy_kwargs
        )
        
        logger.info("PPO agent created successfully")
        
        # Log agent statistics
        stats = self.agent.get_training_stats()
        logger.info(f"Agent stats: {stats}")
    
    def setup_callbacks(self) -> CallbackList:
        """Set up training callbacks."""
        callbacks = []
        
        # Pokemon-specific callback with metrics collection
        pokemon_callback = PokemonPPOCallback(
            metrics_collector=self.metrics_collector,
            save_freq=self.config.save_freq,
            save_path=str(self.models_dir),
            verbose=self.config.verbose
        )
        callbacks.append(pokemon_callback)
        
        # Checkpoint callback
        if self.config.checkpoint_freq > 0:
            checkpoint_callback = CheckpointCallback(
                save_freq=self.config.checkpoint_freq,
                save_path=str(self.models_dir / "checkpoints"),
                name_prefix="pokemon_ppo",
                verbose=self.config.verbose
            )
            callbacks.append(checkpoint_callback)
        
        # Evaluation callback (handled in agent.learn())
        
        return CallbackList(callbacks)
    
    def train(
        self,
        resume_from: Optional[str] = None,
        resume_timesteps: bool = True
    ) -> None:
        """
        Start training the agent.
        
        Args:
            resume_from: Path to model to resume from
            resume_timesteps: Whether to resume timestep counting
        """
        logger.info("Starting RL training...")
        
        # Setup components if not already done
        if self.env is None:
            self.setup_environment()
        
        if self.agent is None:
            self.setup_agent()
        
        # Resume from checkpoint if specified
        if resume_from:
            logger.info(f"Resuming training from: {resume_from}")
            self.agent.load(resume_from, env=self.env)
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Start training
        self.training_start_time = time.time()
        self.is_training = True
        
        try:
            self.agent.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callbacks,
                log_interval=self.config.log_interval,
                eval_env=self.eval_env if self.config.eval_freq > 0 else None,
                eval_freq=self.config.eval_freq,
                n_eval_episodes=self.config.n_eval_episodes,
                tb_log_name=self.config.experiment_name,
                reset_num_timesteps=not resume_timesteps,
                progress_bar=True
            )
            
            # Training completed successfully
            training_time = time.time() - self.training_start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Final evaluation
            self._final_evaluation()
            
            # Generate training report
            self._generate_training_report()
            
            # Upload to cloud storage
            if self.model_manager:
                self._upload_final_model()
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            self.is_training = False
    
    def evaluate(
        self,
        model_path: Optional[str] = None,
        n_episodes: int = 10,
        render: bool = False,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model_path: Path to model (uses current if None)
            n_episodes: Number of episodes to evaluate
            render: Whether to render during evaluation
            deterministic: Use deterministic policy
            
        Returns:
            Dict[str, float]: Evaluation results
        """
        if model_path and self.agent:
            # Load specified model
            self.agent.load(model_path, env=self.eval_env)
        elif not self.agent:
            raise ValueError("No agent available for evaluation")
        
        logger.info(f"Evaluating model for {n_episodes} episodes...")
        
        # Run evaluation
        episode_rewards, episode_lengths = self.agent.evaluate(
            env=self.eval_env,
            n_eval_episodes=n_episodes,
            deterministic=deterministic,
            render=render,
            return_episode_rewards=True
        )
        
        # Calculate statistics
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': sum(1 for r in episode_rewards if r > 0) / len(episode_rewards)
        }
        
        logger.info(f"Evaluation results: {results}")
        
        # Save evaluation results
        eval_file = self.metrics_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(eval_file, 'w') as f:
            json.dump({
                'results': results,
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'config': {
                    'n_episodes': n_episodes,
                    'deterministic': deterministic,
                    'model_path': model_path
                }
            }, f, indent=2)
        
        return results
    
    def _final_evaluation(self) -> None:
        """Run final evaluation after training."""
        logger.info("Running final evaluation...")
        
        try:
            results = self.evaluate(
                n_episodes=20,
                deterministic=True,
                render=False
            )
            
            logger.info(f"Final evaluation: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            
        except Exception as e:
            logger.error(f"Final evaluation failed: {e}")
    
    def _generate_training_report(self) -> None:
        """Generate comprehensive training report."""
        logger.info("Generating training report...")
        
        try:
            # Save metrics
            metrics_file = self.metrics_collector.save_metrics("final_metrics.json")
            
            # Generate visualizations
            episodes = self.metrics_collector.episodes
            if episodes:
                progress_plot = self.metrics_visualizer.plot_training_progress(episodes)
                learning_plot = self.metrics_visualizer.plot_learning_curves(episodes)
                dashboard = self.metrics_visualizer.create_dashboard(episodes)
                
                logger.info(f"Training visualizations saved: {dashboard}")
            
            # Create summary report
            summary = self.metrics_collector.get_training_summary()
            
            report = {
                'experiment_info': {
                    'name': self.config.experiment_name,
                    'description': self.config.description,
                    'tags': self.config.tags,
                    'start_time': self.training_start_time,
                    'duration': time.time() - self.training_start_time if self.training_start_time else 0
                },
                'training_config': vars(self.config),
                'training_summary': summary,
                'agent_stats': self.agent.get_training_stats() if self.agent else {},
                'files': {
                    'metrics': str(metrics_file),
                    'dashboard': dashboard if 'dashboard' in locals() else None
                }
            }
            
            # Save report
            report_file = self.experiment_dir / "training_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Training report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate training report: {e}")
    
    def _upload_final_model(self) -> None:
        """Upload final model to cloud storage."""
        if not self.model_manager or not self.agent:
            return
        
        try:
            # Save final model
            final_model_path = self.models_dir / "pokemon_ppo_final.zip"
            self.agent.save(str(final_model_path))
            
            # Get performance metrics
            summary = self.metrics_collector.get_training_summary()
            performance_metrics = {
                'mean_reward': summary.avg_reward,
                'max_reward': summary.max_reward,
                'total_episodes': summary.total_episodes,
                'total_timesteps': summary.total_timesteps,
                'training_time': summary.training_time.total_seconds()
            }
            
            # Upload to cloud
            success = self.model_manager.save_and_upload_model(
                model_path=str(final_model_path),
                model_id=self.config.experiment_name,
                version="final",
                training_episodes=summary.total_episodes,
                performance_metrics=performance_metrics,
                description=f"Final model for {self.config.experiment_name}",
                tags=self.config.tags + ["final", "pokemon-rl"]
            )
            
            if success:
                logger.info("Final model uploaded to cloud storage")
            else:
                logger.warning("Failed to upload final model to cloud storage")
                
        except Exception as e:
            logger.error(f"Failed to upload final model: {e}")
    
    def _save_config(self) -> None:
        """Save training configuration."""
        config_file = self.experiment_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(vars(self.config), f, indent=2)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.env:
                self.env.close()
            if self.eval_env:
                self.eval_env.close()
            
            # Close W&B if used
            if self.config.use_wandb:
                try:
                    import wandb
                    wandb.finish()
                except:
                    pass
            
            logger.info("Trainer cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


def create_trainer(
    experiment_name: str,
    rom_path: str = "roms/pokemon_leaf_green.gba",
    total_timesteps: int = 1000000,
    **kwargs
) -> RLTrainer:
    """
    Factory function to create RL trainer.
    
    Args:
        experiment_name: Name of the experiment
        rom_path: Path to Pokemon ROM
        total_timesteps: Total training timesteps
        **kwargs: Additional configuration parameters
        
    Returns:
        RLTrainer: Configured trainer
    """
    config = TrainingConfig(
        experiment_name=experiment_name,
        rom_path=rom_path,
        total_timesteps=total_timesteps,
        **kwargs
    )
    
    return RLTrainer(config)


if __name__ == "__main__":
    # Test trainer creation
    print("Testing RL trainer...")
    
    try:
        trainer = create_trainer(
            experiment_name="test_experiment",
            total_timesteps=10000,
            headless=True,
            verbose=1
        )
        
        print(f"Trainer created: {trainer.config.experiment_name}")
        print(f"Experiment directory: {trainer.experiment_dir}")
        
        # Setup environments (without starting training)
        trainer.setup_environment()
        print("Environment setup successful")
        
        trainer.setup_agent()
        print("Agent setup successful")
        
        # Get agent stats
        stats = trainer.agent.get_training_stats()
        print(f"Agent stats: {stats}")
        
        trainer.cleanup()
        print("Trainer test completed!")
        
    except Exception as e:
        print(f"Trainer test failed: {e}")
        import traceback
        traceback.print_exc()
