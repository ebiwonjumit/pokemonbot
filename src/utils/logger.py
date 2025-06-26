"""
Logging utilities for Pokemon RL Bot.
Provides centralized logging configuration and utilities.
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import json


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for different log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """Format log record with colors."""
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_json: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup centralized logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: logs/)
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_json: Use JSON formatting for file logs
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        logging.Logger: Configured root logger
    """
    # Create log directory
    if log_dir is None:
        log_dir = Path("logs")
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Use colored formatter for console
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if enable_file:
        from logging.handlers import RotatingFileHandler
        
        log_file = log_dir / f"pokemon_rl_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Choose formatter
        if enable_json:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Create separate handlers for different modules
    _setup_module_loggers(log_dir, log_level, enable_json)
    
    return root_logger


def _setup_module_loggers(log_dir: Path, log_level: str, enable_json: bool):
    """Setup separate log files for different modules."""
    from logging.handlers import RotatingFileHandler
    
    # Module-specific loggers
    module_configs = {
        'emulator': 'emulator.log',
        'agent': 'agent.log',
        'environment': 'environment.log',
        'rewards': 'rewards.log',
        'web': 'web.log',
        'training': 'training.log'
    }
    
    for module_name, log_file in module_configs.items():
        logger = logging.getLogger(f"pokemon_rl.{module_name}")
        
        if not logger.handlers:  # Only add handler if not already present
            handler = RotatingFileHandler(
                log_dir / log_file,
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=3
            )
            handler.setLevel(getattr(logging, log_level.upper()))
            
            if enable_json:
                formatter = JSONFormatter()
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(getattr(logging, log_level.upper()))


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        logging.Logger: Logger instance
    """
    # Map module names to Pokemon RL structure
    if name.startswith('src.game'):
        mapped_name = f"pokemon_rl.environment"
    elif name.startswith('src.agent'):
        mapped_name = f"pokemon_rl.agent"
    elif name.startswith('src.web'):
        mapped_name = f"pokemon_rl.web"
    elif name.startswith('src.utils'):
        mapped_name = f"pokemon_rl.utils"
    elif 'emulator' in name:
        mapped_name = f"pokemon_rl.emulator"
    elif 'reward' in name:
        mapped_name = f"pokemon_rl.rewards"
    elif 'train' in name:
        mapped_name = f"pokemon_rl.training"
    else:
        mapped_name = f"pokemon_rl.{name}"
    
    return logging.getLogger(mapped_name)


class LoggingMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")


class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self, name: str):
        """Initialize performance logger."""
        self.logger = get_logger(f"performance.{name}")
        self.timers = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        import time
        self.timers[operation] = time.time()
    
    def end_timer(self, operation: str, log_level: str = "INFO"):
        """End timing and log the duration."""
        import time
        if operation in self.timers:
            duration = time.time() - self.timers[operation]
            self.logger.log(
                getattr(logging, log_level.upper()),
                f"{operation} completed in {duration:.4f} seconds"
            )
            del self.timers[operation]
            return duration
        return None
    
    def log_metrics(self, metrics: dict, log_level: str = "INFO"):
        """Log performance metrics."""
        self.logger.log(
            getattr(logging, log_level.upper()),
            f"Performance metrics: {json.dumps(metrics, indent=2)}"
        )


class TrainingLogger:
    """Specialized logger for training metrics."""
    
    def __init__(self, experiment_name: str):
        """Initialize training logger."""
        self.logger = get_logger(f"training.{experiment_name}")
        self.experiment_name = experiment_name
        self.episode_count = 0
    
    def log_episode(
        self,
        episode: int,
        total_reward: float,
        episode_length: int,
        loss: Optional[float] = None,
        metrics: Optional[dict] = None
    ):
        """Log episode training data."""
        log_data = {
            'episode': episode,
            'total_reward': total_reward,
            'episode_length': episode_length,
            'experiment': self.experiment_name
        }
        
        if loss is not None:
            log_data['loss'] = loss
        
        if metrics:
            log_data.update(metrics)
        
        # Create log record with extra data
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=f"Episode {episode} completed",
            args=(),
            exc_info=None
        )
        record.extra_data = log_data
        
        self.logger.handle(record)
    
    def log_checkpoint(self, checkpoint_path: str, metrics: dict):
        """Log model checkpoint information."""
        self.logger.info(
            f"Model checkpoint saved: {checkpoint_path}",
            extra={'checkpoint_metrics': metrics}
        )


def configure_external_loggers():
    """Configure logging for external libraries."""
    # Reduce verbosity of external libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('stable_baselines3').setLevel(logging.INFO)
    logging.getLogger('gym').setLevel(logging.WARNING)
    logging.getLogger('pygame').setLevel(logging.WARNING)


def log_system_info():
    """Log system information for debugging."""
    logger = get_logger("system")
    
    import platform
    import psutil
    import torch
    
    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        system_info['cuda_version'] = torch.version.cuda
        system_info['gpu_count'] = torch.cuda.device_count()
        system_info['gpu_name'] = torch.cuda.get_device_name(0)
    
    logger.info(f"System Information: {json.dumps(system_info, indent=2)}")


# Initialize logging when module is imported
if not logging.getLogger().handlers:
    setup_logging()
    configure_external_loggers()
    log_system_info()


if __name__ == "__main__":
    # Test logging functionality
    setup_logging(log_level="DEBUG", enable_json=False)
    
    logger = get_logger(__name__)
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test performance logger
    perf_logger = PerformanceLogger("test")
    perf_logger.start_timer("test_operation")
    
    import time
    time.sleep(0.1)
    
    perf_logger.end_timer("test_operation")
    perf_logger.log_metrics({"fps": 60, "memory_usage": 1024})
    
    # Test training logger
    training_logger = TrainingLogger("test_experiment")
    training_logger.log_episode(
        episode=1,
        total_reward=100.5,
        episode_length=1000,
        loss=0.01,
        metrics={"badges": 2, "pokemon": 5}
    )
    
    print("Logging test completed!")
