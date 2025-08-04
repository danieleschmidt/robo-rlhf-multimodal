"""
Configuration management for robo-rlhf-multimodal.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field, asdict
import logging

from robo_rlhf.core.exceptions import ConfigurationError


@dataclass
class DataCollectionConfig:
    """Configuration for data collection."""
    output_dir: str = "data/demonstrations"
    recording_fps: int = 30
    buffer_size: int = 10000
    compression: bool = True
    video_format: str = "mp4"
    max_episode_length: int = 1000
    auto_save: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.recording_fps <= 0:
            raise ConfigurationError("recording_fps must be positive")
        if self.buffer_size <= 0:
            raise ConfigurationError("buffer_size must be positive")


@dataclass
class PreferenceConfig:
    """Configuration for preference collection."""
    min_segment_length: int = 10
    max_segment_length: int = 100
    default_segment_length: int = 50
    consensus_threshold: float = 0.7
    min_annotators: int = 2
    max_pairs_per_session: int = 50
    auto_generate_pairs: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0 < self.consensus_threshold <= 1:
            raise ConfigurationError("consensus_threshold must be between 0 and 1")
        if self.min_annotators < 1:
            raise ConfigurationError("min_annotators must be at least 1")


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    device: str = "auto"  # auto, cpu, cuda, mps
    mixed_precision: bool = True
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    checkpoint_every: int = 1000
    validate_every: int = 100
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.learning_rate <= 0:
            raise ConfigurationError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ConfigurationError("batch_size must be positive")


@dataclass
class SecurityConfig:
    """Configuration for security settings."""
    enable_input_validation: bool = True
    enable_file_scanning: bool = True
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_file_types: list = field(default_factory=lambda: [".json", ".pkl", ".npy", ".mp4", ".avi"])
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    enable_auth: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_file_size <= 0:
            raise ConfigurationError("max_file_size must be positive")


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    structured: bool = False
    console: bool = True
    file: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            raise ConfigurationError(f"log level must be one of {valid_levels}")


@dataclass
class Config:
    """Main configuration class."""
    data_collection: DataCollectionConfig = field(default_factory=DataCollectionConfig)
    preferences: PreferenceConfig = field(default_factory=PreferenceConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Global settings
    debug: bool = False
    environment: str = "development"  # development, staging, production
    version: str = "0.1.0"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.environment not in ["development", "staging", "production"]:
            raise ConfigurationError("environment must be development, staging, or production")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        try:
            # Extract nested configs
            data_collection = config_dict.pop("data_collection", {})
            preferences = config_dict.pop("preferences", {})
            models = config_dict.pop("models", {})
            security = config_dict.pop("security", {})
            logging_config = config_dict.pop("logging", {})
            
            return cls(
                data_collection=DataCollectionConfig(**data_collection),
                preferences=PreferenceConfig(**preferences),
                models=ModelConfig(**models),
                security=SecurityConfig(**security),
                logging=LoggingConfig(**logging_config),
                **config_dict
            )
        except TypeError as e:
            raise ConfigurationError(f"Invalid configuration: {e}")
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'Config':
        """Load configuration from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    config_dict = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    config_dict = json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported config file format: {file_path.suffix}")
            
            return cls.from_dict(config_dict)
        
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigurationError(f"Failed to parse config file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load config file: {e}")
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        config_dict = {}
        
        # Global settings
        if os.getenv("ROBO_RLHF_DEBUG"):
            config_dict["debug"] = os.getenv("ROBO_RLHF_DEBUG").lower() == "true"
        if os.getenv("ROBO_RLHF_ENVIRONMENT"):
            config_dict["environment"] = os.getenv("ROBO_RLHF_ENVIRONMENT")
        
        # Data collection
        data_collection = {}
        if os.getenv("ROBO_RLHF_DATA_DIR"):
            data_collection["output_dir"] = os.getenv("ROBO_RLHF_DATA_DIR")
        if os.getenv("ROBO_RLHF_RECORDING_FPS"):
            data_collection["recording_fps"] = int(os.getenv("ROBO_RLHF_RECORDING_FPS"))
        if data_collection:
            config_dict["data_collection"] = data_collection
        
        # Models
        models = {}
        if os.getenv("ROBO_RLHF_DEVICE"):
            models["device"] = os.getenv("ROBO_RLHF_DEVICE")
        if os.getenv("ROBO_RLHF_BATCH_SIZE"):
            models["batch_size"] = int(os.getenv("ROBO_RLHF_BATCH_SIZE"))
        if os.getenv("ROBO_RLHF_LEARNING_RATE"):
            models["learning_rate"] = float(os.getenv("ROBO_RLHF_LEARNING_RATE"))
        if models:
            config_dict["models"] = models
        
        # Logging
        logging_config = {}
        if os.getenv("ROBO_RLHF_LOG_LEVEL"):
            logging_config["level"] = os.getenv("ROBO_RLHF_LOG_LEVEL")
        if os.getenv("ROBO_RLHF_LOG_FILE"):
            logging_config["file"] = os.getenv("ROBO_RLHF_LOG_FILE")
        if os.getenv("ROBO_RLHF_LOG_STRUCTURED"):
            logging_config["structured"] = os.getenv("ROBO_RLHF_LOG_STRUCTURED").lower() == "true"
        if logging_config:
            config_dict["logging"] = logging_config
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save(self, file_path: Union[str, Path], format: str = "yaml") -> None:
        """Save configuration to file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        with open(file_path, 'w') as f:
            if format.lower() == "yaml":
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                json.dump(config_dict, f, indent=2)
            else:
                raise ConfigurationError(f"Unsupported format: {format}")
    
    def update(self, **kwargs) -> 'Config':
        """Create updated configuration."""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.from_dict(config_dict)


# Global configuration instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _global_config
    
    if _global_config is None:
        # Try to load from file
        config_paths = [
            "robo_rlhf_config.yaml",
            "robo_rlhf_config.yml", 
            "robo_rlhf_config.json",
            "config/robo_rlhf.yaml",
            "config/robo_rlhf.yml",
            "config/robo_rlhf.json",
            os.path.expanduser("~/.robo_rlhf/config.yaml"),
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                _global_config = Config.from_file(config_path)
                break
        
        # Fall back to environment variables
        if _global_config is None:
            _global_config = Config.from_env()
    
    return _global_config


def set_config(config: Config) -> None:
    """Set global configuration instance."""
    global _global_config
    _global_config = config


def reload_config() -> Config:
    """Reload configuration from sources."""
    global _global_config
    _global_config = None
    return get_config()