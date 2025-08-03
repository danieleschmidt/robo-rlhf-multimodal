"""
Database models for persistent storage.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    Column, String, Integer, Float, Boolean,
    DateTime, JSON, Text, ForeignKey, Index
)
from sqlalchemy.orm import relationship
from robo_rlhf.database.connection import Base


class DemonstrationRecord(Base):
    """
    Database model for demonstration recordings.
    """
    __tablename__ = "demonstrations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    episode_id = Column(String(100), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Task information
    environment = Column(String(100), nullable=False)
    task = Column(String(100), nullable=False)
    
    # Performance metrics
    success = Column(Boolean, default=False, index=True)
    episode_length = Column(Integer, nullable=False)
    total_reward = Column(Float, nullable=False)
    duration_seconds = Column(Float, nullable=False)
    
    # Data storage paths
    data_path = Column(Text, nullable=False)  # Path to actual data files
    video_path = Column(Text, nullable=True)  # Optional video recording
    
    # Metadata
    annotator_id = Column(String(100), nullable=True, index=True)
    device_type = Column(String(50), nullable=False)  # keyboard, spacemouse, vr
    observation_modalities = Column(JSON, nullable=False)  # List of modalities
    metadata = Column(JSON, nullable=True)  # Additional metadata
    
    # Relationships
    preferences = relationship("PreferenceRecord", back_populates="demonstration")
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_demo_task_success', 'task', 'success'),
        Index('idx_demo_timestamp_desc', timestamp.desc()),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "episode_id": self.episode_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "environment": self.environment,
            "task": self.task,
            "success": self.success,
            "episode_length": self.episode_length,
            "total_reward": self.total_reward,
            "duration_seconds": self.duration_seconds,
            "data_path": self.data_path,
            "video_path": self.video_path,
            "annotator_id": self.annotator_id,
            "device_type": self.device_type,
            "observation_modalities": self.observation_modalities,
            "metadata": self.metadata
        }


class PreferenceRecord(Base):
    """
    Database model for preference annotations.
    """
    __tablename__ = "preferences"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pair_id = Column(String(100), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Segments being compared
    segment_a_demo_id = Column(Integer, ForeignKey("demonstrations.id"))
    segment_a_start = Column(Integer, nullable=False)
    segment_a_end = Column(Integer, nullable=False)
    
    segment_b_demo_id = Column(Integer, ForeignKey("demonstrations.id"))
    segment_b_start = Column(Integer, nullable=False)
    segment_b_end = Column(Integer, nullable=False)
    
    # Annotation information
    annotator_id = Column(String(100), nullable=False, index=True)
    choice = Column(String(20), nullable=False)  # 'a', 'b', 'equal', 'unclear'
    confidence = Column(Float, nullable=False)  # 0-1 confidence score
    time_taken_seconds = Column(Float, nullable=False)
    
    # Session tracking
    session_id = Column(String(100), nullable=True, index=True)
    
    # Metadata
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    demonstration = relationship("DemonstrationRecord", back_populates="preferences")
    
    # Indexes
    __table_args__ = (
        Index('idx_pref_annotator_time', 'annotator_id', timestamp.desc()),
        Index('idx_pref_session', 'session_id'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "pair_id": self.pair_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "segment_a": {
                "demo_id": self.segment_a_demo_id,
                "start": self.segment_a_start,
                "end": self.segment_a_end
            },
            "segment_b": {
                "demo_id": self.segment_b_demo_id,
                "start": self.segment_b_start,
                "end": self.segment_b_end
            },
            "annotator_id": self.annotator_id,
            "choice": self.choice,
            "confidence": self.confidence,
            "time_taken_seconds": self.time_taken_seconds,
            "session_id": self.session_id,
            "metadata": self.metadata
        }


class TrainingRun(Base):
    """
    Database model for training runs.
    """
    __tablename__ = "training_runs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(100), unique=True, nullable=False, index=True)
    
    # Timing
    start_time = Column(DateTime, default=datetime.utcnow, index=True)
    end_time = Column(DateTime, nullable=True)
    
    # Configuration
    model_type = Column(String(100), nullable=False)
    algorithm = Column(String(50), nullable=False)  # 'rlhf', 'bc', 'dagger'
    hyperparameters = Column(JSON, nullable=False)
    
    # Data information
    num_demonstrations = Column(Integer, nullable=False)
    num_preferences = Column(Integer, nullable=False)
    data_split = Column(JSON, nullable=False)  # train/val/test split info
    
    # Status tracking
    status = Column(String(20), nullable=False, index=True)  # running, completed, failed
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, nullable=False)
    
    # Performance metrics
    best_validation_score = Column(Float, nullable=True)
    final_test_score = Column(Float, nullable=True)
    metrics_history = Column(JSON, nullable=True)  # Training curves
    
    # Resource usage
    gpu_hours = Column(Float, nullable=True)
    peak_memory_gb = Column(Float, nullable=True)
    
    # Artifacts
    checkpoint_path = Column(Text, nullable=True)
    tensorboard_path = Column(Text, nullable=True)
    wandb_run_id = Column(String(100), nullable=True)
    
    # Metadata
    git_commit = Column(String(40), nullable=True)
    environment_info = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)
    
    # Relationships
    checkpoints = relationship("ModelCheckpoint", back_populates="training_run")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "model_type": self.model_type,
            "algorithm": self.algorithm,
            "hyperparameters": self.hyperparameters,
            "num_demonstrations": self.num_demonstrations,
            "num_preferences": self.num_preferences,
            "data_split": self.data_split,
            "status": self.status,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "best_validation_score": self.best_validation_score,
            "final_test_score": self.final_test_score,
            "gpu_hours": self.gpu_hours,
            "peak_memory_gb": self.peak_memory_gb,
            "checkpoint_path": self.checkpoint_path,
            "wandb_run_id": self.wandb_run_id,
            "git_commit": self.git_commit,
            "notes": self.notes
        }


class ModelCheckpoint(Base):
    """
    Database model for model checkpoints.
    """
    __tablename__ = "model_checkpoints"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    checkpoint_id = Column(String(100), unique=True, nullable=False, index=True)
    
    # Training association
    training_run_id = Column(Integer, ForeignKey("training_runs.id"))
    
    # Checkpoint information
    epoch = Column(Integer, nullable=False)
    step = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Performance metrics
    validation_score = Column(Float, nullable=False)
    training_loss = Column(Float, nullable=False)
    metrics = Column(JSON, nullable=True)  # Additional metrics
    
    # Storage
    file_path = Column(Text, nullable=False)
    file_size_mb = Column(Float, nullable=False)
    
    # Deployment status
    is_deployed = Column(Boolean, default=False, index=True)
    deployment_timestamp = Column(DateTime, nullable=True)
    deployment_environment = Column(String(50), nullable=True)
    
    # Relationships
    training_run = relationship("TrainingRun", back_populates="checkpoints")
    
    # Indexes
    __table_args__ = (
        Index('idx_checkpoint_run_score', 'training_run_id', validation_score.desc()),
        Index('idx_checkpoint_deployed', 'is_deployed', 'deployment_timestamp'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "checkpoint_id": self.checkpoint_id,
            "training_run_id": self.training_run_id,
            "epoch": self.epoch,
            "step": self.step,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "validation_score": self.validation_score,
            "training_loss": self.training_loss,
            "metrics": self.metrics,
            "file_path": self.file_path,
            "file_size_mb": self.file_size_mb,
            "is_deployed": self.is_deployed,
            "deployment_timestamp": self.deployment_timestamp.isoformat() if self.deployment_timestamp else None,
            "deployment_environment": self.deployment_environment
        }