"""
Data models for preference collection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime
from enum import Enum


class PreferenceChoice(Enum):
    """Preference choice options."""
    SEGMENT_A = "a"
    SEGMENT_B = "b"
    EQUAL = "equal"
    UNCLEAR = "unclear"


@dataclass
class Segment:
    """A segment of a demonstration trajectory."""
    episode_id: str
    start_frame: int
    end_frame: int
    observations: Dict[str, np.ndarray]
    actions: np.ndarray
    rewards: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def length(self) -> int:
        """Get segment length in frames."""
        return self.end_frame - self.start_frame
    
    def get_observation(self, modality: str) -> Optional[np.ndarray]:
        """Get observation for a specific modality."""
        return self.observations.get(modality)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "episode_id": self.episode_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "length": self.length,
            "has_rewards": self.rewards is not None,
            "metadata": self.metadata or {}
        }


@dataclass
class PreferencePair:
    """A pair of segments for preference comparison."""
    pair_id: str
    segment_a: Segment
    segment_b: Segment
    metadata: Optional[Dict[str, Any]] = None
    labels: List['PreferenceLabel'] = field(default_factory=list)
    
    def add_label(self, label: 'PreferenceLabel') -> None:
        """Add a preference label to this pair."""
        self.labels.append(label)
    
    def get_consensus(self, threshold: float = 0.7) -> Optional[PreferenceChoice]:
        """
        Get consensus preference if agreement is above threshold.
        
        Args:
            threshold: Minimum agreement ratio required
            
        Returns:
            Consensus choice or None if no consensus
        """
        if not self.labels:
            return None
        
        # Count votes
        votes = {}
        for label in self.labels:
            choice = label.choice
            votes[choice] = votes.get(choice, 0) + 1
        
        # Find majority choice
        total_votes = len(self.labels)
        for choice, count in votes.items():
            if count / total_votes >= threshold:
                return choice
        
        return None
    
    def get_agreement_score(self) -> float:
        """Calculate inter-annotator agreement score."""
        if len(self.labels) < 2:
            return 1.0
        
        # Calculate pairwise agreement
        agreements = []
        for i in range(len(self.labels)):
            for j in range(i + 1, len(self.labels)):
                if self.labels[i].choice == self.labels[j].choice:
                    agreements.append(1.0)
                else:
                    agreements.append(0.0)
        
        return np.mean(agreements) if agreements else 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "pair_id": self.pair_id,
            "segment_a": self.segment_a.to_dict(),
            "segment_b": self.segment_b.to_dict(),
            "num_labels": len(self.labels),
            "consensus": self.get_consensus(),
            "agreement_score": self.get_agreement_score(),
            "metadata": self.metadata or {}
        }


@dataclass
class PreferenceLabel:
    """A single preference label from an annotator."""
    annotator_id: str
    choice: PreferenceChoice
    confidence: float  # 0-1 confidence score
    timestamp: str
    time_taken: float  # Seconds taken to make decision
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create(
        cls,
        annotator_id: str,
        choice: str,
        confidence: float = 1.0,
        time_taken: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'PreferenceLabel':
        """Create a new preference label."""
        # Convert string to enum
        if isinstance(choice, str):
            choice = PreferenceChoice(choice)
        
        return cls(
            annotator_id=annotator_id,
            choice=choice,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            time_taken=time_taken,
            metadata=metadata or {}
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "annotator_id": self.annotator_id,
            "choice": self.choice.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "time_taken": self.time_taken,
            "metadata": self.metadata or {}
        }


@dataclass
class AnnotationSession:
    """A session of preference annotations."""
    session_id: str
    annotator_id: str
    start_time: str
    end_time: Optional[str] = None
    pairs_annotated: List[str] = field(default_factory=list)
    total_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    def add_annotation(self, pair_id: str, time_taken: float) -> None:
        """Record an annotation in this session."""
        self.pairs_annotated.append(pair_id)
        self.total_time += time_taken
    
    def finish(self) -> None:
        """Mark session as finished."""
        self.end_time = datetime.now().isoformat()
    
    @property
    def average_time_per_pair(self) -> float:
        """Get average annotation time."""
        if not self.pairs_annotated:
            return 0.0
        return self.total_time / len(self.pairs_annotated)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "annotator_id": self.annotator_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "num_pairs": len(self.pairs_annotated),
            "total_time": self.total_time,
            "avg_time": self.average_time_per_pair,
            "metadata": self.metadata or {}
        }


@dataclass
class AnnotatorStats:
    """Statistics for an individual annotator."""
    annotator_id: str
    total_annotations: int = 0
    total_time: float = 0.0
    choice_distribution: Dict[str, int] = field(default_factory=dict)
    confidence_scores: List[float] = field(default_factory=list)
    agreement_scores: List[float] = field(default_factory=list)
    
    def update(self, label: PreferenceLabel, agreement: float) -> None:
        """Update statistics with a new label."""
        self.total_annotations += 1
        self.total_time += label.time_taken
        
        choice_str = label.choice.value
        self.choice_distribution[choice_str] = self.choice_distribution.get(choice_str, 0) + 1
        
        self.confidence_scores.append(label.confidence)
        self.agreement_scores.append(agreement)
    
    @property
    def average_confidence(self) -> float:
        """Get average confidence score."""
        return np.mean(self.confidence_scores) if self.confidence_scores else 0.0
    
    @property
    def average_agreement(self) -> float:
        """Get average agreement with other annotators."""
        return np.mean(self.agreement_scores) if self.agreement_scores else 0.0
    
    @property
    def average_time(self) -> float:
        """Get average time per annotation."""
        if self.total_annotations == 0:
            return 0.0
        return self.total_time / self.total_annotations
    
    def to_dict(self) -> dict:
        """Convert to dictionary for display."""
        return {
            "annotator_id": self.annotator_id,
            "total_annotations": self.total_annotations,
            "total_time": self.total_time,
            "average_time": self.average_time,
            "average_confidence": self.average_confidence,
            "average_agreement": self.average_agreement,
            "choice_distribution": self.choice_distribution
        }