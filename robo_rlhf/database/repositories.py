"""
Repository pattern for data access.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, and_, or_
import json

from robo_rlhf.database.models import (
    DemonstrationRecord,
    PreferenceRecord,
    TrainingRun,
    ModelCheckpoint
)


class BaseRepository:
    """Base repository with common CRUD operations."""
    
    def __init__(self, session: Session, model_class):
        """
        Initialize repository.
        
        Args:
            session: Database session
            model_class: SQLAlchemy model class
        """
        self.session = session
        self.model_class = model_class
    
    def create(self, **kwargs) -> Any:
        """Create new record."""
        instance = self.model_class(**kwargs)
        self.session.add(instance)
        self.session.commit()
        self.session.refresh(instance)
        return instance
    
    def get_by_id(self, record_id: int) -> Optional[Any]:
        """Get record by ID."""
        return self.session.query(self.model_class).filter(
            self.model_class.id == record_id
        ).first()
    
    def get_all(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Any]:
        """Get all records with pagination."""
        query = self.session.query(self.model_class)
        
        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def update(self, record_id: int, **kwargs) -> Optional[Any]:
        """Update record."""
        instance = self.get_by_id(record_id)
        if instance:
            for key, value in kwargs.items():
                setattr(instance, key, value)
            self.session.commit()
            self.session.refresh(instance)
        return instance
    
    def delete(self, record_id: int) -> bool:
        """Delete record."""
        instance = self.get_by_id(record_id)
        if instance:
            self.session.delete(instance)
            self.session.commit()
            return True
        return False
    
    def count(self) -> int:
        """Count total records."""
        return self.session.query(self.model_class).count()


class DemonstrationRepository(BaseRepository):
    """Repository for demonstration records."""
    
    def __init__(self, session: Session):
        """Initialize demonstration repository."""
        super().__init__(session, DemonstrationRecord)
    
    def get_by_episode_id(self, episode_id: str) -> Optional[DemonstrationRecord]:
        """Get demonstration by episode ID."""
        return self.session.query(DemonstrationRecord).filter(
            DemonstrationRecord.episode_id == episode_id
        ).first()
    
    def get_successful(
        self,
        task: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[DemonstrationRecord]:
        """Get successful demonstrations."""
        query = self.session.query(DemonstrationRecord).filter(
            DemonstrationRecord.success == True
        )
        
        if task:
            query = query.filter(DemonstrationRecord.task == task)
        
        query = query.order_by(desc(DemonstrationRecord.timestamp))
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def get_by_annotator(
        self,
        annotator_id: str,
        limit: Optional[int] = None
    ) -> List[DemonstrationRecord]:
        """Get demonstrations by annotator."""
        query = self.session.query(DemonstrationRecord).filter(
            DemonstrationRecord.annotator_id == annotator_id
        ).order_by(desc(DemonstrationRecord.timestamp))
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def get_recent(
        self,
        hours: int = 24,
        task: Optional[str] = None
    ) -> List[DemonstrationRecord]:
        """Get recent demonstrations."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        query = self.session.query(DemonstrationRecord).filter(
            DemonstrationRecord.timestamp >= cutoff_time
        )
        
        if task:
            query = query.filter(DemonstrationRecord.task == task)
        
        return query.order_by(desc(DemonstrationRecord.timestamp)).all()
    
    def get_statistics(self, task: Optional[str] = None) -> Dict[str, Any]:
        """Get demonstration statistics."""
        query = self.session.query(DemonstrationRecord)
        
        if task:
            query = query.filter(DemonstrationRecord.task == task)
        
        demos = query.all()
        
        if not demos:
            return {
                "total": 0,
                "successful": 0,
                "success_rate": 0.0,
                "avg_reward": 0.0,
                "avg_length": 0.0,
                "avg_duration": 0.0
            }
        
        successful = [d for d in demos if d.success]
        
        return {
            "total": len(demos),
            "successful": len(successful),
            "success_rate": len(successful) / len(demos),
            "avg_reward": sum(d.total_reward for d in demos) / len(demos),
            "avg_length": sum(d.episode_length for d in demos) / len(demos),
            "avg_duration": sum(d.duration_seconds for d in demos) / len(demos),
            "device_distribution": self._get_device_distribution(demos),
            "annotator_distribution": self._get_annotator_distribution(demos)
        }
    
    def _get_device_distribution(
        self,
        demos: List[DemonstrationRecord]
    ) -> Dict[str, int]:
        """Get distribution of device types."""
        distribution = {}
        for demo in demos:
            device = demo.device_type
            distribution[device] = distribution.get(device, 0) + 1
        return distribution
    
    def _get_annotator_distribution(
        self,
        demos: List[DemonstrationRecord]
    ) -> Dict[str, int]:
        """Get distribution of annotators."""
        distribution = {}
        for demo in demos:
            if demo.annotator_id:
                distribution[demo.annotator_id] = distribution.get(demo.annotator_id, 0) + 1
        return distribution


class PreferenceRepository(BaseRepository):
    """Repository for preference records."""
    
    def __init__(self, session: Session):
        """Initialize preference repository."""
        super().__init__(session, PreferenceRecord)
    
    def get_by_pair_id(self, pair_id: str) -> Optional[PreferenceRecord]:
        """Get preference by pair ID."""
        return self.session.query(PreferenceRecord).filter(
            PreferenceRecord.pair_id == pair_id
        ).first()
    
    def get_by_annotator(
        self,
        annotator_id: str,
        limit: Optional[int] = None
    ) -> List[PreferenceRecord]:
        """Get preferences by annotator."""
        query = self.session.query(PreferenceRecord).filter(
            PreferenceRecord.annotator_id == annotator_id
        ).order_by(desc(PreferenceRecord.timestamp))
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def get_by_session(self, session_id: str) -> List[PreferenceRecord]:
        """Get preferences from a specific annotation session."""
        return self.session.query(PreferenceRecord).filter(
            PreferenceRecord.session_id == session_id
        ).order_by(asc(PreferenceRecord.timestamp)).all()
    
    def get_high_confidence(
        self,
        min_confidence: float = 0.8,
        limit: Optional[int] = None
    ) -> List[PreferenceRecord]:
        """Get high confidence preferences."""
        query = self.session.query(PreferenceRecord).filter(
            PreferenceRecord.confidence >= min_confidence
        ).order_by(desc(PreferenceRecord.confidence))
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def get_annotator_statistics(
        self,
        annotator_id: str
    ) -> Dict[str, Any]:
        """Get statistics for an annotator."""
        prefs = self.get_by_annotator(annotator_id)
        
        if not prefs:
            return {
                "total_annotations": 0,
                "avg_confidence": 0.0,
                "avg_time_taken": 0.0,
                "choice_distribution": {}
            }
        
        choice_dist = {}
        for pref in prefs:
            choice_dist[pref.choice] = choice_dist.get(pref.choice, 0) + 1
        
        return {
            "total_annotations": len(prefs),
            "avg_confidence": sum(p.confidence for p in prefs) / len(prefs),
            "avg_time_taken": sum(p.time_taken_seconds for p in prefs) / len(prefs),
            "choice_distribution": choice_dist,
            "recent_activity": [p.timestamp for p in prefs[:10]]
        }
    
    def get_agreement_statistics(self) -> Dict[str, Any]:
        """Calculate inter-annotator agreement statistics."""
        # Group preferences by pair_id
        pair_annotations = {}
        
        all_prefs = self.session.query(PreferenceRecord).all()
        
        for pref in all_prefs:
            if pref.pair_id not in pair_annotations:
                pair_annotations[pref.pair_id] = []
            pair_annotations[pref.pair_id].append(pref)
        
        # Calculate agreement
        agreements = []
        for pair_id, annotations in pair_annotations.items():
            if len(annotations) > 1:
                # Count matching choices
                choices = [a.choice for a in annotations]
                most_common = max(set(choices), key=choices.count)
                agreement = choices.count(most_common) / len(choices)
                agreements.append(agreement)
        
        if not agreements:
            return {
                "avg_agreement": 0.0,
                "perfect_agreement_rate": 0.0,
                "num_multi_annotated": 0
            }
        
        return {
            "avg_agreement": sum(agreements) / len(agreements),
            "perfect_agreement_rate": sum(1 for a in agreements if a == 1.0) / len(agreements),
            "num_multi_annotated": len(agreements),
            "total_pairs": len(pair_annotations)
        }


class TrainingRepository(BaseRepository):
    """Repository for training runs and checkpoints."""
    
    def __init__(self, session: Session):
        """Initialize training repository."""
        super().__init__(session, TrainingRun)
    
    def get_by_run_id(self, run_id: str) -> Optional[TrainingRun]:
        """Get training run by run ID."""
        return self.session.query(TrainingRun).filter(
            TrainingRun.run_id == run_id
        ).first()
    
    def get_active_runs(self) -> List[TrainingRun]:
        """Get currently running training runs."""
        return self.session.query(TrainingRun).filter(
            TrainingRun.status == "running"
        ).order_by(desc(TrainingRun.start_time)).all()
    
    def get_best_runs(
        self,
        model_type: Optional[str] = None,
        metric: str = "validation_score",
        limit: int = 10
    ) -> List[TrainingRun]:
        """Get best performing training runs."""
        query = self.session.query(TrainingRun).filter(
            TrainingRun.status == "completed"
        )
        
        if model_type:
            query = query.filter(TrainingRun.model_type == model_type)
        
        if metric == "validation_score":
            query = query.order_by(desc(TrainingRun.best_validation_score))
        elif metric == "test_score":
            query = query.order_by(desc(TrainingRun.final_test_score))
        
        return query.limit(limit).all()
    
    def get_checkpoints(
        self,
        training_run_id: int,
        best_only: bool = False
    ) -> List[ModelCheckpoint]:
        """Get checkpoints for a training run."""
        query = self.session.query(ModelCheckpoint).filter(
            ModelCheckpoint.training_run_id == training_run_id
        )
        
        if best_only:
            query = query.order_by(desc(ModelCheckpoint.validation_score)).limit(1)
        else:
            query = query.order_by(asc(ModelCheckpoint.epoch))
        
        return query.all()
    
    def get_deployed_checkpoints(self) -> List[ModelCheckpoint]:
        """Get currently deployed checkpoints."""
        return self.session.query(ModelCheckpoint).filter(
            ModelCheckpoint.is_deployed == True
        ).order_by(desc(ModelCheckpoint.deployment_timestamp)).all()
    
    def mark_checkpoint_deployed(
        self,
        checkpoint_id: str,
        environment: str
    ) -> Optional[ModelCheckpoint]:
        """Mark a checkpoint as deployed."""
        checkpoint = self.session.query(ModelCheckpoint).filter(
            ModelCheckpoint.checkpoint_id == checkpoint_id
        ).first()
        
        if checkpoint:
            checkpoint.is_deployed = True
            checkpoint.deployment_timestamp = datetime.utcnow()
            checkpoint.deployment_environment = environment
            self.session.commit()
            self.session.refresh(checkpoint)
        
        return checkpoint
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get overall training statistics."""
        all_runs = self.session.query(TrainingRun).all()
        
        if not all_runs:
            return {
                "total_runs": 0,
                "successful_runs": 0,
                "avg_gpu_hours": 0.0,
                "total_gpu_hours": 0.0
            }
        
        completed = [r for r in all_runs if r.status == "completed"]
        
        gpu_hours = [r.gpu_hours for r in completed if r.gpu_hours]
        
        return {
            "total_runs": len(all_runs),
            "successful_runs": len(completed),
            "success_rate": len(completed) / len(all_runs),
            "avg_gpu_hours": sum(gpu_hours) / len(gpu_hours) if gpu_hours else 0.0,
            "total_gpu_hours": sum(gpu_hours),
            "model_distribution": self._get_model_distribution(all_runs),
            "algorithm_distribution": self._get_algorithm_distribution(all_runs)
        }
    
    def _get_model_distribution(
        self,
        runs: List[TrainingRun]
    ) -> Dict[str, int]:
        """Get distribution of model types."""
        distribution = {}
        for run in runs:
            model = run.model_type
            distribution[model] = distribution.get(model, 0) + 1
        return distribution
    
    def _get_algorithm_distribution(
        self,
        runs: List[TrainingRun]
    ) -> Dict[str, int]:
        """Get distribution of algorithms."""
        distribution = {}
        for run in runs:
            algo = run.algorithm
            distribution[algo] = distribution.get(algo, 0) + 1
        return distribution