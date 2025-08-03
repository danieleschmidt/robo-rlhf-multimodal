"""
Seed data for development and testing.
"""

import random
from datetime import datetime, timedelta
import json
import numpy as np
from pathlib import Path

from robo_rlhf.database.connection import get_db
from robo_rlhf.database.models import (
    DemonstrationRecord,
    PreferenceRecord,
    TrainingRun,
    ModelCheckpoint
)
from robo_rlhf.database.repositories import (
    DemonstrationRepository,
    PreferenceRepository,
    TrainingRepository
)


def seed_demonstrations(session, num_demos: int = 100):
    """Seed demonstration data."""
    print(f"Seeding {num_demos} demonstrations...")
    
    repo = DemonstrationRepository(session)
    
    tasks = ["pick_and_place", "stacking", "insertion", "drawer_opening"]
    devices = ["keyboard", "spacemouse", "vr_controller"]
    annotators = ["expert1", "expert2", "expert3", "novice1", "novice2"]
    
    for i in range(num_demos):
        # Generate random demo data
        task = random.choice(tasks)
        device = random.choice(devices)
        annotator = random.choice(annotators) if random.random() > 0.3 else None
        
        # Success rate varies by annotator expertise
        if annotator and "expert" in annotator:
            success = random.random() > 0.2  # 80% success
        else:
            success = random.random() > 0.5  # 50% success
        
        # Create demo record
        demo = repo.create(
            episode_id=f"demo_{i:05d}_{datetime.now().strftime('%Y%m%d')}",
            timestamp=datetime.utcnow() - timedelta(hours=random.randint(0, 720)),
            environment="mujoco",
            task=task,
            success=success,
            episode_length=random.randint(50, 500),
            total_reward=random.uniform(-10, 100) if success else random.uniform(-50, 10),
            duration_seconds=random.uniform(10, 120),
            data_path=f"/data/demonstrations/demo_{i:05d}",
            video_path=f"/data/videos/demo_{i:05d}.mp4" if random.random() > 0.5 else None,
            annotator_id=annotator,
            device_type=device,
            observation_modalities=["rgb", "depth", "proprioception"],
            metadata={
                "seed": random.randint(0, 1000000),
                "fps": 30,
                "resolution": "224x224"
            }
        )
        
        if i % 10 == 0:
            print(f"  Created demo {i+1}/{num_demos}")
    
    print(f"✓ Seeded {num_demos} demonstrations")


def seed_preferences(session, num_pairs: int = 200):
    """Seed preference data."""
    print(f"Seeding {num_pairs} preference pairs...")
    
    demo_repo = DemonstrationRepository(session)
    pref_repo = PreferenceRepository(session)
    
    # Get all demonstrations
    demos = demo_repo.get_all()
    if len(demos) < 2:
        print("Not enough demonstrations to create preferences")
        return
    
    annotators = ["expert1", "expert2", "expert3", "reviewer1", "reviewer2"]
    choices = ["a", "b", "equal", "unclear"]
    choice_weights = [0.4, 0.4, 0.15, 0.05]  # Preference distribution
    
    sessions = {}  # Track annotation sessions
    
    for i in range(num_pairs):
        # Select two random demonstrations
        demo_a, demo_b = random.sample(demos, 2)
        
        # Multiple annotations per pair (1-3)
        num_annotations = random.randint(1, 3)
        
        for j in range(num_annotations):
            annotator = random.choice(annotators)
            
            # Create or get session
            if annotator not in sessions:
                sessions[annotator] = f"session_{annotator}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Generate preference
            choice = random.choices(choices, weights=choice_weights)[0]
            
            # Expert annotators are more confident
            if "expert" in annotator:
                confidence = random.uniform(0.7, 1.0)
                time_taken = random.uniform(5, 20)
            else:
                confidence = random.uniform(0.4, 0.9)
                time_taken = random.uniform(10, 40)
            
            pref = pref_repo.create(
                pair_id=f"pair_{i:05d}_ann_{j}",
                timestamp=datetime.utcnow() - timedelta(hours=random.randint(0, 168)),
                segment_a_demo_id=demo_a.id,
                segment_a_start=random.randint(0, 50),
                segment_a_end=random.randint(51, 100),
                segment_b_demo_id=demo_b.id,
                segment_b_start=random.randint(0, 50),
                segment_b_end=random.randint(51, 100),
                annotator_id=annotator,
                choice=choice,
                confidence=confidence,
                time_taken_seconds=time_taken,
                session_id=sessions[annotator],
                metadata={
                    "interface_version": "1.0",
                    "browser": random.choice(["Chrome", "Firefox", "Safari"])
                }
            )
        
        if i % 20 == 0:
            print(f"  Created preference pair {i+1}/{num_pairs}")
    
    print(f"✓ Seeded {num_pairs} preference pairs")


def seed_training_runs(session, num_runs: int = 20):
    """Seed training run data."""
    print(f"Seeding {num_runs} training runs...")
    
    repo = TrainingRepository(session)
    
    model_types = ["VisionLanguageActor", "MultimodalActor", "BaselineModel"]
    algorithms = ["rlhf", "bc", "dagger", "ppo"]
    statuses = ["completed", "completed", "completed", "running", "failed"]
    
    for i in range(num_runs):
        model_type = random.choice(model_types)
        algorithm = random.choice(algorithms)
        status = random.choice(statuses)
        
        # Create training run
        start_time = datetime.utcnow() - timedelta(days=random.randint(1, 30))
        
        if status == "completed":
            end_time = start_time + timedelta(hours=random.uniform(2, 48))
            current_epoch = 100
            best_val = random.uniform(0.6, 0.95)
            final_test = best_val - random.uniform(0, 0.1)
        elif status == "running":
            end_time = None
            current_epoch = random.randint(1, 99)
            best_val = random.uniform(0.3, 0.7)
            final_test = None
        else:  # failed
            end_time = start_time + timedelta(hours=random.uniform(0.5, 5))
            current_epoch = random.randint(1, 50)
            best_val = random.uniform(0.1, 0.5)
            final_test = None
        
        run = repo.create(
            run_id=f"run_{model_type}_{algorithm}_{i:03d}",
            start_time=start_time,
            end_time=end_time,
            model_type=model_type,
            algorithm=algorithm,
            hyperparameters={
                "learning_rate": random.choice([1e-3, 3e-4, 1e-4]),
                "batch_size": random.choice([16, 32, 64]),
                "hidden_dim": random.choice([256, 512, 1024]),
                "num_layers": random.randint(2, 6)
            },
            num_demonstrations=random.randint(100, 1000),
            num_preferences=random.randint(200, 2000),
            data_split={
                "train": 0.7,
                "val": 0.15,
                "test": 0.15
            },
            status=status,
            current_epoch=current_epoch,
            total_epochs=100,
            best_validation_score=best_val,
            final_test_score=final_test,
            metrics_history={
                "train_loss": [random.uniform(0.5, 2.0) for _ in range(current_epoch)],
                "val_loss": [random.uniform(0.4, 1.8) for _ in range(current_epoch)]
            },
            gpu_hours=random.uniform(2, 100) if status == "completed" else None,
            peak_memory_gb=random.uniform(4, 32),
            checkpoint_path=f"/checkpoints/run_{i:03d}/best.pt",
            tensorboard_path=f"/tensorboard/run_{i:03d}",
            wandb_run_id=f"wandb_{i:08x}" if random.random() > 0.5 else None,
            git_commit="a" * 40,
            environment_info={
                "cuda_version": "11.8",
                "pytorch_version": "2.0.1",
                "num_gpus": random.choice([1, 2, 4, 8])
            },
            notes=f"Experimental run with {algorithm} on {model_type}"
        )
        
        # Create checkpoints for completed runs
        if status == "completed":
            for epoch in [20, 40, 60, 80, 100]:
                checkpoint = ModelCheckpoint(
                    checkpoint_id=f"ckpt_run_{i:03d}_epoch_{epoch}",
                    training_run_id=run.id,
                    epoch=epoch,
                    step=epoch * 1000,
                    timestamp=start_time + timedelta(hours=epoch/5),
                    validation_score=random.uniform(0.5, best_val),
                    training_loss=random.uniform(0.1, 0.5),
                    metrics={
                        "accuracy": random.uniform(0.6, 0.95),
                        "precision": random.uniform(0.6, 0.95),
                        "recall": random.uniform(0.6, 0.95)
                    },
                    file_path=f"/checkpoints/run_{i:03d}/epoch_{epoch}.pt",
                    file_size_mb=random.uniform(100, 500),
                    is_deployed=(epoch == 100 and i < 3),  # Deploy best from first 3 runs
                    deployment_timestamp=datetime.utcnow() if (epoch == 100 and i < 3) else None,
                    deployment_environment="production" if i == 0 else "staging" if i < 3 else None
                )
                session.add(checkpoint)
        
        if i % 5 == 0:
            print(f"  Created training run {i+1}/{num_runs}")
    
    session.commit()
    print(f"✓ Seeded {num_runs} training runs with checkpoints")


def seed_all():
    """Seed all data."""
    print("Starting database seeding...")
    
    # Initialize database
    db = get_db()
    
    with db.get_session() as session:
        # Seed in order of dependencies
        seed_demonstrations(session, 100)
        seed_preferences(session, 200)
        seed_training_runs(session, 20)
    
    print("\n✅ Database seeding complete!")
    
    # Print statistics
    with db.get_session() as session:
        demo_repo = DemonstrationRepository(session)
        pref_repo = PreferenceRepository(session)
        train_repo = TrainingRepository(session)
        
        print("\nDatabase Statistics:")
        print(f"  Demonstrations: {demo_repo.count()}")
        print(f"  Preferences: {pref_repo.count()}")
        print(f"  Training Runs: {train_repo.count()}")
        
        # Get some interesting stats
        demo_stats = demo_repo.get_statistics()
        print(f"\n  Demo Success Rate: {demo_stats['success_rate']:.2%}")
        
        agreement_stats = pref_repo.get_agreement_statistics()
        print(f"  Annotator Agreement: {agreement_stats['avg_agreement']:.2%}")
        
        train_stats = train_repo.get_training_statistics()
        print(f"  Training Success Rate: {train_stats['success_rate']:.2%}")
        print(f"  Total GPU Hours: {train_stats['total_gpu_hours']:.1f}")


if __name__ == "__main__":
    seed_all()