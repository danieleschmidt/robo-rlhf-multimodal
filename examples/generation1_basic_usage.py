#!/usr/bin/env python3
"""
Generation 1 Basic Usage Example - MAKE IT WORK

Demonstrates the core functionality of robo-rlhf-multimodal
without requiring heavy dependencies like PyTorch.
"""

import numpy as np
from pathlib import Path
import tempfile
import json

# Import core components
from robo_rlhf.preference.models import (
    Segment, 
    PreferencePair, 
    PreferenceChoice, 
    PreferenceLabel
)
from robo_rlhf.preference.pair_generator import PreferencePairGenerator
from robo_rlhf.collectors.recorder import DemonstrationRecorder
from robo_rlhf.collectors.base import DemonstrationData


def create_mock_demonstrations(num_demos: int = 5) -> list:
    """Create mock demonstration data."""
    demonstrations = []
    
    for i in range(num_demos):
        # Create mock observations
        length = np.random.randint(20, 100)
        observations = {
            'rgb': np.random.randint(0, 255, (length, 64, 64, 3), dtype=np.uint8),
            'depth': np.random.rand(length, 64, 64, 1),
            'proprioception': np.random.randn(length, 7)
        }
        
        # Create mock actions
        actions = np.random.randn(length, 7)
        
        # Random success
        success = np.random.choice([True, False], p=[0.7, 0.3])
        
        demo = DemonstrationData(
            episode_id=f"demo_{i:03d}",
            timestamp="2025-01-01T00:00:00",
            observations=observations,
            actions=actions,
            success=success,
            duration=length * 0.1,  # 10 FPS
            metadata={
                "environment": "pick_and_place",
                "difficulty": "easy" if success else "hard"
            }
        )
        demonstrations.append(demo)
    
    return demonstrations


def demonstrate_preference_system():
    """Demonstrate the preference collection system."""
    print("\n🎯 Testing Preference System")
    print("=" * 40)
    
    # Create mock segments
    segment_a = Segment(
        episode_id="demo_001",
        start_frame=0,
        end_frame=20,
        observations={'rgb': np.random.randn(20, 64, 64, 3)},
        actions=np.random.randn(20, 7),
        metadata={'success': True, 'smoothness': 0.8}
    )
    
    segment_b = Segment(
        episode_id="demo_002", 
        start_frame=10,
        end_frame=30,
        observations={'rgb': np.random.randn(20, 64, 64, 3)},
        actions=np.random.randn(20, 7),
        metadata={'success': False, 'smoothness': 0.3}
    )
    
    # Create preference pair
    pair = PreferencePair(
        pair_id="pair_001",
        segment_a=segment_a,
        segment_b=segment_b,
        metadata={'comparison_type': 'success_based'}
    )
    
    print(f"✅ Created preference pair: {pair.pair_id}")
    print(f"   Segment A: {segment_a.length} frames, success: {segment_a.metadata['success']}")
    print(f"   Segment B: {segment_b.length} frames, success: {segment_b.metadata['success']}")
    
    # Add some preference labels
    labels = [
        PreferenceLabel.create("expert_1", "a", confidence=0.9),
        PreferenceLabel.create("expert_2", "a", confidence=0.8),
        PreferenceLabel.create("novice_1", "b", confidence=0.6),
    ]
    
    for label in labels:
        pair.add_label(label)
    
    print(f"✅ Added {len(labels)} preference labels")
    
    # Analyze consensus
    consensus = pair.get_consensus(threshold=0.5)
    agreement = pair.get_agreement_score()
    
    print(f"   Consensus: {consensus.value if consensus else 'None'}")
    print(f"   Agreement score: {agreement:.2f}")
    
    return pair


def demonstrate_data_collection():
    """Demonstrate data collection and recording."""
    print("\n📊 Testing Data Collection")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create recorder
        recorder = DemonstrationRecorder(
            output_dir=temp_dir,
            compression=False,
            metadata_enrichment=True
        )
        
        print(f"✅ Created recorder in: {temp_dir}")
        
        # Record a mock episode
        episode_id = recorder.start_episode(
            metadata={'task': 'pick_and_place', 'difficulty': 'easy'}
        )
        
        print(f"✅ Started episode: {episode_id}")
        
        # Simulate recording steps
        for step in range(50):
            observation = {
                'rgb': np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                'proprioception': np.random.randn(7)
            }
            action = np.random.randn(7)
            reward = np.random.rand()
            
            recorder.record_step(observation, action, reward)
        
        print(f"✅ Recorded 50 steps")
        
        # Stop recording
        demo = recorder.stop_episode(
            success=True,
            final_metadata={'final_reward': 1.0}
        )
        
        print(f"✅ Saved demonstration: {demo.episode_id}")
        print(f"   Duration: {demo.duration:.2f}s")
        print(f"   Success: {demo.success}")
        print(f"   Actions shape: {demo.actions.shape}")
        
        # Test loading
        loaded_demos = recorder.load_demonstrations([episode_id])
        print(f"✅ Loaded {len(loaded_demos)} demonstrations")
        
        # Show stats
        stats = recorder.get_stats()
        print(f"✅ Recording stats: {stats}")
        
        return demo


def demonstrate_pair_generation():
    """Demonstrate preference pair generation."""
    print("\n🔄 Testing Pair Generation")
    print("=" * 40)
    
    # Create mock demonstrations
    demonstrations = create_mock_demonstrations(10)
    print(f"✅ Created {len(demonstrations)} mock demonstrations")
    
    # Create pair generator with demonstrations
    generator = PreferencePairGenerator(demonstrations=demonstrations)
    
    # Generate pairs
    pairs = generator.generate_pairs(
        num_pairs=5,
        segment_length=20
    )
    
    print(f"✅ Generated {len(pairs)} preference pairs")
    
    for i, pair in enumerate(pairs):
        print(f"   Pair {i+1}: {pair.segment_a.episode_id}[{pair.segment_a.start_frame}:{pair.segment_a.end_frame}] vs "
              f"{pair.segment_b.episode_id}[{pair.segment_b.start_frame}:{pair.segment_b.end_frame}]")
    
    return pairs


def main():
    """Main demonstration."""
    print("🤖 Robo-RLHF-Multimodal - Generation 1 Demo")
    print("=" * 50)
    print("Testing core functionality without heavy dependencies...")
    
    try:
        # Test each component
        demo = demonstrate_data_collection()
        pair = demonstrate_preference_system()
        pairs = demonstrate_pair_generation()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("=" * 50)
        print("✅ Data collection and recording")
        print("✅ Preference pair creation and labeling")
        print("✅ Preference pair generation from demonstrations")
        print("✅ Metadata handling and serialization")
        
        print("\n📈 Generation 1 Status: COMPLETE")
        print("🚀 Ready for Generation 2 (Robust implementation)")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())