"""
Integration tests for optimized components.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
import time

from robo_rlhf.collectors.base import DemonstrationData
from robo_rlhf.collectors.optimized import (
    OptimizedDataProcessor, OptimizedRecorder, optimize_data_loading
)
from robo_rlhf.preference.models import Segment, PreferencePair, PreferenceLabel
from robo_rlhf.preference.pair_generator import PreferencePairGenerator


class TestOptimizedDataProcessor:
    """Test optimized data processor integration."""
    
    def test_parallel_processing_workflow(self):
        """Test complete parallel processing workflow."""
        processor = OptimizedDataProcessor(batch_size=5, max_workers=2)
        
        # Create test demonstrations
        demonstrations = []
        for i in range(20):
            demo = DemonstrationData(
                episode_id=f"test_demo_{i:03d}",
                timestamp=f"2025-01-01T00:00:{i:02d}",
                observations={
                    'rgb': np.random.randint(0, 255, (10, 32, 32, 3), dtype=np.uint8),
                    'proprioception': np.random.randn(10, 7)
                },
                actions=np.random.randn(10, 7),
                success=i % 3 == 0,
                duration=1.0
            )
            demonstrations.append(demo)
        
        def process_function(demo):
            """Test processing function."""
            processed_obs = processor.preprocess_observations(demo.observations)
            return {
                'episode_id': demo.episode_id,
                'frames': len(demo.actions),
                'success': demo.success,
                'processed_modalities': list(processed_obs.keys())
            }
        
        # Process in parallel
        start_time = time.time()
        results = processor.process_demonstrations_parallel(demonstrations, process_function)
        processing_time = time.time() - start_time
        
        # Verify results
        assert len(results) == len(demonstrations)
        
        for i, result in enumerate(results):
            assert result['episode_id'] == f"test_demo_{i:03d}"
            assert result['frames'] == 10
            assert 'processed_modalities' in result
        
        # Check statistics
        stats = processor.get_stats()
        assert stats['total_processed'] == len(demonstrations)
        assert stats['throughput'] > 0
        assert stats['errors'] == 0
        
        print(f"Processed {len(demonstrations)} demos in {processing_time:.3f}s "
              f"({stats['throughput']:.1f} demos/sec)")
        
        processor.shutdown()
    
    def test_preprocessing_caching(self):
        """Test preprocessing with caching."""
        processor = OptimizedDataProcessor()
        
        # Create identical observations
        observations = {
            'rgb': np.random.randint(0, 255, (5, 16, 16, 3), dtype=np.uint8),
            'proprioception': np.random.randn(5, 7)
        }
        
        # First preprocessing - cache miss
        start_time = time.time()
        result1 = processor.preprocess_observations(observations)
        first_time = time.time() - start_time
        
        # Second preprocessing - cache hit (should be faster)
        start_time = time.time()
        result2 = processor.preprocess_observations(observations)
        second_time = time.time() - start_time
        
        # Verify results are identical
        for key in result1:
            np.testing.assert_array_equal(result1[key], result2[key])
        
        # Cache should make second call faster
        assert second_time < first_time
        
        # Check cache statistics
        cache_stats = processor.preprocess_observations.cache_stats()
        assert cache_stats['hits'] >= 1
        
        processor.shutdown()


class TestOptimizedRecorder:
    """Test optimized recorder integration."""
    
    def test_parallel_saving_workflow(self):
        """Test parallel saving workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = OptimizedRecorder(
                output_dir=temp_dir,
                compression=True,
                enable_parallel_saving=True,
                save_thread_count=3
            )
            
            # Create test demonstrations
            demonstrations = []
            for i in range(15):
                demo = DemonstrationData(
                    episode_id=f"parallel_demo_{i:03d}",
                    timestamp=f"2025-01-01T00:00:{i:02d}",
                    observations={
                        'rgb': np.random.randint(0, 255, (8, 24, 24, 3), dtype=np.uint8),
                        'proprioception': np.random.randn(8, 7)
                    },
                    actions=np.random.randn(8, 7),
                    success=i % 2 == 0,
                    duration=0.8,
                    metadata={'batch_id': i // 5}
                )
                demonstrations.append(demo)
            
            # Save in parallel
            start_time = time.time()
            recorder.batch_save_demonstrations(demonstrations)
            save_time = time.time() - start_time
            
            # Verify files were created
            saved_dirs = list(Path(temp_dir).glob("parallel_demo_*"))
            assert len(saved_dirs) == len(demonstrations)
            
            # Verify compressed format
            for demo_dir in saved_dirs:
                data_file = demo_dir / "data.npz"
                metadata_file = demo_dir / "metadata.json"
                
                assert data_file.exists()
                assert metadata_file.exists()
                
                # Check compressed data can be loaded
                data = np.load(data_file)
                assert 'actions' in data.files
                assert 'obs_rgb' in data.files
                assert 'obs_proprioception' in data.files
            
            print(f"Saved {len(demonstrations)} demos in {save_time:.3f}s "
                  f"({len(demonstrations) / save_time:.1f} demos/sec)")
            
            recorder.shutdown()
    
    def test_save_load_consistency(self):
        """Test save/load consistency."""
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = OptimizedRecorder(
                output_dir=temp_dir,
                compression=True
            )
            
            # Create original demonstration
            original_demo = DemonstrationData(
                episode_id="consistency_test",
                timestamp="2025-01-01T12:00:00",
                observations={
                    'rgb': np.random.randint(0, 255, (5, 16, 16, 3), dtype=np.uint8),
                    'depth': np.random.rand(5, 16, 16, 1),
                    'proprioception': np.random.randn(5, 7)
                },
                actions=np.random.randn(5, 7),
                rewards=np.random.rand(5),
                success=True,
                duration=2.5,
                metadata={'test': 'consistency', 'value': 42}
            )
            
            # Save demonstration
            recorder.save_demonstration_async(original_demo)
            
            # Load demonstrations
            loaded_demos = optimize_data_loading(
                temp_dir,
                parallel_loading=True,
                cache_loaded_data=True
            )
            
            assert len(loaded_demos) == 1
            loaded_demo = loaded_demos[0]
            
            # Verify consistency
            assert loaded_demo.episode_id == original_demo.episode_id
            assert loaded_demo.success == original_demo.success
            assert loaded_demo.duration == original_demo.duration
            
            # Check observations
            for key in original_demo.observations:
                np.testing.assert_array_equal(
                    loaded_demo.observations[key],
                    original_demo.observations[key]
                )
            
            # Check actions and rewards
            np.testing.assert_array_equal(loaded_demo.actions, original_demo.actions)
            np.testing.assert_array_equal(loaded_demo.rewards, original_demo.rewards)
            
            # Check metadata
            assert loaded_demo.metadata['test'] == 'consistency'
            assert loaded_demo.metadata['value'] == 42
            
            recorder.shutdown()


class TestDataLoadingOptimization:
    """Test optimized data loading."""
    
    def test_parallel_vs_sequential_loading(self):
        """Test parallel vs sequential loading performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            demonstrations = []
            for i in range(20):
                demo = DemonstrationData(
                    episode_id=f"perf_test_{i:03d}",
                    timestamp=f"2025-01-01T00:00:{i:02d}",
                    observations={
                        'rgb': np.random.randint(0, 255, (6, 20, 20, 3), dtype=np.uint8),
                        'proprioception': np.random.randn(6, 7)
                    },
                    actions=np.random.randn(6, 7),
                    success=i % 4 == 0,
                    duration=1.2
                )
                demonstrations.append(demo)
            
            # Save demonstrations using optimized recorder
            recorder = OptimizedRecorder(temp_dir, compression=True)
            recorder.batch_save_demonstrations(demonstrations)
            recorder.shutdown()
            
            # Test parallel loading
            start_time = time.time()
            parallel_demos = optimize_data_loading(
                temp_dir,
                parallel_loading=True,
                max_workers=4
            )
            parallel_time = time.time() - start_time
            
            # Test sequential loading
            start_time = time.time()
            sequential_demos = optimize_data_loading(
                temp_dir,
                parallel_loading=False
            )
            sequential_time = time.time() - start_time
            
            # Verify results are identical
            assert len(parallel_demos) == len(sequential_demos) == len(demonstrations)
            
            # Sort by episode_id for comparison
            parallel_demos.sort(key=lambda d: d.episode_id)
            sequential_demos.sort(key=lambda d: d.episode_id)
            
            for p_demo, s_demo in zip(parallel_demos, sequential_demos):
                assert p_demo.episode_id == s_demo.episode_id
                assert p_demo.success == s_demo.success
                np.testing.assert_array_equal(p_demo.actions, s_demo.actions)
            
            # Parallel should be faster (or at least not significantly slower)
            speedup = sequential_time / parallel_time
            print(f"Loading speedup: {speedup:.2f}x "
                  f"(parallel: {parallel_time:.3f}s, sequential: {sequential_time:.3f}s)")
            
            # With small datasets, parallel might not be faster due to overhead,
            # but it shouldn't be significantly slower
            assert speedup > 0.5, "Parallel loading is significantly slower"
    
    def test_cached_loading(self):
        """Test cached data loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save a demonstration
            demo = DemonstrationData(
                episode_id="cache_test",
                timestamp="2025-01-01T10:00:00",
                observations={
                    'rgb': np.random.randint(0, 255, (3, 12, 12, 3), dtype=np.uint8)
                },
                actions=np.random.randn(3, 7),
                success=True,
                duration=1.0
            )
            
            recorder = OptimizedRecorder(temp_dir)
            recorder.save_demonstration_async(demo)
            recorder.shutdown()
            
            # First load - cache miss
            start_time = time.time()
            demos1 = optimize_data_loading(temp_dir, cache_loaded_data=True)
            first_load_time = time.time() - start_time
            
            # Second load - cache hit (should be faster)
            start_time = time.time()
            demos2 = optimize_data_loading(temp_dir, cache_loaded_data=True)
            second_load_time = time.time() - start_time
            
            # Verify results are identical
            assert len(demos1) == len(demos2) == 1
            assert demos1[0].episode_id == demos2[0].episode_id
            
            # Second load should be faster due to caching
            assert second_load_time < first_load_time
            
            print(f"Cache speedup: {first_load_time / second_load_time:.2f}x")


class TestPreferenceIntegration:
    """Test preference system integration."""
    
    def test_preference_pair_generation_workflow(self):
        """Test complete preference pair generation workflow."""
        # Create test demonstrations
        demonstrations = []
        for i in range(30):
            demo = DemonstrationData(
                episode_id=f"pref_demo_{i:03d}",
                timestamp=f"2025-01-01T00:00:{i:02d}",
                observations={
                    'rgb': np.random.randint(0, 255, (15, 28, 28, 3), dtype=np.uint8),
                    'proprioception': np.random.randn(15, 7)
                },
                actions=np.random.randn(15, 7),
                success=np.random.choice([True, False]),
                duration=1.5,
                metadata={'difficulty': np.random.choice(['easy', 'medium', 'hard'])}
            )
            demonstrations.append(demo)
        
        # Generate preference pairs
        generator = PreferencePairGenerator(demonstrations=demonstrations)
        
        start_time = time.time()
        pairs = generator.generate_pairs(num_pairs=10, segment_length=8)
        generation_time = time.time() - start_time
        
        # Verify pairs
        assert len(pairs) > 0
        assert len(pairs) <= 10
        
        for pair in pairs:
            assert isinstance(pair, PreferencePair)
            assert isinstance(pair.segment_a, Segment)
            assert isinstance(pair.segment_b, Segment)
            assert pair.segment_a.length == 8
            assert pair.segment_b.length == 8
            
            # Verify segments have data
            assert len(pair.segment_a.actions) == 8
            assert len(pair.segment_b.actions) == 8
            assert 'rgb' in pair.segment_a.observations
            assert 'proprioception' in pair.segment_a.observations
        
        print(f"Generated {len(pairs)} preference pairs in {generation_time:.3f}s")
    
    def test_preference_labeling_workflow(self):
        """Test preference labeling workflow."""
        # Create test segments
        segment_a = Segment(
            episode_id="demo_001",
            start_frame=0,
            end_frame=10,
            observations={'rgb': np.random.randn(10, 32, 32, 3)},
            actions=np.random.randn(10, 7),
            metadata={'success': True, 'smoothness': 0.8}
        )
        
        segment_b = Segment(
            episode_id="demo_002",
            start_frame=5,
            end_frame=15,
            observations={'rgb': np.random.randn(10, 32, 32, 3)},
            actions=np.random.randn(10, 7),
            metadata={'success': False, 'smoothness': 0.3}
        )
        
        # Create preference pair
        pair = PreferencePair(
            pair_id="test_pair_001",
            segment_a=segment_a,
            segment_b=segment_b,
            metadata={'comparison_type': 'success_based'}
        )
        
        # Add preference labels
        labels = [
            PreferenceLabel.create("expert_1", "a", confidence=0.9),
            PreferenceLabel.create("expert_2", "a", confidence=0.8),
            PreferenceLabel.create("novice_1", "b", confidence=0.6),
            PreferenceLabel.create("expert_3", "a", confidence=0.95)
        ]
        
        for label in labels:
            pair.add_label(label)
        
        # Test consensus
        consensus = pair.get_consensus(threshold=0.6)
        assert consensus is not None
        
        # Test agreement
        agreement = pair.get_agreement_score()
        assert 0 <= agreement <= 1
        
        # Test serialization
        pair_dict = pair.to_dict()
        assert 'pair_id' in pair_dict
        assert 'consensus' in pair_dict
        assert 'agreement_score' in pair_dict
        
        print(f"Consensus: {consensus.value if consensus else 'None'}, "
              f"Agreement: {agreement:.2f}")


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    def test_full_data_collection_and_processing_pipeline(self):
        """Test complete data collection and processing pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Create and save demonstrations
            recorder = OptimizedRecorder(
                output_dir=temp_dir,
                compression=True,
                enable_parallel_saving=True
            )
            
            demonstrations = []
            for i in range(25):
                demo = DemonstrationData(
                    episode_id=f"pipeline_demo_{i:03d}",
                    timestamp=f"2025-01-01T00:00:{i:02d}",
                    observations={
                        'rgb': np.random.randint(0, 255, (12, 24, 24, 3), dtype=np.uint8),
                        'proprioception': np.random.randn(12, 7)
                    },
                    actions=np.random.randn(12, 7),
                    rewards=np.random.rand(12),
                    success=i % 5 == 0,
                    duration=1.2,
                    metadata={'task_id': i % 3}
                )
                demonstrations.append(demo)
            
            # Save demonstrations
            start_time = time.time()
            recorder.batch_save_demonstrations(demonstrations)
            save_time = time.time() - start_time
            recorder.shutdown()
            
            # Step 2: Load demonstrations
            start_time = time.time()
            loaded_demos = optimize_data_loading(
                temp_dir,
                parallel_loading=True,
                cache_loaded_data=True
            )
            load_time = time.time() - start_time
            
            # Step 3: Process demonstrations
            processor = OptimizedDataProcessor(batch_size=8, max_workers=3)
            
            def processing_function(demo):
                processed_obs = processor.preprocess_observations(demo.observations)
                return {
                    'episode_id': demo.episode_id,
                    'success': demo.success,
                    'reward_sum': np.sum(demo.rewards) if demo.rewards is not None else 0,
                    'processed': True
                }
            
            start_time = time.time()
            processed_results = processor.process_demonstrations_parallel(
                loaded_demos, processing_function
            )
            process_time = time.time() - start_time
            
            # Step 4: Generate preference pairs
            generator = PreferencePairGenerator(demonstrations=loaded_demos)
            
            start_time = time.time()
            preference_pairs = generator.generate_pairs(
                num_pairs=8,
                segment_length=6
            )
            pair_time = time.time() - start_time
            
            # Verify complete pipeline
            assert len(loaded_demos) == len(demonstrations)
            assert len(processed_results) > 0  # Some should succeed
            assert len(preference_pairs) > 0
            
            # Print pipeline statistics
            total_time = save_time + load_time + process_time + pair_time
            
            print(f"\n=== Full Pipeline Results ===")
            print(f"Save time: {save_time:.3f}s ({len(demonstrations)} demos)")
            print(f"Load time: {load_time:.3f}s")
            print(f"Process time: {process_time:.3f}s ({len(processed_results)} results)")
            print(f"Pair generation: {pair_time:.3f}s ({len(preference_pairs)} pairs)")
            print(f"Total pipeline time: {total_time:.3f}s")
            
            # Get final statistics
            processor_stats = processor.get_stats()
            print(f"Processing throughput: {processor_stats['throughput']:.1f} demos/sec")
            print(f"Cache hit rate: {processor_stats.get('cache_stats', {}).get('hit_rate', 0):.2f}")
            
            processor.shutdown()
            
            # Verify data integrity throughout pipeline
            for original, loaded in zip(demonstrations, loaded_demos):
                assert original.episode_id == loaded.episode_id
                assert original.success == loaded.success
                np.testing.assert_array_equal(original.actions, loaded.actions)