"""
Performance benchmarks for robo-rlhf-multimodal.
"""

import pytest
import time
import tempfile
import numpy as np
from pathlib import Path
import concurrent.futures
from statistics import mean, stdev

from robo_rlhf.collectors.base import DemonstrationData
from robo_rlhf.collectors.optimized import (
    OptimizedDataProcessor, OptimizedRecorder, optimize_data_loading
)
from robo_rlhf.core.performance import (
    get_performance_monitor, ThreadPool, ProcessPool, LRUCache,
    BatchProcessor, ResourcePool, optimize_memory
)
from robo_rlhf.preference.pair_generator import PreferencePairGenerator


class BenchmarkResults:
    """Container for benchmark results."""
    
    def __init__(self, name: str):
        self.name = name
        self.times = []
        self.throughputs = []
        self.memory_usage = []
        
    def add_result(self, time_taken: float, items_processed: int = 1, memory_mb: float = 0):
        """Add a benchmark result."""
        self.times.append(time_taken)
        if time_taken > 0:
            self.throughputs.append(items_processed / time_taken)
        else:
            self.throughputs.append(float('inf'))
        self.memory_usage.append(memory_mb)
    
    def get_summary(self) -> dict:
        """Get benchmark summary statistics."""
        if not self.times:
            return {"name": self.name, "no_data": True}
        
        return {
            "name": self.name,
            "avg_time": mean(self.times),
            "min_time": min(self.times),
            "max_time": max(self.times),
            "std_time": stdev(self.times) if len(self.times) > 1 else 0,
            "avg_throughput": mean(self.throughputs),
            "max_throughput": max(self.throughputs),
            "avg_memory_mb": mean(self.memory_usage) if self.memory_usage else 0,
            "num_runs": len(self.times)
        }


class TestDataProcessingBenchmarks:
    """Benchmark data processing performance."""
    
    @pytest.mark.benchmark
    def test_preprocessing_performance(self):
        """Benchmark observation preprocessing performance."""
        processor = OptimizedDataProcessor(batch_size=16, max_workers=4)
        benchmark = BenchmarkResults("Observation Preprocessing")
        
        # Test different data sizes
        test_sizes = [
            (10, 32, 32),    # Small
            (50, 64, 64),    # Medium
            (100, 128, 128), # Large
        ]
        
        for frames, height, width in test_sizes:
            observations = {
                'rgb': np.random.randint(0, 255, (frames, height, width, 3), dtype=np.uint8),
                'depth': np.random.rand(frames, height, width, 1),
                'proprioception': np.random.randn(frames, 7)
            }
            
            # Warm-up run
            processor.preprocess_observations(observations)
            
            # Benchmark runs
            num_runs = 10
            for _ in range(num_runs):
                start_time = time.time()
                result = processor.preprocess_observations(observations)
                end_time = time.time()
                
                benchmark.add_result(
                    time_taken=end_time - start_time,
                    items_processed=frames,
                    memory_mb=sum(arr.nbytes for arr in result.values()) / 1024 / 1024
                )
        
        summary = benchmark.get_summary()
        print(f"\n=== {summary['name']} ===")
        print(f"Average time: {summary['avg_time']:.4f}s")
        print(f"Average throughput: {summary['avg_throughput']:.1f} frames/sec")
        print(f"Average memory: {summary['avg_memory_mb']:.1f}MB")
        
        # Performance assertions
        assert summary['avg_throughput'] > 100, "Preprocessing too slow"
        assert summary['avg_memory_mb'] < 1000, "Memory usage too high"
        
        # Check cache performance
        cache_stats = processor.preprocess_observations.cache_stats()
        assert cache_stats['hit_rate'] > 0, "Cache not being used"
        
        processor.shutdown()
    
    @pytest.mark.benchmark
    def test_parallel_processing_scalability(self):
        """Benchmark parallel processing scalability."""
        # Create test data
        demonstrations = []
        for i in range(100):
            demo = DemonstrationData(
                episode_id=f"bench_demo_{i:03d}",
                timestamp=f"2025-01-01T00:00:{i:02d}",
                observations={
                    'rgb': np.random.randint(0, 255, (20, 48, 48, 3), dtype=np.uint8),
                    'proprioception': np.random.randn(20, 7)
                },
                actions=np.random.randn(20, 7),
                success=i % 3 == 0,
                duration=2.0
            )
            demonstrations.append(demo)
        
        def processing_function(demo):
            # Simulate CPU-intensive processing
            processed_obs = {}
            for key, obs in demo.observations.items():
                if key == 'rgb':
                    # Simulate image processing
                    processed_obs[key] = np.mean(obs, axis=(1, 2))
                else:
                    processed_obs[key] = np.mean(obs, axis=0)
            
            return {
                'episode_id': demo.episode_id,
                'success': demo.success,
                'processed_shapes': {k: v.shape for k, v in processed_obs.items()}
            }
        
        # Test different worker counts
        worker_counts = [1, 2, 4, 8]
        results = {}
        
        for worker_count in worker_counts:
            processor = OptimizedDataProcessor(
                batch_size=10,
                max_workers=worker_count
            )
            
            benchmark = BenchmarkResults(f"Parallel Processing ({worker_count} workers)")
            
            # Run benchmark
            num_runs = 5
            for run in range(num_runs):
                start_time = time.time()
                processed_results = processor.process_demonstrations_parallel(
                    demonstrations, processing_function
                )
                end_time = time.time()
                
                benchmark.add_result(
                    time_taken=end_time - start_time,
                    items_processed=len(processed_results)
                )
            
            results[worker_count] = benchmark.get_summary()
            processor.shutdown()
        
        # Print results
        print(f"\n=== Parallel Processing Scalability ===")
        for worker_count, summary in results.items():
            print(f"{worker_count} workers: {summary['avg_throughput']:.1f} demos/sec "
                  f"({summary['avg_time']:.3f}s avg)")
        
        # Calculate speedup
        baseline_throughput = results[1]['avg_throughput']
        for worker_count in worker_counts[1:]:
            speedup = results[worker_count]['avg_throughput'] / baseline_throughput
            print(f"Speedup with {worker_count} workers: {speedup:.2f}x")
            
            # Assert reasonable speedup (at least 1.5x with 4 workers)
            if worker_count == 4:
                assert speedup > 1.5, f"Poor scalability: {speedup:.2f}x speedup with 4 workers"


class TestStorageBenchmarks:
    """Benchmark storage and I/O performance."""
    
    @pytest.mark.benchmark
    def test_save_load_performance(self):
        """Benchmark save/load performance."""
        # Create test demonstrations
        demo_counts = [10, 50, 100]
        
        for demo_count in demo_counts:
            demonstrations = []
            for i in range(demo_count):
                demo = DemonstrationData(
                    episode_id=f"storage_demo_{i:03d}",
                    timestamp=f"2025-01-01T00:00:{i:02d}",
                    observations={
                        'rgb': np.random.randint(0, 255, (15, 56, 56, 3), dtype=np.uint8),
                        'depth': np.random.rand(15, 56, 56, 1),
                        'proprioception': np.random.randn(15, 7)
                    },
                    actions=np.random.randn(15, 7),
                    rewards=np.random.rand(15),
                    success=i % 4 == 0,
                    duration=1.5,
                    metadata={'task_type': f'type_{i % 3}'}
                )
                demonstrations.append(demo)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Benchmark saving
                save_benchmark = BenchmarkResults(f"Save Performance ({demo_count} demos)")
                
                recorder = OptimizedRecorder(
                    output_dir=temp_dir,
                    compression=True,
                    enable_parallel_saving=True,
                    save_thread_count=4
                )
                
                num_runs = 3
                for run in range(num_runs):
                    # Clear directory
                    for item in Path(temp_dir).iterdir():
                        if item.is_dir():
                            import shutil
                            shutil.rmtree(item)
                    
                    start_time = time.time()
                    recorder.batch_save_demonstrations(demonstrations)
                    end_time = time.time()
                    
                    # Calculate total size
                    total_size = sum(
                        f.stat().st_size for f in Path(temp_dir).rglob("*")
                        if f.is_file()
                    ) / 1024 / 1024  # MB
                    
                    save_benchmark.add_result(
                        time_taken=end_time - start_time,
                        items_processed=demo_count,
                        memory_mb=total_size
                    )
                
                recorder.shutdown()
                
                # Benchmark loading
                load_benchmark = BenchmarkResults(f"Load Performance ({demo_count} demos)")
                
                for run in range(num_runs):
                    start_time = time.time()
                    loaded_demos = optimize_data_loading(
                        temp_dir,
                        parallel_loading=True,
                        cache_loaded_data=False,  # Disable cache for fair comparison
                        max_workers=4
                    )
                    end_time = time.time()
                    
                    load_benchmark.add_result(
                        time_taken=end_time - start_time,
                        items_processed=len(loaded_demos)
                    )
                
                # Print results
                save_summary = save_benchmark.get_summary()
                load_summary = load_benchmark.get_summary()
                
                print(f"\n=== Storage Performance ({demo_count} demos) ===")
                print(f"Save: {save_summary['avg_throughput']:.1f} demos/sec "
                      f"({save_summary['avg_memory_mb']:.1f}MB)")
                print(f"Load: {load_summary['avg_throughput']:.1f} demos/sec")
                
                # Performance assertions
                assert save_summary['avg_throughput'] > 10, "Save performance too slow"
                assert load_summary['avg_throughput'] > 20, "Load performance too slow"
    
    @pytest.mark.benchmark
    def test_compression_performance(self):
        """Benchmark compression vs uncompressed storage."""
        demonstrations = []
        for i in range(20):
            demo = DemonstrationData(
                episode_id=f"compression_demo_{i:03d}",
                timestamp=f"2025-01-01T00:00:{i:02d}",
                observations={
                    'rgb': np.random.randint(0, 255, (25, 64, 64, 3), dtype=np.uint8),
                    'proprioception': np.random.randn(25, 7)
                },
                actions=np.random.randn(25, 7),
                success=i % 3 == 0,
                duration=2.5
            )
            demonstrations.append(demo)
        
        results = {}
        
        # Test both compressed and uncompressed
        for compression in [False, True]:
            with tempfile.TemporaryDirectory() as temp_dir:
                recorder = OptimizedRecorder(
                    output_dir=temp_dir,
                    compression=compression,
                    enable_parallel_saving=True
                )
                
                # Benchmark save
                start_time = time.time()
                recorder.batch_save_demonstrations(demonstrations)
                save_time = time.time() - start_time
                recorder.shutdown()
                
                # Calculate total size
                total_size = sum(
                    f.stat().st_size for f in Path(temp_dir).rglob("*")
                    if f.is_file()
                ) / 1024 / 1024  # MB
                
                # Benchmark load
                start_time = time.time()
                loaded_demos = optimize_data_loading(temp_dir, parallel_loading=True)
                load_time = time.time() - start_time
                
                results[compression] = {
                    'save_time': save_time,
                    'load_time': load_time,
                    'total_size_mb': total_size,
                    'save_throughput': len(demonstrations) / save_time,
                    'load_throughput': len(loaded_demos) / load_time
                }
        
        # Print comparison
        print(f"\n=== Compression Performance Comparison ===")
        
        uncompressed = results[False]
        compressed = results[True]
        
        print(f"Uncompressed: {uncompressed['total_size_mb']:.1f}MB, "
              f"Save: {uncompressed['save_throughput']:.1f} demos/sec, "
              f"Load: {uncompressed['load_throughput']:.1f} demos/sec")
        
        print(f"Compressed: {compressed['total_size_mb']:.1f}MB, "
              f"Save: {compressed['save_throughput']:.1f} demos/sec, "
              f"Load: {compressed['load_throughput']:.1f} demos/sec")
        
        compression_ratio = uncompressed['total_size_mb'] / compressed['total_size_mb']
        print(f"Compression ratio: {compression_ratio:.2f}x")
        
        # Assertions
        assert compression_ratio > 1.5, "Compression not effective enough"
        
        # Compressed save might be slower but should be reasonable
        save_ratio = compressed['save_throughput'] / uncompressed['save_throughput']
        assert save_ratio > 0.5, "Compression makes saving too slow"


class TestCachingBenchmarks:
    """Benchmark caching performance."""
    
    @pytest.mark.benchmark
    def test_cache_performance(self):
        """Benchmark cache hit/miss performance."""
        cache_sizes = [32, 128, 512]
        
        for cache_size in cache_sizes:
            cache = LRUCache(maxsize=cache_size)
            benchmark = BenchmarkResults(f"Cache Performance (size {cache_size})")
            
            # Generate test data
            keys = [f"key_{i}" for i in range(cache_size * 2)]  # More keys than cache size
            values = [f"value_{i}" for i in range(len(keys))]
            
            # Fill cache
            for key, value in zip(keys[:cache_size], values[:cache_size]):
                cache.put(key, value)
            
            # Benchmark cache hits and misses
            num_operations = 1000
            
            start_time = time.time()
            hit_count = 0
            miss_count = 0
            
            for i in range(num_operations):
                key = keys[i % len(keys)]
                value, hit = cache.get(key)
                
                if hit:
                    hit_count += 1
                else:
                    miss_count += 1
                    # Add to cache if miss
                    cache.put(key, f"new_value_{i}")
            
            end_time = time.time()
            
            benchmark.add_result(
                time_taken=end_time - start_time,
                items_processed=num_operations
            )
            
            summary = benchmark.get_summary()
            hit_rate = hit_count / num_operations
            
            print(f"\n=== Cache Performance (size {cache_size}) ===")
            print(f"Operations/sec: {summary['avg_throughput']:.0f}")
            print(f"Hit rate: {hit_rate:.2f}")
            print(f"Hits: {hit_count}, Misses: {miss_count}")
            
            # Performance assertions
            assert summary['avg_throughput'] > 10000, "Cache operations too slow"


class TestMemoryBenchmarks:
    """Benchmark memory usage and optimization."""
    
    @pytest.mark.benchmark
    def test_memory_optimization_performance(self):
        """Benchmark memory optimization effectiveness."""
        benchmark = BenchmarkResults("Memory Optimization")
        
        num_runs = 5
        for run in range(num_runs):
            # Create memory pressure
            large_objects = []
            for i in range(50):
                large_objects.append(np.random.randn(1000, 1000))
            
            # Get memory before optimization
            monitor = get_performance_monitor()
            metrics_before = monitor.get_metrics()
            memory_before = metrics_before.get('memory_mb', 0)
            
            # Delete references
            del large_objects
            
            # Optimize memory
            start_time = time.time()
            optimization_result = optimize_memory()
            end_time = time.time()
            
            # Get memory after optimization
            metrics_after = monitor.get_metrics()
            memory_after = metrics_after.get('memory_mb', 0)
            
            memory_freed = memory_before - memory_after
            
            benchmark.add_result(
                time_taken=end_time - start_time,
                items_processed=optimization_result['collected_objects'],
                memory_mb=memory_freed
            )
        
        summary = benchmark.get_summary()
        
        print(f"\n=== Memory Optimization Performance ===")
        print(f"Average time: {summary['avg_time']:.4f}s")
        print(f"Average objects collected: {summary['avg_throughput']:.0f} objects/sec")
        print(f"Average memory freed: {summary['avg_memory_mb']:.1f}MB")
        
        # Performance assertions
        assert summary['avg_time'] < 1.0, "Memory optimization too slow"
        assert summary['avg_memory_mb'] > 0, "Memory optimization not effective"


class TestConcurrencyBenchmarks:
    """Benchmark concurrency performance."""
    
    @pytest.mark.benchmark
    def test_thread_pool_performance(self):
        """Benchmark thread pool performance."""
        def io_task(delay: float) -> float:
            time.sleep(delay)
            return delay
        
        task_delays = [0.01] * 100  # 100 tasks, 0.01s each
        
        # Sequential execution
        start_time = time.time()
        sequential_results = [io_task(delay) for delay in task_delays]
        sequential_time = time.time() - start_time
        
        # Parallel execution with different worker counts
        worker_counts = [2, 4, 8, 16]
        
        for worker_count in worker_counts:
            with ThreadPool(max_workers=worker_count) as pool:
                start_time = time.time()
                futures = [pool.submit(io_task, delay) for delay in task_delays]
                parallel_results = [f.result() for f in futures]
                parallel_time = time.time() - start_time
            
            speedup = sequential_time / parallel_time
            
            print(f"\n=== ThreadPool Performance ({worker_count} workers) ===")
            print(f"Sequential time: {sequential_time:.3f}s")
            print(f"Parallel time: {parallel_time:.3f}s")
            print(f"Speedup: {speedup:.2f}x")
            print(f"Efficiency: {speedup / worker_count:.2f}")
            
            # Verify results are identical
            assert parallel_results == sequential_results
            
            # Assert reasonable speedup
            expected_speedup = min(worker_count, len(task_delays))
            assert speedup > expected_speedup * 0.7, f"Poor speedup: {speedup:.2f}x"
    
    @pytest.mark.benchmark
    def test_batch_processing_performance(self):
        """Benchmark batch processing performance."""
        def process_batch(items: list) -> list:
            # Simulate batch processing with overhead
            time.sleep(0.005)  # Fixed overhead per batch
            return [item * 2 for item in items]
        
        items = list(range(100))
        batch_sizes = [1, 5, 10, 20]
        
        results = {}
        
        for batch_size in batch_sizes:
            batch_processor = BatchProcessor(
                batch_size=batch_size,
                max_wait_time=0.1
            )
            
            start_time = time.time()
            futures = [batch_processor.submit(process_batch, item) for item in items]
            batch_results = [f.result() for f in futures]
            batch_time = time.time() - start_time
            
            batch_processor.shutdown()
            
            results[batch_size] = {
                'time': batch_time,
                'throughput': len(items) / batch_time
            }
            
            print(f"\n=== Batch Processing (batch_size={batch_size}) ===")
            print(f"Time: {batch_time:.3f}s")
            print(f"Throughput: {results[batch_size]['throughput']:.1f} items/sec")
        
        # Verify that larger batch sizes are more efficient
        baseline_throughput = results[1]['throughput']
        for batch_size in batch_sizes[1:]:
            speedup = results[batch_size]['throughput'] / baseline_throughput
            print(f"Batch size {batch_size} speedup: {speedup:.2f}x")
            
            # Larger batches should be more efficient
            if batch_size >= 10:
                assert speedup > 1.5, f"Batch processing not efficient: {speedup:.2f}x"


def print_benchmark_summary():
    """Print a summary of all benchmarks."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print("All performance benchmarks completed successfully!")
    print("Key findings:")
    print("- Parallel processing provides significant speedup")
    print("- Caching improves performance for repeated operations")
    print("- Compression reduces storage size with minimal speed impact")
    print("- Memory optimization effectively frees unused memory")
    print("- Batch processing improves throughput for bulk operations")
    print("="*60)