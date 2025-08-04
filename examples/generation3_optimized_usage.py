#!/usr/bin/env python3
"""
Generation 3 Optimized Usage Example - MAKE IT SCALE

Demonstrates the performance optimization, caching, concurrent processing,
and resource pooling features of robo-rlhf-multimodal.
"""

import time
import tempfile
import numpy as np
from pathlib import Path
from concurrent.futures import as_completed
import asyncio

# Import optimization components
from robo_rlhf.core.performance import (
    get_performance_monitor, timer, measure_time, cached,
    ThreadPool, ProcessPool, BatchProcessor, ResourcePool,
    optimize_memory
)
from robo_rlhf.collectors.optimized import (
    OptimizedDataProcessor, StreamingDataCollector, OptimizedRecorder,
    optimize_data_loading
)
from robo_rlhf.collectors.base import DemonstrationData
from robo_rlhf.core.logging import get_logger


logger = get_logger(__name__)


def test_performance_monitoring():
    """Test the performance monitoring system."""
    print("\n📊 Testing Performance Monitoring")
    print("=" * 40)
    
    perf_monitor = get_performance_monitor()
    
    # Test timing operations
    with timer("test_operation"):
        time.sleep(0.1)  # Simulate work
        
        # Increment counters
        perf_monitor.increment_counter("operations_completed", 5)
        perf_monitor.increment_counter("items_processed", 100)
    
    # Get performance metrics
    metrics = perf_monitor.get_metrics()
    
    print("✅ Performance monitoring active")
    print(f"   Test operation time: {metrics.get('test_operation_avg', 0):.4f}s")
    print(f"   Operations completed: {metrics.get('operations_completed', 0)}")
    print(f"   Items processed: {metrics.get('items_processed', 0)}")
    print(f"   Memory usage: {metrics.get('memory_mb', 0):.1f}MB")
    print(f"   CPU usage: {metrics.get('cpu_percent', 0):.1f}%")
    
    return metrics


def test_caching_system():
    """Test the caching system."""
    print("\n🗄️ Testing Caching System")
    print("=" * 40)
    
    @cached(maxsize=32, ttl=60)
    def expensive_computation(x: int, y: int) -> int:
        """Simulate expensive computation."""
        time.sleep(0.01)  # Simulate work
        return x * y + x ** 2
    
    # Test cache performance
    test_inputs = [(i, i+1) for i in range(10)]
    
    # First run - cache misses
    start_time = time.time()
    results1 = [expensive_computation(x, y) for x, y in test_inputs]
    first_run_time = time.time() - start_time
    
    # Second run - cache hits
    start_time = time.time()
    results2 = [expensive_computation(x, y) for x, y in test_inputs]
    second_run_time = time.time() - start_time
    
    # Verify results are identical
    assert results1 == results2
    
    # Check cache statistics
    cache_stats = expensive_computation.cache_stats()
    
    print("✅ Caching system functional")
    print(f"   First run (cache misses): {first_run_time:.4f}s")
    print(f"   Second run (cache hits): {second_run_time:.4f}s")
    print(f"   Speedup: {first_run_time / second_run_time:.1f}x")
    print(f"   Cache hit rate: {cache_stats['hit_rate']:.2f}")
    print(f"   Cache size: {cache_stats['size']}/{cache_stats['maxsize']}")
    
    return cache_stats


# Module-level functions for pickling compatibility
def cpu_intensive_task(n: int) -> int:
    """CPU-intensive task for testing."""
    total = 0
    for i in range(n * 1000):
        total += i ** 2
    return total

def io_intensive_task(delay: float) -> str:
    """I/O-intensive task for testing."""
    time.sleep(delay)
    return f"Task completed after {delay}s"

def test_parallel_processing():
    """Test parallel processing capabilities."""
    print("\n🚀 Testing Parallel Processing")
    print("=" * 40)
    
    # Test ThreadPool for I/O bound tasks
    print("   Testing ThreadPool (I/O bound tasks)...")
    with ThreadPool(max_workers=4) as thread_pool:
        start_time = time.time()
        
        # Submit I/O tasks
        futures = [thread_pool.submit(io_intensive_task, 0.1) for _ in range(8)]
        
        # Collect results
        results = [future.result() for future in futures]
        
        thread_time = time.time() - start_time
    
    # Test ProcessPool for CPU bound tasks
    print("   Testing ProcessPool (CPU bound tasks)...")
    with ProcessPool(max_workers=2) as process_pool:
        start_time = time.time()
        
        # Submit CPU tasks
        futures = [process_pool.submit(cpu_intensive_task, 100) for _ in range(4)]
        
        # Collect results
        results = [future.result() for future in futures]
        
        process_time = time.time() - start_time
    
    print("✅ Parallel processing functional")
    print(f"   ThreadPool (8 I/O tasks): {thread_time:.2f}s")
    print(f"   ProcessPool (4 CPU tasks): {process_time:.2f}s")
    
    return {"thread_time": thread_time, "process_time": process_time}


def test_batch_processing():
    """Test batch processing system."""
    print("\n📦 Testing Batch Processing")
    print("=" * 40)
    
    def process_batch(items: list) -> list:
        """Process a batch of items."""
        # Simulate batch processing advantage
        time.sleep(0.01)  # Fixed overhead per batch
        return [item * 2 for item in items]
    
    # Create batch processor
    batch_processor = BatchProcessor(batch_size=5, max_wait_time=0.5)
    
    # Submit items for batch processing
    items = list(range(20))
    futures = []
    
    start_time = time.time()
    
    for item in items:
        future = batch_processor.submit(process_batch, item)
        futures.append(future)
    
    # Collect results
    results = [future.result() for future in futures]
    
    batch_time = time.time() - start_time
    
    # Compare with sequential processing
    start_time = time.time()
    sequential_results = []
    for item in items:
        result = process_batch([item])
        sequential_results.extend(result)
    sequential_time = time.time() - start_time
    
    # Verify results are equivalent
    assert results == sequential_results
    
    print("✅ Batch processing functional")
    print(f"   Batch processing: {batch_time:.3f}s")
    print(f"   Sequential processing: {sequential_time:.3f}s")
    print(f"   Speedup: {sequential_time / batch_time:.1f}x")
    
    batch_processor.shutdown()
    
    return {"batch_time": batch_time, "sequential_time": sequential_time}


def test_resource_pool():
    """Test resource pooling system."""
    print("\n🏊 Testing Resource Pool")
    print("=" * 40)
    
    class ExpensiveResource:
        """Simulate an expensive resource like a database connection."""
        def __init__(self):
            time.sleep(0.05)  # Simulate expensive creation
            self.created_at = time.time()
            self.query_count = 0
        
        def query(self, data: str) -> str:
            """Simulate a query operation."""
            self.query_count += 1
            return f"Processed: {data} (query #{self.query_count})"
        
        def reset(self):
            """Reset resource state."""
            self.query_count = 0
    
    # Create resource pool
    resource_pool = ResourcePool(
        create_func=ExpensiveResource,
        reset_func=lambda r: r.reset(),
        max_size=3,
        timeout=5.0
    )
    
    def worker_task(task_id: int) -> str:
        """Worker task that uses pooled resource."""
        with resource_pool.get_resource() as resource:
            return resource.query(f"task_{task_id}")
    
    # Test concurrent resource usage
    start_time = time.time()
    
    with ThreadPool(max_workers=5) as pool:
        futures = [pool.submit(worker_task, i) for i in range(10)]
        results = [future.result() for future in futures]
    
    pool_time = time.time() - start_time
    
    # Get pool statistics
    pool_stats = resource_pool.stats()
    
    print("✅ Resource pool functional")
    print(f"   Processed 10 tasks in: {pool_time:.3f}s")
    print(f"   Pool size: {pool_stats['pool_size']}")
    print(f"   Created resources: {pool_stats['created_count']}")
    print(f"   Max pool size: {pool_stats['max_size']}")
    
    return pool_stats


def test_optimized_data_processing():
    """Test optimized data processing."""
    print("\n⚡ Testing Optimized Data Processing")
    print("=" * 40)
    
    # Create mock demonstration data
    def create_mock_demo(i: int) -> DemonstrationData:
        return DemonstrationData(
            episode_id=f"demo_{i:03d}",
            timestamp=f"2025-01-01T00:00:{i:02d}",
            observations={
                'rgb': np.random.randint(0, 255, (20, 64, 64, 3), dtype=np.uint8),
                'proprioception': np.random.randn(20, 7)
            },
            actions=np.random.randn(20, 7),
            success=np.random.choice([True, False]),
            duration=2.0
        )
    
    demonstrations = [create_mock_demo(i) for i in range(50)]
    
    # Create optimized processor
    processor = OptimizedDataProcessor(batch_size=10, max_workers=4)
    
    def process_demo(demo: DemonstrationData) -> dict:
        """Process a single demonstration."""
        # Simulate processing
        processed_obs = processor.preprocess_observations(demo.observations)
        return {
            'episode_id': demo.episode_id,
            'processed_frames': len(demo.actions),
            'success': demo.success
        }
    
    # Test parallel processing
    start_time = time.time()
    results = processor.process_demonstrations_parallel(demonstrations, process_demo)
    parallel_time = time.time() - start_time
    
    # Get processing statistics
    stats = processor.get_stats()
    
    print("✅ Optimized data processing functional")
    print(f"   Processed {len(results)} demonstrations")
    print(f"   Processing time: {parallel_time:.3f}s")
    print(f"   Throughput: {stats['throughput']:.1f} demos/sec")
    print(f"   Average time per demo: {stats['avg_processing_time']:.4f}s")
    print(f"   Cache hit rate: {stats['cache_stats']['hit_rate']:.2f}")
    
    processor.shutdown()
    
    return stats


def test_optimized_recording():
    """Test optimized demonstration recording."""
    print("\n💾 Testing Optimized Recording")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create optimized recorder
        recorder = OptimizedRecorder(
            output_dir=temp_dir,
            compression=True,
            enable_parallel_saving=True,
            save_thread_count=3
        )
        
        # Create test demonstrations
        demonstrations = []
        for i in range(20):
            demo = DemonstrationData(
                episode_id=f"optimized_demo_{i:03d}",
                timestamp=f"2025-01-01T00:00:{i:02d}",
                observations={
                    'rgb': np.random.randint(0, 255, (10, 32, 32, 3), dtype=np.uint8),
                    'proprioception': np.random.randn(10, 7)
                },
                actions=np.random.randn(10, 7),
                success=i % 3 == 0,  # Every 3rd is successful
                duration=1.0,
                metadata={'test_id': i}
            )
            demonstrations.append(demo)
        
        # Test batch saving
        start_time = time.time()
        recorder.batch_save_demonstrations(demonstrations)
        save_time = time.time() - start_time
        
        # Verify saved files
        saved_count = len(list(Path(temp_dir).glob("**/data.npz")))
        
        print("✅ Optimized recording functional")
        print(f"   Saved {saved_count} demonstrations")
        print(f"   Save time: {save_time:.3f}s")
        print(f"   Throughput: {saved_count / save_time:.1f} demos/sec")
        
        # Test optimized loading
        start_time = time.time()
        loaded_demos = optimize_data_loading(
            temp_dir,
            parallel_loading=True,
            cache_loaded_data=True,
            max_workers=4
        )
        load_time = time.time() - start_time
        
        print(f"   Loaded {len(loaded_demos)} demonstrations")
        print(f"   Load time: {load_time:.3f}s")
        print(f"   Load throughput: {len(loaded_demos) / load_time:.1f} demos/sec")
        
        # Verify data integrity
        assert len(loaded_demos) == len(demonstrations)
        
        recorder.shutdown()
        
        return {
            "saved_count": saved_count,
            "save_time": save_time,
            "load_time": load_time
        }


def test_memory_optimization():
    """Test memory optimization features."""
    print("\n🧠 Testing Memory Optimization")
    print("=" * 40)
    
    # Get initial memory usage
    initial_metrics = get_performance_monitor().get_metrics()
    initial_memory = initial_metrics.get('memory_mb', 0)
    
    # Create some large objects
    large_arrays = []
    for i in range(100):
        large_arrays.append(np.random.randn(1000, 1000))
    
    # Check memory after allocation
    mid_metrics = get_performance_monitor().get_metrics()
    mid_memory = mid_metrics.get('memory_mb', 0)
    
    # Delete references
    del large_arrays
    
    # Optimize memory
    optimization_result = optimize_memory()
    
    # Check final memory
    final_metrics = get_performance_monitor().get_metrics()
    final_memory = final_metrics.get('memory_mb', 0)
    
    print("✅ Memory optimization functional")
    print(f"   Initial memory: {initial_memory:.1f}MB")
    print(f"   After allocation: {mid_memory:.1f}MB")
    print(f"   After optimization: {final_memory:.1f}MB")
    print(f"   Memory freed: {mid_memory - final_memory:.1f}MB")
    print(f"   Objects collected: {optimization_result['collected_objects']}")
    
    return optimization_result


def main():
    """Main demonstration of Generation 3 optimization features."""
    print("⚡ Robo-RLHF-Multimodal - Generation 3 Demo")
    print("=" * 50)
    print("Testing performance optimization, caching, and concurrent processing...")
    
    try:
        # Test all optimization features
        perf_metrics = test_performance_monitoring()
        cache_stats = test_caching_system()
        parallel_stats = test_parallel_processing()
        batch_stats = test_batch_processing()
        pool_stats = test_resource_pool()
        processing_stats = test_optimized_data_processing()
        recording_stats = test_optimized_recording()
        memory_stats = test_memory_optimization()
        
        print("\n🎉 ALL OPTIMIZATION TESTS PASSED!")
        print("=" * 50)
        print("✅ Performance monitoring and metrics collection")
        print("✅ Intelligent caching with TTL support")
        print("✅ Parallel processing (ThreadPool + ProcessPool)")
        print("✅ Batch processing for improved throughput")
        print("✅ Resource pooling for expensive objects")
        print("✅ Optimized data processing with concurrency")
        print("✅ High-performance data recording and loading")
        print("✅ Memory optimization and garbage collection")
        
        # Summary statistics
        print("\n📈 Performance Summary:")
        print(f"   Cache speedup: {cache_stats.get('hit_rate', 0) * 100:.0f}% hit rate")
        print(f"   Parallel I/O speedup: ~{8 * 0.1 / parallel_stats['thread_time']:.1f}x")
        print(f"   Batch processing speedup: {batch_stats['sequential_time'] / batch_stats['batch_time']:.1f}x")
        print(f"   Data processing: {processing_stats['throughput']:.1f} demos/sec")
        print(f"   Recording throughput: {recording_stats['saved_count'] / recording_stats['save_time']:.1f} demos/sec")
        print(f"   Memory optimization: {memory_stats['collected_objects']} objects freed")
        
        print("\n📈 Generation 3 Status: COMPLETE")
        print("🏁 Ready for comprehensive testing and deployment!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during optimization testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())