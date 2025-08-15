"""
Performance benchmark tests for self-healing pipeline components.

Tests performance characteristics, scalability, and resource usage
of pipeline components under various load conditions.
"""

import asyncio
import pytest
import time
import statistics
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import psutil
import sys

from robo_rlhf.pipeline import (
    PipelineGuard, PipelineComponent, MetricsCollector,
    IntelligentCache, AutoScaler, LoadBalancer
)


class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self, name: str):
        self.name = name
        self.results: Dict[str, Any] = {}
    
    def measure_time(self, func_name: str):
        """Decorator to measure execution time."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = await func(*args, **kwargs)
                end_time = time.perf_counter()
                
                execution_time = end_time - start_time
                if func_name not in self.results:
                    self.results[func_name] = []
                self.results[func_name].append(execution_time)
                
                return result
            return wrapper
        return decorator
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics."""
        stats = {}
        
        for operation, times in self.results.items():
            if times:
                stats[operation] = {
                    "count": len(times),
                    "mean": statistics.mean(times),
                    "median": statistics.median(times),
                    "min": min(times),
                    "max": max(times),
                    "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
                    "p95": times[int(0.95 * len(times))] if len(times) > 1 else times[0],
                    "p99": times[int(0.99 * len(times))] if len(times) > 1 else times[0]
                }
        
        return stats


class TestMetricsCollectorPerformance(PerformanceBenchmark):
    """Performance tests for MetricsCollector."""
    
    def __init__(self):
        super().__init__("MetricsCollector")
        self.collector = MetricsCollector(max_points_per_metric=10000)
    
    @pytest.mark.benchmark
    def test_metric_recording_throughput(self):
        """Test throughput of metric recording operations."""
        
        @self.measure_time("record_single_metric")
        async def record_single_metric():
            self.collector.record_metric("test_metric", 1.0, {"tag": "value"})
        
        # Record metrics sequentially
        start_time = time.perf_counter()
        
        for i in range(10000):
            self.collector.record_metric(f"metric_{i % 100}", float(i), {"iteration": str(i)})
        
        end_time = time.perf_counter()
        
        throughput = 10000 / (end_time - start_time)
        
        # Should handle at least 50k metrics per second
        assert throughput > 50000, f"Throughput too low: {throughput:.0f} metrics/sec"
        
        print(f"Metrics recording throughput: {throughput:.0f} metrics/sec")
    
    @pytest.mark.benchmark
    def test_batch_recording_performance(self):
        """Test performance of batch metric recording."""
        batch_sizes = [10, 100, 1000]
        
        for batch_size in batch_sizes:
            metrics_batch = [
                {
                    "name": f"batch_metric_{i}",
                    "value": float(i),
                    "tags": {"batch_size": str(batch_size), "index": str(i)}
                }
                for i in range(batch_size)
            ]
            
            start_time = time.perf_counter()
            self.collector.record_batch(metrics_batch)
            end_time = time.perf_counter()
            
            batch_time = end_time - start_time
            
            print(f"Batch size {batch_size}: {batch_time:.4f}s ({batch_size/batch_time:.0f} metrics/sec)")
            
            # Batch recording should be efficient
            assert batch_time < 0.1, f"Batch recording too slow for size {batch_size}"
    
    @pytest.mark.benchmark
    def test_metric_retrieval_performance(self):
        """Test performance of metric retrieval operations."""
        # Populate with test data
        for i in range(1000):
            self.collector.record_metric("retrieval_test", float(i), {"index": str(i)})
        
        # Test latest metrics retrieval
        start_time = time.perf_counter()
        
        for _ in range(1000):
            latest = self.collector.get_latest_metrics(["retrieval_test"])
        
        end_time = time.perf_counter()
        
        retrieval_time = end_time - start_time
        retrieval_rate = 1000 / retrieval_time
        
        print(f"Metric retrieval rate: {retrieval_rate:.0f} retrievals/sec")
        
        # Should handle at least 10k retrievals per second
        assert retrieval_rate > 10000, f"Retrieval rate too low: {retrieval_rate:.0f}/sec"
    
    @pytest.mark.benchmark
    def test_summary_calculation_performance(self):
        """Test performance of metric summary calculations."""
        # Populate with substantial data
        for i in range(5000):
            self.collector.record_metric("summary_test", float(i % 100), {"index": str(i)})
        
        start_time = time.perf_counter()
        
        for _ in range(100):
            summary = self.collector.get_metric_summary("summary_test")
        
        end_time = time.perf_counter()
        
        summary_time = end_time - start_time
        summary_rate = 100 / summary_time
        
        print(f"Summary calculation rate: {summary_rate:.0f} summaries/sec")
        
        # Should handle at least 100 summaries per second
        assert summary_rate > 100, f"Summary calculation too slow: {summary_rate:.0f}/sec"


class TestCachePerformance(PerformanceBenchmark):
    """Performance tests for caching system."""
    
    def __init__(self):
        super().__init__("IntelligentCache")
        self.cache = IntelligentCache("performance_test", max_size_mb=100)
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_cache_write_performance(self):
        """Test cache write performance."""
        data_sizes = [
            ("small", "x" * 100),      # 100 bytes
            ("medium", "x" * 10000),   # 10KB
            ("large", "x" * 1000000)   # 1MB
        ]
        
        for size_name, data in data_sizes:
            times = []
            
            for i in range(100):
                start_time = time.perf_counter()
                await self.cache.set(f"{size_name}_key_{i}", data)
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
            
            avg_time = statistics.mean(times)
            throughput = 1.0 / avg_time
            
            print(f"Cache write ({size_name}): {avg_time:.4f}s avg, {throughput:.0f} ops/sec")
            
            # Performance requirements
            if size_name == "small":
                assert throughput > 50000, f"Small write throughput too low: {throughput:.0f}/sec"
            elif size_name == "medium":
                assert throughput > 1000, f"Medium write throughput too low: {throughput:.0f}/sec"
            elif size_name == "large":
                assert throughput > 100, f"Large write throughput too low: {throughput:.0f}/sec"
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_cache_read_performance(self):
        """Test cache read performance."""
        # Populate cache
        test_data = {
            f"read_key_{i}": f"test_value_{i}" * 100  # ~1KB per entry
            for i in range(1000)
        }
        
        for key, value in test_data.items():
            await self.cache.set(key, value)
        
        # Test read performance
        keys = list(test_data.keys())
        times = []
        
        for _ in range(10000):
            key = keys[_ % len(keys)]
            
            start_time = time.perf_counter()
            value = await self.cache.get(key)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            assert value is not None  # Should be cache hit
        
        avg_time = statistics.mean(times)
        throughput = 1.0 / avg_time
        
        print(f"Cache read performance: {avg_time:.6f}s avg, {throughput:.0f} ops/sec")
        
        # Should handle at least 100k reads per second
        assert throughput > 100000, f"Read throughput too low: {throughput:.0f}/sec"
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_cache_eviction_performance(self):
        """Test cache eviction performance under memory pressure."""
        # Fill cache to capacity
        large_data = "x" * 10000  # 10KB per entry
        
        # This should trigger evictions
        start_time = time.perf_counter()
        
        for i in range(2000):  # More than cache capacity
            await self.cache.set(f"eviction_key_{i}", large_data)
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time_per_set = total_time / 2000
        
        print(f"Cache with eviction: {avg_time_per_set:.4f}s avg per set")
        
        # Even with evictions, should maintain reasonable performance
        assert avg_time_per_set < 0.001, f"Eviction performance too slow: {avg_time_per_set:.4f}s"
        
        # Verify cache stayed within size limits
        cache_size_mb = self.cache.get_size_mb()
        assert cache_size_mb <= 110, f"Cache exceeded size limit: {cache_size_mb:.1f}MB"  # Allow 10% overhead


class TestHealthCheckPerformance(PerformanceBenchmark):
    """Performance tests for health checking operations."""
    
    def __init__(self):
        super().__init__("HealthCheck")
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self):
        """Test performance of concurrent health checks."""
        # Create multiple mock components
        async def fast_health_check():
            await asyncio.sleep(0.001)  # 1ms simulated check time
            return {"status": "ok", "response_time": 0.001}
        
        async def slow_health_check():
            await asyncio.sleep(0.1)  # 100ms simulated check time
            return {"status": "ok", "response_time": 0.1}
        
        components = []
        
        # Mix of fast and slow components
        for i in range(50):
            health_check = fast_health_check if i % 2 == 0 else slow_health_check
            component = PipelineComponent(
                name=f"service_{i}",
                endpoint=f"http://service_{i}:8080",
                health_check=health_check
            )
            components.append(component)
        
        guard = PipelineGuard(components=components, check_interval=60)
        
        # Test concurrent health checks
        start_time = time.perf_counter()
        
        check_tasks = [
            guard._check_component_health(comp.name, comp)
            for comp in components
        ]
        
        reports = await asyncio.gather(*check_tasks)
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        
        print(f"Concurrent health checks: {total_time:.3f}s for {len(components)} components")
        print(f"Average time per component: {total_time/len(components):.4f}s")
        
        # Concurrent execution should be much faster than sequential
        assert total_time < 2.0, f"Concurrent health checks too slow: {total_time:.3f}s"
        
        # All reports should be successful
        assert len(reports) == len(components)
        for report in reports:
            assert report.status.value in ["healthy", "degraded"]  # Allow some variation
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_health_check_with_failures(self):
        """Test performance when some health checks fail."""
        async def failing_health_check():
            if hasattr(failing_health_check, 'call_count'):
                failing_health_check.call_count += 1
            else:
                failing_health_check.call_count = 1
            
            # Fail every 3rd call
            if failing_health_check.call_count % 3 == 0:
                raise Exception("Simulated failure")
            
            return {"status": "ok"}
        
        components = [
            PipelineComponent(
                name=f"failing_service_{i}",
                endpoint=f"http://failing_{i}:8080",
                health_check=failing_health_check
            )
            for i in range(20)
        ]
        
        guard = PipelineGuard(components=components)
        
        # Run multiple health check cycles
        cycle_times = []
        
        for cycle in range(10):
            start_time = time.perf_counter()
            
            check_tasks = [
                guard._check_component_health(comp.name, comp)
                for comp in components
            ]
            
            reports = await asyncio.gather(*check_tasks)
            
            end_time = time.perf_counter()
            cycle_times.append(end_time - start_time)
        
        avg_cycle_time = statistics.mean(cycle_times)
        
        print(f"Health check cycles with failures: {avg_cycle_time:.3f}s average")
        
        # Even with failures, should maintain reasonable performance
        assert avg_cycle_time < 1.0, f"Health check with failures too slow: {avg_cycle_time:.3f}s"


class TestScalingPerformance(PerformanceBenchmark):
    """Performance tests for auto-scaling operations."""
    
    def __init__(self):
        super().__init__("AutoScaler")
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_scaling_decision_performance(self):
        """Test performance of scaling decision algorithms."""
        from robo_rlhf.pipeline.scaling import AutoScaler, ScalingStrategy, ScalingRule, ResourceType
        
        scaler = AutoScaler("test_service", ScalingStrategy.HYBRID)
        
        # Add multiple scaling rules
        rules = [
            ScalingRule("cpu_usage", 0.8, 0.2, resource_type=ResourceType.CPU),
            ScalingRule("memory_usage", 0.85, 0.3, resource_type=ResourceType.MEMORY),
            ScalingRule("response_time", 2.0, 0.5, resource_type=ResourceType.INSTANCES),
            ScalingRule("error_rate", 0.05, 0.01, resource_type=ResourceType.INSTANCES)
        ]
        
        for rule in rules:
            scaler.add_scaling_rule(rule)
        
        # Populate with metric history
        current_time = time.time()
        for i in range(1000):
            timestamp = current_time - (1000 - i) * 60  # 1000 minutes of history
            
            scaler.update_metric("cpu_usage", 0.3 + (i % 100) / 200, timestamp)
            scaler.update_metric("memory_usage", 0.4 + (i % 80) / 160, timestamp)
            scaler.update_metric("response_time", 0.5 + (i % 60) / 120, timestamp)
            scaler.update_metric("error_rate", 0.01 + (i % 40) / 4000, timestamp)
        
        # Test scaling evaluation performance
        times = []
        
        for _ in range(100):
            start_time = time.perf_counter()
            scaling_event = await scaler.evaluate_scaling()
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
        
        avg_time = statistics.mean(times)
        
        print(f"Scaling decision time: {avg_time:.4f}s average")
        
        # Scaling decisions should be fast
        assert avg_time < 0.01, f"Scaling decision too slow: {avg_time:.4f}s"
    
    @pytest.mark.benchmark
    def test_load_balancer_routing_performance(self):
        """Test load balancer routing performance."""
        from robo_rlhf.pipeline.scaling import LoadBalancer
        
        balancer = LoadBalancer("test_service")
        
        # Add multiple instances
        for i in range(10):
            balancer.add_instance(
                f"instance_{i}",
                f"http://instance_{i}:8080",
                weight=1.0 + i * 0.1
            )
        
        # Test routing performance
        start_time = time.perf_counter()
        
        for _ in range(10000):
            selected = asyncio.run(balancer.route_request({"user_id": "test"}))
            assert selected is not None
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        routing_rate = 10000 / total_time
        
        print(f"Load balancer routing rate: {routing_rate:.0f} routes/sec")
        
        # Should handle at least 50k routes per second
        assert routing_rate > 50000, f"Routing rate too low: {routing_rate:.0f}/sec"


class TestMemoryUsage:
    """Memory usage tests for pipeline components."""
    
    @pytest.mark.benchmark
    def test_metrics_collector_memory_usage(self):
        """Test memory usage of MetricsCollector with large datasets."""
        collector = MetricsCollector(max_points_per_metric=100000)
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Add large number of metrics
        for i in range(100000):
            collector.record_metric(
                f"metric_{i % 1000}",
                float(i),
                tags={"iteration": str(i), "batch": str(i // 1000)}
            )
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage for 100k metrics: {memory_increase:.1f}MB")
        
        # Should use reasonable amount of memory
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.1f}MB"
        
        # Test memory cleanup
        collector._cleanup_old_metrics()
        
        # Memory should not grow unbounded
        stats = collector.get_stats()
        assert stats["memory_usage_mb"] < 200, f"Memory not properly managed: {stats['memory_usage_mb']:.1f}MB"
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_cache_memory_efficiency(self):
        """Test memory efficiency of caching system."""
        cache = IntelligentCache("memory_test", max_size_mb=50)
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Fill cache with data
        data_size = 1024  # 1KB per entry
        test_data = "x" * data_size
        
        for i in range(1000):
            await cache.set(f"memory_key_{i}", test_data)
        
        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Get cache reported size
        cache_size = cache.get_size_mb()
        
        print(f"Cache reported size: {cache_size:.1f}MB")
        print(f"Actual memory increase: {memory_increase:.1f}MB")
        
        # Memory overhead should be reasonable (within 2x of reported size)
        assert memory_increase < cache_size * 2, f"Memory overhead too high: {memory_increase:.1f}MB vs {cache_size:.1f}MB"


@pytest.mark.benchmark
class TestConcurrencyPerformance:
    """Test performance under concurrent load."""
    
    @pytest.mark.asyncio
    async def test_concurrent_pipeline_operations(self):
        """Test pipeline performance under concurrent operations."""
        # Create components with varying response times
        async def variable_health_check():
            delay = 0.001 + (time.time() % 100) / 100000  # 1-10ms
            await asyncio.sleep(delay)
            return {"status": "ok", "response_time": delay}
        
        components = [
            PipelineComponent(
                name=f"concurrent_service_{i}",
                endpoint=f"http://service_{i}:8080",
                health_check=variable_health_check
            )
            for i in range(100)
        ]
        
        guard = PipelineGuard(components=components, check_interval=1)
        
        # Run concurrent operations
        start_time = time.perf_counter()
        
        # Multiple monitoring cycles in parallel
        cycle_tasks = []
        for cycle in range(10):
            check_tasks = [
                guard._check_component_health(comp.name, comp)
                for comp in components
            ]
            cycle_task = asyncio.gather(*check_tasks)
            cycle_tasks.append(cycle_task)
        
        all_results = await asyncio.gather(*cycle_tasks)
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        total_checks = len(components) * 10  # 100 components * 10 cycles
        
        print(f"Concurrent operations: {total_checks} checks in {total_time:.3f}s")
        print(f"Throughput: {total_checks/total_time:.0f} checks/sec")
        
        # Should handle high concurrency efficiently
        assert total_time < 5.0, f"Concurrent operations too slow: {total_time:.3f}s"
        
        # Verify all checks completed successfully
        for cycle_results in all_results:
            assert len(cycle_results) == len(components)


def run_performance_suite():
    """Run the complete performance benchmark suite."""
    print("=== Performance Benchmark Suite ===")
    
    benchmarks = [
        TestMetricsCollectorPerformance(),
        TestCachePerformance(),
        TestHealthCheckPerformance(),
        TestScalingPerformance()
    ]
    
    for benchmark in benchmarks:
        print(f"\n--- {benchmark.name} Performance ---")
        
        # Run benchmark methods
        for method_name in dir(benchmark):
            if method_name.startswith("test_") and hasattr(getattr(benchmark, method_name), "__pytest_marks__"):
                method = getattr(benchmark, method_name)
                if any(mark.name == "benchmark" for mark in method.__pytest_marks__):
                    print(f"Running {method_name}...")
                    try:
                        if asyncio.iscoroutinefunction(method):
                            asyncio.run(method())
                        else:
                            method()
                    except Exception as e:
                        print(f"  ERROR: {e}")
        
        # Print statistics
        stats = benchmark.get_statistics()
        if stats:
            print(f"\n{benchmark.name} Statistics:")
            for operation, metrics in stats.items():
                print(f"  {operation}:")
                print(f"    Mean: {metrics['mean']:.4f}s")
                print(f"    P95:  {metrics['p95']:.4f}s")
                print(f"    P99:  {metrics['p99']:.4f}s")


if __name__ == "__main__":
    # Run with pytest for proper test discovery and reporting
    pytest.main([__file__, "-v", "-m", "benchmark"])