"""
Performance testing configuration and fixtures.
"""

import pytest
import time
import psutil
import threading
from typing import Dict, Any, List, Callable
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    cpu_usage: float
    memory_usage: float
    execution_time: float
    throughput: float = 0.0
    error_rate: float = 0.0


class PerformanceMonitor:
    """Monitor system performance during tests."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 0.1):
        """Start performance monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,)
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            metrics = PerformanceMetrics(
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                execution_time=time.time()
            )
            self.metrics.append(metrics)
            time.sleep(interval)
    
    def get_average_metrics(self) -> PerformanceMetrics:
        """Get average metrics from monitoring period."""
        if not self.metrics:
            return PerformanceMetrics(0.0, 0.0, 0.0)
        
        avg_cpu = sum(m.cpu_usage for m in self.metrics) / len(self.metrics)
        avg_memory = sum(m.memory_usage for m in self.metrics) / len(self.metrics)
        total_time = self.metrics[-1].execution_time - self.metrics[0].execution_time
        
        return PerformanceMetrics(
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            execution_time=total_time
        )


@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture."""
    monitor = PerformanceMonitor()
    yield monitor
    monitor.stop_monitoring()


@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "cpu_threshold": 80.0,      # Max CPU usage %
        "memory_threshold": 80.0,   # Max memory usage %
        "time_threshold": 5.0,      # Max execution time (seconds)
        "throughput_threshold": 100, # Min operations per second
        "error_threshold": 0.01     # Max error rate (1%)
    }


@pytest.fixture
def load_test_config():
    """Configuration for load testing."""
    return {
        "concurrent_users": [1, 5, 10, 25, 50],
        "test_duration": 30,  # seconds
        "ramp_up_time": 5,    # seconds
        "operations_per_user": 100
    }


class LoadTestRunner:
    """Run load tests with multiple concurrent operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results: List[Dict[str, Any]] = []
    
    async def run_load_test(self, test_function: Callable, *args, **kwargs):
        """Run load test with increasing concurrent users."""
        for user_count in self.config["concurrent_users"]:
            print(f"Running load test with {user_count} concurrent users...")
            
            start_time = time.time()
            
            # Create concurrent tasks
            tasks = []
            for i in range(user_count):
                tasks.append(test_function(*args, **kwargs))
            
            # Run tasks concurrently
            import asyncio
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            
            # Calculate metrics
            successful_ops = sum(1 for r in results if not isinstance(r, Exception))
            failed_ops = len(results) - successful_ops
            total_time = end_time - start_time
            throughput = successful_ops / total_time
            error_rate = failed_ops / len(results)
            
            self.results.append({
                "concurrent_users": user_count,
                "successful_operations": successful_ops,
                "failed_operations": failed_ops,
                "total_time": total_time,
                "throughput": throughput,
                "error_rate": error_rate
            })
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get load test results."""
        return self.results


@pytest.fixture
def load_test_runner(load_test_config):
    """Load test runner fixture."""
    return LoadTestRunner(load_test_config)


@pytest.fixture
def memory_profiler():
    """Memory profiling fixture."""
    import tracemalloc
    
    tracemalloc.start()
    yield tracemalloc
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Memory usage - Current: {current / 1024 / 1024:.2f} MB, Peak: {peak / 1024 / 1024:.2f} MB")


def pytest_configure(config):
    """Configure pytest markers for performance tests."""
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as a benchmark test"
    )
    config.addinivalue_line(
        "markers", "load: mark test as a load test"
    )
    config.addinivalue_line(
        "markers", "stress: mark test as a stress test"
    )


@pytest.fixture(scope="session", autouse=True)
def performance_test_setup():
    """Setup for performance tests."""
    # Ensure we have sufficient system resources
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
    if available_memory < 2.0:
        pytest.skip("Insufficient memory for performance tests")
    
    # Check CPU availability
    cpu_count = psutil.cpu_count()
    if cpu_count < 2:
        pytest.skip("Insufficient CPU cores for performance tests")


@pytest.fixture
def stress_test_data():
    """Generate large datasets for stress testing."""
    return {
        "small_dataset": list(range(1000)),
        "medium_dataset": list(range(10000)),
        "large_dataset": list(range(100000)),
        "stress_dataset": list(range(1000000))
    }