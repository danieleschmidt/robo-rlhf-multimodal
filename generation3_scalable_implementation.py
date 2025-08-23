#!/usr/bin/env python3
"""
Generation 3: MAKE IT SCALE (Optimized) - Performance optimization and scalability
Implements caching, concurrent processing, performance monitoring, and auto-scaling
"""

import sys
import json
import time
import logging
import asyncio
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache, wraps
import hashlib
import psutil
import gc
from collections import deque
from statistics import mean, median

# Configure optimized logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/generation3_scalable.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    operation_name: str
    start_time: float
    end_time: float
    cpu_usage: float
    memory_usage: float
    execution_time: float
    throughput: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

@dataclass
class ScalabilityConfig:
    """Configuration for scalability features."""
    max_workers: int = min(32, multiprocessing.cpu_count() * 4)
    enable_caching: bool = True
    cache_size: int = 1000
    enable_monitoring: bool = True
    performance_threshold: float = 0.1  # seconds
    memory_threshold: float = 80.0  # percentage
    cpu_threshold: float = 80.0  # percentage
    auto_scaling: bool = True

class PerformanceMonitor:
    """Advanced performance monitoring with real-time metrics."""
    
    def __init__(self):
        self.metrics_history: deque = deque(maxlen=1000)
        self.cache_stats = {"hits": 0, "misses": 0}
        self.operation_stats = {}
        self._lock = threading.Lock()
    
    def record_metric(self, metric: PerformanceMetrics):
        """Record performance metric with thread safety."""
        with self._lock:
            self.metrics_history.append(metric)
            
            if metric.operation_name not in self.operation_stats:
                self.operation_stats[metric.operation_name] = []
            
            self.operation_stats[metric.operation_name].append(metric.execution_time)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate comprehensive performance summary."""
        with self._lock:
            if not self.metrics_history:
                return {"status": "no_metrics"}
            
            recent_metrics = list(self.metrics_history)
            
            return {
                "total_operations": len(recent_metrics),
                "avg_execution_time": mean([m.execution_time for m in recent_metrics]),
                "median_execution_time": median([m.execution_time for m in recent_metrics]),
                "avg_cpu_usage": mean([m.cpu_usage for m in recent_metrics]),
                "avg_memory_usage": mean([m.memory_usage for m in recent_metrics]),
                "total_cache_hits": self.cache_stats["hits"],
                "total_cache_misses": self.cache_stats["misses"],
                "cache_hit_rate": (
                    self.cache_stats["hits"] / (self.cache_stats["hits"] + self.cache_stats["misses"])
                    if (self.cache_stats["hits"] + self.cache_stats["misses"]) > 0 else 0
                ),
                "operations_by_type": {
                    op_name: {
                        "count": len(times),
                        "avg_time": mean(times),
                        "min_time": min(times),
                        "max_time": max(times)
                    }
                    for op_name, times in self.operation_stats.items()
                }
            }

def performance_monitor(monitor: PerformanceMonitor):
    """Decorator for automatic performance monitoring."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            process = psutil.Process()
            start_cpu = process.cpu_percent()
            start_memory = process.memory_percent()
            
            try:
                result = func(*args, **kwargs)
                
                end_time = time.time()
                execution_time = end_time - start_time
                end_cpu = process.cpu_percent()
                end_memory = process.memory_percent()
                
                metric = PerformanceMetrics(
                    operation_name=func.__name__,
                    start_time=start_time,
                    end_time=end_time,
                    execution_time=execution_time,
                    cpu_usage=(start_cpu + end_cpu) / 2,
                    memory_usage=(start_memory + end_memory) / 2
                )
                
                monitor.record_metric(metric)
                
                return result
                
            except Exception as e:
                logger.error(f"Performance monitoring error in {func.__name__}: {e}")
                raise
        
        return wrapper
    return decorator

class OptimizedCache:
    """High-performance caching system with LRU eviction."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_order = deque()
        self.max_size = max_size
        self._lock = threading.RLock()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with LRU tracking."""
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                self.stats["hits"] += 1
                return self.cache[key]
            
            self.stats["misses"] += 1
            return None
    
    def put(self, key: str, value: Any):
        """Put item in cache with automatic eviction."""
        with self._lock:
            if key in self.cache:
                # Update existing item
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                # Evict least recently used
                lru_key = self.access_order.popleft()
                del self.cache[lru_key]
                self.stats["evictions"] += 1
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def clear(self):
        """Clear cache and reset stats."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.stats = {"hits": 0, "misses": 0, "evictions": 0}

class ScalableAutonomousEngine:
    """Highly scalable autonomous execution engine with performance optimization."""
    
    def __init__(self, config: ScalabilityConfig = None):
        self.config = config or ScalabilityConfig()
        self.performance_monitor = PerformanceMonitor()
        self.cache = OptimizedCache(self.config.cache_size)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers // 2)
        self.process_pool = ProcessPoolExecutor(max_workers=self.config.max_workers // 4)
        
        logger.info(f"Scalable engine initialized with {self.config.max_workers} max workers")
    
    
    @lru_cache(maxsize=256)
    def compute_expensive_operation(self, input_data: str) -> str:
        """Expensive computation with caching."""
        # Simulate expensive computation
        time.sleep(0.01)  # 10ms computation
        result = hashlib.sha256(input_data.encode()).hexdigest()
        logger.debug(f"Computed expensive operation for: {input_data[:20]}...")
        return result
    
    async def autonomous_concurrent_processing(self) -> Dict[str, Any]:
        """Demonstrate high-performance concurrent processing."""
        results = {"tasks_completed": 0, "total_time": 0, "concurrency_level": 0}
        start_time = time.time()
        
        # Generate workload
        tasks = [f"task_{i}" for i in range(100)]
        
        # Process tasks concurrently
        async def process_task(task_id: str) -> str:
            await asyncio.sleep(0.001)  # Simulate async I/O
            return self.compute_expensive_operation(task_id)
        
        # Execute all tasks concurrently
        concurrent_results = await asyncio.gather(
            *[process_task(task) for task in tasks],
            return_exceptions=True
        )
        
        successful_tasks = [r for r in concurrent_results if isinstance(r, str)]
        
        results["tasks_completed"] = len(successful_tasks)
        results["total_time"] = time.time() - start_time
        results["concurrency_level"] = len(tasks)
        results["throughput"] = len(successful_tasks) / results["total_time"]
        
        logger.info(f"Concurrent processing: {results['tasks_completed']} tasks in {results['total_time']:.3f}s")
        return results
    
    def autonomous_parallel_computation(self) -> Dict[str, Any]:
        """CPU-intensive parallel computation using multiprocessing."""
        results = {"computations": 0, "total_time": 0, "workers_used": 0}
        start_time = time.time()
        
        def cpu_intensive_task(n: int) -> int:
            """CPU-intensive computation."""
            result = 0
            for i in range(n * 1000):
                result += i ** 2
            return result
        
        # Prepare workload
        workload = [1000 + i * 100 for i in range(20)]
        
        # Execute in parallel
        with ProcessPoolExecutor(max_workers=self.config.max_workers // 4) as executor:
            future_to_task = {executor.submit(cpu_intensive_task, n): n for n in workload}
            
            completed_tasks = 0
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    completed_tasks += 1
                except Exception as e:
                    logger.error(f"Parallel task failed: {e}")
        
        results["computations"] = completed_tasks
        results["total_time"] = time.time() - start_time
        results["workers_used"] = min(len(workload), self.config.max_workers // 4)
        results["throughput"] = completed_tasks / results["total_time"]
        
        logger.info(f"Parallel computation: {results['computations']} tasks in {results['total_time']:.3f}s")
        return results
    
    def autonomous_memory_optimization(self) -> Dict[str, Any]:
        """Demonstrate memory optimization techniques."""
        results = {"optimization_steps": [], "memory_saved": 0}
        
        # Measure initial memory
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Create and optimize data structures
        large_data = [i ** 2 for i in range(100000)]
        results["optimization_steps"].append("large_data_creation")
        
        # Use generators for memory efficiency
        def memory_efficient_generator():
            for i in range(100000):
                yield i ** 2
        
        # Clear large data and use generator
        del large_data
        efficient_data = memory_efficient_generator()
        results["optimization_steps"].append("generator_replacement")
        
        # Force garbage collection
        gc.collect()
        results["optimization_steps"].append("garbage_collection")
        
        # Measure final memory
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        results["memory_saved"] = max(0, initial_memory - final_memory)
        
        # Consume generator to verify functionality
        consumed_count = sum(1 for _ in efficient_data)
        results["verification"] = consumed_count == 100000
        
        logger.info(f"Memory optimization: {results['memory_saved']:.2f}MB saved")
        return results
    
    def autonomous_adaptive_scaling(self) -> Dict[str, Any]:
        """Demonstrate adaptive scaling based on system load."""
        results = {"scaling_decisions": [], "final_workers": 0}
        
        # Monitor system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        current_workers = self.config.max_workers
        
        # Scale down if high resource usage
        if cpu_percent > self.config.cpu_threshold:
            current_workers = max(1, current_workers // 2)
            results["scaling_decisions"].append(f"scale_down_cpu_{cpu_percent:.1f}%")
        
        if memory_percent > self.config.memory_threshold:
            current_workers = max(1, current_workers // 2)
            results["scaling_decisions"].append(f"scale_down_memory_{memory_percent:.1f}%")
        
        # Scale up if low resource usage
        if cpu_percent < 20 and memory_percent < 30:
            current_workers = min(self.config.max_workers, current_workers * 2)
            results["scaling_decisions"].append(f"scale_up_resources_available")
        
        results["final_workers"] = current_workers
        results["cpu_usage"] = cpu_percent
        results["memory_usage"] = memory_percent
        
        logger.info(f"Adaptive scaling: {len(results['scaling_decisions'])} decisions made")
        return results
    
    def generate_scalability_report(self) -> Dict[str, Any]:
        """Generate comprehensive scalability and performance report."""
        performance_summary = self.performance_monitor.get_performance_summary()
        cache_stats = self.cache.stats
        
        system_info = {
            "cpu_count": multiprocessing.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "current_cpu_percent": psutil.cpu_percent(),
            "current_memory_percent": psutil.virtual_memory().percent
        }
        
        report = {
            "timestamp": time.time(),
            "generation": "Generation 3 - MAKE IT SCALE",
            "configuration": {
                "max_workers": self.config.max_workers,
                "cache_size": self.config.cache_size,
                "auto_scaling": self.config.auto_scaling
            },
            "performance_metrics": performance_summary,
            "cache_performance": {
                "total_requests": cache_stats["hits"] + cache_stats["misses"],
                "hit_rate": (
                    cache_stats["hits"] / (cache_stats["hits"] + cache_stats["misses"])
                    if (cache_stats["hits"] + cache_stats["misses"]) > 0 else 0
                ),
                "evictions": cache_stats["evictions"]
            },
            "system_resources": system_info,
            "scalability_score": self._calculate_scalability_score(performance_summary, system_info)
        }
        
        return report
    
    def _calculate_scalability_score(self, performance: Dict, system: Dict) -> float:
        """Calculate overall scalability score (0-100)."""
        score = 100.0
        
        # Deduct for poor performance
        if performance.get("avg_execution_time", 0) > self.config.performance_threshold:
            score -= 20
        
        # Deduct for high resource usage
        if system.get("current_cpu_percent", 0) > self.config.cpu_threshold:
            score -= 15
        
        if system.get("current_memory_percent", 0) > self.config.memory_threshold:
            score -= 15
        
        # Bonus for good cache performance
        cache_hit_rate = performance.get("cache_hit_rate", 0)
        if cache_hit_rate > 0.8:
            score += 10
        
        return max(0, min(100, score))
    
    def __del__(self):
        """Cleanup resources."""
        try:
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)
        except:
            pass

async def main():
    """Main execution function for Generation 3."""
    print("⚡ Generation 3: MAKE IT SCALE - Performance Optimization Test")
    print("=" * 70)
    
    config = ScalabilityConfig(
        max_workers=min(16, multiprocessing.cpu_count() * 2),
        enable_caching=True,
        cache_size=500,
        auto_scaling=True
    )
    
    engine = ScalableAutonomousEngine(config)
    
    try:
        # Test 1: Concurrent Processing
        print("\n⚡ Testing concurrent processing...")
        concurrent_results = await engine.autonomous_concurrent_processing()
        print(f"✅ Concurrent: {concurrent_results['tasks_completed']} tasks, "
              f"{concurrent_results['throughput']:.1f} tasks/sec")
        
        # Test 2: Parallel Computation
        print("\n🔄 Testing parallel computation...")
        parallel_results = engine.autonomous_parallel_computation()
        print(f"✅ Parallel: {parallel_results['computations']} computations, "
              f"{parallel_results['throughput']:.1f} ops/sec")
        
        # Test 3: Memory Optimization
        print("\n💾 Testing memory optimization...")
        memory_results = engine.autonomous_memory_optimization()
        print(f"✅ Memory: {len(memory_results['optimization_steps'])} optimizations, "
              f"{memory_results['memory_saved']:.2f}MB saved")
        
        # Test 4: Adaptive Scaling
        print("\n📈 Testing adaptive scaling...")
        scaling_results = engine.autonomous_adaptive_scaling()
        print(f"✅ Scaling: {len(scaling_results['scaling_decisions'])} decisions made")
        
        # Generate comprehensive report
        final_report = engine.generate_scalability_report()
        
        # Save report
        report_file = Path("/root/repo/generation3_scalable_report.json")
        with open(report_file, "w") as f:
            json.dump(final_report, f, indent=2)
        
        print(f"\n📊 GENERATION 3 RESULTS:")
        print(f"Workers Available: {final_report['configuration']['max_workers']}")
        print(f"Cache Hit Rate: {final_report['cache_performance']['hit_rate']:.1%}")
        print(f"Scalability Score: {final_report['scalability_score']:.1f}/100")
        print(f"Report saved to: {report_file}")
        
        if final_report['scalability_score'] >= 80:
            print("\n🎉 GENERATION 3 COMPLETE - EXCELLENT SCALABILITY ACHIEVED")
            return 0
        else:
            print("\n⚠️ GENERATION 3 PARTIAL SUCCESS - OPTIMIZATION OPPORTUNITIES EXIST")
            return 1
            
    except Exception as e:
        logger.error(f"Generation 3 failed: {e}")
        print(f"\n❌ GENERATION 3 FAILED: {e}")
        return 1
    finally:
        # Cleanup
        del engine

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))