"""
Performance optimization utilities for robo-rlhf-multimodal.
"""

import time
import threading
import multiprocessing as mp
from functools import wraps, lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging
import asyncio
import concurrent.futures
from contextlib import contextmanager
import psutil
import gc

from robo_rlhf.core.logging import get_logger


logger = get_logger(__name__)


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {}
        self.timers = {}
        self.counters = {}
        self.lock = threading.Lock()
    
    def start_timer(self, name: str) -> None:
        """Start a performance timer."""
        with self.lock:
            self.timers[name] = time.perf_counter()
    
    def stop_timer(self, name: str) -> float:
        """Stop a timer and record duration."""
        end_time = time.perf_counter()
        with self.lock:
            if name in self.timers:
                duration = end_time - self.timers[name]
                if name not in self.metrics:
                    self.metrics[name] = []
                self.metrics[name].append(duration)
                del self.timers[name]
                return duration
        return 0.0
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        with self.lock:
            self.counters[name] = self.counters.get(name, 0) + value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self.lock:
            metrics_summary = {}
            
            # Timer statistics
            for name, durations in self.metrics.items():
                if durations:
                    metrics_summary[f"{name}_avg"] = sum(durations) / len(durations)
                    metrics_summary[f"{name}_min"] = min(durations)
                    metrics_summary[f"{name}_max"] = max(durations)
                    metrics_summary[f"{name}_count"] = len(durations)
            
            # Counter values
            metrics_summary.update(self.counters)
            
            # System metrics
            process = psutil.Process()
            metrics_summary.update({
                "cpu_percent": process.cpu_percent(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "memory_percent": process.memory_percent(),
                "num_threads": process.num_threads()
            })
            
            return metrics_summary
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self.lock:
            self.metrics.clear()
            self.timers.clear()
            self.counters.clear()


# Global performance monitor
_perf_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor."""
    return _perf_monitor


@contextmanager
def timer(name: str):
    """Context manager for timing operations."""
    _perf_monitor.start_timer(name)
    try:
        yield
    finally:
        duration = _perf_monitor.stop_timer(name)
        logger.debug(f"Timer '{name}' completed in {duration:.4f}s")


def measure_time(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"
        with timer(func_name):
            return func(*args, **kwargs)
    return wrapper


def measure_async_time(func: Callable) -> Callable:
    """Decorator to measure async function execution time."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"
        with timer(func_name):
            return await func(*args, **kwargs)
    return wrapper


class LRUCache:
    """Thread-safe LRU cache with TTL support."""
    
    def __init__(self, maxsize: int = 128, ttl: Optional[float] = None):
        """
        Initialize LRU cache.
        
        Args:
            maxsize: Maximum number of cached items
            ttl: Time-to-live in seconds (None for no expiration)
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache = {}
        self.access_order = []
        self.timestamps = {} if ttl else None
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def get(self, key: Any) -> Tuple[Any, bool]:
        """
        Get item from cache.
        
        Returns:
            (value, hit) tuple where hit indicates cache hit
        """
        with self.lock:
            current_time = time.time()
            
            # Check if key exists
            if key in self.cache:
                # Check TTL expiration
                if self.timestamps and current_time - self.timestamps[key] > self.ttl:
                    self._remove_key(key)
                    self.misses += 1
                    return None, False
                
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                self.hits += 1
                return self.cache[key], True
            
            self.misses += 1
            return None, False
    
    def put(self, key: Any, value: Any) -> None:
        """Put item in cache."""
        with self.lock:
            current_time = time.time()
            
            if key in self.cache:
                # Update existing
                self.cache[key] = value
                if self.timestamps:
                    self.timestamps[key] = current_time
                # Move to end
                self.access_order.remove(key)
                self.access_order.append(key)
            else:
                # Add new
                if len(self.cache) >= self.maxsize:
                    # Remove least recently used
                    lru_key = self.access_order.pop(0)
                    self._remove_key(lru_key)
                
                self.cache[key] = value
                self.access_order.append(key)
                if self.timestamps:
                    self.timestamps[key] = current_time
    
    def _remove_key(self, key: Any) -> None:
        """Remove key from cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_order:
            self.access_order.remove(key)
        if self.timestamps and key in self.timestamps:
            del self.timestamps[key]
    
    def clear(self) -> None:
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            if self.timestamps:
                self.timestamps.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "size": len(self.cache),
                "maxsize": self.maxsize
            }


def cached(maxsize: int = 128, ttl: Optional[float] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        cache = LRUCache(maxsize, ttl)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = (args, tuple(sorted(kwargs.items())))
            
            # Try to get from cache
            result, hit = cache.get(key)
            if hit:
                _perf_monitor.increment_counter(f"{func.__name__}_cache_hits")
                return result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.put(key, result)
            _perf_monitor.increment_counter(f"{func.__name__}_cache_misses")
            
            return result
        
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.stats
        return wrapper
    
    return decorator


class ThreadPool:
    """Optimized thread pool for I/O bound tasks."""
    
    def __init__(self, max_workers: Optional[int] = None, thread_name_prefix: str = ""):
        """
        Initialize thread pool.
        
        Args:
            max_workers: Maximum number of worker threads
            thread_name_prefix: Prefix for thread names
        """
        if max_workers is None:
            max_workers = min(32, (psutil.cpu_count() or 1) * 2)
        
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix
        )
        self.max_workers = max_workers
        logger.info(f"ThreadPool initialized with {max_workers} workers")
    
    def submit(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit a function for execution."""
        return self.executor.submit(fn, *args, **kwargs)
    
    def map(self, fn: Callable, *iterables) -> concurrent.futures.Future:
        """Map function over iterables."""
        return self.executor.map(fn, *iterables)
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool."""
        self.executor.shutdown(wait=wait)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class ProcessPool:
    """Optimized process pool for CPU bound tasks."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize process pool.
        
        Args:
            max_workers: Maximum number of worker processes
        """
        if max_workers is None:
            max_workers = psutil.cpu_count() or 1
        
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        )
        self.max_workers = max_workers
        logger.info(f"ProcessPool initialized with {max_workers} workers")
    
    def submit(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit a function for execution."""
        return self.executor.submit(fn, *args, **kwargs)
    
    def map(self, fn: Callable, *iterables) -> concurrent.futures.Future:
        """Map function over iterables."""
        return self.executor.map(fn, *iterables)
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the process pool."""
        self.executor.shutdown(wait=wait)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class BatchProcessor:
    """Batch processor for efficient bulk operations."""
    
    def __init__(
        self,
        batch_size: int = 32,
        max_wait_time: float = 1.0,
        max_workers: Optional[int] = None
    ):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Size of batches to process
            max_wait_time: Maximum time to wait for batch to fill
            max_workers: Number of worker threads
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.thread_pool = ThreadPool(max_workers)
        
        self.pending_items = []
        self.pending_futures = []
        self.last_batch_time = time.time()
        self.lock = threading.Lock()
        
        # Start batch processing thread
        self.running = True
        self.process_thread = threading.Thread(target=self._process_batches, daemon=True)
        self.process_thread.start()
    
    def submit(self, process_func: Callable, item: Any) -> concurrent.futures.Future:
        """
        Submit item for batch processing.
        
        Args:
            process_func: Function to process batch of items
            item: Item to add to batch
            
        Returns:
            Future that will contain the result
        """
        with self.lock:
            future = concurrent.futures.Future()
            self.pending_items.append((process_func, item))
            self.pending_futures.append(future)
            
            # Process batch if full
            if len(self.pending_items) >= self.batch_size:
                self._submit_batch()
            
            return future
    
    def _process_batches(self) -> None:
        """Background thread to process batches."""
        while self.running:
            with self.lock:
                current_time = time.time()
                if (self.pending_items and 
                    current_time - self.last_batch_time >= self.max_wait_time):
                    self._submit_batch()
            
            time.sleep(0.1)
    
    def _submit_batch(self) -> None:
        """Submit current batch for processing."""
        if not self.pending_items:
            return
        
        items = self.pending_items.copy()
        futures = self.pending_futures.copy()
        
        self.pending_items.clear()
        self.pending_futures.clear()
        self.last_batch_time = time.time()
        
        # Group by process function
        batches = {}
        for i, (process_func, item) in enumerate(items):
            func_key = process_func
            if func_key not in batches:
                batches[func_key] = {"items": [], "futures": []}
            batches[func_key]["items"].append(item)
            batches[func_key]["futures"].append(futures[i])
        
        # Process each batch
        for process_func, batch_data in batches.items():
            batch_future = self.thread_pool.submit(
                self._process_single_batch,
                process_func,
                batch_data["items"],
                batch_data["futures"]
            )
    
    def _process_single_batch(
        self,
        process_func: Callable,
        items: List[Any],
        futures: List[concurrent.futures.Future]
    ) -> None:
        """Process a single batch."""
        try:
            results = process_func(items)
            
            # Set results for individual futures
            if isinstance(results, list) and len(results) == len(futures):
                for future, result in zip(futures, results):
                    future.set_result(result)
            else:
                # Single result for all items
                for future in futures:
                    future.set_result(results)
        
        except Exception as e:
            # Set exception for all futures
            for future in futures:
                future.set_exception(e)
    
    def shutdown(self) -> None:
        """Shutdown batch processor."""
        self.running = False
        if self.process_thread.is_alive():
            self.process_thread.join()
        self.thread_pool.shutdown()


class ResourcePool:
    """Generic resource pool for expensive objects."""
    
    def __init__(
        self,
        create_func: Callable,
        reset_func: Optional[Callable] = None,
        validate_func: Optional[Callable] = None,
        max_size: int = 10,
        timeout: float = 30.0
    ):
        """
        Initialize resource pool.
        
        Args:
            create_func: Function to create new resources
            reset_func: Function to reset resource state
            validate_func: Function to validate resource health
            max_size: Maximum pool size
            timeout: Timeout for acquiring resources
        """
        self.create_func = create_func
        self.reset_func = reset_func
        self.validate_func = validate_func
        self.max_size = max_size
        self.timeout = timeout
        
        self.pool = []
        self.in_use = set()
        self.created_count = 0
        self.lock = threading.Lock()
        self.available = threading.Condition(self.lock)
    
    def acquire(self) -> Any:
        """Acquire a resource from the pool."""
        with self.available:
            # Wait for available resource or ability to create new one
            deadline = time.time() + self.timeout
            
            while True:
                # Try to get from pool
                if self.pool:
                    resource = self.pool.pop()
                    
                    # Validate if validator provided
                    if self.validate_func and not self.validate_func(resource):
                        continue  # Resource is invalid, try next
                    
                    self.in_use.add(id(resource))
                    return resource
                
                # Create new resource if under limit
                if self.created_count < self.max_size:
                    resource = self.create_func()
                    self.created_count += 1
                    self.in_use.add(id(resource))
                    logger.debug(f"Created new resource, pool size: {self.created_count}")
                    return resource
                
                # Wait for resource to be returned
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("Timeout acquiring resource from pool")
                
                self.available.wait(timeout=remaining)
    
    def release(self, resource: Any) -> None:
        """Release a resource back to the pool."""
        with self.available:
            resource_id = id(resource)
            
            if resource_id not in self.in_use:
                logger.warning("Attempted to release resource not in use")
                return
            
            self.in_use.remove(resource_id)
            
            # Reset resource if reset function provided
            if self.reset_func:
                try:
                    self.reset_func(resource)
                except Exception as e:
                    logger.warning(f"Failed to reset resource: {e}")
                    self.created_count -= 1  # Don't return to pool
                    self.available.notify()
                    return
            
            # Return to pool
            self.pool.append(resource)
            self.available.notify()
    
    @contextmanager
    def get_resource(self):
        """Context manager for acquiring and releasing resources."""
        resource = self.acquire()
        try:
            yield resource
        finally:
            self.release(resource)
    
    def stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            return {
                "pool_size": len(self.pool),
                "in_use": len(self.in_use),
                "created_count": self.created_count,
                "max_size": self.max_size
            }


def optimize_memory():
    """Perform memory optimization."""
    # Force garbage collection
    collected = gc.collect()
    
    # Get memory info
    process = psutil.Process()
    memory_info = process.memory_info()
    
    logger.info(f"Memory optimization: collected {collected} objects, "
               f"RSS: {memory_info.rss / 1024 / 1024:.1f}MB")
    
    return {
        "collected_objects": collected,
        "memory_mb": memory_info.rss / 1024 / 1024
    }