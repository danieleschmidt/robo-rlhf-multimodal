"""
Reliability Layer: Advanced reliability patterns and fault tolerance for pipeline operations.

Implements circuit breakers, retry logic, bulkheads, and graceful degradation.
"""

import asyncio
import logging
import time
import random
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class RetryStrategy(Enum):
    """Retry strategies."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    RANDOM_JITTER = "random_jitter"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True
    backoff_multiplier: float = 2.0


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3  # For half-open -> closed transition
    timeout: float = 30.0


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead isolation."""
    max_concurrent: int = 10
    queue_size: int = 100
    timeout: float = 30.0


class ReliabilityPattern(ABC):
    """Abstract base class for reliability patterns."""
    
    @abstractmethod
    async def execute(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with reliability pattern applied."""
        pass


class RetryPattern(ReliabilityPattern):
    """
    Retry pattern implementation with various strategies.
    
    Automatically retries failed operations with configurable backoff strategies.
    """
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.retry_stats: Dict[str, List[float]] = {}
    
    async def execute(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with retry logic."""
        operation_name = getattr(operation, "__name__", "unknown")
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                start_time = time.time()
                result = await operation(*args, **kwargs)
                
                # Record successful execution time
                execution_time = time.time() - start_time
                self._record_retry_stat(operation_name, execution_time)
                
                if attempt > 0:
                    logger.info(f"Operation {operation_name} succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt == self.config.max_attempts - 1:
                    # Last attempt failed
                    logger.error(
                        f"Operation {operation_name} failed after {self.config.max_attempts} attempts: {e}"
                    )
                    break
                
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt)
                
                logger.warning(
                    f"Operation {operation_name} failed on attempt {attempt + 1}, "
                    f"retrying in {delay:.2f}s: {e}"
                )
                
                await asyncio.sleep(delay)
        
        # All attempts failed
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for next retry attempt."""
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** attempt)
        
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * (attempt + 1)
        
        elif self.config.strategy == RetryStrategy.RANDOM_JITTER:
            base_delay = self.config.base_delay * (self.config.backoff_multiplier ** attempt)
            jitter = random.uniform(0, base_delay * 0.1)
            delay = base_delay + jitter
        
        else:
            delay = self.config.base_delay
        
        # Apply max delay limit
        delay = min(delay, self.config.max_delay)
        
        # Apply jitter if enabled
        if self.config.jitter and self.config.strategy != RetryStrategy.RANDOM_JITTER:
            jitter = random.uniform(-delay * 0.1, delay * 0.1)
            delay = max(0, delay + jitter)
        
        return delay
    
    def _record_retry_stat(self, operation_name: str, execution_time: float) -> None:
        """Record retry statistics."""
        if operation_name not in self.retry_stats:
            self.retry_stats[operation_name] = []
        
        self.retry_stats[operation_name].append(execution_time)
        
        # Keep only recent stats
        if len(self.retry_stats[operation_name]) > 100:
            self.retry_stats[operation_name] = self.retry_stats[operation_name][-100:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics."""
        stats = {}
        for operation, times in self.retry_stats.items():
            if times:
                stats[operation] = {
                    "total_executions": len(times),
                    "avg_execution_time": sum(times) / len(times),
                    "min_execution_time": min(times),
                    "max_execution_time": max(times)
                }
        return stats


class CircuitBreakerPattern(ReliabilityPattern):
    """
    Circuit breaker pattern implementation.
    
    Prevents cascading failures by failing fast when a service is unreliable.
    """
    
    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.operation_stats: Dict[str, Dict[str, int]] = {}
    
    async def execute(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation through circuit breaker."""
        operation_name = getattr(operation, "__name__", "unknown")
        
        # Initialize stats for operation
        if operation_name not in self.operation_stats:
            self.operation_stats[operation_name] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "circuit_open_calls": 0
            }
        
        stats = self.operation_stats[operation_name]
        stats["total_calls"] += 1
        
        # Check circuit state
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker for {operation_name} moved to HALF_OPEN")
            else:
                # Circuit is open, fail fast
                stats["circuit_open_calls"] += 1
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            # Execute operation with timeout
            result = await asyncio.wait_for(operation(*args, **kwargs), timeout=self.config.timeout)
            
            # Success
            stats["successful_calls"] += 1
            self._on_success(operation_name)
            return result
            
        except Exception as e:
            # Failure
            stats["failed_calls"] += 1
            self._on_failure(operation_name)
            raise
    
    def _on_success(self, operation_name: str) -> None:
        """Handle successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            if self.success_count >= self.config.success_threshold:
                # Enough successes, close the circuit
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker for {operation_name} CLOSED - service recovered")
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def _on_failure(self, operation_name: str) -> None:
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                # Too many failures, open the circuit
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker for {operation_name} OPENED - "
                    f"failure threshold ({self.config.failure_threshold}) exceeded"
                )
        
        elif self.state == CircuitState.HALF_OPEN:
            # Failure during half-open, go back to open
            self.state = CircuitState.OPEN
            self.success_count = 0
            logger.warning(f"Circuit breaker for {operation_name} back to OPEN - test failed")
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state
    
    def force_open(self) -> None:
        """Manually open the circuit."""
        self.state = CircuitState.OPEN
        self.last_failure_time = time.time()
        logger.warning("Circuit breaker manually OPENED")
    
    def force_close(self) -> None:
        """Manually close the circuit."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info("Circuit breaker manually CLOSED")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "operations": self.operation_stats
        }


class BulkheadPattern(ReliabilityPattern):
    """
    Bulkhead pattern implementation.
    
    Isolates resources to prevent failures in one area from affecting others.
    """
    
    def __init__(self, config: BulkheadConfig = None):
        self.config = config or BulkheadConfig()
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self.queue = asyncio.Queue(maxsize=self.config.queue_size)
        self.active_operations = 0
        self.total_operations = 0
        self.rejected_operations = 0
    
    async def execute(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with bulkhead isolation."""
        self.total_operations += 1
        
        try:
            # Try to acquire semaphore with timeout
            await asyncio.wait_for(
                self.semaphore.acquire(),
                timeout=self.config.timeout
            )
        except asyncio.TimeoutError:
            self.rejected_operations += 1
            raise Exception("Bulkhead capacity exceeded - operation rejected")
        
        try:
            self.active_operations += 1
            
            # Execute operation with timeout
            result = await asyncio.wait_for(
                operation(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            return result
            
        finally:
            self.active_operations -= 1
            self.semaphore.release()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        return {
            "max_concurrent": self.config.max_concurrent,
            "active_operations": self.active_operations,
            "total_operations": self.total_operations,
            "rejected_operations": self.rejected_operations,
            "utilization": self.active_operations / self.config.max_concurrent
        }


class TimeoutPattern(ReliabilityPattern):
    """
    Timeout pattern implementation.
    
    Prevents operations from hanging indefinitely.
    """
    
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.timeout_count = 0
        self.successful_operations = 0
    
    async def execute(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with timeout."""
        try:
            result = await asyncio.wait_for(operation(*args, **kwargs), timeout=self.timeout)
            self.successful_operations += 1
            return result
            
        except asyncio.TimeoutError:
            self.timeout_count += 1
            operation_name = getattr(operation, "__name__", "unknown")
            logger.warning(f"Operation {operation_name} timed out after {self.timeout}s")
            raise Exception(f"Operation timed out after {self.timeout}s")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get timeout statistics."""
        total = self.successful_operations + self.timeout_count
        return {
            "timeout_seconds": self.timeout,
            "successful_operations": self.successful_operations,
            "timeout_count": self.timeout_count,
            "timeout_rate": self.timeout_count / total if total > 0 else 0.0
        }


class ReliabilityManager:
    """
    Manages multiple reliability patterns for pipeline operations.
    
    Combines retry, circuit breaker, bulkhead, and timeout patterns for comprehensive
    fault tolerance.
    """
    
    def __init__(self):
        self.patterns: Dict[str, List[ReliabilityPattern]] = {}
        self.default_patterns: List[ReliabilityPattern] = []
        
        # Default patterns
        self.default_retry = RetryPattern()
        self.default_circuit_breaker = CircuitBreakerPattern()
        self.default_bulkhead = BulkheadPattern()
        self.default_timeout = TimeoutPattern()
        
        # Statistics
        self.operation_stats: Dict[str, Dict[str, Any]] = {}
    
    def configure_operation(
        self,
        operation_name: str,
        patterns: List[ReliabilityPattern]
    ) -> None:
        """Configure reliability patterns for specific operation."""
        self.patterns[operation_name] = patterns
        logger.info(f"Configured {len(patterns)} reliability patterns for {operation_name}")
    
    def add_default_pattern(self, pattern: ReliabilityPattern) -> None:
        """Add pattern to default reliability stack."""
        self.default_patterns.append(pattern)
    
    async def execute_with_reliability(
        self,
        operation: Callable,
        operation_name: Optional[str] = None,
        custom_patterns: Optional[List[ReliabilityPattern]] = None,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute operation with reliability patterns applied.
        
        Args:
            operation: The operation to execute
            operation_name: Name for statistics tracking
            custom_patterns: Custom patterns for this execution
            *args, **kwargs: Arguments for the operation
        """
        op_name = operation_name or getattr(operation, "__name__", "unknown")
        
        # Initialize stats
        if op_name not in self.operation_stats:
            self.operation_stats[op_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "avg_execution_time": 0.0
            }
        
        stats = self.operation_stats[op_name]
        stats["total_executions"] += 1
        
        # Determine patterns to apply
        if custom_patterns:
            patterns = custom_patterns
        elif op_name in self.patterns:
            patterns = self.patterns[op_name]
        else:
            patterns = [
                self.default_timeout,
                self.default_circuit_breaker,
                self.default_bulkhead,
                self.default_retry
            ]
        
        # Apply patterns in reverse order (innermost first)
        wrapped_operation = operation
        for pattern in reversed(patterns):
            current_op = wrapped_operation
            wrapped_operation = lambda *a, **kw, p=pattern, op=current_op: p.execute(op, *a, **kw)
        
        # Execute with timing
        start_time = time.time()
        try:
            result = await wrapped_operation(*args, **kwargs)
            
            # Update success stats
            execution_time = time.time() - start_time
            stats["successful_executions"] += 1
            stats["avg_execution_time"] = (
                (stats["avg_execution_time"] * (stats["successful_executions"] - 1) + execution_time)
                / stats["successful_executions"]
            )
            
            return result
            
        except Exception as e:
            stats["failed_executions"] += 1
            logger.error(f"Operation {op_name} failed with reliability patterns: {e}")
            raise
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics for all reliability patterns."""
        return {
            "default_retry": self.default_retry.get_stats(),
            "default_circuit_breaker": self.default_circuit_breaker.get_stats(),
            "default_bulkhead": self.default_bulkhead.get_stats(),
            "default_timeout": self.default_timeout.get_stats(),
            "operation_stats": self.operation_stats
        }
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.operation_stats.clear()
        
        # Reset pattern stats (simplified - would need pattern-specific reset methods)
        self.default_retry.retry_stats.clear()
        self.default_circuit_breaker.operation_stats.clear()


class GracefulDegradation:
    """
    Graceful degradation system for maintaining service availability.
    
    Provides fallback mechanisms when primary services fail.
    """
    
    def __init__(self):
        self.fallback_handlers: Dict[str, List[Callable]] = {}
        self.degradation_stats: Dict[str, int] = {}
    
    def register_fallback(
        self,
        operation_name: str,
        fallback_handler: Callable,
        priority: int = 1
    ) -> None:
        """Register fallback handler for an operation."""
        if operation_name not in self.fallback_handlers:
            self.fallback_handlers[operation_name] = []
        
        # Insert based on priority (lower number = higher priority)
        self.fallback_handlers[operation_name].append((priority, fallback_handler))
        self.fallback_handlers[operation_name].sort(key=lambda x: x[0])
        
        logger.info(f"Registered fallback handler for {operation_name} (priority: {priority})")
    
    async def execute_with_fallback(
        self,
        operation: Callable,
        operation_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with fallback on failure."""
        try:
            # Try primary operation
            return await operation(*args, **kwargs)
            
        except Exception as primary_error:
            logger.warning(f"Primary operation {operation_name} failed: {primary_error}")
            
            # Try fallback handlers in priority order
            if operation_name in self.fallback_handlers:
                for priority, fallback_handler in self.fallback_handlers[operation_name]:
                    try:
                        logger.info(f"Trying fallback handler for {operation_name} (priority: {priority})")
                        result = await fallback_handler(*args, **kwargs)
                        
                        # Track degradation
                        self.degradation_stats[operation_name] = (
                            self.degradation_stats.get(operation_name, 0) + 1
                        )
                        
                        return result
                        
                    except Exception as fallback_error:
                        logger.warning(f"Fallback handler failed for {operation_name}: {fallback_error}")
                        continue
            
            # All fallbacks failed
            logger.error(f"All fallback options exhausted for {operation_name}")
            raise primary_error
    
    def get_degradation_stats(self) -> Dict[str, Any]:
        """Get graceful degradation statistics."""
        return {
            "registered_fallbacks": {
                name: len(handlers) for name, handlers in self.fallback_handlers.items()
            },
            "degradation_counts": self.degradation_stats.copy()
        }