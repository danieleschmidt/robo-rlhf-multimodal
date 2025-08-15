"""
Self-Healer: Intelligent recovery and healing mechanisms for pipeline components.

Implements advanced recovery strategies, adaptive healing, and failure pattern analysis.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RESTART = "restart"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MIGRATE = "migrate"
    ROLLBACK = "rollback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CUSTOM = "custom"


class RecoveryResult(Enum):
    """Results of recovery attempts."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RecoveryAction:
    """Represents a single recovery action."""
    strategy: RecoveryStrategy
    component: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1=highest, 10=lowest
    timeout: float = 300.0  # 5 minutes default
    retry_count: int = 3
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class RecoveryResult:
    """Result of a recovery operation."""
    action: RecoveryAction
    result: RecoveryResult
    duration: float
    message: str
    timestamp: float
    error: Optional[str] = None


class RecoveryExecutor(ABC):
    """Abstract base class for recovery executors."""
    
    @abstractmethod
    async def execute(self, action: RecoveryAction) -> RecoveryResult:
        """Execute a recovery action."""
        pass
    
    @abstractmethod
    def can_handle(self, strategy: RecoveryStrategy) -> bool:
        """Check if this executor can handle the strategy."""
        pass


class RestartExecutor(RecoveryExecutor):
    """Executor for restart recovery strategy."""
    
    def __init__(self, restart_handler: Optional[Callable] = None):
        self.restart_handler = restart_handler
    
    async def execute(self, action: RecoveryAction) -> RecoveryResult:
        """Execute restart recovery."""
        start_time = time.time()
        
        try:
            if self.restart_handler:
                await self.restart_handler(action.component, action.parameters)
            else:
                # Default restart simulation
                logger.info(f"Restarting component: {action.component}")
                await asyncio.sleep(2)  # Simulate restart time
            
            return RecoveryResult(
                action=action,
                result=RecoveryResult.SUCCESS,
                duration=time.time() - start_time,
                message=f"Successfully restarted {action.component}",
                timestamp=time.time()
            )
            
        except Exception as e:
            return RecoveryResult(
                action=action,
                result=RecoveryResult.FAILED,
                duration=time.time() - start_time,
                message=f"Failed to restart {action.component}",
                timestamp=time.time(),
                error=str(e)
            )
    
    def can_handle(self, strategy: RecoveryStrategy) -> bool:
        """Check if can handle restart strategy."""
        return strategy == RecoveryStrategy.RESTART


class ScaleExecutor(RecoveryExecutor):
    """Executor for scaling recovery strategies."""
    
    def __init__(self, scale_handler: Optional[Callable] = None):
        self.scale_handler = scale_handler
    
    async def execute(self, action: RecoveryAction) -> RecoveryResult:
        """Execute scaling recovery."""
        start_time = time.time()
        
        try:
            scale_factor = action.parameters.get("scale_factor", 2)
            direction = "up" if action.strategy == RecoveryStrategy.SCALE_UP else "down"
            
            if self.scale_handler:
                await self.scale_handler(
                    action.component, 
                    scale_factor, 
                    direction,
                    action.parameters
                )
            else:
                # Default scaling simulation
                logger.info(f"Scaling {direction} component {action.component} by {scale_factor}x")
                await asyncio.sleep(5)  # Simulate scaling time
            
            return RecoveryResult(
                action=action,
                result=RecoveryResult.SUCCESS,
                duration=time.time() - start_time,
                message=f"Successfully scaled {direction} {action.component}",
                timestamp=time.time()
            )
            
        except Exception as e:
            return RecoveryResult(
                action=action,
                result=RecoveryResult.FAILED,
                duration=time.time() - start_time,
                message=f"Failed to scale {action.component}",
                timestamp=time.time(),
                error=str(e)
            )
    
    def can_handle(self, strategy: RecoveryStrategy) -> bool:
        """Check if can handle scaling strategies."""
        return strategy in [RecoveryStrategy.SCALE_UP, RecoveryStrategy.SCALE_DOWN]


class CircuitBreakerExecutor(RecoveryExecutor):
    """Executor for circuit breaker pattern."""
    
    def __init__(self):
        self.circuit_states: Dict[str, str] = {}  # component -> state
        self.failure_counts: Dict[str, int] = {}
        self.last_failure_time: Dict[str, float] = {}
    
    async def execute(self, action: RecoveryAction) -> RecoveryResult:
        """Execute circuit breaker activation."""
        start_time = time.time()
        component = action.component
        
        try:
            # Activate circuit breaker
            self.circuit_states[component] = "open"
            timeout = action.parameters.get("timeout", 60)  # 1 minute default
            
            logger.warning(f"Circuit breaker opened for {component} (timeout: {timeout}s)")
            
            # Schedule automatic reset
            asyncio.create_task(self._schedule_reset(component, timeout))
            
            return RecoveryResult(
                action=action,
                result=RecoveryResult.SUCCESS,
                duration=time.time() - start_time,
                message=f"Circuit breaker activated for {component}",
                timestamp=time.time()
            )
            
        except Exception as e:
            return RecoveryResult(
                action=action,
                result=RecoveryResult.FAILED,
                duration=time.time() - start_time,
                message=f"Failed to activate circuit breaker for {component}",
                timestamp=time.time(),
                error=str(e)
            )
    
    async def _schedule_reset(self, component: str, timeout: float) -> None:
        """Schedule circuit breaker reset."""
        await asyncio.sleep(timeout)
        self.circuit_states[component] = "half-open"
        logger.info(f"Circuit breaker for {component} moved to half-open state")
    
    def can_handle(self, strategy: RecoveryStrategy) -> bool:
        """Check if can handle circuit breaker strategy."""
        return strategy == RecoveryStrategy.CIRCUIT_BREAKER
    
    def is_circuit_open(self, component: str) -> bool:
        """Check if circuit is open for component."""
        return self.circuit_states.get(component) == "open"


class SelfHealer:
    """
    Intelligent self-healing system for pipeline components.
    
    Features:
    - Multiple recovery strategies
    - Adaptive strategy selection
    - Failure pattern analysis
    - Recovery coordination
    - Success rate tracking
    """
    
    def __init__(self):
        self.executors: List[RecoveryExecutor] = []
        self.recovery_history: List[RecoveryResult] = []
        self.strategy_success_rates: Dict[RecoveryStrategy, List[bool]] = {}
        self.component_preferences: Dict[str, List[RecoveryStrategy]] = {}
        
        # Initialize default executors
        self._setup_default_executors()
        
        # Quantum integration if available
        try:
            from robo_rlhf.quantum import QuantumOptimizer
            self.quantum_optimizer = QuantumOptimizer()
            self.quantum_enabled = True
            logger.info("Quantum-enhanced self-healing initialized")
        except ImportError:
            self.quantum_enabled = False
            logger.info("Standard self-healing initialized")
    
    def _setup_default_executors(self) -> None:
        """Setup default recovery executors."""
        self.executors = [
            RestartExecutor(),
            ScaleExecutor(),
            CircuitBreakerExecutor()
        ]
    
    def register_executor(self, executor: RecoveryExecutor) -> None:
        """Register a custom recovery executor."""
        self.executors.append(executor)
        logger.info(f"Registered recovery executor: {type(executor).__name__}")
    
    def set_component_preferences(
        self, 
        component: str, 
        strategies: List[RecoveryStrategy]
    ) -> None:
        """Set preferred recovery strategies for a component."""
        self.component_preferences[component] = strategies
        logger.info(f"Set recovery preferences for {component}: {strategies}")
    
    async def heal(
        self, 
        component: str, 
        failure_context: Dict[str, Any],
        suggested_strategies: Optional[List[RecoveryStrategy]] = None
    ) -> List[RecoveryResult]:
        """
        Initiate healing process for a failed component.
        
        Args:
            component: Name of the failed component
            failure_context: Context about the failure (metrics, errors, etc.)
            suggested_strategies: Optional list of strategies to try
            
        Returns:
            List of recovery results from executed actions
        """
        logger.info(f"Starting healing process for component: {component}")
        
        # Determine recovery actions
        if self.quantum_enabled:
            actions = await self._quantum_strategy_selection(component, failure_context)
        else:
            actions = await self._standard_strategy_selection(
                component, failure_context, suggested_strategies
            )
        
        # Execute recovery actions
        results = await self._execute_recovery_plan(actions)
        
        # Update success tracking
        self._update_success_tracking(results)
        
        # Store results
        self.recovery_history.extend(results)
        
        # Keep only recent history (last 1000 attempts)
        if len(self.recovery_history) > 1000:
            self.recovery_history = self.recovery_history[-1000:]
        
        logger.info(
            f"Healing process completed for {component}. "
            f"Successful actions: {sum(1 for r in results if r.result == RecoveryResult.SUCCESS)}/{len(results)}"
        )
        
        return results
    
    async def _quantum_strategy_selection(
        self, 
        component: str, 
        failure_context: Dict[str, Any]
    ) -> List[RecoveryAction]:
        """Use quantum optimization for strategy selection."""
        try:
            # Get optimal recovery plan from quantum optimizer
            optimization_result = await self.quantum_optimizer.optimize_recovery_plan(
                component=component,
                failure_context=failure_context,
                success_history=self._get_component_success_history(component),
                available_strategies=[
                    strategy for executor in self.executors 
                    for strategy in RecoveryStrategy 
                    if executor.can_handle(strategy)
                ]
            )
            
            actions = []
            for strategy_info in optimization_result.get("strategies", []):
                action = RecoveryAction(
                    strategy=RecoveryStrategy(strategy_info["strategy"]),
                    component=component,
                    parameters=strategy_info.get("parameters", {}),
                    priority=strategy_info.get("priority", 1),
                    timeout=strategy_info.get("timeout", 300.0)
                )
                actions.append(action)
            
            return actions
            
        except Exception as e:
            logger.error(f"Quantum strategy selection failed: {e}")
            # Fallback to standard selection
            return await self._standard_strategy_selection(component, failure_context)
    
    async def _standard_strategy_selection(
        self, 
        component: str, 
        failure_context: Dict[str, Any],
        suggested_strategies: Optional[List[RecoveryStrategy]] = None
    ) -> List[RecoveryAction]:
        """Standard strategy selection based on heuristics."""
        actions = []
        
        # Use suggested strategies if provided
        if suggested_strategies:
            strategies = suggested_strategies
        # Use component preferences if set
        elif component in self.component_preferences:
            strategies = self.component_preferences[component]
        else:
            # Default strategy order based on failure context
            strategies = self._analyze_failure_and_suggest_strategies(failure_context)
        
        # Create recovery actions
        for i, strategy in enumerate(strategies):
            # Check if we have an executor for this strategy
            if not any(executor.can_handle(strategy) for executor in self.executors):
                continue
            
            parameters = self._get_strategy_parameters(strategy, failure_context)
            
            action = RecoveryAction(
                strategy=strategy,
                component=component,
                parameters=parameters,
                priority=i + 1,  # Earlier strategies have higher priority
                timeout=parameters.get("timeout", 300.0)
            )
            actions.append(action)
        
        return actions
    
    def _analyze_failure_and_suggest_strategies(
        self, 
        failure_context: Dict[str, Any]
    ) -> List[RecoveryStrategy]:
        """Analyze failure context and suggest appropriate strategies."""
        strategies = []
        
        # Get failure indicators
        high_cpu = failure_context.get("cpu_usage", 0) > 0.8
        high_memory = failure_context.get("memory_usage", 0) > 0.8
        high_error_rate = failure_context.get("error_rate", 0) > 0.1
        slow_response = failure_context.get("response_time", 0) > 5.0
        
        # Strategy selection logic
        if high_error_rate and not (high_cpu or high_memory):
            # Likely application-level issue, restart might help
            strategies.append(RecoveryStrategy.RESTART)
            strategies.append(RecoveryStrategy.CIRCUIT_BREAKER)
        
        elif high_cpu or high_memory or slow_response:
            # Resource issues, scaling might help
            strategies.append(RecoveryStrategy.SCALE_UP)
            strategies.append(RecoveryStrategy.RESTART)
        
        else:
            # Generic failure, try restart first
            strategies.append(RecoveryStrategy.RESTART)
            strategies.append(RecoveryStrategy.SCALE_UP)
            strategies.append(RecoveryStrategy.CIRCUIT_BREAKER)
        
        return strategies
    
    def _get_strategy_parameters(
        self, 
        strategy: RecoveryStrategy, 
        failure_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get parameters for a specific strategy based on failure context."""
        if strategy == RecoveryStrategy.SCALE_UP:
            # Determine scale factor based on load
            cpu_usage = failure_context.get("cpu_usage", 0)
            scale_factor = 2 if cpu_usage > 0.9 else 1.5
            return {"scale_factor": scale_factor}
        
        elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            # Timeout based on error rate
            error_rate = failure_context.get("error_rate", 0)
            timeout = 120 if error_rate > 0.5 else 60
            return {"timeout": timeout}
        
        elif strategy == RecoveryStrategy.RESTART:
            # Graceful vs force restart
            return {"graceful": True, "timeout": 30}
        
        return {}
    
    async def _execute_recovery_plan(
        self, 
        actions: List[RecoveryAction]
    ) -> List[RecoveryResult]:
        """Execute recovery plan with proper coordination."""
        results = []
        
        # Sort actions by priority
        actions.sort(key=lambda a: a.priority)
        
        for action in actions:
            # Find appropriate executor
            executor = None
            for exec_candidate in self.executors:
                if exec_candidate.can_handle(action.strategy):
                    executor = exec_candidate
                    break
            
            if not executor:
                logger.error(f"No executor found for strategy: {action.strategy}")
                continue
            
            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    executor.execute(action),
                    timeout=action.timeout
                )
                results.append(result)
                
                # If successful, we might not need to try other strategies
                if result.result == RecoveryResult.SUCCESS:
                    logger.info(f"Recovery successful with strategy: {action.strategy}")
                    break
                    
            except asyncio.TimeoutError:
                result = RecoveryResult(
                    action=action,
                    result=RecoveryResult.FAILED,
                    duration=action.timeout,
                    message=f"Recovery action timed out after {action.timeout}s",
                    timestamp=time.time(),
                    error="Timeout"
                )
                results.append(result)
                
            except Exception as e:
                result = RecoveryResult(
                    action=action,
                    result=RecoveryResult.FAILED,
                    duration=0.0,
                    message=f"Recovery action failed with exception",
                    timestamp=time.time(),
                    error=str(e)
                )
                results.append(result)
        
        return results
    
    def _update_success_tracking(self, results: List[RecoveryResult]) -> None:
        """Update success rate tracking for strategies."""
        for result in results:
            strategy = result.action.strategy
            success = result.result == RecoveryResult.SUCCESS
            
            if strategy not in self.strategy_success_rates:
                self.strategy_success_rates[strategy] = []
            
            self.strategy_success_rates[strategy].append(success)
            
            # Keep only recent results (last 100)
            if len(self.strategy_success_rates[strategy]) > 100:
                self.strategy_success_rates[strategy] = self.strategy_success_rates[strategy][-100:]
    
    def _get_component_success_history(self, component: str) -> List[Dict[str, Any]]:
        """Get success history for a specific component."""
        return [
            {
                "strategy": result.action.strategy.value,
                "success": result.result == RecoveryResult.SUCCESS,
                "duration": result.duration,
                "timestamp": result.timestamp
            }
            for result in self.recovery_history
            if result.action.component == component
        ]
    
    def get_strategy_success_rate(self, strategy: RecoveryStrategy) -> float:
        """Get success rate for a specific strategy."""
        if strategy not in self.strategy_success_rates:
            return 0.0
        
        successes = sum(self.strategy_success_rates[strategy])
        total = len(self.strategy_success_rates[strategy])
        
        return successes / total if total > 0 else 0.0
    
    def get_healing_stats(self) -> Dict[str, Any]:
        """Get comprehensive healing statistics."""
        total_attempts = len(self.recovery_history)
        successful_attempts = sum(
            1 for result in self.recovery_history 
            if result.result == RecoveryResult.SUCCESS
        )
        
        strategy_stats = {}
        for strategy in RecoveryStrategy:
            strategy_stats[strategy.value] = {
                "success_rate": self.get_strategy_success_rate(strategy),
                "total_uses": sum(
                    1 for result in self.recovery_history 
                    if result.action.strategy == strategy
                )
            }
        
        return {
            "total_healing_attempts": total_attempts,
            "successful_healings": successful_attempts,
            "overall_success_rate": successful_attempts / total_attempts if total_attempts > 0 else 0.0,
            "strategy_performance": strategy_stats,
            "quantum_enabled": self.quantum_enabled,
            "registered_executors": len(self.executors)
        }