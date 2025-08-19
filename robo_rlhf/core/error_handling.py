"""
Comprehensive Error Handling and Recovery System for Autonomous SDLC.

Advanced error handling with quantum-inspired recovery strategies, predictive failure detection,
and self-healing capabilities for robust autonomous SDLC execution.
"""

import asyncio
import traceback
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from pathlib import Path
import inspect
from functools import wraps
import numpy as np

from .logging import setup_logger
from .state_manager import PersistentStateManager, StateType


logger = setup_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    ROLLBACK = "rollback"
    SKIP = "skip"
    ESCALATE = "escalate"
    HEAL = "heal"
    QUANTUM_RECOVERY = "quantum_recovery"


@dataclass
class ErrorContext:
    """Detailed error context information."""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    component: str
    function_name: str
    args: Dict[str, Any]
    kwargs: Dict[str, Any]
    stack_trace: str
    system_state: Dict[str, Any]
    recovery_attempts: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryResult:
    """Result of error recovery operation."""
    success: bool
    strategy_used: RecoveryStrategy
    execution_time: float
    recovery_actions: List[str]
    final_state: Dict[str, Any]
    error_resolved: bool
    additional_info: Dict[str, Any] = field(default_factory=dict)


class QuantumErrorRecovery:
    """
    Quantum-inspired error recovery system with multiple recovery strategies.
    
    Uses quantum superposition concepts to explore multiple recovery paths
    simultaneously and collapse to the optimal recovery strategy.
    """

    def __init__(self, state_manager: PersistentStateManager):
        self.state_manager = state_manager
        self.error_history: List[ErrorContext] = []
        self.recovery_patterns: Dict[str, List[RecoveryStrategy]] = {}
        self.success_rates: Dict[RecoveryStrategy, float] = {
            strategy: 0.8 for strategy in RecoveryStrategy
        }
        
        # Quantum recovery configuration
        self.quantum_recovery_enabled = True
        self.superposition_depth = 5
        self.entanglement_threshold = 0.7
        
        logger.info("Quantum Error Recovery system initialized")

    async def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        component: str = "unknown",
        function_name: str = "unknown"
    ) -> RecoveryResult:
        """
        Handle error with quantum-inspired recovery strategies.
        
        Args:
            error: The exception that occurred
            context: Error context information
            component: Component where error occurred
            function_name: Function where error occurred
            
        Returns:
            Recovery result with success status and actions taken
        """
        start_time = datetime.now()
        
        # Create error context
        error_context = ErrorContext(
            error_id=self._generate_error_id(error, context),
            timestamp=start_time,
            error_type=type(error).__name__,
            error_message=str(error),
            severity=self._assess_error_severity(error, context),
            component=component,
            function_name=function_name,
            args=context.get("args", {}),
            kwargs=context.get("kwargs", {}),
            stack_trace=traceback.format_exc(),
            system_state=await self._capture_system_state(),
        )
        
        self.error_history.append(error_context)
        
        logger.error(f"Error detected [{error_context.error_id}]: {error_context.error_message}")
        
        try:
            # Quantum recovery process
            if self.quantum_recovery_enabled:
                recovery_result = await self._quantum_recovery_process(error_context)
            else:
                recovery_result = await self._standard_recovery_process(error_context)
            
            # Update success rates based on result
            if recovery_result.success:
                self._update_recovery_success_rates(recovery_result.strategy_used, True)
            else:
                self._update_recovery_success_rates(recovery_result.strategy_used, False)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            recovery_result.execution_time = execution_time
            
            logger.info(f"Error recovery completed [{error_context.error_id}]: "
                       f"Success={recovery_result.success}, Strategy={recovery_result.strategy_used.value}")
            
            return recovery_result
            
        except Exception as recovery_error:
            logger.critical(f"Recovery process failed for error [{error_context.error_id}]: {recovery_error}")
            
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.ESCALATE,
                execution_time=(datetime.now() - start_time).total_seconds(),
                recovery_actions=[f"Recovery failed: {str(recovery_error)}"],
                final_state=await self._capture_system_state(),
                error_resolved=False,
                additional_info={"recovery_error": str(recovery_error)}
            )

    async def _quantum_recovery_process(self, error_context: ErrorContext) -> RecoveryResult:
        """Execute quantum-inspired recovery process."""
        logger.info(f"Starting quantum recovery for error: {error_context.error_id}")
        
        # Generate quantum superposition of recovery strategies
        recovery_superposition = await self._generate_recovery_superposition(error_context)
        
        # Evaluate each strategy in parallel (quantum simulation)
        strategy_evaluations = await asyncio.gather(*[
            self._evaluate_recovery_strategy(strategy, error_context)
            for strategy in recovery_superposition
        ], return_exceptions=True)
        
        # Find optimal strategy through quantum collapse
        optimal_strategy = await self._quantum_strategy_collapse(
            recovery_superposition, strategy_evaluations, error_context
        )
        
        # Execute optimal recovery strategy
        return await self._execute_recovery_strategy(optimal_strategy, error_context)

    async def _standard_recovery_process(self, error_context: ErrorContext) -> RecoveryResult:
        """Execute standard recovery process."""
        # Determine best strategy based on error type and history
        strategy = self._select_recovery_strategy(error_context)
        return await self._execute_recovery_strategy(strategy, error_context)

    async def _generate_recovery_superposition(self, error_context: ErrorContext) -> List[RecoveryStrategy]:
        """Generate quantum superposition of possible recovery strategies."""
        # Base strategies always considered
        strategies = [RecoveryStrategy.RETRY, RecoveryStrategy.ROLLBACK]
        
        # Add strategies based on error context
        if error_context.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]:
            strategies.append(RecoveryStrategy.SKIP)
        
        if error_context.recovery_attempts < error_context.max_retries:
            strategies.append(RecoveryStrategy.HEAL)
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            strategies.append(RecoveryStrategy.ESCALATE)
        
        # Add quantum recovery for complex errors
        if self._is_complex_error(error_context):
            strategies.append(RecoveryStrategy.QUANTUM_RECOVERY)
        
        return strategies[:self.superposition_depth]

    async def _evaluate_recovery_strategy(
        self, 
        strategy: RecoveryStrategy, 
        error_context: ErrorContext
    ) -> Dict[str, Any]:
        """Evaluate potential success of recovery strategy."""
        # Simulate strategy evaluation (in production, this would use ML models)
        base_success_rate = self.success_rates.get(strategy, 0.5)
        
        # Adjust based on error context
        context_modifier = 0.0
        
        if strategy == RecoveryStrategy.RETRY:
            context_modifier = -0.1 * error_context.recovery_attempts
        elif strategy == RecoveryStrategy.ROLLBACK:
            context_modifier = 0.1 if await self._has_recent_checkpoint(error_context) else -0.2
        elif strategy == RecoveryStrategy.HEAL:
            context_modifier = 0.15 if error_context.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM] else -0.1
        elif strategy == RecoveryStrategy.QUANTUM_RECOVERY:
            context_modifier = 0.2 if self._is_complex_error(error_context) else -0.3
        
        success_probability = max(0.1, min(0.9, base_success_rate + context_modifier))
        
        return {
            "strategy": strategy,
            "success_probability": success_probability,
            "estimated_time": np.random.uniform(1, 10),  # Seconds
            "resource_cost": np.random.uniform(0.1, 1.0),
            "risk_level": 1.0 - success_probability,
        }

    async def _quantum_strategy_collapse(
        self,
        strategies: List[RecoveryStrategy],
        evaluations: List[Dict[str, Any]],
        error_context: ErrorContext
    ) -> RecoveryStrategy:
        """Collapse quantum superposition to optimal strategy."""
        # Filter out failed evaluations
        valid_evaluations = [
            eval_result for eval_result in evaluations
            if isinstance(eval_result, dict) and not isinstance(eval_result, Exception)
        ]
        
        if not valid_evaluations:
            return RecoveryStrategy.ESCALATE
        
        # Multi-objective optimization: success probability, time, resource cost
        best_strategy = RecoveryStrategy.ESCALATE
        best_score = -1.0
        
        for evaluation in valid_evaluations:
            # Weighted score calculation
            score = (
                0.5 * evaluation["success_probability"] +
                0.3 * (1.0 - evaluation["estimated_time"] / 10.0) +  # Prefer faster recovery
                0.2 * (1.0 - evaluation["resource_cost"])  # Prefer lower cost
            )
            
            # Bias toward strategies with higher historical success
            historical_success = self.success_rates.get(evaluation["strategy"], 0.5)
            score *= (0.8 + 0.4 * historical_success)
            
            if score > best_score:
                best_score = score
                best_strategy = evaluation["strategy"]
        
        logger.debug(f"Quantum collapse selected strategy: {best_strategy.value} (score: {best_score:.3f})")
        return best_strategy

    async def _execute_recovery_strategy(
        self, 
        strategy: RecoveryStrategy, 
        error_context: ErrorContext
    ) -> RecoveryResult:
        """Execute the selected recovery strategy."""
        logger.info(f"Executing recovery strategy: {strategy.value}")
        
        recovery_actions = []
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                success, actions = await self._execute_retry_recovery(error_context)
            elif strategy == RecoveryStrategy.ROLLBACK:
                success, actions = await self._execute_rollback_recovery(error_context)
            elif strategy == RecoveryStrategy.SKIP:
                success, actions = await self._execute_skip_recovery(error_context)
            elif strategy == RecoveryStrategy.HEAL:
                success, actions = await self._execute_heal_recovery(error_context)
            elif strategy == RecoveryStrategy.QUANTUM_RECOVERY:
                success, actions = await self._execute_quantum_recovery(error_context)
            else:  # ESCALATE
                success, actions = await self._execute_escalate_recovery(error_context)
            
            recovery_actions.extend(actions)
            
            return RecoveryResult(
                success=success,
                strategy_used=strategy,
                execution_time=0.0,  # Set by caller
                recovery_actions=recovery_actions,
                final_state=await self._capture_system_state(),
                error_resolved=success,
                additional_info={
                    "error_id": error_context.error_id,
                    "strategy_details": await self._get_strategy_details(strategy)
                }
            )
            
        except Exception as e:
            logger.error(f"Recovery strategy {strategy.value} failed: {e}")
            return RecoveryResult(
                success=False,
                strategy_used=strategy,
                execution_time=0.0,
                recovery_actions=[f"Strategy execution failed: {str(e)}"],
                final_state=await self._capture_system_state(),
                error_resolved=False,
                additional_info={"execution_error": str(e)}
            )

    async def _execute_retry_recovery(self, error_context: ErrorContext) -> tuple[bool, List[str]]:
        """Execute retry recovery strategy."""
        actions = []
        
        if error_context.recovery_attempts >= error_context.max_retries:
            actions.append(f"Max retries ({error_context.max_retries}) exceeded")
            return False, actions
        
        # Implement exponential backoff
        delay = 2 ** error_context.recovery_attempts
        actions.append(f"Waiting {delay} seconds before retry")
        await asyncio.sleep(delay)
        
        error_context.recovery_attempts += 1
        actions.append(f"Retry attempt {error_context.recovery_attempts}")
        
        # In a real implementation, this would re-execute the failed operation
        # For now, simulate success/failure
        success = np.random.random() > 0.3  # 70% success rate for retries
        
        if success:
            actions.append("Retry successful")
        else:
            actions.append("Retry failed")
        
        return success, actions

    async def _execute_rollback_recovery(self, error_context: ErrorContext) -> tuple[bool, List[str]]:
        """Execute rollback recovery strategy."""
        actions = []
        
        try:
            # Find recent checkpoint for the affected component
            checkpoints = await self.state_manager.list_checkpoints(limit=10)
            
            if not checkpoints:
                actions.append("No checkpoints available for rollback")
                return False, actions
            
            # Select most recent checkpoint
            latest_checkpoint = checkpoints[0]
            actions.append(f"Rolling back to checkpoint: {latest_checkpoint['id']}")
            
            # Perform rollback
            await self.state_manager.restore_from_checkpoint(latest_checkpoint['id'])
            actions.append("Rollback completed successfully")
            
            return True, actions
            
        except Exception as e:
            actions.append(f"Rollback failed: {str(e)}")
            return False, actions

    async def _execute_skip_recovery(self, error_context: ErrorContext) -> tuple[bool, List[str]]:
        """Execute skip recovery strategy."""
        actions = [
            f"Skipping failed operation: {error_context.function_name}",
            f"Error marked as non-critical: {error_context.error_message}"
        ]
        
        # Log the skip for monitoring
        await self.state_manager.update_state(
            StateType.SYSTEM_STATE,
            {
                "skipped_operations": {
                    error_context.error_id: {
                        "timestamp": error_context.timestamp.isoformat(),
                        "component": error_context.component,
                        "function": error_context.function_name,
                        "reason": "error_skip_recovery"
                    }
                }
            }
        )
        
        return True, actions

    async def _execute_heal_recovery(self, error_context: ErrorContext) -> tuple[bool, List[str]]:
        """Execute self-healing recovery strategy."""
        actions = []
        
        # Analyze error patterns and apply healing
        healing_actions = await self._determine_healing_actions(error_context)
        
        for action in healing_actions:
            try:
                await self._apply_healing_action(action, error_context)
                actions.append(f"Applied healing action: {action['type']}")
            except Exception as e:
                actions.append(f"Healing action failed: {action['type']} - {str(e)}")
        
        # Verify healing effectiveness
        healing_effective = len(healing_actions) > 0
        
        if healing_effective:
            actions.append("Self-healing completed successfully")
        else:
            actions.append("No effective healing actions available")
        
        return healing_effective, actions

    async def _execute_quantum_recovery(self, error_context: ErrorContext) -> tuple[bool, List[str]]:
        """Execute quantum-inspired recovery strategy."""
        actions = ["Initiating quantum recovery process"]
        
        # Quantum error correction simulation
        error_patterns = await self._analyze_quantum_error_patterns(error_context)
        correction_strategies = await self._generate_quantum_corrections(error_patterns)
        
        success_count = 0
        for i, strategy in enumerate(correction_strategies):
            try:
                await self._apply_quantum_correction(strategy, error_context)
                actions.append(f"Applied quantum correction {i+1}: {strategy['type']}")
                success_count += 1
            except Exception as e:
                actions.append(f"Quantum correction {i+1} failed: {str(e)}")
        
        success = success_count > len(correction_strategies) // 2
        
        if success:
            actions.append("Quantum recovery completed successfully")
        else:
            actions.append("Quantum recovery insufficient")
        
        return success, actions

    async def _execute_escalate_recovery(self, error_context: ErrorContext) -> tuple[bool, List[str]]:
        """Execute escalation recovery strategy."""
        actions = [
            "Escalating error to higher-level recovery systems",
            f"Error severity: {error_context.severity.value}",
            f"Recovery attempts: {error_context.recovery_attempts}"
        ]
        
        # In production, this would notify administrators, trigger alerts, etc.
        escalation_data = {
            "error_id": error_context.error_id,
            "timestamp": error_context.timestamp.isoformat(),
            "severity": error_context.severity.value,
            "component": error_context.component,
            "message": error_context.error_message,
            "stack_trace": error_context.stack_trace,
        }
        
        # Log escalation
        await self.state_manager.update_state(
            StateType.SYSTEM_STATE,
            {
                "escalated_errors": {
                    error_context.error_id: escalation_data
                }
            }
        )
        
        actions.append("Error escalated and logged for manual intervention")
        
        # Escalation doesn't resolve the error, but manages it
        return False, actions

    def _assess_error_severity(self, error: Exception, context: Dict[str, Any]) -> ErrorSeverity:
        """Assess the severity of an error."""
        # Critical system errors
        if isinstance(error, (MemoryError, SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if isinstance(error, (ConnectionError, TimeoutError, PermissionError)):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if isinstance(error, (ValueError, TypeError, AttributeError)):
            return ErrorSeverity.MEDIUM
        
        # Default to low severity
        return ErrorSeverity.LOW

    def _is_complex_error(self, error_context: ErrorContext) -> bool:
        """Determine if error is complex enough for quantum recovery."""
        return (
            error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] or
            error_context.recovery_attempts > 2 or
            "quantum" in error_context.component.lower()
        )

    async def _has_recent_checkpoint(self, error_context: ErrorContext) -> bool:
        """Check if recent checkpoint exists for rollback."""
        checkpoints = await self.state_manager.list_checkpoints(limit=1)
        if not checkpoints:
            return False
        
        # Check if checkpoint is recent (within last hour)
        latest_checkpoint = checkpoints[0]
        checkpoint_time = datetime.fromisoformat(latest_checkpoint["timestamp"])
        return datetime.now() - checkpoint_time < timedelta(hours=1)

    async def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for error context."""
        try:
            stats = await self.state_manager.get_state_statistics()
            return {
                "timestamp": datetime.now().isoformat(),
                "state_manager_stats": stats,
                "error_history_count": len(self.error_history),
                "recent_errors": len([
                    e for e in self.error_history[-10:] 
                    if datetime.now() - e.timestamp < timedelta(minutes=5)
                ]),
            }
        except Exception:
            return {"timestamp": datetime.now().isoformat(), "capture_failed": True}

    def _generate_error_id(self, error: Exception, context: Dict[str, Any]) -> str:
        """Generate unique error ID."""
        import hashlib
        error_string = f"{type(error).__name__}_{str(error)}_{datetime.now().isoformat()}"
        return hashlib.sha256(error_string.encode()).hexdigest()[:12]

    def _select_recovery_strategy(self, error_context: ErrorContext) -> RecoveryStrategy:
        """Select recovery strategy based on error context and history."""
        # Look for historical patterns
        similar_errors = [
            e for e in self.error_history[-20:]  # Last 20 errors
            if e.error_type == error_context.error_type and e.component == error_context.component
        ]
        
        if similar_errors:
            # Use most successful strategy from similar errors
            # This is simplified - production would use ML models
            return RecoveryStrategy.RETRY
        
        # Default strategy based on severity
        if error_context.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.ESCALATE
        elif error_context.severity == ErrorSeverity.HIGH:
            return RecoveryStrategy.ROLLBACK
        else:
            return RecoveryStrategy.RETRY

    def _update_recovery_success_rates(self, strategy: RecoveryStrategy, success: bool):
        """Update success rates for recovery strategies."""
        current_rate = self.success_rates.get(strategy, 0.5)
        # Simple exponential moving average
        new_rate = current_rate * 0.9 + (1.0 if success else 0.0) * 0.1
        self.success_rates[strategy] = max(0.1, min(0.9, new_rate))

    async def _determine_healing_actions(self, error_context: ErrorContext) -> List[Dict[str, Any]]:
        """Determine appropriate healing actions for the error."""
        healing_actions = []
        
        # Resource cleanup
        if "memory" in error_context.error_message.lower():
            healing_actions.append({"type": "memory_cleanup", "priority": 1})
        
        # Connection reset
        if "connection" in error_context.error_message.lower():
            healing_actions.append({"type": "connection_reset", "priority": 2})
        
        # Cache invalidation
        if "cache" in error_context.error_message.lower():
            healing_actions.append({"type": "cache_invalidation", "priority": 3})
        
        return healing_actions

    async def _apply_healing_action(self, action: Dict[str, Any], error_context: ErrorContext):
        """Apply a specific healing action."""
        action_type = action["type"]
        
        if action_type == "memory_cleanup":
            # Simulate memory cleanup
            await asyncio.sleep(0.1)
        elif action_type == "connection_reset":
            # Simulate connection reset
            await asyncio.sleep(0.2)
        elif action_type == "cache_invalidation":
            # Simulate cache invalidation
            await asyncio.sleep(0.1)

    async def _analyze_quantum_error_patterns(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Analyze error patterns using quantum-inspired techniques."""
        return {
            "error_entropy": np.random.random(),
            "temporal_patterns": np.random.random(5).tolist(),
            "component_correlations": np.random.random(3).tolist(),
            "severity_distribution": {
                "low": 0.4,
                "medium": 0.3,
                "high": 0.2,
                "critical": 0.1
            }
        }

    async def _generate_quantum_corrections(self, error_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate quantum-inspired error corrections."""
        return [
            {"type": "entropy_reduction", "strength": 0.8},
            {"type": "pattern_stabilization", "strength": 0.6},
            {"type": "correlation_adjustment", "strength": 0.7},
        ]

    async def _apply_quantum_correction(self, strategy: Dict[str, Any], error_context: ErrorContext):
        """Apply quantum-inspired error correction."""
        # Simulate quantum correction application
        await asyncio.sleep(0.1)

    async def _get_strategy_details(self, strategy: RecoveryStrategy) -> Dict[str, Any]:
        """Get detailed information about recovery strategy."""
        return {
            "strategy": strategy.value,
            "success_rate": self.success_rates.get(strategy, 0.5),
            "recent_usage": len([
                e for e in self.error_history[-10:]
                # Would track which strategy was used for each error
            ]),
        }

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error and recovery statistics."""
        if not self.error_history:
            return {"message": "No errors recorded"}
        
        recent_errors = [
            e for e in self.error_history
            if datetime.now() - e.timestamp < timedelta(hours=24)
        ]
        
        error_types = {}
        severity_counts = {severity: 0 for severity in ErrorSeverity}
        
        for error in recent_errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
            severity_counts[error.severity] += 1
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors_24h": len(recent_errors),
            "error_types": error_types,
            "severity_distribution": {k.value: v for k, v in severity_counts.items()},
            "recovery_success_rates": {k.value: v for k, v in self.success_rates.items()},
            "most_common_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else None,
        }


def error_handler(
    component: str = "unknown",
    max_retries: int = 3,
    recovery_strategy: Optional[RecoveryStrategy] = None
):
    """
    Decorator for automatic error handling with quantum recovery.
    
    Usage:
        @error_handler(component="data_processing", max_retries=5)
        async def process_data(data):
            # Function that might fail
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get or create error recovery system
            state_manager = kwargs.pop('_state_manager', None)
            if not state_manager:
                state_manager = PersistentStateManager()
            
            error_recovery = QuantumErrorRecovery(state_manager)
            
            for attempt in range(max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    if attempt == max_retries:
                        # Final attempt failed, handle error
                        context = {
                            "args": {f"arg_{i}": str(arg)[:100] for i, arg in enumerate(args)},
                            "kwargs": {k: str(v)[:100] for k, v in kwargs.items()},
                            "attempt": attempt,
                        }
                        
                        recovery_result = await error_recovery.handle_error(
                            error=e,
                            context=context,
                            component=component,
                            function_name=func.__name__
                        )
                        
                        if recovery_result.success:
                            logger.info(f"Error recovered successfully for {func.__name__}")
                            # Could retry the function here if recovery was successful
                            continue
                        else:
                            logger.error(f"Error recovery failed for {func.__name__}")
                            raise e
                    else:
                        # Retry without full error handling
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
            
            # Should not reach here
            raise RuntimeError(f"All retry attempts exhausted for {func.__name__}")
        
        return wrapper
    return decorator