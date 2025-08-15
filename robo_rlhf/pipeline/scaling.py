"""
Pipeline Scaling: Advanced auto-scaling and performance optimization for pipeline components.

Implements intelligent scaling algorithms, resource optimization, and performance monitoring.
"""

import asyncio
import logging
import time
import math
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import statistics

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling directions."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingStrategy(Enum):
    """Scaling strategies."""
    REACTIVE = "reactive"           # Scale based on current metrics
    PREDICTIVE = "predictive"       # Scale based on predictions
    SCHEDULED = "scheduled"         # Scale based on schedule
    HYBRID = "hybrid"              # Combine multiple strategies


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU = "cpu"
    MEMORY = "memory"
    INSTANCES = "instances"
    BANDWIDTH = "bandwidth"
    STORAGE = "storage"


@dataclass
class ScalingRule:
    """Configuration for a scaling rule."""
    metric_name: str
    threshold_up: float
    threshold_down: float
    scale_factor: float = 1.5
    cooldown_period: int = 300  # 5 minutes
    min_instances: int = 1
    max_instances: int = 10
    resource_type: ResourceType = ResourceType.INSTANCES


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    component: str
    direction: ScalingDirection
    factor: float
    reason: str
    timestamp: float
    before_value: float
    after_value: float
    success: bool
    duration: float


@dataclass
class PredictionModel:
    """Model for predicting resource needs."""
    lookback_window: int = 60  # minutes
    prediction_horizon: int = 30  # minutes
    accuracy_threshold: float = 0.8
    last_trained: float = 0.0
    training_interval: float = 3600.0  # 1 hour


class AutoScaler:
    """
    Intelligent auto-scaling system for pipeline components.
    
    Features:
    - Multi-metric scaling decisions
    - Predictive scaling based on historical patterns
    - Resource optimization and cost management
    - Scaling event tracking and analysis
    - Integration with quantum optimization
    """
    
    def __init__(
        self,
        component_name: str,
        scaling_strategy: ScalingStrategy = ScalingStrategy.HYBRID
    ):
        self.component_name = component_name
        self.scaling_strategy = scaling_strategy
        
        # Scaling configuration
        self.scaling_rules: List[ScalingRule] = []
        self.current_instances = 1
        self.current_resources: Dict[ResourceType, float] = {
            ResourceType.CPU: 1.0,
            ResourceType.MEMORY: 1.0,
            ResourceType.BANDWIDTH: 1.0,
            ResourceType.STORAGE: 1.0
        }
        
        # Metrics and history
        self.metrics_history: Dict[str, deque] = {}
        self.scaling_events: List[ScalingEvent] = []
        self.last_scaling_time: Dict[str, float] = {}
        
        # Prediction models
        self.prediction_models: Dict[str, PredictionModel] = {}
        
        # Performance tracking
        self.scaling_effectiveness: Dict[str, List[float]] = {}
        self.cost_optimization_score = 0.0
        
        # Quantum integration if available
        try:
            from robo_rlhf.quantum import QuantumOptimizer, PredictiveAnalytics
            self.quantum_optimizer = QuantumOptimizer()
            self.predictive_analytics = PredictiveAnalytics()
            self.quantum_enabled = True
            logger.info(f"Quantum-enhanced auto-scaler for {component_name}")
        except ImportError:
            self.quantum_enabled = False
            logger.info(f"Standard auto-scaler for {component_name}")
    
    def add_scaling_rule(self, rule: ScalingRule) -> None:
        """Add a scaling rule."""
        self.scaling_rules.append(rule)
        
        # Initialize history tracking for this metric
        if rule.metric_name not in self.metrics_history:
            self.metrics_history[rule.metric_name] = deque(maxlen=1000)
        
        logger.info(f"Added scaling rule for {self.component_name}: {rule.metric_name}")
    
    def update_metric(self, metric_name: str, value: float, timestamp: float) -> None:
        """Update metric value for scaling decisions."""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = deque(maxlen=1000)
        
        self.metrics_history[metric_name].append({
            "value": value,
            "timestamp": timestamp
        })
    
    async def evaluate_scaling(self) -> Optional[ScalingEvent]:
        """Evaluate if scaling is needed and execute if required."""
        if self.scaling_strategy == ScalingStrategy.REACTIVE:
            return await self._reactive_scaling()
        elif self.scaling_strategy == ScalingStrategy.PREDICTIVE:
            return await self._predictive_scaling()
        elif self.scaling_strategy == ScalingStrategy.SCHEDULED:
            return await self._scheduled_scaling()
        elif self.scaling_strategy == ScalingStrategy.HYBRID:
            return await self._hybrid_scaling()
        
        return None
    
    async def _reactive_scaling(self) -> Optional[ScalingEvent]:
        """Reactive scaling based on current metrics."""
        scaling_decisions = []
        
        for rule in self.scaling_rules:
            decision = self._evaluate_scaling_rule(rule)
            if decision:
                scaling_decisions.append(decision)
        
        # If multiple decisions, prioritize most critical
        if scaling_decisions:
            # Sort by urgency (higher values = more urgent)
            scaling_decisions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            direction, urgency, rule = scaling_decisions[0]
            return await self._execute_scaling(direction, rule, f"Reactive: {rule.metric_name}")
        
        return None
    
    async def _predictive_scaling(self) -> Optional[ScalingEvent]:
        """Predictive scaling based on forecasted demand."""
        try:
            if self.quantum_enabled:
                return await self._quantum_predictive_scaling()
            else:
                return await self._statistical_predictive_scaling()
        except Exception as e:
            logger.error(f"Predictive scaling failed for {self.component_name}: {e}")
            return None
    
    async def _quantum_predictive_scaling(self) -> Optional[ScalingEvent]:
        """Quantum-enhanced predictive scaling."""
        try:
            # Prepare historical data
            metrics_data = {}
            for metric_name, history in self.metrics_history.items():
                if len(history) >= 30:  # Need sufficient data
                    metrics_data[metric_name] = [
                        point["value"] for point in list(history)[-100:]
                    ]
            
            if not metrics_data:
                return None
            
            # Get quantum predictions
            predictions = await self.predictive_analytics.predict_resource_demand(
                component=self.component_name,
                metrics_data=metrics_data,
                prediction_horizon=30  # 30 minutes
            )
            
            # Analyze predictions for scaling decisions
            for metric_name, predicted_values in predictions.get("metrics", {}).items():
                rule = self._get_rule_for_metric(metric_name)
                if not rule:
                    continue
                
                # Get max predicted value in next period
                max_predicted = max(predicted_values)
                
                # Check if scaling needed based on prediction
                if max_predicted > rule.threshold_up:
                    return await self._execute_scaling(
                        ScalingDirection.UP,
                        rule,
                        f"Quantum prediction: {metric_name} will reach {max_predicted:.2f}"
                    )
                elif max_predicted < rule.threshold_down:
                    return await self._execute_scaling(
                        ScalingDirection.DOWN,
                        rule,
                        f"Quantum prediction: {metric_name} will drop to {max_predicted:.2f}"
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Quantum predictive scaling failed: {e}")
            return await self._statistical_predictive_scaling()
    
    async def _statistical_predictive_scaling(self) -> Optional[ScalingEvent]:
        """Statistical predictive scaling using time series analysis."""
        for rule in self.scaling_rules:
            history = self.metrics_history.get(rule.metric_name)
            if not history or len(history) < 30:
                continue
            
            # Simple trend analysis
            recent_values = [point["value"] for point in list(history)[-30:]]
            
            if len(recent_values) >= 10:
                # Calculate trend using linear regression
                trend = self._calculate_trend(recent_values)
                current_value = recent_values[-1]
                
                # Predict value in next period (simplified)
                predicted_value = current_value + (trend * 10)  # 10 time periods ahead
                
                # Check if scaling needed based on prediction
                if predicted_value > rule.threshold_up:
                    return await self._execute_scaling(
                        ScalingDirection.UP,
                        rule,
                        f"Trend prediction: {rule.metric_name} trending to {predicted_value:.2f}"
                    )
                elif predicted_value < rule.threshold_down and current_value > rule.threshold_down:
                    return await self._execute_scaling(
                        ScalingDirection.DOWN,
                        rule,
                        f"Trend prediction: {rule.metric_name} trending to {predicted_value:.2f}"
                    )
        
        return None
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend using simple linear regression."""
        n = len(values)
        if n < 2:
            return 0.0
        
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    async def _scheduled_scaling(self) -> Optional[ScalingEvent]:
        """Scheduled scaling based on time patterns."""
        # This would implement time-based scaling rules
        # For now, return None (placeholder implementation)
        return None
    
    async def _hybrid_scaling(self) -> Optional[ScalingEvent]:
        """Hybrid scaling combining multiple strategies."""
        # Try predictive first
        predictive_result = await self._predictive_scaling()
        if predictive_result:
            return predictive_result
        
        # Fall back to reactive
        reactive_result = await self._reactive_scaling()
        if reactive_result:
            return reactive_result
        
        # Check scheduled rules
        scheduled_result = await self._scheduled_scaling()
        return scheduled_result
    
    def _evaluate_scaling_rule(self, rule: ScalingRule) -> Optional[Tuple[ScalingDirection, float, ScalingRule]]:
        """Evaluate a single scaling rule."""
        history = self.metrics_history.get(rule.metric_name)
        if not history:
            return None
        
        # Check cooldown period
        last_scaling = self.last_scaling_time.get(rule.metric_name, 0)
        if time.time() - last_scaling < rule.cooldown_period:
            return None
        
        # Get current value
        current_value = history[-1]["value"]
        
        # Calculate urgency based on how far we are from threshold
        if current_value > rule.threshold_up:
            urgency = (current_value - rule.threshold_up) / rule.threshold_up
            return (ScalingDirection.UP, urgency, rule)
        elif current_value < rule.threshold_down:
            urgency = (rule.threshold_down - current_value) / rule.threshold_down
            return (ScalingDirection.DOWN, urgency, rule)
        
        return None
    
    def _get_rule_for_metric(self, metric_name: str) -> Optional[ScalingRule]:
        """Get scaling rule for a specific metric."""
        for rule in self.scaling_rules:
            if rule.metric_name == metric_name:
                return rule
        return None
    
    async def _execute_scaling(
        self,
        direction: ScalingDirection,
        rule: ScalingRule,
        reason: str
    ) -> ScalingEvent:
        """Execute scaling operation."""
        start_time = time.time()
        
        # Get current value
        before_value = self.current_instances if rule.resource_type == ResourceType.INSTANCES else \
                      self.current_resources[rule.resource_type]
        
        # Calculate new value
        if direction == ScalingDirection.UP:
            scale_factor = rule.scale_factor
            new_value = before_value * scale_factor
            
            # Apply max limits
            if rule.resource_type == ResourceType.INSTANCES:
                new_value = min(new_value, rule.max_instances)
            else:
                new_value = min(new_value, 10.0)  # Arbitrary max for other resources
        
        else:  # DOWN
            scale_factor = 1.0 / rule.scale_factor
            new_value = before_value * scale_factor
            
            # Apply min limits
            if rule.resource_type == ResourceType.INSTANCES:
                new_value = max(new_value, rule.min_instances)
            else:
                new_value = max(new_value, 0.1)  # Arbitrary min for other resources
        
        # Execute the scaling (this would integrate with actual infrastructure)
        success = await self._perform_scaling(rule.resource_type, new_value)
        
        # Update internal state
        if success:
            if rule.resource_type == ResourceType.INSTANCES:
                self.current_instances = int(new_value)
            else:
                self.current_resources[rule.resource_type] = new_value
            
            self.last_scaling_time[rule.metric_name] = time.time()
        
        # Create scaling event
        event = ScalingEvent(
            component=self.component_name,
            direction=direction,
            factor=new_value / before_value,
            reason=reason,
            timestamp=start_time,
            before_value=before_value,
            after_value=new_value if success else before_value,
            success=success,
            duration=time.time() - start_time
        )
        
        self.scaling_events.append(event)
        
        # Keep only recent events
        if len(self.scaling_events) > 100:
            self.scaling_events = self.scaling_events[-100:]
        
        logger.info(
            f"Scaling {direction.value} for {self.component_name}: "
            f"{before_value:.1f} -> {new_value:.1f} ({reason})"
        )
        
        return event
    
    async def _perform_scaling(self, resource_type: ResourceType, new_value: float) -> bool:
        """Perform actual scaling operation."""
        try:
            # This is where integration with infrastructure would happen
            # For now, simulate scaling with a delay
            await asyncio.sleep(0.1)
            
            # Simulate occasional scaling failures
            import random
            if random.random() < 0.05:  # 5% failure rate
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Scaling operation failed: {e}")
            return False
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        total_events = len(self.scaling_events)
        successful_events = sum(1 for event in self.scaling_events if event.success)
        
        # Calculate average scaling time
        successful_durations = [
            event.duration for event in self.scaling_events if event.success
        ]
        avg_scaling_time = statistics.mean(successful_durations) if successful_durations else 0.0
        
        # Count events by direction
        up_events = sum(1 for event in self.scaling_events if event.direction == ScalingDirection.UP)
        down_events = sum(1 for event in self.scaling_events if event.direction == ScalingDirection.DOWN)
        
        return {
            "component": self.component_name,
            "strategy": self.scaling_strategy.value,
            "current_instances": self.current_instances,
            "current_resources": {rt.value: val for rt, val in self.current_resources.items()},
            "total_scaling_events": total_events,
            "successful_scaling_events": successful_events,
            "success_rate": successful_events / total_events if total_events > 0 else 0.0,
            "average_scaling_time": avg_scaling_time,
            "scaling_events_up": up_events,
            "scaling_events_down": down_events,
            "quantum_enabled": self.quantum_enabled,
            "active_rules": len(self.scaling_rules)
        }


class PerformanceOptimizer:
    """
    Performance optimization system for pipeline components.
    
    Optimizes resource allocation, caching strategies, and execution patterns.
    """
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        
        # Performance metrics
        self.response_times: deque = deque(maxlen=1000)
        self.throughput_history: deque = deque(maxlen=1000)
        self.resource_usage: Dict[str, deque] = {
            "cpu": deque(maxlen=1000),
            "memory": deque(maxlen=1000),
            "io": deque(maxlen=1000)
        }
        
        # Optimization state
        self.current_optimizations: Dict[str, Any] = {}
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Cache management
        self.cache_hit_rate = 0.0
        self.cache_size = 0
        self.optimal_cache_size = 100  # MB
        
        # Connection pooling
        self.connection_pool_size = 10
        self.optimal_pool_size = 10
        
        # Quantum integration
        try:
            from robo_rlhf.quantum import QuantumOptimizer
            self.quantum_optimizer = QuantumOptimizer()
            self.quantum_enabled = True
        except ImportError:
            self.quantum_enabled = False
    
    def record_performance_metric(
        self,
        metric_type: str,
        value: float,
        timestamp: float
    ) -> None:
        """Record performance metric."""
        metric_data = {"value": value, "timestamp": timestamp}
        
        if metric_type == "response_time":
            self.response_times.append(metric_data)
        elif metric_type == "throughput":
            self.throughput_history.append(metric_data)
        elif metric_type in self.resource_usage:
            self.resource_usage[metric_type].append(metric_data)
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Perform comprehensive performance optimization."""
        optimizations = {}
        
        # Cache optimization
        cache_opt = await self._optimize_cache()
        if cache_opt:
            optimizations["cache"] = cache_opt
        
        # Connection pool optimization
        pool_opt = await self._optimize_connection_pool()
        if pool_opt:
            optimizations["connection_pool"] = pool_opt
        
        # Resource allocation optimization
        resource_opt = await self._optimize_resource_allocation()
        if resource_opt:
            optimizations["resource_allocation"] = resource_opt
        
        # Quantum optimization if available
        if self.quantum_enabled:
            quantum_opt = await self._quantum_optimization()
            if quantum_opt:
                optimizations["quantum"] = quantum_opt
        
        # Record optimization
        if optimizations:
            self.optimization_history.append({
                "timestamp": time.time(),
                "optimizations": optimizations
            })
            
            # Keep only recent history
            if len(self.optimization_history) > 50:
                self.optimization_history = self.optimization_history[-50:]
        
        return optimizations
    
    async def _optimize_cache(self) -> Optional[Dict[str, Any]]:
        """Optimize caching strategy."""
        if not self.response_times:
            return None
        
        # Calculate current performance metrics
        recent_response_times = [rt["value"] for rt in list(self.response_times)[-100:]]
        avg_response_time = statistics.mean(recent_response_times)
        
        # Determine optimal cache size based on performance
        if avg_response_time > 1.0 and self.cache_hit_rate < 0.8:
            # Increase cache size
            new_cache_size = min(self.optimal_cache_size * 1.5, 500)  # Max 500MB
            
            if new_cache_size != self.optimal_cache_size:
                self.optimal_cache_size = new_cache_size
                
                return {
                    "action": "increase_cache_size",
                    "new_size_mb": new_cache_size,
                    "reason": f"Low hit rate ({self.cache_hit_rate:.2f}) and slow responses ({avg_response_time:.2f}s)"
                }
        
        elif avg_response_time < 0.1 and self.cache_hit_rate > 0.95:
            # Decrease cache size to save memory
            new_cache_size = max(self.optimal_cache_size * 0.8, 10)  # Min 10MB
            
            if new_cache_size != self.optimal_cache_size:
                self.optimal_cache_size = new_cache_size
                
                return {
                    "action": "decrease_cache_size",
                    "new_size_mb": new_cache_size,
                    "reason": f"High hit rate ({self.cache_hit_rate:.2f}) and fast responses ({avg_response_time:.2f}s)"
                }
        
        return None
    
    async def _optimize_connection_pool(self) -> Optional[Dict[str, Any]]:
        """Optimize connection pool size."""
        if not self.throughput_history:
            return None
        
        # Calculate throughput trend
        recent_throughput = [th["value"] for th in list(self.throughput_history)[-50:]]
        
        if len(recent_throughput) < 10:
            return None
        
        avg_throughput = statistics.mean(recent_throughput)
        
        # Simple heuristic for pool sizing
        optimal_pool_size = max(5, min(int(avg_throughput / 10), 50))
        
        if optimal_pool_size != self.optimal_pool_size:
            old_size = self.optimal_pool_size
            self.optimal_pool_size = optimal_pool_size
            
            return {
                "action": "adjust_pool_size",
                "old_size": old_size,
                "new_size": optimal_pool_size,
                "reason": f"Throughput-based optimization (avg: {avg_throughput:.1f} req/s)"
            }
        
        return None
    
    async def _optimize_resource_allocation(self) -> Optional[Dict[str, Any]]:
        """Optimize CPU and memory allocation."""
        optimizations = {}
        
        # CPU optimization
        if "cpu" in self.resource_usage and self.resource_usage["cpu"]:
            recent_cpu = [cpu["value"] for cpu in list(self.resource_usage["cpu"])[-50:]]
            avg_cpu = statistics.mean(recent_cpu)
            
            if avg_cpu > 0.8:
                optimizations["cpu"] = {
                    "action": "increase_cpu_allocation",
                    "current_usage": avg_cpu,
                    "recommendation": "Scale up CPU resources"
                }
            elif avg_cpu < 0.2:
                optimizations["cpu"] = {
                    "action": "decrease_cpu_allocation",
                    "current_usage": avg_cpu,
                    "recommendation": "Scale down CPU resources"
                }
        
        # Memory optimization
        if "memory" in self.resource_usage and self.resource_usage["memory"]:
            recent_memory = [mem["value"] for mem in list(self.resource_usage["memory"])[-50:]]
            avg_memory = statistics.mean(recent_memory)
            
            if avg_memory > 0.85:
                optimizations["memory"] = {
                    "action": "increase_memory_allocation",
                    "current_usage": avg_memory,
                    "recommendation": "Scale up memory resources"
                }
            elif avg_memory < 0.3:
                optimizations["memory"] = {
                    "action": "decrease_memory_allocation", 
                    "current_usage": avg_memory,
                    "recommendation": "Scale down memory resources"
                }
        
        return optimizations if optimizations else None
    
    async def _quantum_optimization(self) -> Optional[Dict[str, Any]]:
        """Quantum-enhanced performance optimization."""
        try:
            # Prepare performance data
            performance_data = {
                "response_times": [rt["value"] for rt in list(self.response_times)[-100:]],
                "throughput": [th["value"] for th in list(self.throughput_history)[-100:]],
                "resource_usage": {
                    name: [res["value"] for res in list(history)[-100:]]
                    for name, history in self.resource_usage.items()
                }
            }
            
            # Get quantum optimization recommendations
            optimization_result = await self.quantum_optimizer.optimize_performance(
                component=self.component_name,
                performance_data=performance_data,
                current_config=self.current_optimizations
            )
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Quantum optimization failed: {e}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance optimization statistics."""
        # Calculate current performance metrics
        current_metrics = {}
        
        if self.response_times:
            recent_rt = [rt["value"] for rt in list(self.response_times)[-50:]]
            current_metrics["avg_response_time"] = statistics.mean(recent_rt)
            current_metrics["p95_response_time"] = sorted(recent_rt)[int(0.95 * len(recent_rt))] if recent_rt else 0
        
        if self.throughput_history:
            recent_throughput = [th["value"] for th in list(self.throughput_history)[-50:]]
            current_metrics["avg_throughput"] = statistics.mean(recent_throughput)
        
        for resource_name, history in self.resource_usage.items():
            if history:
                recent_usage = [res["value"] for res in list(history)[-50:]]
                current_metrics[f"avg_{resource_name}_usage"] = statistics.mean(recent_usage)
        
        return {
            "component": self.component_name,
            "current_metrics": current_metrics,
            "cache_optimization": {
                "hit_rate": self.cache_hit_rate,
                "current_size_mb": self.cache_size,
                "optimal_size_mb": self.optimal_cache_size
            },
            "connection_pool": {
                "current_size": self.connection_pool_size,
                "optimal_size": self.optimal_pool_size
            },
            "total_optimizations": len(self.optimization_history),
            "quantum_enabled": self.quantum_enabled
        }


class LoadBalancer:
    """
    Intelligent load balancer for distributing requests across pipeline components.
    
    Implements multiple load balancing algorithms and adaptive routing.
    """
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.instances: Dict[str, Dict[str, Any]] = {}
        self.routing_stats: Dict[str, int] = {}
        self.health_check_interval = 30
        
        # Load balancing algorithms
        self.algorithms = {
            "round_robin": self._round_robin,
            "weighted_round_robin": self._weighted_round_robin,
            "least_connections": self._least_connections,
            "weighted_response_time": self._weighted_response_time,
            "quantum_optimal": self._quantum_optimal
        }
        
        self.current_algorithm = "weighted_response_time"
        self.round_robin_index = 0
        
        # Quantum integration
        try:
            from robo_rlhf.quantum import QuantumOptimizer
            self.quantum_optimizer = QuantumOptimizer()
            self.quantum_enabled = True
        except ImportError:
            self.quantum_enabled = False
    
    def add_instance(
        self,
        instance_id: str,
        endpoint: str,
        weight: float = 1.0
    ) -> None:
        """Add an instance to the load balancer."""
        self.instances[instance_id] = {
            "endpoint": endpoint,
            "weight": weight,
            "active_connections": 0,
            "total_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "healthy": True,
            "last_health_check": 0.0
        }
        
        self.routing_stats[instance_id] = 0
        logger.info(f"Added instance {instance_id} to load balancer")
    
    def remove_instance(self, instance_id: str) -> bool:
        """Remove an instance from the load balancer."""
        if instance_id in self.instances:
            del self.instances[instance_id]
            if instance_id in self.routing_stats:
                del self.routing_stats[instance_id]
            logger.info(f"Removed instance {instance_id} from load balancer")
            return True
        return False
    
    async def route_request(self, request_context: Dict[str, Any]) -> Optional[str]:
        """Route request to optimal instance."""
        healthy_instances = [
            instance_id for instance_id, instance in self.instances.items()
            if instance["healthy"]
        ]
        
        if not healthy_instances:
            logger.error("No healthy instances available")
            return None
        
        # Select instance using current algorithm
        algorithm = self.algorithms.get(self.current_algorithm, self._weighted_response_time)
        selected_instance = await algorithm(healthy_instances, request_context)
        
        if selected_instance:
            # Update routing stats
            self.routing_stats[selected_instance] += 1
            self.instances[selected_instance]["active_connections"] += 1
            self.instances[selected_instance]["total_requests"] += 1
        
        return selected_instance
    
    def complete_request(
        self,
        instance_id: str,
        success: bool,
        response_time: float
    ) -> None:
        """Mark request as completed and update metrics."""
        if instance_id not in self.instances:
            return
        
        instance = self.instances[instance_id]
        instance["active_connections"] = max(0, instance["active_connections"] - 1)
        
        if not success:
            instance["failed_requests"] += 1
        
        # Update average response time (exponential moving average)
        alpha = 0.1  # Smoothing factor
        instance["avg_response_time"] = (
            alpha * response_time + (1 - alpha) * instance["avg_response_time"]
        )
    
    async def _round_robin(
        self,
        healthy_instances: List[str],
        request_context: Dict[str, Any]
    ) -> str:
        """Simple round-robin load balancing."""
        if not healthy_instances:
            return None
        
        selected = healthy_instances[self.round_robin_index % len(healthy_instances)]
        self.round_robin_index += 1
        return selected
    
    async def _weighted_round_robin(
        self,
        healthy_instances: List[str],
        request_context: Dict[str, Any]
    ) -> str:
        """Weighted round-robin based on instance weights."""
        if not healthy_instances:
            return None
        
        # Create weighted list
        weighted_instances = []
        for instance_id in healthy_instances:
            weight = int(self.instances[instance_id]["weight"] * 10)
            weighted_instances.extend([instance_id] * weight)
        
        if not weighted_instances:
            return healthy_instances[0]
        
        selected = weighted_instances[self.round_robin_index % len(weighted_instances)]
        self.round_robin_index += 1
        return selected
    
    async def _least_connections(
        self,
        healthy_instances: List[str],
        request_context: Dict[str, Any]
    ) -> str:
        """Route to instance with least active connections."""
        if not healthy_instances:
            return None
        
        return min(
            healthy_instances,
            key=lambda x: self.instances[x]["active_connections"]
        )
    
    async def _weighted_response_time(
        self,
        healthy_instances: List[str],
        request_context: Dict[str, Any]
    ) -> str:
        """Route based on weighted response time."""
        if not healthy_instances:
            return None
        
        # Calculate scores (lower is better)
        scores = {}
        for instance_id in healthy_instances:
            instance = self.instances[instance_id]
            
            # Combine response time and active connections
            response_score = instance["avg_response_time"]
            connection_score = instance["active_connections"] * 0.1
            weight_factor = 1.0 / instance["weight"]
            
            scores[instance_id] = (response_score + connection_score) * weight_factor
        
        # Select instance with best (lowest) score
        return min(scores.keys(), key=lambda x: scores[x])
    
    async def _quantum_optimal(
        self,
        healthy_instances: List[str],
        request_context: Dict[str, Any]
    ) -> str:
        """Quantum-optimized routing."""
        if not self.quantum_enabled:
            return await self._weighted_response_time(healthy_instances, request_context)
        
        try:
            # Prepare instance data for quantum optimization
            instance_data = {}
            for instance_id in healthy_instances:
                instance = self.instances[instance_id]
                instance_data[instance_id] = {
                    "response_time": instance["avg_response_time"],
                    "active_connections": instance["active_connections"],
                    "weight": instance["weight"],
                    "error_rate": instance["failed_requests"] / max(instance["total_requests"], 1)
                }
            
            # Get quantum routing decision
            optimal_instance = await self.quantum_optimizer.select_optimal_instance(
                instances=instance_data,
                request_context=request_context
            )
            
            return optimal_instance if optimal_instance in healthy_instances else healthy_instances[0]
            
        except Exception as e:
            logger.error(f"Quantum routing failed: {e}")
            return await self._weighted_response_time(healthy_instances, request_context)
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        total_requests = sum(self.routing_stats.values())
        
        instance_stats = {}
        for instance_id, instance in self.instances.items():
            requests = self.routing_stats.get(instance_id, 0)
            instance_stats[instance_id] = {
                "requests_routed": requests,
                "request_percentage": (requests / total_requests * 100) if total_requests > 0 else 0,
                "active_connections": instance["active_connections"],
                "avg_response_time": instance["avg_response_time"],
                "error_rate": instance["failed_requests"] / max(instance["total_requests"], 1),
                "healthy": instance["healthy"]
            }
        
        return {
            "component": self.component_name,
            "algorithm": self.current_algorithm,
            "total_instances": len(self.instances),
            "healthy_instances": sum(1 for i in self.instances.values() if i["healthy"]),
            "total_requests_routed": total_requests,
            "instance_stats": instance_stats,
            "quantum_enabled": self.quantum_enabled
        }