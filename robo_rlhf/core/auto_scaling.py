"""
Autonomous Auto-Scaling and Self-Healing Infrastructure for Quantum RLHF.

Intelligent infrastructure that automatically scales resources, heals failures,
and optimizes performance based on real-time workload patterns and quantum computation requirements.
"""

import asyncio
import numpy as np
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import statistics
import psutil
from pathlib import Path

from robo_rlhf.core import get_logger, get_config
from robo_rlhf.core.exceptions import RoboRLHFError, ValidationError
from robo_rlhf.core.performance import PerformanceMonitor
from robo_rlhf.core.advanced_monitoring import QuantumRLHFMonitor, Alert, AlertSeverity


class ScalingDirection(Enum):
    """Scaling directions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"  # Horizontal scaling
    SCALE_IN = "scale_in"    # Horizontal scaling in


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class ScalingTrigger(Enum):
    """Triggers for scaling actions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    QUANTUM_RESOURCE_EXHAUSTION = "quantum_resource_exhaustion"
    PREDICTION_BASED = "prediction_based"
    WORKLOAD_PATTERN = "workload_pattern"


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    name: str
    trigger: ScalingTrigger
    metric_name: str
    scale_up_threshold: float
    scale_down_threshold: float
    scale_up_adjustment: int = 1
    scale_down_adjustment: int = 1
    cooldown_seconds: float = 300.0
    evaluation_periods: int = 2
    enabled: bool = True
    min_instances: int = 1
    max_instances: int = 10


@dataclass
class ResourceInstance:
    """Resource instance representation."""
    instance_id: str
    instance_type: str
    cpu_cores: int
    memory_gb: float
    gpu_count: int = 0
    status: HealthStatus = HealthStatus.HEALTHY
    created_at: float = field(default_factory=time.time)
    last_health_check: float = field(default_factory=time.time)
    workload: float = 0.0
    error_count: int = 0
    success_count: int = 0


@dataclass
class WorkloadPrediction:
    """Workload prediction model."""
    timestamp: float
    predicted_cpu: float
    predicted_memory: float
    predicted_queue_length: float
    confidence: float
    prediction_horizon_minutes: int


class PredictiveModel:
    """Machine learning model for workload prediction."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.historical_data = deque(maxlen=window_size)
        self.model_params = {"alpha": 0.3, "beta": 0.3, "gamma": 0.4}  # Exponential smoothing
        
    def add_observation(self, timestamp: float, cpu_util: float, memory_util: float, queue_length: float):
        """Add new observation to the model."""
        self.historical_data.append({
            "timestamp": timestamp,
            "cpu_util": cpu_util,
            "memory_util": memory_util,
            "queue_length": queue_length
        })
    
    def predict(self, horizon_minutes: int = 15) -> WorkloadPrediction:
        """Predict future workload."""
        if len(self.historical_data) < 5:
            # Not enough data for prediction
            return WorkloadPrediction(
                timestamp=time.time(),
                predicted_cpu=0.5,
                predicted_memory=0.5,
                predicted_queue_length=10.0,
                confidence=0.1,
                prediction_horizon_minutes=horizon_minutes
            )
        
        # Simple exponential smoothing prediction
        recent_data = list(self.historical_data)[-20:]  # Last 20 observations
        
        # Calculate trends
        cpu_values = [d["cpu_util"] for d in recent_data]
        memory_values = [d["memory_util"] for d in recent_data]
        queue_values = [d["queue_length"] for d in recent_data]
        
        # Exponential smoothing prediction
        alpha = self.model_params["alpha"]
        
        predicted_cpu = self._exponential_smooth(cpu_values, alpha)
        predicted_memory = self._exponential_smooth(memory_values, alpha)
        predicted_queue = self._exponential_smooth(queue_values, alpha)
        
        # Add seasonal/trend adjustments
        predicted_cpu = self._apply_trend_adjustment(predicted_cpu, cpu_values)
        predicted_memory = self._apply_trend_adjustment(predicted_memory, memory_values)
        predicted_queue = self._apply_trend_adjustment(predicted_queue, queue_values)
        
        # Calculate confidence based on data variance
        confidence = self._calculate_confidence(recent_data)
        
        return WorkloadPrediction(
            timestamp=time.time(),
            predicted_cpu=max(0.0, min(1.0, predicted_cpu)),
            predicted_memory=max(0.0, min(1.0, predicted_memory)),
            predicted_queue_length=max(0.0, predicted_queue),
            confidence=confidence,
            prediction_horizon_minutes=horizon_minutes
        )
    
    def _exponential_smooth(self, values: List[float], alpha: float) -> float:
        """Apply exponential smoothing."""
        if not values:
            return 0.5
        
        smoothed = values[0]
        for value in values[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        
        return smoothed
    
    def _apply_trend_adjustment(self, base_prediction: float, historical_values: List[float]) -> float:
        """Apply trend adjustment to prediction."""
        if len(historical_values) < 5:
            return base_prediction
        
        # Calculate simple linear trend
        x = list(range(len(historical_values)))
        y = historical_values
        
        # Simple linear regression
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi ** 2 for xi in x)
        
        if n * sum_x2 - sum_x ** 2 != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            
            # Apply trend adjustment (project forward)
            trend_adjustment = slope * 5  # Project 5 steps forward
            return base_prediction + trend_adjustment
        
        return base_prediction
    
    def _calculate_confidence(self, data: List[Dict[str, Any]]) -> float:
        """Calculate prediction confidence based on data quality."""
        if len(data) < 5:
            return 0.2
        
        # Calculate variance in each metric
        cpu_values = [d["cpu_util"] for d in data]
        memory_values = [d["memory_util"] for d in data]
        queue_values = [d["queue_length"] for d in data]
        
        cpu_variance = statistics.variance(cpu_values) if len(cpu_values) > 1 else 1.0
        memory_variance = statistics.variance(memory_values) if len(memory_values) > 1 else 1.0
        queue_variance = statistics.variance(queue_values) if len(queue_values) > 1 else 1.0
        
        # Lower variance = higher confidence
        avg_variance = (cpu_variance + memory_variance + queue_variance / 100) / 3
        confidence = max(0.1, min(0.95, 1.0 - avg_variance))
        
        return confidence


class AutoScaler:
    """Intelligent auto-scaling system for quantum RLHF infrastructure."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.config = config or get_config().to_dict()
        
        # Auto-scaling configuration
        self.scaling_enabled = self.config.get("autoscaling", {}).get("enabled", True)
        self.check_interval = self.config.get("autoscaling", {}).get("check_interval", 30.0)
        self.prediction_enabled = self.config.get("autoscaling", {}).get("prediction_enabled", True)
        
        # Resource management
        self.instances = {}
        self.scaling_policies = {}
        self.scaling_history = deque(maxlen=1000)
        self.instance_lock = threading.RLock()
        
        # Predictive modeling
        self.predictive_model = PredictiveModel()
        self.prediction_history = deque(maxlen=100)
        
        # Monitoring integration
        self.monitor = QuantumRLHFMonitor(config)
        
        # Background tasks
        self.scaling_active = False
        self.scaling_thread = None
        self.health_check_thread = None
        
        # Performance tracking
        self.performance_monitor = PerformanceMonitor()
        self.scaling_stats = defaultdict(int)
        
        # Initialize default policies
        self._initialize_default_policies()
        
        self.logger.info("Auto-scaler initialized")
    
    def _initialize_default_policies(self):
        """Initialize default scaling policies."""
        default_policies = [
            ScalingPolicy(
                name="cpu_utilization",
                trigger=ScalingTrigger.CPU_UTILIZATION,
                metric_name="system.cpu_percent",
                scale_up_threshold=80.0,
                scale_down_threshold=30.0,
                scale_up_adjustment=2,
                scale_down_adjustment=1,
                cooldown_seconds=300.0,
                min_instances=1,
                max_instances=20
            ),
            ScalingPolicy(
                name="memory_utilization",
                trigger=ScalingTrigger.MEMORY_UTILIZATION,
                metric_name="system.memory_percent",
                scale_up_threshold=85.0,
                scale_down_threshold=40.0,
                scale_up_adjustment=1,
                scale_down_adjustment=1,
                cooldown_seconds=300.0,
                min_instances=1,
                max_instances=15
            ),
            ScalingPolicy(
                name="quantum_queue_length",
                trigger=ScalingTrigger.QUEUE_LENGTH,
                metric_name="quantum.queue_length",
                scale_up_threshold=50.0,
                scale_down_threshold=5.0,
                scale_up_adjustment=3,
                scale_down_adjustment=1,
                cooldown_seconds=120.0,
                min_instances=2,
                max_instances=50
            ),
            ScalingPolicy(
                name="response_time",
                trigger=ScalingTrigger.RESPONSE_TIME,
                metric_name="rlhf.response_time_seconds",
                scale_up_threshold=5.0,
                scale_down_threshold=1.0,
                scale_up_adjustment=2,
                scale_down_adjustment=1,
                cooldown_seconds=180.0,
                min_instances=1,
                max_instances=25
            ),
            ScalingPolicy(
                name="error_rate",
                trigger=ScalingTrigger.ERROR_RATE,
                metric_name="system.error_rate",
                scale_up_threshold=0.05,  # 5% error rate
                scale_down_threshold=0.01,  # 1% error rate
                scale_up_adjustment=2,
                scale_down_adjustment=1,
                cooldown_seconds=60.0,
                min_instances=2,
                max_instances=30
            )
        ]
        
        for policy in default_policies:
            self.scaling_policies[policy.name] = policy
    
    def start_autoscaling(self):
        """Start auto-scaling background processes."""
        if self.scaling_active:
            return
        
        self.scaling_active = True
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Start scaling thread
        self.scaling_thread = threading.Thread(
            target=self._run_scaling_loop,
            daemon=True
        )
        self.scaling_thread.start()
        
        # Start health check thread
        self.health_check_thread = threading.Thread(
            target=self._run_health_checks,
            daemon=True
        )
        self.health_check_thread.start()
        
        self.logger.info("Auto-scaling started")
    
    def stop_autoscaling(self):
        """Stop auto-scaling processes."""
        self.scaling_active = False
        
        if self.scaling_thread:
            self.scaling_thread.join(timeout=10.0)
        
        if self.health_check_thread:
            self.health_check_thread.join(timeout=10.0)
        
        self.monitor.stop_monitoring()
        
        self.logger.info("Auto-scaling stopped")
    
    def _run_scaling_loop(self):
        """Main scaling loop."""
        while self.scaling_active:
            try:
                current_time = time.time()
                
                # Collect current metrics
                self._collect_metrics_for_prediction()
                
                # Check scaling policies
                if self.scaling_enabled:
                    self._evaluate_scaling_policies()
                
                # Predictive scaling
                if self.prediction_enabled:
                    self._evaluate_predictive_scaling()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in scaling loop: {e}")
                time.sleep(60.0)  # Back off on error
    
    def _run_health_checks(self):
        """Run periodic health checks on instances."""
        while self.scaling_active:
            try:
                self._perform_health_checks()
                time.sleep(30.0)  # Health checks every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in health checks: {e}")
                time.sleep(60.0)
    
    def _collect_metrics_for_prediction(self):
        """Collect metrics for predictive modeling."""
        # Get current system metrics
        dashboard_data = self.monitor.get_dashboard_data()
        system_metrics = dashboard_data.get("system_metrics", {})
        
        cpu_util = system_metrics.get("system.cpu_percent", {}).get("value", 0) / 100.0
        memory_util = system_metrics.get("system.memory_percent", {}).get("value", 0) / 100.0
        
        # Estimate queue length (in production, this would be actual queue metrics)
        queue_length = len(self.instances) * 10 + np.random.poisson(5)
        
        # Add to predictive model
        self.predictive_model.add_observation(
            timestamp=time.time(),
            cpu_util=cpu_util,
            memory_util=memory_util,
            queue_length=queue_length
        )
    
    def _evaluate_scaling_policies(self):
        """Evaluate all scaling policies and take actions."""
        current_time = time.time()
        
        for policy_name, policy in self.scaling_policies.items():
            if not policy.enabled:
                continue
            
            try:
                should_scale, direction, adjustment = self._should_scale_based_on_policy(policy)
                
                if should_scale:
                    # Check cooldown period
                    last_scaling = self._get_last_scaling_time(policy_name)
                    if current_time - last_scaling < policy.cooldown_seconds:
                        continue  # Still in cooldown
                    
                    # Execute scaling action
                    success = self._execute_scaling_action(policy, direction, adjustment)
                    
                    if success:
                        self._record_scaling_action(policy_name, direction, adjustment, current_time)
                
            except Exception as e:
                self.logger.error(f"Error evaluating policy {policy_name}: {e}")
    
    def _should_scale_based_on_policy(self, policy: ScalingPolicy) -> Tuple[bool, ScalingDirection, int]:
        """Check if scaling is needed based on policy."""
        # Get recent metric values
        metric_history = self.monitor.get_metric_history(policy.metric_name, hours=0.1)  # Last 6 minutes
        
        if len(metric_history) < policy.evaluation_periods:
            return False, ScalingDirection.SCALE_UP, 0
        
        # Get recent values for evaluation
        recent_values = [metric.value for metric in metric_history[-policy.evaluation_periods:]]
        avg_value = statistics.mean(recent_values)
        
        # Check scaling conditions
        if avg_value > policy.scale_up_threshold:
            current_instances = len(self._get_healthy_instances())
            if current_instances < policy.max_instances:
                return True, ScalingDirection.SCALE_UP, policy.scale_up_adjustment
        
        elif avg_value < policy.scale_down_threshold:
            current_instances = len(self._get_healthy_instances())
            if current_instances > policy.min_instances:
                return True, ScalingDirection.SCALE_DOWN, policy.scale_down_adjustment
        
        return False, ScalingDirection.SCALE_UP, 0
    
    def _evaluate_predictive_scaling(self):
        """Evaluate predictive scaling opportunities."""
        try:
            # Generate prediction
            prediction = self.predictive_model.predict(horizon_minutes=15)
            self.prediction_history.append(prediction)
            
            # Only act on high-confidence predictions
            if prediction.confidence < 0.7:
                return
            
            current_instances = len(self._get_healthy_instances())
            
            # Predictive scale-up
            if (prediction.predicted_cpu > 0.8 or 
                prediction.predicted_memory > 0.85 or 
                prediction.predicted_queue_length > 100):
                
                if current_instances < 50:  # Max predictive instances
                    self._execute_predictive_scaling(ScalingDirection.SCALE_UP, 1, prediction)
            
            # Predictive scale-down
            elif (prediction.predicted_cpu < 0.3 and 
                  prediction.predicted_memory < 0.4 and 
                  prediction.predicted_queue_length < 10):
                
                if current_instances > 2:  # Min predictive instances
                    self._execute_predictive_scaling(ScalingDirection.SCALE_DOWN, 1, prediction)
        
        except Exception as e:
            self.logger.error(f"Error in predictive scaling: {e}")
    
    def _execute_scaling_action(self, policy: ScalingPolicy, direction: ScalingDirection, adjustment: int) -> bool:
        """Execute scaling action."""
        try:
            if direction == ScalingDirection.SCALE_UP:
                return self._scale_up(adjustment, f"Policy: {policy.name}")
            elif direction == ScalingDirection.SCALE_DOWN:
                return self._scale_down(adjustment, f"Policy: {policy.name}")
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing scaling action: {e}")
            return False
    
    def _execute_predictive_scaling(self, direction: ScalingDirection, adjustment: int, 
                                  prediction: WorkloadPrediction) -> bool:
        """Execute predictive scaling action."""
        reason = f"Predictive (confidence: {prediction.confidence:.2f})"
        
        if direction == ScalingDirection.SCALE_UP:
            return self._scale_up(adjustment, reason)
        elif direction == ScalingDirection.SCALE_DOWN:
            return self._scale_down(adjustment, reason)
        
        return False
    
    def _scale_up(self, count: int, reason: str) -> bool:
        """Scale up by adding instances."""
        with self.instance_lock:
            successful_launches = 0
            
            for i in range(count):
                instance = self._launch_instance(reason)
                if instance:
                    successful_launches += 1
            
            if successful_launches > 0:
                self.scaling_stats["scale_up_actions"] += 1
                self.scaling_stats["instances_launched"] += successful_launches
                self.logger.info(f"Scaled up by {successful_launches} instances. Reason: {reason}")
                return True
            
            return False
    
    def _scale_down(self, count: int, reason: str) -> bool:
        """Scale down by removing instances."""
        with self.instance_lock:
            healthy_instances = self._get_healthy_instances()
            
            if len(healthy_instances) <= count:
                return False  # Can't scale down below minimum
            
            # Select instances to terminate (least utilized first)
            instances_to_terminate = sorted(healthy_instances, key=lambda x: x.workload)[:count]
            
            successful_terminations = 0
            for instance in instances_to_terminate:
                if self._terminate_instance(instance.instance_id, reason):
                    successful_terminations += 1
            
            if successful_terminations > 0:
                self.scaling_stats["scale_down_actions"] += 1
                self.scaling_stats["instances_terminated"] += successful_terminations
                self.logger.info(f"Scaled down by {successful_terminations} instances. Reason: {reason}")
                return True
            
            return False
    
    def _launch_instance(self, reason: str) -> Optional[ResourceInstance]:
        """Launch a new instance."""
        try:
            instance_id = f"instance_{int(time.time() * 1000)}"
            
            # In production, this would interact with cloud APIs
            # For simulation, create a mock instance
            instance = ResourceInstance(
                instance_id=instance_id,
                instance_type="quantum_worker",
                cpu_cores=4,
                memory_gb=8.0,
                gpu_count=0,
                status=HealthStatus.HEALTHY
            )
            
            self.instances[instance_id] = instance
            
            self.logger.info(f"Launched instance {instance_id}. Reason: {reason}")
            return instance
            
        except Exception as e:
            self.logger.error(f"Failed to launch instance: {e}")
            return None
    
    def _terminate_instance(self, instance_id: str, reason: str) -> bool:
        """Terminate an instance."""
        try:
            if instance_id in self.instances:
                # In production, this would gracefully shut down the instance
                # For simulation, just remove from tracking
                del self.instances[instance_id]
                
                self.logger.info(f"Terminated instance {instance_id}. Reason: {reason}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to terminate instance {instance_id}: {e}")
            return False
    
    def _get_healthy_instances(self) -> List[ResourceInstance]:
        """Get list of healthy instances."""
        return [
            instance for instance in self.instances.values()
            if instance.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        ]
    
    def _perform_health_checks(self):
        """Perform health checks on all instances."""
        with self.instance_lock:
            for instance_id, instance in list(self.instances.items()):
                try:
                    health_status = self._check_instance_health(instance)
                    instance.status = health_status
                    instance.last_health_check = time.time()
                    
                    # Auto-heal unhealthy instances
                    if health_status == HealthStatus.CRITICAL:
                        self._auto_heal_instance(instance)
                
                except Exception as e:
                    self.logger.error(f"Health check failed for {instance_id}: {e}")
                    instance.status = HealthStatus.UNHEALTHY
    
    def _check_instance_health(self, instance: ResourceInstance) -> HealthStatus:
        """Check health of a specific instance."""
        # In production, this would check actual instance metrics
        # For simulation, use random health based on error rates
        
        success_rate = instance.success_count / max(1, instance.success_count + instance.error_count)
        
        if success_rate > 0.95:
            return HealthStatus.HEALTHY
        elif success_rate > 0.85:
            return HealthStatus.DEGRADED
        elif success_rate > 0.70:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.CRITICAL
    
    def _auto_heal_instance(self, instance: ResourceInstance):
        """Attempt to automatically heal an unhealthy instance."""
        try:
            self.logger.warning(f"Auto-healing instance {instance.instance_id}")
            
            # Restart instance (in production, this would restart services)
            instance.error_count = 0
            instance.status = HealthStatus.DEGRADED
            
            # If auto-healing fails repeatedly, replace instance
            if instance.error_count > 10:
                self.logger.error(f"Replacing critically unhealthy instance {instance.instance_id}")
                
                # Launch replacement
                replacement = self._launch_instance("Auto-healing replacement")
                
                # Terminate unhealthy instance
                if replacement:
                    self._terminate_instance(instance.instance_id, "Auto-healing replacement")
            
            self.scaling_stats["auto_heal_attempts"] += 1
            
        except Exception as e:
            self.logger.error(f"Auto-healing failed for {instance.instance_id}: {e}")
    
    def _get_last_scaling_time(self, policy_name: str) -> float:
        """Get the timestamp of the last scaling action for a policy."""
        for record in reversed(self.scaling_history):
            if record.get("policy") == policy_name:
                return record.get("timestamp", 0)
        
        return 0  # No previous scaling action
    
    def _record_scaling_action(self, policy_name: str, direction: ScalingDirection, 
                             adjustment: int, timestamp: float):
        """Record scaling action in history."""
        record = {
            "timestamp": timestamp,
            "policy": policy_name,
            "direction": direction.value,
            "adjustment": adjustment,
            "instances_before": len(self.instances),
            "instances_after": len(self.instances)  # This would be updated after scaling
        }
        
        self.scaling_history.append(record)
    
    def add_scaling_policy(self, policy: ScalingPolicy):
        """Add or update a scaling policy."""
        self.scaling_policies[policy.name] = policy
        self.logger.info(f"Added scaling policy: {policy.name}")
    
    def remove_scaling_policy(self, policy_name: str):
        """Remove a scaling policy."""
        if policy_name in self.scaling_policies:
            del self.scaling_policies[policy_name]
            self.logger.info(f"Removed scaling policy: {policy_name}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and statistics."""
        healthy_instances = self._get_healthy_instances()
        
        return {
            "scaling_enabled": self.scaling_enabled,
            "prediction_enabled": self.prediction_enabled,
            "total_instances": len(self.instances),
            "healthy_instances": len(healthy_instances),
            "instance_distribution": {
                status.value: len([i for i in self.instances.values() if i.status == status])
                for status in HealthStatus
            },
            "scaling_policies": {
                name: {
                    "enabled": policy.enabled,
                    "trigger": policy.trigger.value,
                    "scale_up_threshold": policy.scale_up_threshold,
                    "scale_down_threshold": policy.scale_down_threshold
                }
                for name, policy in self.scaling_policies.items()
            },
            "scaling_statistics": dict(self.scaling_stats),
            "recent_predictions": [
                {
                    "timestamp": p.timestamp,
                    "predicted_cpu": p.predicted_cpu,
                    "predicted_memory": p.predicted_memory,
                    "confidence": p.confidence
                }
                for p in list(self.prediction_history)[-5:]
            ],
            "recent_scaling_actions": list(self.scaling_history)[-10:]
        }
    
    def simulate_load_test(self, duration_minutes: int = 30, load_pattern: str = "gradual_increase"):
        """
        Simulate load test to validate auto-scaling behavior.
        
        Args:
            duration_minutes: Duration of load test
            load_pattern: Load pattern ("gradual_increase", "spike", "oscillating")
        """
        self.logger.info(f"Starting load test: {load_pattern} for {duration_minutes} minutes")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        test_thread = threading.Thread(
            target=self._run_load_test,
            args=(start_time, end_time, load_pattern),
            daemon=True
        )
        test_thread.start()
    
    def _run_load_test(self, start_time: float, end_time: float, load_pattern: str):
        """Run load test simulation."""
        while time.time() < end_time:
            try:
                progress = (time.time() - start_time) / (end_time - start_time)
                
                # Generate synthetic load based on pattern
                if load_pattern == "gradual_increase":
                    cpu_load = progress * 0.9  # Gradually increase to 90%
                    memory_load = progress * 0.8
                    queue_length = progress * 100
                
                elif load_pattern == "spike":
                    # Sudden spike at 50% progress
                    if 0.4 < progress < 0.6:
                        cpu_load = 0.95
                        memory_load = 0.90
                        queue_length = 200
                    else:
                        cpu_load = 0.3
                        memory_load = 0.4
                        queue_length = 10
                
                elif load_pattern == "oscillating":
                    # Oscillating load
                    cycle = np.sin(progress * 4 * np.pi) * 0.5 + 0.5
                    cpu_load = 0.3 + cycle * 0.6
                    memory_load = 0.25 + cycle * 0.65
                    queue_length = 10 + cycle * 90
                
                else:
                    # Default steady load
                    cpu_load = 0.5
                    memory_load = 0.5
                    queue_length = 25
                
                # Record synthetic metrics
                self.monitor.record_metric("system.cpu_percent", cpu_load * 100)
                self.monitor.record_metric("system.memory_percent", memory_load * 100)
                self.monitor.record_metric("quantum.queue_length", queue_length)
                
                # Update instance workloads
                with self.instance_lock:
                    for instance in self.instances.values():
                        instance.workload = cpu_load + np.random.normal(0, 0.1)
                        
                        # Simulate some successes/errors
                        if np.random.random() < 0.9:  # 90% success rate
                            instance.success_count += 1
                        else:
                            instance.error_count += 1
                
                time.sleep(5.0)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in load test: {e}")
                break
        
        self.logger.info("Load test completed")
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for optimizing auto-scaling configuration."""
        recommendations = []
        
        # Analyze scaling history
        if len(self.scaling_history) > 10:
            recent_actions = list(self.scaling_history)[-50:]
            
            # Check for thrashing (too frequent scaling)
            scale_up_count = len([a for a in recent_actions if a["direction"] == "scale_up"])
            scale_down_count = len([a for a in recent_actions if a["direction"] == "scale_down"])
            
            if abs(scale_up_count - scale_down_count) < 2 and len(recent_actions) > 20:
                recommendations.append({
                    "type": "thrashing_detected",
                    "severity": "medium",
                    "description": "Frequent up/down scaling detected. Consider adjusting thresholds or cooldown periods.",
                    "suggestion": "Increase cooldown periods by 50% or adjust thresholds to create larger deadband."
                })
        
        # Check prediction accuracy
        if len(self.prediction_history) > 10:
            avg_confidence = np.mean([p.confidence for p in self.prediction_history])
            
            if avg_confidence < 0.5:
                recommendations.append({
                    "type": "low_prediction_confidence",
                    "severity": "low",
                    "description": f"Prediction confidence is low ({avg_confidence:.2f}). Predictive scaling may not be effective.",
                    "suggestion": "Consider disabling predictive scaling or collecting more historical data."
                })
        
        # Check instance health patterns
        unhealthy_rate = len([i for i in self.instances.values() if i.status != HealthStatus.HEALTHY]) / max(1, len(self.instances))
        
        if unhealthy_rate > 0.2:
            recommendations.append({
                "type": "high_unhealthy_rate",
                "severity": "high",
                "description": f"High rate of unhealthy instances ({unhealthy_rate:.1%}). May indicate resource constraints or configuration issues.",
                "suggestion": "Check resource allocation and instance health check parameters."
            })
        
        return recommendations
    
    def __enter__(self):
        """Context manager entry."""
        self.start_autoscaling()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_autoscaling()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.stop_autoscaling()
        except:
            pass