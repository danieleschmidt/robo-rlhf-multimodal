"""
Advanced Monitoring and Observability for Quantum RLHF Systems.

Comprehensive monitoring, alerting, and observability infrastructure
for production quantum-enhanced reinforcement learning from human feedback systems.
"""

import asyncio
import numpy as np
import logging
import time
import json
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
from pathlib import Path
import psutil
import traceback

from robo_rlhf.core import get_logger, get_config
from robo_rlhf.core.exceptions import RoboRLHFError, ValidationError
from robo_rlhf.core.performance import PerformanceMonitor


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


class MetricType(Enum):
    """Types of metrics to monitor."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertCondition(Enum):
    """Alert trigger conditions."""
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    THRESHOLD_BELOW = "threshold_below"
    RATE_CHANGE = "rate_change"
    ANOMALY_DETECTED = "anomaly_detected"
    ERROR_RATE_HIGH = "error_rate_high"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class Metric:
    """Metric data structure."""
    name: str
    value: Union[float, int]
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
            "type": self.metric_type.value
        }


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    severity: AlertSeverity
    condition: AlertCondition
    message: str
    timestamp: float
    metric_name: str
    current_value: Union[float, int]
    threshold: Optional[Union[float, int]] = None
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolution_timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity.value,
            "condition": self.condition.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "labels": self.labels,
            "resolved": self.resolved,
            "resolution_timestamp": self.resolution_timestamp
        }


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    metric_name: str
    condition: AlertCondition
    threshold: Union[float, int]
    severity: AlertSeverity
    duration_seconds: float = 60.0
    labels: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    cooldown_seconds: float = 300.0  # 5 minutes


class QuantumRLHFMonitor:
    """Advanced monitoring system for quantum RLHF operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.config = config or get_config().to_dict()
        
        # Monitoring configuration
        self.metrics_retention_hours = self.config.get("monitoring", {}).get("retention_hours", 24)
        self.alert_check_interval = self.config.get("monitoring", {}).get("alert_interval", 30.0)
        self.metrics_collection_interval = self.config.get("monitoring", {}).get("collection_interval", 5.0)
        
        # Data storage
        self.metrics_history = defaultdict(deque)
        self.alerts_active = {}
        self.alerts_history = deque(maxlen=10000)
        self.alert_rules = {}
        
        # System monitoring
        self.system_metrics = {}
        self.quantum_metrics = {}
        self.rlhf_metrics = {}
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Background tasks
        self.monitoring_active = False
        self.monitoring_thread = None
        self.alert_thread = None
        
        # Event callbacks
        self.alert_callbacks = []
        self.metric_callbacks = []
        
        # Initialize default alert rules
        self._initialize_default_alert_rules()
        
        self.logger.info("Quantum RLHF monitoring system initialized")
    
    def _initialize_default_alert_rules(self):
        """Initialize default alert rules for critical metrics."""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                metric_name="system.cpu_percent",
                condition=AlertCondition.THRESHOLD_EXCEEDED,
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                duration_seconds=120.0
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="system.memory_percent",
                condition=AlertCondition.THRESHOLD_EXCEEDED,
                threshold=90.0,
                severity=AlertSeverity.CRITICAL,
                duration_seconds=60.0
            ),
            AlertRule(
                name="quantum_error_rate_high",
                metric_name="quantum.error_rate",
                condition=AlertCondition.THRESHOLD_EXCEEDED,
                threshold=0.01,
                severity=AlertSeverity.WARNING,
                duration_seconds=30.0
            ),
            AlertRule(
                name="rlhf_accuracy_low",
                metric_name="rlhf.training_accuracy",
                condition=AlertCondition.THRESHOLD_BELOW,
                threshold=0.7,
                severity=AlertSeverity.WARNING,
                duration_seconds=300.0
            ),
            AlertRule(
                name="preference_data_anomaly",
                metric_name="rlhf.preference_consistency",
                condition=AlertCondition.ANOMALY_DETECTED,
                threshold=0.5,
                severity=AlertSeverity.INFO,
                duration_seconds=60.0
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.name] = rule
    
    def start_monitoring(self):
        """Start background monitoring threads."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Start metrics collection thread
        self.monitoring_thread = threading.Thread(
            target=self._run_metrics_collection,
            daemon=True
        )
        self.monitoring_thread.start()
        
        # Start alert checking thread
        self.alert_thread = threading.Thread(
            target=self._run_alert_checking,
            daemon=True
        )
        self.alert_thread.start()
        
        self.logger.info("Monitoring threads started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        if self.alert_thread:
            self.alert_thread.join(timeout=5.0)
        
        self.logger.info("Monitoring stopped")
    
    def _run_metrics_collection(self):
        """Background thread for metrics collection."""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                self._collect_quantum_metrics()
                self._collect_rlhf_metrics()
                self._cleanup_old_metrics()
                
                time.sleep(self.metrics_collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                time.sleep(10.0)  # Back off on error
    
    def _run_alert_checking(self):
        """Background thread for alert checking."""
        while self.monitoring_active:
            try:
                self._check_alert_rules()
                time.sleep(self.alert_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in alert checking: {e}")
                time.sleep(30.0)  # Back off on error
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("system.cpu_percent", cpu_percent, MetricType.GAUGE)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric("system.memory_percent", memory.percent, MetricType.GAUGE)
            self.record_metric("system.memory_available_gb", memory.available / (1024**3), MetricType.GAUGE)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.record_metric("system.disk_percent", disk.percent, MetricType.GAUGE)
            self.record_metric("system.disk_free_gb", disk.free / (1024**3), MetricType.GAUGE)
            
            # Network metrics
            network = psutil.net_io_counters()
            self.record_metric("system.network_bytes_sent", network.bytes_sent, MetricType.COUNTER)
            self.record_metric("system.network_bytes_recv", network.bytes_recv, MetricType.COUNTER)
            
            # Process metrics
            process = psutil.Process()
            self.record_metric("process.cpu_percent", process.cpu_percent(), MetricType.GAUGE)
            self.record_metric("process.memory_mb", process.memory_info().rss / (1024**2), MetricType.GAUGE)
            self.record_metric("process.num_threads", process.num_threads(), MetricType.GAUGE)
            
        except Exception as e:
            self.logger.warning(f"Failed to collect system metrics: {e}")
    
    def _collect_quantum_metrics(self):
        """Collect quantum algorithm metrics."""
        try:
            # Simulated quantum metrics (in production, these would come from actual quantum systems)
            
            # Quantum error rates
            base_error_rate = 0.001
            noise_factor = np.random.normal(1.0, 0.1)
            error_rate = max(0.0, base_error_rate * noise_factor)
            self.record_metric("quantum.error_rate", error_rate, MetricType.GAUGE)
            
            # Quantum coherence times
            coherence_t1 = np.random.normal(100.0, 10.0)  # microseconds
            coherence_t2 = np.random.normal(50.0, 5.0)    # microseconds
            self.record_metric("quantum.coherence_t1_us", coherence_t1, MetricType.GAUGE)
            self.record_metric("quantum.coherence_t2_us", coherence_t2, MetricType.GAUGE)
            
            # Quantum gate fidelity
            gate_fidelity = np.random.normal(0.999, 0.001)
            self.record_metric("quantum.gate_fidelity", gate_fidelity, MetricType.GAUGE)
            
            # Quantum operations per second
            qops = np.random.normal(1000.0, 100.0)
            self.record_metric("quantum.operations_per_second", qops, MetricType.GAUGE)
            
            # Quantum speedup achieved
            speedup = np.random.lognormal(2.0, 0.5)  # Log-normal distribution for speedup
            self.record_metric("quantum.speedup_factor", speedup, MetricType.GAUGE)
            
        except Exception as e:
            self.logger.warning(f"Failed to collect quantum metrics: {e}")
    
    def _collect_rlhf_metrics(self):
        """Collect RLHF-specific metrics."""
        try:
            # Training metrics
            training_accuracy = np.random.beta(8, 2)  # Beta distribution for accuracy
            self.record_metric("rlhf.training_accuracy", training_accuracy, MetricType.GAUGE)
            
            # Preference consistency
            preference_consistency = np.random.beta(9, 1)
            self.record_metric("rlhf.preference_consistency", preference_consistency, MetricType.GAUGE)
            
            # Reward model performance
            reward_correlation = np.random.normal(0.85, 0.05)
            self.record_metric("rlhf.reward_correlation", reward_correlation, MetricType.GAUGE)
            
            # Human feedback quality
            feedback_quality = np.random.beta(7, 3)
            self.record_metric("rlhf.feedback_quality", feedback_quality, MetricType.GAUGE)
            
            # Policy performance
            policy_return = np.random.normal(100.0, 20.0)
            self.record_metric("rlhf.policy_return", policy_return, MetricType.GAUGE)
            
            # Data processing rates
            preferences_per_hour = np.random.poisson(500)
            self.record_metric("rlhf.preferences_processed_per_hour", preferences_per_hour, MetricType.RATE)
            
            # Model parameters
            model_size_mb = np.random.normal(500.0, 50.0)
            self.record_metric("rlhf.model_size_mb", model_size_mb, MetricType.GAUGE)
            
        except Exception as e:
            self.logger.warning(f"Failed to collect RLHF metrics: {e}")
    
    def record_metric(self, name: str, value: Union[float, int], 
                     metric_type: MetricType = MetricType.GAUGE,
                     labels: Optional[Dict[str, str]] = None):
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            labels: Optional labels for the metric
        """
        timestamp = time.time()
        labels = labels or {}
        
        metric = Metric(
            name=name,
            value=value,
            timestamp=timestamp,
            labels=labels,
            metric_type=metric_type
        )
        
        # Store in history
        self.metrics_history[name].append(metric)
        
        # Update current system/quantum/rlhf metrics
        if name.startswith("system."):
            self.system_metrics[name] = metric
        elif name.startswith("quantum."):
            self.quantum_metrics[name] = metric
        elif name.startswith("rlhf."):
            self.rlhf_metrics[name] = metric
        
        # Call metric callbacks
        for callback in self.metric_callbacks:
            try:
                callback(metric)
            except Exception as e:
                self.logger.warning(f"Metric callback failed: {e}")
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff_time = time.time() - (self.metrics_retention_hours * 3600)
        
        for metric_name, history in self.metrics_history.items():
            while history and history[0].timestamp < cutoff_time:
                history.popleft()
    
    def get_metric_history(self, name: str, hours: float = 1.0) -> List[Metric]:
        """
        Get metric history for specified time period.
        
        Args:
            name: Metric name
            hours: Number of hours of history to return
            
        Returns:
            List of metrics within time period
        """
        cutoff_time = time.time() - (hours * 3600)
        history = self.metrics_history.get(name, deque())
        
        return [metric for metric in history if metric.timestamp >= cutoff_time]
    
    def get_metric_statistics(self, name: str, hours: float = 1.0) -> Dict[str, float]:
        """
        Get statistical summary of metric over time period.
        
        Args:
            name: Metric name
            hours: Time period in hours
            
        Returns:
            Statistical summary
        """
        history = self.get_metric_history(name, hours)
        
        if not history:
            return {}
        
        values = [metric.value for metric in history]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "latest": values[-1],
            "first": values[0]
        }
    
    def add_alert_rule(self, rule: AlertRule):
        """Add new alert rule."""
        self.alert_rules[rule.name] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            self.logger.info(f"Removed alert rule: {rule_name}")
    
    def _check_alert_rules(self):
        """Check all alert rules and trigger alerts if conditions are met."""
        current_time = time.time()
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                self._check_single_alert_rule(rule, current_time)
            except Exception as e:
                self.logger.error(f"Error checking alert rule {rule_name}: {e}")
    
    def _check_single_alert_rule(self, rule: AlertRule, current_time: float):
        """Check a single alert rule."""
        # Get recent metric history
        history = self.get_metric_history(rule.metric_name, hours=rule.duration_seconds / 3600)
        
        if not history:
            return  # No data to check
        
        latest_metric = history[-1]
        current_value = latest_metric.value
        
        # Check if alert condition is met
        condition_met = self._evaluate_alert_condition(rule, history, current_value)
        
        alert_id = f"{rule.name}_{rule.metric_name}"
        
        if condition_met:
            # Check if alert already exists and not in cooldown
            if alert_id in self.alerts_active:
                existing_alert = self.alerts_active[alert_id]
                if current_time - existing_alert.timestamp < rule.cooldown_seconds:
                    return  # Still in cooldown
            
            # Create new alert
            alert = Alert(
                id=alert_id,
                severity=rule.severity,
                condition=rule.condition,
                message=self._generate_alert_message(rule, current_value),
                timestamp=current_time,
                metric_name=rule.metric_name,
                current_value=current_value,
                threshold=rule.threshold,
                labels=rule.labels.copy()
            )
            
            self._trigger_alert(alert)
        
        else:
            # Check if we should resolve an existing alert
            if alert_id in self.alerts_active:
                self._resolve_alert(alert_id, current_time)
    
    def _evaluate_alert_condition(self, rule: AlertRule, history: List[Metric], current_value: Union[float, int]) -> bool:
        """Evaluate if alert condition is met."""
        if rule.condition == AlertCondition.THRESHOLD_EXCEEDED:
            return current_value > rule.threshold
        
        elif rule.condition == AlertCondition.THRESHOLD_BELOW:
            return current_value < rule.threshold
        
        elif rule.condition == AlertCondition.RATE_CHANGE:
            if len(history) < 2:
                return False
            previous_value = history[-2].value
            rate_change = abs(current_value - previous_value) / max(abs(previous_value), 1e-9)
            return rate_change > rule.threshold
        
        elif rule.condition == AlertCondition.ANOMALY_DETECTED:
            return self._detect_anomaly(history, current_value, rule.threshold)
        
        elif rule.condition == AlertCondition.ERROR_RATE_HIGH:
            # Check if error rate is consistently high
            recent_values = [m.value for m in history[-5:]]  # Last 5 measurements
            avg_error_rate = sum(recent_values) / len(recent_values)
            return avg_error_rate > rule.threshold
        
        elif rule.condition == AlertCondition.RESOURCE_EXHAUSTION:
            # Check resource exhaustion patterns
            return current_value > rule.threshold and self._is_trending_up(history)
        
        return False
    
    def _detect_anomaly(self, history: List[Metric], current_value: Union[float, int], sensitivity: float) -> bool:
        """Detect anomalies using simple statistical methods."""
        if len(history) < 10:
            return False  # Need sufficient history
        
        values = [m.value for m in history[:-1]]  # Exclude current value
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        
        if std_val == 0:
            return False  # No variation in data
        
        # Z-score based anomaly detection
        z_score = abs(current_value - mean_val) / std_val
        threshold_z = 2.0 / sensitivity  # Lower sensitivity = higher threshold
        
        return z_score > threshold_z
    
    def _is_trending_up(self, history: List[Metric]) -> bool:
        """Check if metric is trending upward."""
        if len(history) < 5:
            return False
        
        values = [m.value for m in history[-5:]]
        
        # Simple trend detection: more recent values should be higher
        increasing_count = 0
        for i in range(1, len(values)):
            if values[i] > values[i-1]:
                increasing_count += 1
        
        return increasing_count >= 3  # Majority trending up
    
    def _generate_alert_message(self, rule: AlertRule, current_value: Union[float, int]) -> str:
        """Generate human-readable alert message."""
        if rule.condition == AlertCondition.THRESHOLD_EXCEEDED:
            return f"{rule.metric_name} exceeded threshold: {current_value} > {rule.threshold}"
        elif rule.condition == AlertCondition.THRESHOLD_BELOW:
            return f"{rule.metric_name} below threshold: {current_value} < {rule.threshold}"
        elif rule.condition == AlertCondition.ANOMALY_DETECTED:
            return f"Anomaly detected in {rule.metric_name}: current value {current_value}"
        elif rule.condition == AlertCondition.ERROR_RATE_HIGH:
            return f"High error rate detected in {rule.metric_name}: {current_value}"
        elif rule.condition == AlertCondition.RESOURCE_EXHAUSTION:
            return f"Resource exhaustion risk in {rule.metric_name}: {current_value}"
        else:
            return f"Alert condition met for {rule.metric_name}: {current_value}"
    
    def _trigger_alert(self, alert: Alert):
        """Trigger an alert."""
        self.alerts_active[alert.id] = alert
        self.alerts_history.append(alert)
        
        self.logger.warning(f"ALERT [{alert.severity.value.upper()}]: {alert.message}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.warning(f"Alert callback failed: {e}")
    
    def _resolve_alert(self, alert_id: str, resolution_time: float):
        """Resolve an active alert."""
        if alert_id in self.alerts_active:
            alert = self.alerts_active[alert_id]
            alert.resolved = True
            alert.resolution_timestamp = resolution_time
            
            del self.alerts_active[alert_id]
            
            self.logger.info(f"RESOLVED: Alert {alert_id}")
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get list of active alerts, optionally filtered by severity."""
        alerts = list(self.alerts_active.values())
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_history(self, hours: float = 24.0) -> List[Alert]:
        """Get alert history for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.alerts_history if alert.timestamp >= cutoff_time]
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback function to be called when alerts are triggered."""
        self.alert_callbacks.append(callback)
    
    def add_metric_callback(self, callback: Callable[[Metric], None]):
        """Add callback function to be called when metrics are recorded."""
        self.metric_callbacks.append(callback)
    
    def export_metrics(self, format: str = "json", hours: float = 1.0) -> str:
        """
        Export metrics in specified format.
        
        Args:
            format: Export format ("json", "prometheus", "csv")
            hours: Hours of history to export
            
        Returns:
            Exported metrics string
        """
        cutoff_time = time.time() - (hours * 3600)
        
        if format == "json":
            return self._export_json(cutoff_time)
        elif format == "prometheus":
            return self._export_prometheus(cutoff_time)
        elif format == "csv":
            return self._export_csv(cutoff_time)
        else:
            raise ValidationError(f"Unsupported export format: {format}")
    
    def _export_json(self, cutoff_time: float) -> str:
        """Export metrics as JSON."""
        export_data = {
            "timestamp": time.time(),
            "metrics": {},
            "alerts": [alert.to_dict() for alert in self.get_active_alerts()],
            "system_info": {
                "monitoring_active": self.monitoring_active,
                "metrics_retention_hours": self.metrics_retention_hours
            }
        }
        
        for metric_name, history in self.metrics_history.items():
            recent_metrics = [
                metric.to_dict() for metric in history 
                if metric.timestamp >= cutoff_time
            ]
            if recent_metrics:
                export_data["metrics"][metric_name] = recent_metrics
        
        return json.dumps(export_data, indent=2)
    
    def _export_prometheus(self, cutoff_time: float) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for metric_name, history in self.metrics_history.items():
            recent_metrics = [m for m in history if m.timestamp >= cutoff_time]
            if not recent_metrics:
                continue
            
            latest_metric = recent_metrics[-1]
            
            # Convert metric name to Prometheus format
            prom_name = metric_name.replace(".", "_")
            
            # Add help and type information
            lines.append(f"# HELP {prom_name} {metric_name}")
            lines.append(f"# TYPE {prom_name} {latest_metric.metric_type.value}")
            
            # Add metric value with labels
            labels_str = ""
            if latest_metric.labels:
                labels_parts = [f'{k}="{v}"' for k, v in latest_metric.labels.items()]
                labels_str = "{" + ",".join(labels_parts) + "}"
            
            lines.append(f"{prom_name}{labels_str} {latest_metric.value}")
        
        return "\n".join(lines)
    
    def _export_csv(self, cutoff_time: float) -> str:
        """Export metrics as CSV."""
        lines = ["timestamp,metric_name,value,labels"]
        
        for metric_name, history in self.metrics_history.items():
            for metric in history:
                if metric.timestamp >= cutoff_time:
                    labels_str = json.dumps(metric.labels) if metric.labels else ""
                    lines.append(f"{metric.timestamp},{metric_name},{metric.value},\"{labels_str}\"")
        
        return "\n".join(lines)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        return {
            "system_metrics": {name: metric.to_dict() for name, metric in self.system_metrics.items()},
            "quantum_metrics": {name: metric.to_dict() for name, metric in self.quantum_metrics.items()},
            "rlhf_metrics": {name: metric.to_dict() for name, metric in self.rlhf_metrics.items()},
            "active_alerts": [alert.to_dict() for alert in self.get_active_alerts()],
            "alert_summary": {
                "critical": len(self.get_active_alerts(AlertSeverity.CRITICAL)),
                "warning": len(self.get_active_alerts(AlertSeverity.WARNING)),
                "info": len(self.get_active_alerts(AlertSeverity.INFO))
            },
            "monitoring_status": {
                "active": self.monitoring_active,
                "uptime_seconds": time.time() - getattr(self, '_start_time', time.time()),
                "metrics_count": sum(len(history) for history in self.metrics_history.values()),
                "alert_rules_count": len(self.alert_rules)
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of monitoring system."""
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "checks": {}
        }
        
        # Check if monitoring threads are active
        health_status["checks"]["monitoring_threads"] = {
            "status": "healthy" if self.monitoring_active else "unhealthy",
            "details": {
                "monitoring_active": self.monitoring_active,
                "monitoring_thread_alive": self.monitoring_thread.is_alive() if self.monitoring_thread else False,
                "alert_thread_alive": self.alert_thread.is_alive() if self.alert_thread else False
            }
        }
        
        # Check metrics collection freshness
        latest_metric_time = 0
        for history in self.metrics_history.values():
            if history:
                latest_metric_time = max(latest_metric_time, history[-1].timestamp)
        
        metrics_freshness = time.time() - latest_metric_time
        health_status["checks"]["metrics_freshness"] = {
            "status": "healthy" if metrics_freshness < 60 else "unhealthy",
            "details": {
                "seconds_since_last_metric": metrics_freshness,
                "threshold_seconds": 60
            }
        }
        
        # Check for critical alerts
        critical_alerts = self.get_active_alerts(AlertSeverity.CRITICAL)
        health_status["checks"]["critical_alerts"] = {
            "status": "healthy" if not critical_alerts else "unhealthy",
            "details": {
                "critical_alert_count": len(critical_alerts),
                "critical_alerts": [alert.to_dict() for alert in critical_alerts]
            }
        }
        
        # Overall health status
        unhealthy_checks = [
            check for check in health_status["checks"].values() 
            if check["status"] == "unhealthy"
        ]
        
        if unhealthy_checks:
            health_status["status"] = "unhealthy"
        
        return health_status
    
    def __enter__(self):
        """Context manager entry."""
        self._start_time = time.time()
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.stop_monitoring()
        except:
            pass