"""
Advanced Monitoring and Observability System for Autonomous SDLC.

Comprehensive monitoring with quantum-inspired analytics, predictive alerting,
and real-time system health tracking for robust autonomous execution.
"""

import asyncio
import time
import psutil
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import threading
from pathlib import Path
import numpy as np
from collections import deque, defaultdict
import aiohttp
import socket

from .logging import setup_logger
from .state_manager import PersistentStateManager, StateType
from .config import get_config


logger = setup_logger(__name__)


class MetricType(Enum):
    """Types of metrics to collect."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert information."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    source: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """System health check result."""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime
    response_time: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class QuantumMonitoringEngine:
    """
    Quantum-inspired monitoring engine with predictive analytics.
    
    Features:
    - Real-time metric collection and aggregation
    - Predictive alerting using quantum-inspired algorithms
    - Automated health checks and dependency monitoring
    - Performance trend analysis and anomaly detection
    - Integration with external monitoring systems
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get("monitoring", {})
        
        # Metric storage
        self.metrics_buffer: deque = deque(maxlen=10000)
        self.metric_aggregates: Dict[str, List[float]] = defaultdict(list)
        self.alerts: List[Alert] = []
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_tasks: List[asyncio.Task] = []
        self.metric_locks: Dict[str, threading.RLock] = defaultdict(threading.RLock)
        
        # Configuration
        self.collection_interval = self.config.get("collection_interval", 30)  # seconds
        self.retention_period = self.config.get("retention_period", 86400)  # 24 hours
        self.alert_cooldown = self.config.get("alert_cooldown", 300)  # 5 minutes
        self.health_check_interval = self.config.get("health_check_interval", 60)  # 1 minute
        
        # Quantum-inspired parameters
        self.anomaly_detection_threshold = self.config.get("anomaly_threshold", 2.0)  # standard deviations
        self.prediction_horizon = self.config.get("prediction_horizon", 3600)  # 1 hour
        self.quantum_correlation_enabled = self.config.get("quantum_correlation", True)
        
        # Alert tracking
        self.alert_history: deque = deque(maxlen=1000)
        self.last_alert_times: Dict[str, datetime] = {}
        
        logger.info("Quantum Monitoring Engine initialized")

    async def start_monitoring(self):
        """Start the monitoring system."""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._system_metrics_collector()),
            asyncio.create_task(self._performance_metrics_collector()),
            asyncio.create_task(self._health_check_monitor()),
            asyncio.create_task(self._anomaly_detector()),
            asyncio.create_task(self._alert_processor()),
            asyncio.create_task(self._metric_aggregator()),
        ]
        
        logger.info("Monitoring system started with {} tasks", len(self.monitoring_tasks))

    async def stop_monitoring(self):
        """Stop the monitoring system."""
        if not self.is_monitoring:
            logger.warning("Monitoring not running")
            return
        
        self.is_monitoring = False
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        self.monitoring_tasks.clear()
        logger.info("Monitoring system stopped")

    async def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a custom metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            labels=labels or {},
            metadata=metadata or {}
        )
        
        with self.metric_locks[name]:
            self.metrics_buffer.append(metric)
            self.metric_aggregates[name].append(value)
            
            # Keep only recent values for aggregation
            if len(self.metric_aggregates[name]) > 1000:
                self.metric_aggregates[name] = self.metric_aggregates[name][-1000:]
        
        # Check for anomalies
        await self._check_metric_anomaly(metric)

    async def create_alert(
        self,
        title: str,
        description: str,
        severity: AlertSeverity,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new alert."""
        alert_id = f"{source}_{int(time.time())}"
        
        # Check cooldown period
        last_alert_key = f"{source}_{title}"
        if last_alert_key in self.last_alert_times:
            time_since_last = datetime.now() - self.last_alert_times[last_alert_key]
            if time_since_last.total_seconds() < self.alert_cooldown:
                logger.debug(f"Alert suppressed due to cooldown: {title}")
                return alert_id
        
        alert = Alert(
            id=alert_id,
            title=title,
            description=description,
            severity=severity,
            source=source,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        self.alert_history.append(alert)
        self.last_alert_times[last_alert_key] = datetime.now()
        
        logger.warning(f"Alert created [{alert.severity.value}]: {alert.title}")
        
        # Send to external systems if configured
        await self._send_external_alert(alert)
        
        return alert_id

    async def resolve_alert(self, alert_id: str, resolution_note: Optional[str] = None):
        """Resolve an existing alert."""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                if resolution_note:
                    alert.metadata["resolution_note"] = resolution_note
                
                logger.info(f"Alert resolved: {alert.title}")
                return True
        
        return False

    async def register_health_check(
        self,
        component: str,
        check_function: Callable[[], Any],
        timeout: float = 5.0
    ):
        """Register a health check for a component."""
        health_check_key = f"health_check_{component}"
        
        async def wrapped_check():
            start_time = time.time()
            try:
                if asyncio.iscoroutinefunction(check_function):
                    result = await asyncio.wait_for(check_function(), timeout=timeout)
                else:
                    result = check_function()
                
                response_time = time.time() - start_time
                
                health_check = HealthCheck(
                    component=component,
                    status="healthy",
                    timestamp=datetime.now(),
                    response_time=response_time,
                    details={"result": str(result)[:200]}
                )
                
                self.health_checks[component] = health_check
                
            except Exception as e:
                response_time = time.time() - start_time
                
                health_check = HealthCheck(
                    component=component,
                    status="unhealthy",
                    timestamp=datetime.now(),
                    response_time=response_time,
                    error_message=str(e)
                )
                
                self.health_checks[component] = health_check
                
                # Create alert for failed health check
                await self.create_alert(
                    title=f"Health check failed: {component}",
                    description=f"Component {component} health check failed: {str(e)}",
                    severity=AlertSeverity.ERROR,
                    source="health_monitor",
                    metadata={"component": component, "error": str(e)}
                )
        
        # Store the check function for periodic execution
        if not hasattr(self, "_health_check_functions"):
            self._health_check_functions = {}
        self._health_check_functions[component] = wrapped_check

    async def get_metrics_summary(
        self,
        time_range: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get aggregated metrics summary."""
        time_range = time_range or timedelta(hours=1)
        cutoff_time = datetime.now() - time_range
        
        # Filter metrics by time range
        recent_metrics = [
            m for m in self.metrics_buffer
            if m.timestamp >= cutoff_time
        ]
        
        # Aggregate by metric name
        metric_summaries = {}
        for metric in recent_metrics:
            if metric.name not in metric_summaries:
                metric_summaries[metric.name] = {
                    "values": [],
                    "count": 0,
                    "metric_type": metric.metric_type.value,
                    "labels": metric.labels
                }
            
            metric_summaries[metric.name]["values"].append(metric.value)
            metric_summaries[metric.name]["count"] += 1
        
        # Calculate statistics
        summary = {}
        for name, data in metric_summaries.items():
            values = data["values"]
            if values:
                summary[name] = {
                    "count": data["count"],
                    "metric_type": data["metric_type"],
                    "latest_value": values[-1],
                    "average": np.mean(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "std_dev": np.std(values),
                    "percentiles": {
                        "p50": np.percentile(values, 50),
                        "p90": np.percentile(values, 90),
                        "p95": np.percentile(values, 95),
                        "p99": np.percentile(values, 99)
                    }
                }
        
        return summary

    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        current_time = datetime.now()
        
        # Count health check statuses
        health_status_counts = {"healthy": 0, "degraded": 0, "unhealthy": 0}
        component_details = {}
        
        for component, check in self.health_checks.items():
            # Check if health check is stale
            time_since_check = current_time - check.timestamp
            if time_since_check > timedelta(minutes=5):
                status = "degraded"
            else:
                status = check.status
            
            health_status_counts[status] += 1
            component_details[component] = {
                "status": status,
                "last_check": check.timestamp.isoformat(),
                "response_time": check.response_time,
                "error": check.error_message
            }
        
        # Determine overall health
        total_components = sum(health_status_counts.values())
        if total_components == 0:
            overall_health = "unknown"
        elif health_status_counts["unhealthy"] > 0:
            overall_health = "unhealthy"
        elif health_status_counts["degraded"] > 0:
            overall_health = "degraded"
        else:
            overall_health = "healthy"
        
        # Get recent alerts
        recent_alerts = [
            {
                "id": alert.id,
                "title": alert.title,
                "severity": alert.severity.value,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved
            }
            for alert in self.alerts[-10:]  # Last 10 alerts
        ]
        
        return {
            "overall_health": overall_health,
            "component_count": total_components,
            "health_distribution": health_status_counts,
            "component_details": component_details,
            "recent_alerts": recent_alerts,
            "monitoring_active": self.is_monitoring,
            "last_updated": current_time.isoformat()
        }

    async def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights using quantum-inspired analytics."""
        insights = {
            "trends": await self._analyze_trends(),
            "anomalies": await self._detect_anomalies(),
            "predictions": await self._generate_predictions(),
            "correlations": await self._find_correlations(),
            "recommendations": await self._generate_recommendations()
        }
        
        return insights

    async def export_metrics(self, filepath: str, format: str = "json"):
        """Export collected metrics to file."""
        if format.lower() == "json":
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "metric_count": len(self.metrics_buffer),
                "alert_count": len(self.alerts),
                "health_check_count": len(self.health_checks),
                "metrics": [asdict(m) for m in self.metrics_buffer],
                "alerts": [asdict(a) for a in self.alerts],
                "health_checks": [asdict(h) for h in self.health_checks.values()],
                "summary": await self.get_metrics_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to {filepath}")

    # Background monitoring tasks
    
    async def _system_metrics_collector(self):
        """Collect system-level metrics."""
        while self.is_monitoring:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                await self.record_metric("system.cpu.usage_percent", cpu_percent)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                await self.record_metric("system.memory.usage_percent", memory.percent)
                await self.record_metric("system.memory.used_bytes", memory.used)
                await self.record_metric("system.memory.available_bytes", memory.available)
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                await self.record_metric("system.disk.usage_percent", disk.percent)
                await self.record_metric("system.disk.used_bytes", disk.used)
                await self.record_metric("system.disk.free_bytes", disk.free)
                
                # Network metrics
                network = psutil.net_io_counters()
                await self.record_metric("system.network.bytes_sent", network.bytes_sent, MetricType.COUNTER)
                await self.record_metric("system.network.bytes_recv", network.bytes_recv, MetricType.COUNTER)
                
                # Process metrics
                process = psutil.Process()
                await self.record_metric("process.cpu.usage_percent", process.cpu_percent())
                await self.record_metric("process.memory.rss_bytes", process.memory_info().rss)
                await self.record_metric("process.threads.count", process.num_threads())
                
            except Exception as e:
                logger.error(f"System metrics collection failed: {e}")
            
            await asyncio.sleep(self.collection_interval)

    async def _performance_metrics_collector(self):
        """Collect application performance metrics."""
        while self.is_monitoring:
            try:
                # Application-specific metrics would go here
                # For now, simulate some performance metrics
                
                # Response time simulation
                response_time = np.random.normal(0.1, 0.02)  # 100ms ± 20ms
                await self.record_metric("app.response_time_seconds", max(0, response_time), MetricType.TIMER)
                
                # Throughput simulation
                throughput = np.random.poisson(50)  # ~50 requests per interval
                await self.record_metric("app.requests_per_minute", throughput, MetricType.COUNTER)
                
                # Error rate simulation
                error_rate = np.random.exponential(0.01)  # Low error rate
                await self.record_metric("app.error_rate_percent", min(100, error_rate * 100))
                
                # Queue lengths
                queue_length = max(0, np.random.normal(5, 2))
                await self.record_metric("app.queue_length", queue_length)
                
            except Exception as e:
                logger.error(f"Performance metrics collection failed: {e}")
            
            await asyncio.sleep(self.collection_interval)

    async def _health_check_monitor(self):
        """Execute registered health checks periodically."""
        while self.is_monitoring:
            try:
                if hasattr(self, "_health_check_functions"):
                    for component, check_func in self._health_check_functions.items():
                        try:
                            await check_func()
                        except Exception as e:
                            logger.error(f"Health check failed for {component}: {e}")
                
            except Exception as e:
                logger.error(f"Health check monitor failed: {e}")
            
            await asyncio.sleep(self.health_check_interval)

    async def _anomaly_detector(self):
        """Detect anomalies in metrics using quantum-inspired algorithms."""
        while self.is_monitoring:
            try:
                # Analyze each metric for anomalies
                for metric_name, values in self.metric_aggregates.items():
                    if len(values) >= 10:  # Need sufficient data
                        anomalies = await self._detect_metric_anomalies(metric_name, values)
                        
                        for anomaly in anomalies:
                            await self.create_alert(
                                title=f"Anomaly detected: {metric_name}",
                                description=f"Metric {metric_name} shows anomalous behavior: {anomaly['description']}",
                                severity=AlertSeverity.WARNING,
                                source="anomaly_detector",
                                metadata=anomaly
                            )
                
            except Exception as e:
                logger.error(f"Anomaly detection failed: {e}")
            
            await asyncio.sleep(60)  # Check every minute

    async def _alert_processor(self):
        """Process and manage alerts."""
        while self.is_monitoring:
            try:
                # Auto-resolve old alerts
                current_time = datetime.now()
                for alert in self.alerts:
                    if not alert.resolved:
                        age = current_time - alert.timestamp
                        if age > timedelta(hours=24):  # Auto-resolve after 24 hours
                            alert.resolved = True
                            alert.resolution_time = current_time
                            alert.metadata["auto_resolved"] = True
                
                # Clean up old resolved alerts
                self.alerts = [
                    alert for alert in self.alerts
                    if not alert.resolved or (current_time - alert.timestamp) < timedelta(days=7)
                ]
                
            except Exception as e:
                logger.error(f"Alert processing failed: {e}")
            
            await asyncio.sleep(300)  # Process every 5 minutes

    async def _metric_aggregator(self):
        """Aggregate and clean up old metrics."""
        while self.is_monitoring:
            try:
                cutoff_time = datetime.now() - timedelta(seconds=self.retention_period)
                
                # Remove old metrics from buffer
                self.metrics_buffer = deque(
                    [m for m in self.metrics_buffer if m.timestamp >= cutoff_time],
                    maxlen=self.metrics_buffer.maxlen
                )
                
                # Clean up metric aggregates
                for metric_name in list(self.metric_aggregates.keys()):
                    values = self.metric_aggregates[metric_name]
                    if len(values) > 1000:
                        self.metric_aggregates[metric_name] = values[-1000:]
                
            except Exception as e:
                logger.error(f"Metric aggregation failed: {e}")
            
            await asyncio.sleep(300)  # Aggregate every 5 minutes

    # Analytics methods
    
    async def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze metric trends."""
        trends = {}
        
        for metric_name, values in self.metric_aggregates.items():
            if len(values) >= 10:
                # Simple linear trend analysis
                x = np.arange(len(values))
                y = np.array(values)
                
                # Calculate trend slope
                slope = np.polyfit(x, y, 1)[0]
                
                # Determine trend direction
                if abs(slope) < 0.01:
                    direction = "stable"
                elif slope > 0:
                    direction = "increasing"
                else:
                    direction = "decreasing"
                
                trends[metric_name] = {
                    "direction": direction,
                    "slope": slope,
                    "recent_values": values[-5:],
                    "confidence": min(1.0, len(values) / 100.0)
                }
        
        return trends

    async def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies across all metrics."""
        anomalies = []
        
        for metric_name, values in self.metric_aggregates.items():
            if len(values) >= 20:
                metric_anomalies = await self._detect_metric_anomalies(metric_name, values)
                anomalies.extend(metric_anomalies)
        
        return anomalies

    async def _detect_metric_anomalies(self, metric_name: str, values: List[float]) -> List[Dict[str, Any]]:
        """Detect anomalies in a specific metric."""
        anomalies = []
        
        if len(values) < 10:
            return anomalies
        
        # Statistical anomaly detection
        mean = np.mean(values)
        std = np.std(values)
        
        # Z-score based detection
        for i, value in enumerate(values[-10:]):  # Check last 10 values
            z_score = abs(value - mean) / (std + 1e-8)
            if z_score > self.anomaly_detection_threshold:
                anomalies.append({
                    "metric": metric_name,
                    "value": value,
                    "z_score": z_score,
                    "mean": mean,
                    "std": std,
                    "description": f"Value {value:.3f} is {z_score:.2f} standard deviations from mean {mean:.3f}"
                })
        
        return anomalies

    async def _generate_predictions(self) -> Dict[str, Any]:
        """Generate predictions for metric values."""
        predictions = {}
        
        for metric_name, values in self.metric_aggregates.items():
            if len(values) >= 20:
                # Simple moving average prediction
                recent_values = values[-10:]
                prediction = np.mean(recent_values)
                
                # Trend-adjusted prediction
                if len(values) >= 20:
                    x = np.arange(len(values[-20:]))
                    y = np.array(values[-20:])
                    slope = np.polyfit(x, y, 1)[0]
                    prediction += slope * 5  # Project 5 time steps ahead
                
                predictions[metric_name] = {
                    "predicted_value": prediction,
                    "confidence": min(1.0, len(values) / 100.0),
                    "time_horizon": "5 intervals"
                }
        
        return predictions

    async def _find_correlations(self) -> Dict[str, Any]:
        """Find correlations between metrics."""
        correlations = {}
        
        if not self.quantum_correlation_enabled:
            return correlations
        
        metric_names = list(self.metric_aggregates.keys())
        
        for i, metric1 in enumerate(metric_names):
            for metric2 in metric_names[i+1:]:
                values1 = self.metric_aggregates[metric1]
                values2 = self.metric_aggregates[metric2]
                
                if len(values1) >= 10 and len(values2) >= 10:
                    # Align lengths
                    min_len = min(len(values1), len(values2))
                    corr_coef = np.corrcoef(values1[-min_len:], values2[-min_len:])[0, 1]
                    
                    if not np.isnan(corr_coef) and abs(corr_coef) > 0.7:
                        correlations[f"{metric1}_vs_{metric2}"] = {
                            "correlation": corr_coef,
                            "strength": "strong" if abs(corr_coef) > 0.8 else "moderate",
                            "direction": "positive" if corr_coef > 0 else "negative"
                        }
        
        return correlations

    async def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Analyze system metrics for recommendations
        system_metrics = {
            name: values for name, values in self.metric_aggregates.items()
            if name.startswith("system.")
        }
        
        for metric_name, values in system_metrics.items():
            if len(values) >= 5:
                recent_avg = np.mean(values[-5:])
                
                if "cpu.usage_percent" in metric_name and recent_avg > 80:
                    recommendations.append("High CPU usage detected. Consider scaling or optimization.")
                elif "memory.usage_percent" in metric_name and recent_avg > 85:
                    recommendations.append("High memory usage detected. Consider memory optimization.")
                elif "disk.usage_percent" in metric_name and recent_avg > 90:
                    recommendations.append("High disk usage detected. Consider cleanup or expansion.")
        
        return recommendations

    async def _check_metric_anomaly(self, metric: Metric):
        """Check if a single metric value is anomalous."""
        if metric.name in self.metric_aggregates:
            values = self.metric_aggregates[metric.name]
            if len(values) >= 10:
                mean = np.mean(values[:-1])  # Exclude current value
                std = np.std(values[:-1])
                
                if std > 0:
                    z_score = abs(metric.value - mean) / std
                    if z_score > self.anomaly_detection_threshold:
                        await self.create_alert(
                            title=f"Real-time anomaly: {metric.name}",
                            description=f"Value {metric.value:.3f} is anomalous (z-score: {z_score:.2f})",
                            severity=AlertSeverity.WARNING,
                            source="real_time_anomaly",
                            metadata={
                                "metric_name": metric.name,
                                "value": metric.value,
                                "z_score": z_score,
                                "mean": mean,
                                "std": std
                            }
                        )

    async def _send_external_alert(self, alert: Alert):
        """Send alert to external monitoring systems."""
        # Webhook notifications
        webhook_url = self.config.get("webhook_url")
        if webhook_url:
            try:
                payload = {
                    "alert_id": alert.id,
                    "title": alert.title,
                    "description": alert.description,
                    "severity": alert.severity.value,
                    "source": alert.source,
                    "timestamp": alert.timestamp.isoformat(),
                    "metadata": alert.metadata
                }
                
                async with aiohttp.ClientSession() as session:
                    await session.post(webhook_url, json=payload, timeout=10)
                
                logger.debug(f"Alert sent to webhook: {alert.title}")
                
            except Exception as e:
                logger.error(f"Failed to send alert to webhook: {e}")


# Factory function for easy instantiation
def create_monitoring_engine(config: Optional[Dict[str, Any]] = None) -> QuantumMonitoringEngine:
    """Create monitoring engine instance."""
    return QuantumMonitoringEngine(config)


# Decorator for monitoring function execution
def monitor_execution(metric_prefix: str = "app"):
    """
    Decorator to monitor function execution metrics.
    
    Usage:
        @monitor_execution("data_processing")
        async def process_data():
            # Function execution will be monitored
            pass
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            # Get monitoring engine from global state or create one
            monitoring = kwargs.pop('_monitoring_engine', None)
            if not monitoring:
                monitoring = create_monitoring_engine()
            
            start_time = time.time()
            function_name = func.__name__
            
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                # Record success metrics
                execution_time = time.time() - start_time
                await monitoring.record_metric(f"{metric_prefix}.{function_name}.execution_time", execution_time, MetricType.TIMER)
                await monitoring.record_metric(f"{metric_prefix}.{function_name}.success_count", 1, MetricType.COUNTER)
                
                return result
                
            except Exception as e:
                # Record error metrics
                execution_time = time.time() - start_time
                await monitoring.record_metric(f"{metric_prefix}.{function_name}.execution_time", execution_time, MetricType.TIMER)
                await monitoring.record_metric(f"{metric_prefix}.{function_name}.error_count", 1, MetricType.COUNTER)
                
                # Create alert for function errors
                await monitoring.create_alert(
                    title=f"Function error: {function_name}",
                    description=f"Function {function_name} failed: {str(e)}",
                    severity=AlertSeverity.ERROR,
                    source="function_monitor",
                    metadata={"function": function_name, "error": str(e)}
                )
                
                raise
        
        return wrapper
    return decorator