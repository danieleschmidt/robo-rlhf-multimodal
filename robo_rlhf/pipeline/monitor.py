"""
Pipeline Monitor: Advanced monitoring and metrics collection for pipeline components.

Provides real-time monitoring, metrics aggregation, and health assessment capabilities.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str]
    unit: str = ""


@dataclass
class MetricsSummary:
    """Statistical summary of metrics over time."""
    name: str
    count: int
    min_value: float
    max_value: float
    mean: float
    median: float
    std_dev: float
    percentile_95: float
    percentile_99: float


class MetricsCollector:
    """
    Collects and aggregates metrics from pipeline components.
    
    Features:
    - Real-time metric collection
    - Statistical aggregation
    - Time-series storage
    - Alert thresholds
    """
    
    def __init__(self, retention_hours: int = 24, max_points_per_metric: int = 10000):
        self.retention_hours = retention_hours
        self.max_points = max_points_per_metric
        
        # Storage for metrics
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self._alert_thresholds: Dict[str, Dict[str, float]] = {}
        self._alert_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Performance counters
        self._collection_stats = {
            "total_metrics": 0,
            "alerts_triggered": 0,
            "last_cleanup": time.time()
        }
    
    def record_metric(
        self, 
        name: str, 
        value: float, 
        tags: Optional[Dict[str, str]] = None,
        unit: str = ""
    ) -> None:
        """Record a single metric data point."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            unit=unit
        )
        
        self._metrics[name].append(metric)
        self._collection_stats["total_metrics"] += 1
        
        # Check alert thresholds
        self._check_alerts(metric)
        
        # Periodic cleanup
        if time.time() - self._collection_stats["last_cleanup"] > 3600:  # Every hour
            self._cleanup_old_metrics()
    
    def record_batch(self, metrics: List[Dict[str, Any]]) -> None:
        """Record multiple metrics efficiently."""
        for metric_data in metrics:
            self.record_metric(
                name=metric_data["name"],
                value=metric_data["value"],
                tags=metric_data.get("tags"),
                unit=metric_data.get("unit", "")
            )
    
    def get_metric_summary(
        self, 
        name: str, 
        since_hours: Optional[float] = None
    ) -> Optional[MetricsSummary]:
        """Get statistical summary for a metric."""
        if name not in self._metrics:
            return None
        
        cutoff_time = time.time() - (since_hours * 3600) if since_hours else 0
        values = [
            m.value for m in self._metrics[name] 
            if m.timestamp >= cutoff_time
        ]
        
        if not values:
            return None
        
        # Calculate statistics
        values.sort()
        count = len(values)
        
        try:
            return MetricsSummary(
                name=name,
                count=count,
                min_value=min(values),
                max_value=max(values),
                mean=statistics.mean(values),
                median=statistics.median(values),
                std_dev=statistics.stdev(values) if count > 1 else 0.0,
                percentile_95=values[int(0.95 * count)] if count > 0 else 0.0,
                percentile_99=values[int(0.99 * count)] if count > 0 else 0.0
            )
        except Exception as e:
            logger.error(f"Error calculating summary for {name}: {e}")
            return None
    
    def get_latest_metrics(self, names: List[str]) -> Dict[str, Optional[Metric]]:
        """Get latest values for specified metrics."""
        result = {}
        for name in names:
            if name in self._metrics and self._metrics[name]:
                result[name] = self._metrics[name][-1]
            else:
                result[name] = None
        return result
    
    def set_alert_threshold(
        self, 
        metric_name: str, 
        threshold_type: str, 
        value: float,
        callback: Optional[Callable] = None
    ) -> None:
        """Set alert threshold for a metric."""
        if metric_name not in self._alert_thresholds:
            self._alert_thresholds[metric_name] = {}
        
        self._alert_thresholds[metric_name][threshold_type] = value
        
        if callback:
            self._alert_callbacks[metric_name].append(callback)
    
    def _check_alerts(self, metric: Metric) -> None:
        """Check if metric triggers any alerts."""
        if metric.name not in self._alert_thresholds:
            return
        
        thresholds = self._alert_thresholds[metric.name]
        alerts_triggered = []
        
        # Check various threshold types
        if "max" in thresholds and metric.value > thresholds["max"]:
            alerts_triggered.append(f"exceeds maximum threshold ({thresholds['max']})")
        
        if "min" in thresholds and metric.value < thresholds["min"]:
            alerts_triggered.append(f"below minimum threshold ({thresholds['min']})")
        
        if "rate_change" in thresholds:
            # Check rate of change (requires previous values)
            metrics_list = list(self._metrics[metric.name])
            if len(metrics_list) >= 2:
                prev_metric = metrics_list[-2]
                time_diff = metric.timestamp - prev_metric.timestamp
                if time_diff > 0:
                    rate = abs(metric.value - prev_metric.value) / time_diff
                    if rate > thresholds["rate_change"]:
                        alerts_triggered.append(f"rapid change rate ({rate:.2f}/s)")
        
        # Trigger callbacks for alerts
        if alerts_triggered:
            self._collection_stats["alerts_triggered"] += len(alerts_triggered)
            
            for callback in self._alert_callbacks[metric.name]:
                try:
                    callback(metric, alerts_triggered)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        cutoff_time = time.time() - (self.retention_hours * 3600)
        cleaned = 0
        
        for name, metrics_deque in self._metrics.items():
            # Remove old metrics from the left side
            while metrics_deque and metrics_deque[0].timestamp < cutoff_time:
                metrics_deque.popleft()
                cleaned += 1
        
        self._collection_stats["last_cleanup"] = time.time()
        logger.debug(f"Cleaned {cleaned} old metrics")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        total_stored = sum(len(deque) for deque in self._metrics.values())
        
        return {
            **self._collection_stats,
            "metrics_stored": total_stored,
            "unique_metrics": len(self._metrics),
            "alert_thresholds": len(self._alert_thresholds),
            "memory_usage_mb": total_stored * 0.1  # Rough estimate
        }


class PipelineMonitor:
    """
    Advanced pipeline monitoring with health assessment and adaptive thresholds.
    
    Provides:
    - Continuous component monitoring
    - Adaptive threshold management
    - Health scoring algorithms
    - Performance trend analysis
    """
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics = metrics_collector or MetricsCollector()
        self._monitors: Dict[str, ComponentMonitor] = {}
        self._running = False
        
        # Health scoring weights
        self.health_weights = {
            "response_time": 0.3,
            "error_rate": 0.4,
            "resource_usage": 0.2,
            "availability": 0.1
        }
    
    def add_component_monitor(
        self, 
        name: str, 
        health_check: Callable,
        check_interval: int = 30,
        custom_metrics: Optional[List[str]] = None
    ) -> None:
        """Add a new component to monitor."""
        monitor = ComponentMonitor(
            name=name,
            health_check=health_check,
            check_interval=check_interval,
            metrics_collector=self.metrics,
            custom_metrics=custom_metrics or []
        )
        
        self._monitors[name] = monitor
        logger.info(f"Added monitor for component: {name}")
    
    async def start_monitoring(self) -> None:
        """Start monitoring all components."""
        if self._running:
            return
        
        self._running = True
        logger.info("Starting pipeline monitoring")
        
        # Start all component monitors
        monitor_tasks = [
            monitor.start() for monitor in self._monitors.values()
        ]
        
        try:
            await asyncio.gather(*monitor_tasks)
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
            raise
        finally:
            self._running = False
    
    async def stop_monitoring(self) -> None:
        """Stop all monitoring."""
        self._running = False
        
        # Stop all monitors
        stop_tasks = [
            monitor.stop() for monitor in self._monitors.values()
        ]
        
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        logger.info("Pipeline monitoring stopped")
    
    def get_health_score(self, component_name: str) -> float:
        """Calculate health score for a component (0-100)."""
        if component_name not in self._monitors:
            return 0.0
        
        monitor = self._monitors[component_name]
        recent_metrics = self._get_recent_health_metrics(component_name)
        
        if not recent_metrics:
            return 0.0
        
        # Calculate component scores
        scores = {}
        
        # Response time score (lower is better)
        if "response_time" in recent_metrics:
            avg_response = recent_metrics["response_time"]["mean"]
            scores["response_time"] = max(0, 100 - (avg_response * 20))  # 5s = 0 score
        
        # Error rate score (lower is better)
        if "error_rate" in recent_metrics:
            avg_error_rate = recent_metrics["error_rate"]["mean"]
            scores["error_rate"] = max(0, 100 - (avg_error_rate * 100))
        
        # Resource usage score
        if "cpu_usage" in recent_metrics:
            avg_cpu = recent_metrics["cpu_usage"]["mean"]
            scores["resource_usage"] = max(0, 100 - avg_cpu)
        
        # Availability score
        if "availability" in recent_metrics:
            scores["availability"] = recent_metrics["availability"]["mean"] * 100
        
        # Calculate weighted average
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in self.health_weights.items():
            if metric in scores:
                total_score += scores[metric] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _get_recent_health_metrics(self, component_name: str) -> Dict[str, Dict[str, float]]:
        """Get recent health metrics for a component."""
        health_metrics = [
            "response_time", "error_rate", "cpu_usage", 
            "memory_usage", "availability"
        ]
        
        result = {}
        for metric in health_metrics:
            full_metric_name = f"{component_name}.{metric}"
            summary = self.metrics.get_metric_summary(full_metric_name, since_hours=1)
            if summary:
                result[metric] = {
                    "mean": summary.mean,
                    "max": summary.max_value,
                    "min": summary.min_value,
                    "std_dev": summary.std_dev
                }
        
        return result
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system monitoring overview."""
        overview = {
            "timestamp": time.time(),
            "components": {},
            "overall_health": 0.0,
            "alerts": [],
            "metrics_stats": self.metrics.get_stats()
        }
        
        health_scores = []
        
        for name, monitor in self._monitors.items():
            health_score = self.get_health_score(name)
            health_scores.append(health_score)
            
            overview["components"][name] = {
                "health_score": health_score,
                "status": monitor.get_status(),
                "last_check": monitor.last_check_time,
                "check_count": monitor.check_count
            }
        
        # Calculate overall system health
        overview["overall_health"] = statistics.mean(health_scores) if health_scores else 0.0
        
        return overview


class ComponentMonitor:
    """Monitor for individual pipeline component."""
    
    def __init__(
        self,
        name: str,
        health_check: Callable,
        check_interval: int,
        metrics_collector: MetricsCollector,
        custom_metrics: List[str]
    ):
        self.name = name
        self.health_check = health_check
        self.check_interval = check_interval
        self.metrics = metrics_collector
        self.custom_metrics = custom_metrics
        
        self.last_check_time = 0.0
        self.check_count = 0
        self.status = "stopped"
        self._running = False
    
    async def start(self) -> None:
        """Start monitoring this component."""
        if self._running:
            return
        
        self._running = True
        self.status = "running"
        logger.info(f"Starting monitor for {self.name}")
        
        while self._running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Monitor error for {self.name}: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def stop(self) -> None:
        """Stop monitoring this component."""
        self._running = False
        self.status = "stopped"
        logger.info(f"Stopped monitor for {self.name}")
    
    async def _perform_health_check(self) -> None:
        """Perform health check and record metrics."""
        start_time = time.time()
        
        try:
            # Execute health check
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.health_check)
            
            response_time = time.time() - start_time
            self.last_check_time = time.time()
            self.check_count += 1
            
            # Record basic metrics
            self.metrics.record_metric(
                f"{self.name}.response_time", 
                response_time,
                tags={"component": self.name}
            )
            
            self.metrics.record_metric(
                f"{self.name}.availability",
                1.0,  # Successful check
                tags={"component": self.name}
            )
            
            # Record custom metrics from health check result
            if isinstance(result, dict):
                for metric_name, value in result.items():
                    if metric_name in self.custom_metrics or metric_name.startswith("metric_"):
                        self.metrics.record_metric(
                            f"{self.name}.{metric_name}",
                            float(value),
                            tags={"component": self.name}
                        )
        
        except Exception as e:
            # Record failure metrics
            self.metrics.record_metric(
                f"{self.name}.availability",
                0.0,  # Failed check
                tags={"component": self.name, "error": str(e)}
            )
            
            logger.error(f"Health check failed for {self.name}: {e}")
    
    def get_status(self) -> str:
        """Get current monitor status."""
        return self.status