"""
Pipeline Orchestrator: Central coordination for self-healing pipeline operations.

Integrates all pipeline components into a unified, production-ready system.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum

from .guard import PipelineGuard, PipelineComponent, HealthStatus
from .monitor import PipelineMonitor, MetricsCollector
from .healer import SelfHealer, RecoveryStrategy, RecoveryAction
from .detector import AnomalyDetector, FailurePredictor, Anomaly
from .security import SecurityManager, SecurityContext, Permission
from .reliability import ReliabilityManager, GracefulDegradation

logger = logging.getLogger(__name__)


class PipelineMode(Enum):
    """Pipeline operating modes."""
    NORMAL = "normal"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PipelineAlert:
    """Pipeline alert message."""
    level: AlertLevel
    component: str
    message: str
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


@dataclass
class PipelineConfig:
    """Configuration for pipeline orchestrator."""
    monitoring_interval: int = 30
    healing_enabled: bool = True
    security_enabled: bool = True
    max_concurrent_healings: int = 3
    alert_cooldown: int = 300  # 5 minutes
    auto_scaling_enabled: bool = True
    quantum_enhanced: bool = True


class PipelineOrchestrator:
    """
    Central orchestrator for self-healing pipeline operations.
    
    Integrates all pipeline components:
    - Health monitoring and metrics collection
    - Anomaly detection and failure prediction
    - Intelligent self-healing and recovery
    - Security and access control
    - Reliability patterns and fault tolerance
    - Alert management and notifications
    - Performance optimization and scaling
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        components: Optional[List[PipelineComponent]] = None
    ):
        self.config = config or PipelineConfig()
        self.components = {comp.name: comp for comp in (components or [])}
        
        # Initialize core systems
        self.metrics_collector = MetricsCollector()
        self.monitor = PipelineMonitor(self.metrics_collector)
        self.healer = SelfHealer()
        self.reliability_manager = ReliabilityManager()
        self.graceful_degradation = GracefulDegradation()
        
        # Initialize security if enabled
        if self.config.security_enabled:
            self.security_manager = SecurityManager()
        else:
            self.security_manager = None
        
        # Component-specific systems
        self.anomaly_detectors: Dict[str, AnomalyDetector] = {}
        self.failure_predictors: Dict[str, FailurePredictor] = {}
        
        # Pipeline state
        self.mode = PipelineMode.NORMAL
        self.active_healings: Set[str] = set()
        self.alerts: List[PipelineAlert] = []
        self.alert_history: Dict[str, float] = {}  # component -> last_alert_time
        
        # Performance tracking
        self.performance_metrics = {
            "healing_success_rate": 0.0,
            "average_recovery_time": 0.0,
            "uptime_percentage": 100.0,
            "false_positive_rate": 0.0
        }
        
        # Quantum integration if available
        try:
            from robo_rlhf.quantum import AutonomousSDLCExecutor
            self.autonomous_executor = AutonomousSDLCExecutor()
            self.quantum_enabled = True
            logger.info("Quantum-enhanced pipeline orchestrator initialized")
        except ImportError:
            self.quantum_enabled = False
            logger.info("Standard pipeline orchestrator initialized")
        
        # Runtime state
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        logger.info(f"Pipeline orchestrator initialized with {len(self.components)} components")
    
    def add_component(self, component: PipelineComponent) -> None:
        """Add a new component to the pipeline."""
        self.components[component.name] = component
        
        # Initialize component-specific systems
        self.anomaly_detectors[component.name] = AnomalyDetector(
            component_name=component.name,
            enable_ml_detection=True
        )
        
        self.failure_predictors[component.name] = FailurePredictor()
        
        # Add to monitor
        self.monitor.add_component_monitor(
            name=component.name,
            health_check=component.health_check,
            check_interval=self.config.monitoring_interval
        )
        
        logger.info(f"Added component to pipeline: {component.name}")
    
    def remove_component(self, component_name: str) -> bool:
        """Remove a component from the pipeline."""
        if component_name in self.components:
            del self.components[component_name]
            
            # Cleanup component-specific systems
            if component_name in self.anomaly_detectors:
                del self.anomaly_detectors[component_name]
            
            if component_name in self.failure_predictors:
                del self.failure_predictors[component_name]
            
            logger.info(f"Removed component from pipeline: {component_name}")
            return True
        
        return False
    
    async def start(self, context: Optional[SecurityContext] = None) -> None:
        """Start the pipeline orchestrator."""
        # Security check
        if self.security_manager and context:
            if not self.security_manager.authorize(context, Permission.MANAGE_COMPONENTS):
                raise PermissionError("Insufficient permissions to start pipeline")
        
        if self._running:
            logger.warning("Pipeline orchestrator already running")
            return
        
        self._running = True
        logger.info("Starting pipeline orchestrator")
        
        try:
            # Start core monitoring
            monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._tasks.append(monitoring_task)
            
            # Start anomaly detection
            detection_task = asyncio.create_task(self._detection_loop())
            self._tasks.append(detection_task)
            
            # Start healing coordination
            healing_task = asyncio.create_task(self._healing_loop())
            self._tasks.append(healing_task)
            
            # Start alert management
            alert_task = asyncio.create_task(self._alert_loop())
            self._tasks.append(alert_task)
            
            # Start quantum autonomous execution if enabled
            if self.quantum_enabled and self.config.quantum_enhanced:
                quantum_task = asyncio.create_task(self._quantum_execution_loop())
                self._tasks.append(quantum_task)
            
            # Wait for tasks to complete
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Pipeline orchestrator failed: {e}")
            raise
        finally:
            self._running = False
    
    async def stop(self, context: Optional[SecurityContext] = None) -> None:
        """Stop the pipeline orchestrator."""
        # Security check
        if self.security_manager and context:
            if not self.security_manager.authorize(context, Permission.MANAGE_COMPONENTS):
                raise PermissionError("Insufficient permissions to stop pipeline")
        
        if not self._running:
            return
        
        logger.info("Stopping pipeline orchestrator")
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks, return_exceptions=True),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.warning("Some tasks did not stop within timeout")
        
        self._tasks.clear()
        logger.info("Pipeline orchestrator stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Starting monitoring loop")
        
        while self._running:
            try:
                # Collect health metrics for all components
                health_reports = {}
                
                for component_name in self.components:
                    component = self.components[component_name]
                    
                    try:
                        # Execute health check with reliability patterns
                        health_result = await self.reliability_manager.execute_with_reliability(
                            component.health_check,
                            operation_name=f"{component_name}_health_check"
                        )
                        
                        health_reports[component_name] = health_result
                        
                        # Record metrics
                        if isinstance(health_result, dict):
                            for metric_name, value in health_result.items():
                                if isinstance(value, (int, float)):
                                    self.metrics_collector.record_metric(
                                        f"{component_name}.{metric_name}",
                                        float(value),
                                        tags={"component": component_name}
                                    )
                    
                    except Exception as e:
                        logger.error(f"Health check failed for {component_name}: {e}")
                        health_reports[component_name] = None
                
                # Update performance metrics
                await self._update_performance_metrics(health_reports)
                
                await asyncio.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def _detection_loop(self) -> None:
        """Anomaly detection loop."""
        logger.info("Starting anomaly detection loop")
        
        while self._running:
            try:
                detection_interval = 60  # 1 minute
                
                for component_name, detector in self.anomaly_detectors.items():
                    try:
                        # Run anomaly detection
                        anomalies = await detector.detect_anomalies()
                        
                        # Process detected anomalies
                        for anomaly in anomalies:
                            await self._handle_anomaly(anomaly)
                        
                        # Failure prediction
                        if component_name in self.failure_predictors:
                            predictor = self.failure_predictors[component_name]
                            
                            # Get current metrics
                            current_metrics = self._get_component_current_metrics(component_name)
                            
                            # Predict failure risk
                            risk_assessment = await predictor.predict_failure_risk(
                                component_name, current_metrics
                            )
                            
                            # Handle high risk predictions
                            if risk_assessment.get("risk_score", 0) > 0.7:
                                await self._handle_high_failure_risk(component_name, risk_assessment)
                    
                    except Exception as e:
                        logger.error(f"Detection error for {component_name}: {e}")
                
                await asyncio.sleep(detection_interval)
                
            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                await asyncio.sleep(5)
    
    async def _healing_loop(self) -> None:
        """Healing coordination loop."""
        logger.info("Starting healing coordination loop")
        
        while self._running:
            try:
                healing_interval = 30  # 30 seconds
                
                # Check for components needing healing
                for component_name, component in self.components.items():
                    if component_name in self.active_healings:
                        continue  # Already healing
                    
                    if len(self.active_healings) >= self.config.max_concurrent_healings:
                        break  # Too many concurrent healings
                    
                    # Check if component needs healing
                    needs_healing = await self._assess_healing_need(component_name)
                    
                    if needs_healing:
                        await self._initiate_healing(component_name)
                
                await asyncio.sleep(healing_interval)
                
            except Exception as e:
                logger.error(f"Healing loop error: {e}")
                await asyncio.sleep(5)
    
    async def _alert_loop(self) -> None:
        """Alert management loop."""
        logger.info("Starting alert management loop")
        
        while self._running:
            try:
                alert_interval = 10  # 10 seconds
                
                # Process and manage alerts
                await self._process_alerts()
                await self._cleanup_old_alerts()
                
                await asyncio.sleep(alert_interval)
                
            except Exception as e:
                logger.error(f"Alert loop error: {e}")
                await asyncio.sleep(5)
    
    async def _quantum_execution_loop(self) -> None:
        """Quantum autonomous execution loop."""
        logger.info("Starting quantum autonomous execution loop")
        
        while self._running:
            try:
                execution_interval = 300  # 5 minutes
                
                # Execute quantum autonomous optimizations
                if hasattr(self, 'autonomous_executor'):
                    optimization_tasks = [
                        "performance_optimization",
                        "predictive_scaling",
                        "intelligent_recovery_planning"
                    ]
                    
                    for task in optimization_tasks:
                        try:
                            await self.autonomous_executor.execute_autonomous_task(
                                task_type=task,
                                context={
                                    "pipeline_state": self.get_pipeline_status(),
                                    "performance_metrics": self.performance_metrics,
                                    "active_components": list(self.components.keys())
                                }
                            )
                        except Exception as e:
                            logger.error(f"Quantum autonomous task {task} failed: {e}")
                
                await asyncio.sleep(execution_interval)
                
            except Exception as e:
                logger.error(f"Quantum execution loop error: {e}")
                await asyncio.sleep(30)
    
    async def _handle_anomaly(self, anomaly: Anomaly) -> None:
        """Handle detected anomaly."""
        logger.warning(f"Anomaly detected: {anomaly.component} - {anomaly.description}")
        
        # Create alert
        alert = PipelineAlert(
            level=AlertLevel.WARNING if anomaly.severity.value in ["low", "medium"] else AlertLevel.ERROR,
            component=anomaly.component,
            message=anomaly.description,
            timestamp=anomaly.timestamp,
            details={
                "anomaly_type": anomaly.type.value,
                "confidence": anomaly.confidence,
                "metrics": anomaly.metrics,
                "recommendations": anomaly.recommendations
            }
        )
        
        await self._add_alert(alert)
        
        # Consider automatic healing for high-severity anomalies
        if anomaly.severity.value in ["high", "critical"] and self.config.healing_enabled:
            await self._trigger_healing(anomaly.component, anomaly)
    
    async def _handle_high_failure_risk(self, component_name: str, risk_assessment: Dict[str, Any]) -> None:
        """Handle high failure risk prediction."""
        risk_score = risk_assessment.get("risk_score", 0)
        
        logger.warning(
            f"High failure risk predicted for {component_name}: {risk_score:.2f}"
        )
        
        # Create predictive alert
        alert = PipelineAlert(
            level=AlertLevel.WARNING,
            component=component_name,
            message=f"High failure risk predicted (score: {risk_score:.2f})",
            timestamp=time.time(),
            details=risk_assessment
        )
        
        await self._add_alert(alert)
        
        # Consider proactive healing
        if risk_score > 0.8 and self.config.healing_enabled:
            await self._trigger_proactive_healing(component_name, risk_assessment)
    
    async def _assess_healing_need(self, component_name: str) -> bool:
        """Assess if component needs healing."""
        # Check recent alerts
        recent_alerts = [
            alert for alert in self.alerts[-10:]
            if alert.component == component_name and 
               alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL] and
               not alert.resolved
        ]
        
        if recent_alerts:
            return True
        
        # Check component health status
        if component_name in self.anomaly_detectors:
            active_anomalies = self.anomaly_detectors[component_name].get_active_anomalies()
            critical_anomalies = [
                a for a in active_anomalies 
                if a.severity.value in ["high", "critical"]
            ]
            
            if critical_anomalies:
                return True
        
        return False
    
    async def _initiate_healing(self, component_name: str) -> None:
        """Initiate healing process for component."""
        if component_name in self.active_healings:
            return
        
        self.active_healings.add(component_name)
        
        try:
            logger.info(f"Initiating healing for component: {component_name}")
            
            # Gather failure context
            failure_context = self._gather_failure_context(component_name)
            
            # Execute healing with reliability patterns
            healing_results = await self.reliability_manager.execute_with_reliability(
                self.healer.heal,
                operation_name=f"{component_name}_healing",
                component=component_name,
                failure_context=failure_context
            )
            
            # Process healing results
            successful_healings = [
                r for r in healing_results 
                if r.result.value == "success"
            ]
            
            if successful_healings:
                logger.info(f"Healing successful for {component_name}")
                await self._resolve_component_alerts(component_name)
            else:
                logger.error(f"Healing failed for {component_name}")
                
                # Escalate to emergency mode if critical component
                if self.components[component_name].critical:
                    self.mode = PipelineMode.EMERGENCY
        
        finally:
            self.active_healings.discard(component_name)
    
    async def _trigger_healing(self, component_name: str, anomaly: Anomaly) -> None:
        """Trigger healing based on anomaly."""
        if component_name not in self.active_healings:
            failure_context = {
                "anomaly_type": anomaly.type.value,
                "severity": anomaly.severity.value,
                "confidence": anomaly.confidence,
                "metrics": anomaly.metrics
            }
            
            # Add to healing queue
            asyncio.create_task(self._initiate_healing(component_name))
    
    async def _trigger_proactive_healing(self, component_name: str, risk_assessment: Dict[str, Any]) -> None:
        """Trigger proactive healing based on failure prediction."""
        logger.info(f"Triggering proactive healing for {component_name}")
        
        failure_context = {
            "type": "predicted_failure",
            "risk_score": risk_assessment.get("risk_score", 0),
            "risk_factors": risk_assessment.get("risk_factors", []),
            "predicted_time": risk_assessment.get("predicted_time_to_failure")
        }
        
        # Execute proactive healing
        asyncio.create_task(self._initiate_healing(component_name))
    
    def _gather_failure_context(self, component_name: str) -> Dict[str, Any]:
        """Gather comprehensive failure context for healing."""
        context = {}
        
        # Recent metrics
        recent_metrics = self.metrics_collector.get_latest_metrics([
            f"{component_name}.response_time",
            f"{component_name}.error_rate",
            f"{component_name}.cpu_usage",
            f"{component_name}.memory_usage"
        ])
        
        context["recent_metrics"] = {
            name.split(".")[-1]: metric.value if metric else 0
            for name, metric in recent_metrics.items()
        }
        
        # Active anomalies
        if component_name in self.anomaly_detectors:
            active_anomalies = self.anomaly_detectors[component_name].get_active_anomalies()
            context["active_anomalies"] = [
                {
                    "type": anomaly.type.value,
                    "severity": anomaly.severity.value,
                    "description": anomaly.description
                }
                for anomaly in active_anomalies
            ]
        
        return context
    
    def _get_component_current_metrics(self, component_name: str) -> Dict[str, float]:
        """Get current metrics for a component."""
        metric_names = [
            f"{component_name}.response_time",
            f"{component_name}.error_rate",
            f"{component_name}.cpu_usage",
            f"{component_name}.memory_usage",
            f"{component_name}.availability"
        ]
        
        recent_metrics = self.metrics_collector.get_latest_metrics(metric_names)
        
        return {
            name.split(".")[-1]: metric.value if metric else 0.0
            for name, metric in recent_metrics.items()
        }
    
    async def _add_alert(self, alert: PipelineAlert) -> None:
        """Add alert with cooldown management."""
        # Check cooldown
        last_alert_time = self.alert_history.get(alert.component, 0)
        if time.time() - last_alert_time < self.config.alert_cooldown:
            return  # Still in cooldown
        
        self.alerts.append(alert)
        self.alert_history[alert.component] = alert.timestamp
        
        logger.info(f"Alert added: {alert.level.value} - {alert.component} - {alert.message}")
    
    async def _resolve_component_alerts(self, component_name: str) -> None:
        """Resolve all alerts for a component."""
        for alert in self.alerts:
            if alert.component == component_name and not alert.resolved:
                alert.resolved = True
        
        logger.info(f"Resolved alerts for component: {component_name}")
    
    async def _process_alerts(self) -> None:
        """Process and manage alerts."""
        # Count unresolved alerts by level
        unresolved_counts = {}
        for alert in self.alerts:
            if not alert.resolved:
                level = alert.level.value
                unresolved_counts[level] = unresolved_counts.get(level, 0) + 1
        
        # Escalate pipeline mode based on alert levels
        if unresolved_counts.get("critical", 0) > 0:
            self.mode = PipelineMode.EMERGENCY
        elif unresolved_counts.get("error", 0) >= 3:
            self.mode = PipelineMode.DEGRADED
        elif sum(unresolved_counts.values()) == 0:
            self.mode = PipelineMode.NORMAL
    
    async def _cleanup_old_alerts(self) -> None:
        """Cleanup old resolved alerts."""
        cutoff_time = time.time() - 3600  # 1 hour
        
        self.alerts = [
            alert for alert in self.alerts
            if not alert.resolved or alert.timestamp > cutoff_time
        ]
    
    async def _update_performance_metrics(self, health_reports: Dict[str, Any]) -> None:
        """Update overall pipeline performance metrics."""
        # Calculate uptime percentage
        healthy_components = sum(1 for report in health_reports.values() if report is not None)
        total_components = len(health_reports)
        
        if total_components > 0:
            self.performance_metrics["uptime_percentage"] = (
                healthy_components / total_components * 100
            )
        
        # Update healing success rate
        healing_stats = self.healer.get_healing_stats()
        self.performance_metrics["healing_success_rate"] = healing_stats.get("overall_success_rate", 0.0)
        
        # Calculate average recovery time (simplified)
        recent_healings = [
            result for result in self.healer.recovery_history[-10:]
            if result.result.value == "success"
        ]
        
        if recent_healings:
            avg_recovery_time = sum(r.duration for r in recent_healings) / len(recent_healings)
            self.performance_metrics["average_recovery_time"] = avg_recovery_time
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        # Component health summary
        component_health = {}
        for name in self.components:
            if name in self.anomaly_detectors:
                active_anomalies = len(self.anomaly_detectors[name].get_active_anomalies())
            else:
                active_anomalies = 0
            
            component_health[name] = {
                "active_anomalies": active_anomalies,
                "in_healing": name in self.active_healings
            }
        
        # Alert summary
        unresolved_alerts = [alert for alert in self.alerts if not alert.resolved]
        alert_summary = {}
        for alert in unresolved_alerts:
            level = alert.level.value
            alert_summary[level] = alert_summary.get(level, 0) + 1
        
        return {
            "mode": self.mode.value,
            "running": self._running,
            "components": component_health,
            "alerts": alert_summary,
            "active_healings": len(self.active_healings),
            "performance_metrics": self.performance_metrics,
            "quantum_enabled": self.quantum_enabled,
            "security_enabled": self.config.security_enabled,
            "timestamp": time.time()
        }
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics and statistics."""
        return {
            "pipeline_status": self.get_pipeline_status(),
            "metrics_collector": self.metrics_collector.get_stats(),
            "reliability_patterns": self.reliability_manager.get_pattern_stats(),
            "healing_stats": self.healer.get_healing_stats(),
            "security_stats": self.security_manager.get_security_stats() if self.security_manager else {},
            "anomaly_detection": {
                name: detector.get_detection_stats()
                for name, detector in self.anomaly_detectors.items()
            },
            "graceful_degradation": self.graceful_degradation.get_degradation_stats()
        }