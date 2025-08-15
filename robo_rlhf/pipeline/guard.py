"""
Pipeline Guard: Central orchestrator for self-healing pipeline operations.

Coordinates monitoring, detection, and healing components to maintain pipeline health.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Pipeline health status indicators."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class PipelineComponent:
    """Represents a monitored pipeline component."""
    name: str
    endpoint: str
    health_check: Callable
    critical: bool = False
    recovery_strategy: Optional[str] = None


@dataclass
class HealthReport:
    """Health status report for a component or system."""
    component: str
    status: HealthStatus
    timestamp: float
    metrics: Dict[str, Any]
    issues: List[str]
    recommendations: List[str]


class PipelineGuard:
    """
    Self-healing pipeline guard that monitors, detects, and recovers from failures.
    
    Core responsibilities:
    - Continuous health monitoring of pipeline components
    - Intelligent failure detection and prediction
    - Automated recovery and healing operations
    - Performance optimization and adaptive scaling
    """
    
    def __init__(
        self,
        components: List[PipelineComponent],
        check_interval: int = 30,
        healing_enabled: bool = True,
        max_workers: int = 4
    ):
        self.components = {comp.name: comp for comp in components}
        self.check_interval = check_interval
        self.healing_enabled = healing_enabled
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        self._health_history: Dict[str, List[HealthReport]] = {}
        self._recovery_attempts: Dict[str, int] = {}
        self._running = False
        
        # Initialize quantum integration if available
        try:
            from robo_rlhf.quantum import QuantumTaskPlanner, PredictiveAnalytics
            self.quantum_planner = QuantumTaskPlanner()
            self.predictive_analytics = PredictiveAnalytics()
            self.quantum_enabled = True
            logger.info("Quantum-enhanced pipeline guard initialized")
        except ImportError:
            self.quantum_enabled = False
            logger.info("Standard pipeline guard initialized")
    
    async def start_monitoring(self) -> None:
        """Start continuous pipeline monitoring."""
        if self._running:
            logger.warning("Pipeline guard already running")
            return
            
        self._running = True
        logger.info("Starting pipeline guard monitoring")
        
        try:
            await self._monitoring_loop()
        except Exception as e:
            logger.error(f"Pipeline guard monitoring failed: {e}")
            raise
        finally:
            self._running = False
    
    async def stop_monitoring(self) -> None:
        """Stop pipeline monitoring."""
        self._running = False
        self.executor.shutdown(wait=True)
        logger.info("Pipeline guard monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Concurrent health checks for all components
                check_tasks = [
                    self._check_component_health(name, component)
                    for name, component in self.components.items()
                ]
                
                health_reports = await asyncio.gather(*check_tasks, return_exceptions=True)
                
                # Process health reports and trigger healing if needed
                await self._process_health_reports(health_reports)
                
                # Quantum-enhanced predictive analysis
                if self.quantum_enabled:
                    await self._quantum_analysis()
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _check_component_health(
        self, name: str, component: PipelineComponent
    ) -> HealthReport:
        """Check health of a single component."""
        start_time = time.time()
        
        try:
            # Execute health check
            loop = asyncio.get_event_loop()
            health_result = await loop.run_in_executor(
                self.executor, component.health_check
            )
            
            response_time = time.time() - start_time
            
            # Analyze health result
            status, metrics, issues, recommendations = self._analyze_health_result(
                health_result, response_time, component
            )
            
            report = HealthReport(
                component=name,
                status=status,
                timestamp=time.time(),
                metrics=metrics,
                issues=issues,
                recommendations=recommendations
            )
            
            # Store in history
            if name not in self._health_history:
                self._health_history[name] = []
            self._health_history[name].append(report)
            
            # Keep only recent history (last 100 checks)
            self._health_history[name] = self._health_history[name][-100:]
            
            return report
            
        except Exception as e:
            logger.error(f"Health check failed for {name}: {e}")
            return HealthReport(
                component=name,
                status=HealthStatus.FAILED,
                timestamp=time.time(),
                metrics={"error": str(e)},
                issues=[f"Health check failed: {e}"],
                recommendations=["Investigate component failure", "Check connectivity"]
            )
    
    def _analyze_health_result(
        self, result: Any, response_time: float, component: PipelineComponent
    ) -> tuple:
        """Analyze health check result and determine status."""
        metrics = {"response_time": response_time}
        issues = []
        recommendations = []
        
        # Default healthy status
        status = HealthStatus.HEALTHY
        
        # Response time analysis
        if response_time > 5.0:
            status = HealthStatus.DEGRADED
            issues.append(f"Slow response time: {response_time:.2f}s")
            recommendations.append("Investigate performance bottlenecks")
        
        # Analyze result based on type
        if isinstance(result, dict):
            metrics.update(result)
            
            # Check common health indicators
            if "status" in result:
                if result["status"] not in ["ok", "healthy", "up"]:
                    status = HealthStatus.CRITICAL
                    issues.append(f"Component reports unhealthy status: {result['status']}")
            
            if "error_rate" in result and result["error_rate"] > 0.05:
                status = HealthStatus.DEGRADED
                issues.append(f"High error rate: {result['error_rate']:.1%}")
                
            if "memory_usage" in result and result["memory_usage"] > 0.9:
                status = HealthStatus.DEGRADED
                issues.append(f"High memory usage: {result['memory_usage']:.1%}")
                recommendations.append("Consider memory optimization or scaling")
        
        elif isinstance(result, bool):
            if not result:
                status = HealthStatus.FAILED
                issues.append("Component health check returned False")
        
        elif result is None:
            status = HealthStatus.FAILED
            issues.append("Component health check returned None")
        
        return status, metrics, issues, recommendations
    
    async def _process_health_reports(self, reports: List[HealthReport]) -> None:
        """Process health reports and trigger healing if needed."""
        critical_components = []
        degraded_components = []
        
        for report in reports:
            if isinstance(report, Exception):
                logger.error(f"Health check exception: {report}")
                continue
                
            if report.status == HealthStatus.FAILED:
                critical_components.append(report)
            elif report.status in [HealthStatus.CRITICAL, HealthStatus.DEGRADED]:
                degraded_components.append(report)
        
        # Trigger healing for critical components
        if critical_components and self.healing_enabled:
            healing_tasks = [
                self._initiate_healing(report) for report in critical_components
            ]
            await asyncio.gather(*healing_tasks, return_exceptions=True)
        
        # Log degraded components for analysis
        for report in degraded_components:
            logger.warning(
                f"Component {report.component} degraded: {', '.join(report.issues)}"
            )
    
    async def _initiate_healing(self, report: HealthReport) -> None:
        """Initiate healing process for a failed component."""
        component_name = report.component
        component = self.components[component_name]
        
        # Track recovery attempts
        attempts = self._recovery_attempts.get(component_name, 0)
        self._recovery_attempts[component_name] = attempts + 1
        
        logger.warning(
            f"Initiating healing for {component_name} "
            f"(attempt {attempts + 1}): {', '.join(report.issues)}"
        )
        
        try:
            if self.quantum_enabled:
                # Quantum-enhanced recovery strategy
                recovery_plan = await self._quantum_recovery_planning(report)
                await self._execute_recovery_plan(recovery_plan)
            else:
                # Standard recovery strategies
                await self._standard_recovery(component, report)
                
        except Exception as e:
            logger.error(f"Healing failed for {component_name}: {e}")
    
    async def _quantum_recovery_planning(self, report: HealthReport) -> Dict[str, Any]:
        """Use quantum planning for optimal recovery strategy."""
        try:
            recovery_plan = await self.quantum_planner.plan_recovery(
                component=report.component,
                failure_symptoms=report.issues,
                metrics=report.metrics,
                history=self._health_history.get(report.component, [])
            )
            return recovery_plan
        except Exception as e:
            logger.error(f"Quantum recovery planning failed: {e}")
            return {"strategy": "restart", "priority": "high"}
    
    async def _execute_recovery_plan(self, plan: Dict[str, Any]) -> None:
        """Execute quantum-generated recovery plan."""
        strategy = plan.get("strategy", "restart")
        
        if strategy == "restart":
            await self._restart_component(plan.get("component"))
        elif strategy == "scale":
            await self._scale_component(plan.get("component"), plan.get("scale_factor", 2))
        elif strategy == "migrate":
            await self._migrate_component(plan.get("component"), plan.get("target_node"))
        else:
            logger.warning(f"Unknown recovery strategy: {strategy}")
    
    async def _standard_recovery(
        self, component: PipelineComponent, report: HealthReport
    ) -> None:
        """Execute standard recovery strategies."""
        if component.recovery_strategy == "restart":
            await self._restart_component(component.name)
        elif component.recovery_strategy == "scale":
            await self._scale_component(component.name)
        else:
            # Default: attempt restart
            await self._restart_component(component.name)
    
    async def _restart_component(self, component_name: str) -> None:
        """Restart a failed component."""
        logger.info(f"Restarting component: {component_name}")
        # Implementation depends on deployment environment
        # This is a placeholder for actual restart logic
        await asyncio.sleep(1)  # Simulate restart time
    
    async def _scale_component(self, component_name: str, factor: int = 2) -> None:
        """Scale a component to handle load."""
        logger.info(f"Scaling component {component_name} by factor {factor}")
        # Implementation depends on orchestration system
        await asyncio.sleep(1)  # Simulate scaling time
    
    async def _migrate_component(self, component_name: str, target: str) -> None:
        """Migrate component to healthier node."""
        logger.info(f"Migrating component {component_name} to {target}")
        await asyncio.sleep(1)  # Simulate migration time
    
    async def _quantum_analysis(self) -> None:
        """Perform quantum-enhanced predictive analysis."""
        try:
            predictions = await self.predictive_analytics.predict_failures(
                self._health_history
            )
            
            for component, risk_score in predictions.items():
                if risk_score > 0.7:
                    logger.warning(
                        f"High failure risk predicted for {component}: {risk_score:.2f}"
                    )
                    # Proactive healing could be triggered here
                    
        except Exception as e:
            logger.error(f"Quantum analysis failed: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self._health_history:
            return {"status": "unknown", "components": {}}
        
        component_statuses = {}
        overall_status = HealthStatus.HEALTHY
        
        for name, history in self._health_history.items():
            if history:
                latest = history[-1]
                component_statuses[name] = {
                    "status": latest.status.value,
                    "timestamp": latest.timestamp,
                    "issues": latest.issues,
                    "metrics": latest.metrics
                }
                
                # Determine overall status (worst component status)
                if latest.status == HealthStatus.FAILED:
                    overall_status = HealthStatus.FAILED
                elif latest.status == HealthStatus.CRITICAL and overall_status != HealthStatus.FAILED:
                    overall_status = HealthStatus.CRITICAL
                elif latest.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
        
        return {
            "status": overall_status.value,
            "timestamp": time.time(),
            "components": component_statuses,
            "recovery_attempts": self._recovery_attempts.copy()
        }