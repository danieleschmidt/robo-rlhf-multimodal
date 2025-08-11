"""
Autonomous SDLC execution engine with self-optimization capabilities.

Implements autonomous decision-making for continuous integration, deployment,
testing, and system optimization using quantum-inspired algorithms.
"""

import asyncio
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import subprocess
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from robo_rlhf.core import get_logger, get_config
from robo_rlhf.core.exceptions import RoboRLHFError, SecurityError, ValidationError
from robo_rlhf.core.security import sanitize_input, check_file_safety, RateLimiter
from robo_rlhf.core.validators import validate_path, validate_numeric
from robo_rlhf.quantum.planner import QuantumTaskPlanner, QuantumTask, TaskState, TaskPriority
from robo_rlhf.quantum.optimizer import QuantumOptimizer, OptimizationObjective, OptimizationProblem


class SDLCPhase(Enum):
    """SDLC phases for autonomous execution."""
    ANALYSIS = "analysis"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    INTEGRATION = "integration"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"


class ExecutionStatus(Enum):
    """Execution status for autonomous tasks."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    OPTIMIZING = "optimizing"
    ROLLED_BACK = "rolled_back"


@dataclass
class AutonomousAction:
    """Autonomous action definition."""
    id: str
    name: str
    phase: SDLCPhase
    command: str
    prerequisites: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    rollback_command: Optional[str] = None
    timeout: float = 300.0
    retry_count: int = 3
    critical: bool = False
    auto_approve: bool = False


@dataclass
class ExecutionContext:
    """Execution context for autonomous actions."""
    project_path: Path
    environment: str
    configuration: Dict[str, Any]
    resource_limits: Dict[str, float]
    quality_gates: Dict[str, float]
    monitoring_config: Dict[str, Any]


@dataclass
class ExecutionResult:
    """Result of autonomous execution."""
    action_id: str
    status: ExecutionStatus
    start_time: float
    end_time: Optional[float] = None
    output: str = ""
    error_output: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    optimization_applied: bool = False
    rollback_performed: bool = False


class AutonomousSDLCExecutor:
    """Autonomous SDLC execution engine."""
    
    def __init__(self, 
                 project_path: Path,
                 config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.config = config or get_config().to_dict()
        
        # Validate and secure project path
        try:
            self.project_path = validate_path(project_path, must_exist=True, must_be_dir=True)
            # Security check on project directory - skip for directories
            if self.project_path.is_file():
                security_scan = check_file_safety(
                    self.project_path, 
                    scan_content=False,
                    max_size=None
                )
                if not security_scan["is_safe"]:
                    raise SecurityError(
                        f"Project directory failed security scan: {security_scan['threats']}",
                        threat_type="unsafe_directory"
                    )
            # For directories, just validate they exist and are readable
            elif not (self.project_path.exists() and os.access(self.project_path, os.R_OK)):
                raise SecurityError(
                    f"Project directory not accessible: {self.project_path}",
                    threat_type="inaccessible_directory"
                )
        except ValidationError as e:
            raise SecurityError(f"Invalid project path: {e}", threat_type="path_validation")
        
        # Initialize rate limiter for command execution
        self.rate_limiter = RateLimiter(
            max_requests=self.config.get("security", {}).get("max_commands_per_minute", 30),
            time_window=60
        )
        
        # Security configuration
        self.security_config = self.config.get("security", {})
        self.allowed_commands = self.security_config.get("allowed_commands", [
            "python", "pytest", "mypy", "bandit", "docker", "npm", "pip"
        ])
        self.command_timeout_limit = self.security_config.get("max_command_timeout", 1800)  # 30 minutes
        self.max_output_size = self.security_config.get("max_output_size", 10 * 1024 * 1024)  # 10MB
        
        # Initialize quantum components
        self.task_planner = QuantumTaskPlanner(config)
        self.optimizer = QuantumOptimizer(config)
        
        # Execution parameters
        self.max_parallel_actions = self.config.get("autonomous", {}).get("max_parallel", 3)
        self.auto_rollback = self.config.get("autonomous", {}).get("auto_rollback", True)
        self.quality_threshold = self.config.get("autonomous", {}).get("quality_threshold", 0.85)
        self.optimization_frequency = self.config.get("autonomous", {}).get("optimization_frequency", 10)
        
        # State tracking
        self.execution_history: List[ExecutionResult] = []
        self.current_context: Optional[ExecutionContext] = None
        self.performance_metrics: Dict[str, List[float]] = {}
        self.optimization_counter = 0
        
        # Predefined SDLC actions
        self.sdlc_actions = self._initialize_sdlc_actions()
        
        self.logger.info(f"AutonomousSDLCExecutor initialized for project: {project_path}")
    
    def _initialize_sdlc_actions(self) -> Dict[str, AutonomousAction]:
        """Initialize predefined SDLC actions."""
        actions = {}
        
        # Analysis phase
        actions["code_analysis"] = AutonomousAction(
            id="code_analysis",
            name="Static Code Analysis",
            phase=SDLCPhase.ANALYSIS,
            command="python -m mypy robo_rlhf/ --ignore-missing-imports",
            success_criteria={"max_errors": 0},
            timeout=120.0,
            auto_approve=True
        )
        
        actions["security_scan"] = AutonomousAction(
            id="security_scan",
            name="Security Vulnerability Scan",
            phase=SDLCPhase.ANALYSIS,
            command="python -m bandit -r robo_rlhf/ -f json",
            success_criteria={"max_high_severity": 0},
            timeout=180.0,
            critical=True
        )
        
        # Testing phase
        actions["unit_tests"] = AutonomousAction(
            id="unit_tests",
            name="Unit Test Execution",
            phase=SDLCPhase.TESTING,
            command="python -m pytest tests/unit/ -v --cov=robo_rlhf --cov-report=json",
            prerequisites=["code_analysis"],
            success_criteria={"min_coverage": 0.8, "max_failures": 0},
            timeout=300.0,
            critical=True
        )
        
        actions["integration_tests"] = AutonomousAction(
            id="integration_tests",
            name="Integration Test Execution", 
            phase=SDLCPhase.TESTING,
            command="python -m pytest tests/integration/ -v",
            prerequisites=["unit_tests"],
            success_criteria={"max_failures": 0},
            timeout=600.0,
            critical=True
        )
        
        actions["performance_tests"] = AutonomousAction(
            id="performance_tests",
            name="Performance Benchmark Tests",
            phase=SDLCPhase.TESTING,
            command="python -m pytest tests/performance/ --benchmark-only",
            prerequisites=["integration_tests"],
            success_criteria={"max_regression": 0.1},
            timeout=900.0
        )
        
        # Build phase
        actions["build_package"] = AutonomousAction(
            id="build_package",
            name="Build Python Package",
            phase=SDLCPhase.IMPLEMENTATION,
            command="python -m build",
            prerequisites=["unit_tests"],
            success_criteria={"build_success": True},
            timeout=300.0,
            critical=True
        )
        
        # Deployment phase
        actions["build_docker"] = AutonomousAction(
            id="build_docker",
            name="Build Docker Container",
            phase=SDLCPhase.DEPLOYMENT,
            command="docker build -t robo-rlhf:latest .",
            prerequisites=["build_package"],
            success_criteria={"build_success": True},
            rollback_command="docker rmi robo-rlhf:latest",
            timeout=600.0
        )
        
        actions["security_container_scan"] = AutonomousAction(
            id="security_container_scan",
            name="Container Security Scan",
            phase=SDLCPhase.DEPLOYMENT,
            command="docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy image robo-rlhf:latest",
            prerequisites=["build_docker"],
            success_criteria={"max_critical_vulns": 0},
            timeout=300.0,
            critical=True
        )
        
        # Quality gates
        actions["quality_gate"] = AutonomousAction(
            id="quality_gate",
            name="Quality Gate Validation",
            phase=SDLCPhase.INTEGRATION,
            command="python scripts/collect_metrics.py --validate-quality",
            prerequisites=["performance_tests", "security_scan"],
            success_criteria={"quality_score": 0.85},
            timeout=120.0,
            critical=True
        )
        
        return actions
    
    async def execute_autonomous_sdlc(self,
                                    target_phases: Optional[List[SDLCPhase]] = None,
                                    context: Optional[ExecutionContext] = None) -> Dict[str, Any]:
        """Execute autonomous SDLC pipeline."""
        self.logger.info("Starting autonomous SDLC execution")
        
        if context:
            self.current_context = context
        else:
            self.current_context = self._create_default_context()
        
        target_phases = target_phases or list(SDLCPhase)
        
        execution_summary = {
            "start_time": time.time(),
            "target_phases": [phase.value for phase in target_phases],
            "total_actions": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "optimizations_applied": 0,
            "rollbacks_performed": 0,
            "overall_success": False,
            "execution_time": 0.0,
            "quality_score": 0.0
        }
        
        try:
            # Create quantum execution plan
            plan = await self._create_execution_plan(target_phases)
            execution_summary["total_actions"] = len(plan.tasks)
            
            # Execute plan with autonomous decision-making
            results = await self._execute_quantum_plan(plan)
            
            # Process results
            for result in results["task_results"].values():
                if result.get("success", False):
                    execution_summary["successful_actions"] += 1
                else:
                    execution_summary["failed_actions"] += 1
                
                if result.get("optimization_applied", False):
                    execution_summary["optimizations_applied"] += 1
                
                if result.get("rollback_performed", False):
                    execution_summary["rollbacks_performed"] += 1
            
            # Calculate overall success
            success_rate = execution_summary["successful_actions"] / execution_summary["total_actions"]
            execution_summary["overall_success"] = success_rate >= self.quality_threshold
            
            # Calculate quality score
            execution_summary["quality_score"] = await self._calculate_quality_score()
            
            # Apply autonomous optimizations if needed
            if not execution_summary["overall_success"] or execution_summary["quality_score"] < self.quality_threshold:
                await self._apply_autonomous_optimizations(results)
            
            execution_summary["execution_time"] = time.time() - execution_summary["start_time"]
            
            self.logger.info(f"Autonomous SDLC execution completed: success={execution_summary['overall_success']}, quality={execution_summary['quality_score']:.3f}")
            
        except Exception as e:
            self.logger.error(f"Autonomous SDLC execution failed: {str(e)}")
            execution_summary["error"] = str(e)
            execution_summary["execution_time"] = time.time() - execution_summary["start_time"]
        
        return execution_summary
    
    async def _create_execution_plan(self, target_phases: List[SDLCPhase]) -> Any:
        """Create quantum execution plan for SDLC phases."""
        # Filter actions by target phases
        relevant_actions = [
            action for action in self.sdlc_actions.values()
            if action.phase in target_phases
        ]
        
        # Convert to quantum tasks
        quantum_tasks = []
        for action in relevant_actions:
            task = QuantumTask(
                id=action.id,
                name=action.name,
                description=f"Autonomous execution of {action.name}",
                priority=TaskPriority.CRITICAL if action.critical else TaskPriority.HIGH,
                dependencies=set(action.prerequisites),
                resource_requirements={
                    "cpu": 0.3,
                    "memory": 0.2,
                    "time": action.timeout
                }
            )
            quantum_tasks.append(task)
        
        # Create execution plan using quantum planner
        objective = f"Execute SDLC phases: {', '.join(phase.value for phase in target_phases)}"
        requirements = [task.name for task in quantum_tasks]
        
        plan = self.task_planner.create_quantum_plan(objective, requirements)
        plan.tasks = quantum_tasks  # Replace with our SDLC tasks
        
        return plan
    
    async def _execute_quantum_plan(self, plan: Any) -> Dict[str, Any]:
        """Execute quantum plan with autonomous decision-making."""
        results = {"task_results": {}, "autonomous_decisions": []}
        
        # Execute tasks according to quantum plan
        for task in plan.tasks:
            task_result = await self._execute_autonomous_action(task)
            results["task_results"][task.id] = task_result
            
            # Autonomous decision-making for failures
            if not task_result.get("success", False):
                decision = await self._make_autonomous_failure_decision(task, task_result)
                results["autonomous_decisions"].append(decision)
                
                if decision.get("action") == "retry":
                    retry_result = await self._execute_autonomous_action(task)
                    results["task_results"][f"{task.id}_retry"] = retry_result
                elif decision.get("action") == "rollback":
                    rollback_result = await self._perform_autonomous_rollback(task)
                    results["task_results"][f"{task.id}_rollback"] = rollback_result
        
        return results
    
    async def _execute_autonomous_action(self, task: QuantumTask) -> Dict[str, Any]:
        """Execute autonomous action with monitoring and optimization."""
        action = self.sdlc_actions.get(task.id)
        if not action:
            return {"success": False, "error": f"Action {task.id} not found"}
        
        self.logger.info(f"Executing autonomous action: {action.name}")
        
        result = ExecutionResult(
            action_id=action.id,
            status=ExecutionStatus.RUNNING,
            start_time=time.time()
        )
        
        try:
            # Pre-execution optimization check
            if self.optimization_counter >= self.optimization_frequency:
                await self._apply_real_time_optimization(action)
                result.optimization_applied = True
                self.optimization_counter = 0
            
            # Execute command
            process = await asyncio.create_subprocess_shell(
                action.command,
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=action.timeout
                )
                
                result.output = stdout.decode()
                result.error_output = stderr.decode()
                result.status = ExecutionStatus.SUCCESS if process.returncode == 0 else ExecutionStatus.FAILED
                
            except asyncio.TimeoutError:
                process.kill()
                result.status = ExecutionStatus.FAILED
                result.error_output = f"Command timed out after {action.timeout} seconds"
            
            result.end_time = time.time()
            
            # Evaluate success criteria
            success = await self._evaluate_success_criteria(action, result)
            result.status = ExecutionStatus.SUCCESS if success else ExecutionStatus.FAILED
            
            # Extract metrics
            result.metrics = await self._extract_execution_metrics(action, result)
            
            # Store in history
            self.execution_history.append(result)
            
            # Update performance tracking
            self._update_performance_metrics(action.id, result)
            
            self.optimization_counter += 1
            
            self.logger.info(f"Autonomous action completed: {action.name}, status={result.status.value}")
            
        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error_output = str(e)
            result.end_time = time.time()
            
            self.logger.error(f"Autonomous action failed: {action.name}, error={str(e)}")
        
        return {
            "success": result.status == ExecutionStatus.SUCCESS,
            "action_id": result.action_id,
            "execution_time": (result.end_time or time.time()) - result.start_time,
            "output": result.output,
            "error": result.error_output,
            "metrics": result.metrics,
            "optimization_applied": result.optimization_applied,
            "rollback_performed": result.rollback_performed
        }
    
    async def _make_autonomous_failure_decision(self, task: QuantumTask, task_result: Dict[str, Any]) -> Dict[str, Any]:
        """Make autonomous decision on how to handle failures."""
        action = self.sdlc_actions.get(task.id)
        if not action:
            return {"action": "skip", "reason": "Action not found"}
        
        self.logger.info(f"Making autonomous failure decision for: {action.name}")
        
        # Analyze failure
        failure_analysis = await self._analyze_failure(action, task_result)
        
        # Decision criteria
        criteria = {
            "is_critical": action.critical,
            "has_rollback": action.rollback_command is not None,
            "retry_count": task_result.get("retry_count", 0),
            "max_retries": action.retry_count,
            "failure_type": failure_analysis.get("type", "unknown"),
            "confidence": failure_analysis.get("confidence", 0.5)
        }
        
        # Decision logic
        if criteria["is_critical"] and criteria["retry_count"] < criteria["max_retries"]:
            if failure_analysis.get("recoverable", False):
                return {"action": "retry", "reason": "Critical task with recoverable failure"}
        
        if criteria["has_rollback"] and criteria["is_critical"]:
            return {"action": "rollback", "reason": "Critical failure with rollback available"}
        
        if not criteria["is_critical"]:
            return {"action": "continue", "reason": "Non-critical failure, continuing pipeline"}
        
        return {"action": "abort", "reason": "Unrecoverable critical failure"}
    
    async def _analyze_failure(self, action: AutonomousAction, task_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze failure to determine recovery options."""
        error_output = task_result.get("error", "")
        
        # Pattern-based failure analysis
        failure_patterns = {
            "dependency": ["ModuleNotFoundError", "ImportError", "command not found"],
            "timeout": ["timeout", "timed out", "TimeoutError"],
            "permission": ["Permission denied", "PermissionError"],
            "resource": ["OutOfMemoryError", "No space left", "ResourceExhausted"],
            "network": ["ConnectionError", "NetworkError", "timeout"],
            "configuration": ["ConfigError", "InvalidConfiguration"]
        }
        
        failure_type = "unknown"
        confidence = 0.3
        
        for pattern_type, patterns in failure_patterns.items():
            for pattern in patterns:
                if pattern in error_output:
                    failure_type = pattern_type
                    confidence = 0.8
                    break
            if failure_type != "unknown":
                break
        
        # Determine if recoverable
        recoverable_types = ["dependency", "configuration", "network"]
        recoverable = failure_type in recoverable_types
        
        return {
            "type": failure_type,
            "confidence": confidence,
            "recoverable": recoverable,
            "error_snippet": error_output[:200],
            "suggested_fix": self._suggest_fix(failure_type)
        }
    
    def _suggest_fix(self, failure_type: str) -> str:
        """Suggest fix based on failure type."""
        suggestions = {
            "dependency": "Install missing dependencies or check import paths",
            "timeout": "Increase timeout or optimize performance",
            "permission": "Check file permissions or run with appropriate privileges",
            "resource": "Allocate more resources or optimize memory usage",
            "network": "Check network connectivity or retry with backoff",
            "configuration": "Validate configuration parameters"
        }
        return suggestions.get(failure_type, "Review logs and fix underlying issue")
    
    async def _perform_autonomous_rollback(self, task: QuantumTask) -> Dict[str, Any]:
        """Perform autonomous rollback operation."""
        action = self.sdlc_actions.get(task.id)
        if not action or not action.rollback_command:
            return {"success": False, "error": "No rollback command available"}
        
        self.logger.info(f"Performing autonomous rollback: {action.name}")
        
        try:
            process = await asyncio.create_subprocess_shell(
                action.rollback_command,
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            success = process.returncode == 0
            
            return {
                "success": success,
                "rollback_performed": True,
                "output": stdout.decode(),
                "error": stderr.decode() if not success else ""
            }
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {str(e)}")
            return {"success": False, "error": str(e), "rollback_performed": False}
    
    async def _apply_real_time_optimization(self, action: AutonomousAction) -> None:
        """Apply real-time optimization based on performance history."""
        if action.id not in self.performance_metrics:
            return
        
        metrics = self.performance_metrics[action.id]
        
        # Analyze performance trends
        if len(metrics) >= 5:
            recent_times = metrics[-5:]
            avg_time = np.mean(recent_times)
            
            # Adjust timeout based on performance
            if avg_time > action.timeout * 0.8:
                action.timeout *= 1.2  # Increase timeout
                self.logger.info(f"Increased timeout for {action.name} to {action.timeout}s")
            
            elif avg_time < action.timeout * 0.3:
                action.timeout *= 0.9  # Decrease timeout
                self.logger.info(f"Decreased timeout for {action.name} to {action.timeout}s")
    
    async def _evaluate_success_criteria(self, action: AutonomousAction, result: ExecutionResult) -> bool:
        """Evaluate success criteria for action."""
        if result.status == ExecutionStatus.FAILED:
            return False
        
        for criterion, expected_value in action.success_criteria.items():
            if criterion == "max_errors" and "error" in result.output.lower():
                # Count errors in output (simplified)
                error_count = result.output.lower().count("error")
                if error_count > expected_value:
                    return False
            
            elif criterion == "min_coverage":
                # Extract coverage from pytest output (simplified)
                if "TOTAL" in result.output and "%" in result.output:
                    # Extract coverage percentage (this is simplified)
                    lines = result.output.split('\n')
                    for line in lines:
                        if "TOTAL" in line and "%" in line:
                            try:
                                coverage = float(line.split('%')[0].split()[-1]) / 100
                                if coverage < expected_value:
                                    return False
                            except:
                                pass
        
        return True
    
    async def _extract_execution_metrics(self, action: AutonomousAction, result: ExecutionResult) -> Dict[str, float]:
        """Extract metrics from execution result."""
        metrics = {}
        
        # Basic metrics
        if result.end_time:
            metrics["execution_time"] = result.end_time - result.start_time
            metrics["success"] = 1.0 if result.status == ExecutionStatus.SUCCESS else 0.0
        
        # Phase-specific metrics
        if action.phase == SDLCPhase.TESTING:
            # Extract test metrics
            output = result.output
            if "passed" in output:
                try:
                    passed = int(output.split("passed")[0].split()[-1])
                    metrics["tests_passed"] = passed
                except:
                    pass
            
            if "failed" in output:
                try:
                    failed = int(output.split("failed")[0].split()[-1])
                    metrics["tests_failed"] = failed
                except:
                    pass
        
        elif action.phase == SDLCPhase.DEPLOYMENT:
            # Extract deployment metrics
            metrics["deployment_size"] = len(result.output)  # Simplified
        
        return metrics
    
    def _update_performance_metrics(self, action_id: str, result: ExecutionResult) -> None:
        """Update performance tracking metrics."""
        if action_id not in self.performance_metrics:
            self.performance_metrics[action_id] = []
        
        if result.end_time:
            execution_time = result.end_time - result.start_time
            self.performance_metrics[action_id].append(execution_time)
            
            # Keep only recent metrics
            if len(self.performance_metrics[action_id]) > 20:
                self.performance_metrics[action_id] = self.performance_metrics[action_id][-10:]
    
    async def _calculate_quality_score(self) -> float:
        """Calculate overall quality score."""
        if not self.execution_history:
            return 0.0
        
        # Recent executions
        recent_results = self.execution_history[-10:]
        
        # Success rate
        success_count = sum(1 for r in recent_results if r.status == ExecutionStatus.SUCCESS)
        success_rate = success_count / len(recent_results)
        
        # Performance score (based on execution times)
        avg_execution_time = np.mean([
            (r.end_time or r.start_time) - r.start_time 
            for r in recent_results if r.end_time
        ])
        
        # Normalize execution time (assuming 300s baseline)
        performance_score = max(0.0, 1.0 - (avg_execution_time / 300.0))
        
        # Optimization effectiveness
        optimization_applied = sum(1 for r in recent_results if r.optimization_applied)
        optimization_score = min(1.0, optimization_applied / len(recent_results) * 2)
        
        # Weighted quality score
        quality_score = (success_rate * 0.5 + performance_score * 0.3 + optimization_score * 0.2)
        
        return min(1.0, quality_score)
    
    async def _apply_autonomous_optimizations(self, results: Dict[str, Any]) -> None:
        """Apply autonomous optimizations based on execution results."""
        self.logger.info("Applying autonomous optimizations")
        
        # Analyze failure patterns
        failed_tasks = [
            task_id for task_id, result in results["task_results"].items()
            if not result.get("success", False)
        ]
        
        for task_id in failed_tasks:
            if task_id in self.sdlc_actions:
                action = self.sdlc_actions[task_id]
                
                # Apply optimizations based on failure analysis
                optimization_applied = False
                
                # Increase timeout for timeout failures
                if "timeout" in results["task_results"][task_id].get("error", "").lower():
                    action.timeout *= 1.5
                    optimization_applied = True
                    self.logger.info(f"Increased timeout for {action.name} to {action.timeout}s")
                
                # Increase retry count for transient failures
                if "network" in results["task_results"][task_id].get("error", "").lower():
                    action.retry_count += 1
                    optimization_applied = True
                    self.logger.info(f"Increased retry count for {action.name} to {action.retry_count}")
                
                if optimization_applied:
                    # Re-execute optimized action
                    quantum_task = QuantumTask(
                        id=action.id,
                        name=action.name,
                        description=f"Optimized execution of {action.name}",
                        priority=TaskPriority.HIGH
                    )
                    
                    optimized_result = await self._execute_autonomous_action(quantum_task)
                    results["task_results"][f"{task_id}_optimized"] = optimized_result
    
    def _create_default_context(self) -> ExecutionContext:
        """Create default execution context."""
        return ExecutionContext(
            project_path=self.project_path,
            environment="development",
            configuration=self.config,
            resource_limits={"cpu": 1.0, "memory": 2.0, "storage": 10.0},
            quality_gates={"test_coverage": 0.8, "success_rate": 0.9},
            monitoring_config={"enabled": True, "interval": 60}
        )
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of autonomous execution history."""
        if not self.execution_history:
            return {"message": "No execution history available"}
        
        recent_executions = self.execution_history[-20:]
        
        total_executions = len(recent_executions)
        successful_executions = sum(1 for r in recent_executions if r.status == ExecutionStatus.SUCCESS)
        success_rate = successful_executions / total_executions
        
        avg_execution_time = np.mean([
            (r.end_time or r.start_time) - r.start_time 
            for r in recent_executions if r.end_time
        ])
        
        # Phase distribution
        phase_counts = {}
        for result in recent_executions:
            if result.action_id in self.sdlc_actions:
                phase = self.sdlc_actions[result.action_id].phase.value
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        # Calculate quality score synchronously for this summary
        quality_score = 0.0
        if self.execution_history:
            recent_results = self.execution_history[-10:]
            success_count = sum(1 for r in recent_results if r.status == ExecutionStatus.SUCCESS)
            quality_score = success_count / len(recent_results)
        
        return {
            "total_executions": total_executions,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "phase_distribution": phase_counts,
            "performance_metrics": len(self.performance_metrics),
            "optimizations_applied": sum(1 for r in recent_executions if r.optimization_applied),
            "current_quality_score": quality_score
        }
    
    def _create_secure_action(self, **kwargs) -> AutonomousAction:
        """Create a secure autonomous action with validation."""
        # Validate timeout
        if "timeout" in kwargs:
            kwargs["timeout"] = validate_numeric(
                kwargs["timeout"],
                min_value=1.0,
                max_value=self.command_timeout_limit,
                must_be_positive=True
            )
        
        # Sanitize command
        if "command" in kwargs:
            kwargs["command"] = self._sanitize_command(kwargs["command"])
        
        # Sanitize name and id
        if "name" in kwargs:
            kwargs["name"] = sanitize_input(
                kwargs["name"],
                max_length=200,
                allowed_chars="a-zA-Z0-9 _-"
            )
        
        if "id" in kwargs:
            kwargs["id"] = sanitize_input(
                kwargs["id"],
                max_length=100,
                allowed_chars="a-zA-Z0-9_"
            )
        
        return AutonomousAction(**kwargs)
    
    def _sanitize_command(self, command: str) -> str:
        """Sanitize and validate command for security."""
        # Basic input sanitization
        command = sanitize_input(command, max_length=1000)
        
        # Extract command executable
        cmd_parts = command.strip().split()
        if not cmd_parts:
            raise SecurityError("Empty command", threat_type="invalid_command")
        
        executable = cmd_parts[0]
        
        # Check if command is allowed
        if not any(allowed in executable for allowed in self.allowed_commands):
            raise SecurityError(
                f"Command '{executable}' not in allowed list: {self.allowed_commands}",
                threat_type="forbidden_command"
            )
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'\|', r'&', r';', r'\$\(', r'`', r'>', r'<',  # Shell operators
            r'rm\s+-rf', r'sudo', r'chmod\s+777',  # Dangerous commands  
            r'\.\./.*', r'/etc/', r'/root/'  # Path traversal
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                raise SecurityError(
                    f"Command contains dangerous pattern: {pattern}",
                    threat_type="dangerous_command"
                )
        
        return command
    
    def _validate_action_security(self, action: AutonomousAction) -> AutonomousAction:
        """Validate action for security compliance."""
        # Validate command security
        try:
            action.command = self._sanitize_command(action.command)
        except SecurityError as e:
            raise SecurityError(f"Action {action.id} failed command validation: {e}")
        
        # Validate rollback command if present
        if action.rollback_command:
            try:
                action.rollback_command = self._sanitize_command(action.rollback_command)
            except SecurityError as e:
                self.logger.warning(f"Removing unsafe rollback command for {action.id}: {e}")
                action.rollback_command = None
        
        # Validate timeout bounds
        if action.timeout > self.command_timeout_limit:
            self.logger.warning(f"Reducing timeout for {action.id} from {action.timeout} to {self.command_timeout_limit}")
            action.timeout = self.command_timeout_limit
        
        return action
    
    async def _execute_secure_command(self, action: AutonomousAction) -> asyncio.subprocess.Process:
        """Execute command with security constraints."""
        # Create secure environment
        secure_env = os.environ.copy()
        
        # Remove potentially dangerous environment variables
        dangerous_env_vars = ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH']
        for var in dangerous_env_vars:
            secure_env.pop(var, None)
        
        # Add security markers
        secure_env['ROBO_RLHF_SECURE_MODE'] = '1'
        
        try:
            process = await asyncio.create_subprocess_shell(
                action.command,
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=secure_env,
                # Additional security: limit process resources
                preexec_fn=None,  # Disable preexec for security
                shell=True  # We've already validated the command
            )
            return process
        except Exception as e:
            raise SecurityError(f"Failed to execute secure command: {e}", threat_type="execution_failure")
    
    def _sanitize_output(self, output: str) -> str:
        """Sanitize command output for security."""
        if len(output) > self.max_output_size:
            self.logger.warning(f"Output truncated from {len(output)} to {self.max_output_size} bytes")
            output = output[:self.max_output_size] + "\n[OUTPUT TRUNCATED FOR SECURITY]"
        
        # Remove potential secrets (basic patterns)
        secret_patterns = [
            (r'password[=:]\s*["\'][^"\s]+["\']', 'password=***'),
            (r'token[=:]\s*["\'][^"\s]+["\']', 'token=***'),
            (r'key[=:]\s*["\'][^"\s]+["\']', 'key=***'),
            (r'secret[=:]\s*["\'][^"\s]+["\']', 'secret=***')
        ]
        
        for pattern, replacement in secret_patterns:
            output = re.sub(pattern, replacement, output, flags=re.IGNORECASE)
        
        return output
    
    def _validate_command_output(self, stdout: str, stderr: str) -> None:
        """Validate command output for security threats."""
        # Check for suspicious patterns in output
        suspicious_patterns = [
            r'(?i)(password|token|secret|key)\s*[:=]\s*[\w\d@#$%]+',
            r'(?i)connection\s+established.*backdoor',
            r'(?i)reverse\s+shell',
            r'(?i)malicious\s+code\s+detected'
        ]
        
        combined_output = stdout + "\n" + stderr
        
        for pattern in suspicious_patterns:
            if re.search(pattern, combined_output):
                raise SecurityError(
                    f"Suspicious pattern detected in command output",
                    threat_type="malicious_output"
                )