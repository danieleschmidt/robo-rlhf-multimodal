"""
Quantum-inspired task planning for autonomous SDLC execution.

Implements quantum superposition principles for exploring multiple solution paths
simultaneously and quantum entanglement for coordinated task execution.
"""

import asyncio
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from robo_rlhf.core import get_logger, get_config
from robo_rlhf.core.exceptions import RoboRLHFError


class TaskState(Enum):
    """Quantum-inspired task states."""
    SUPERPOSITION = "superposition"  # Multiple potential solutions
    ENTANGLED = "entangled"  # Coordinated with other tasks
    COLLAPSED = "collapsed"  # Solution determined
    COMPLETED = "completed"  # Task finished
    FAILED = "failed"  # Task failed


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class QuantumTask:
    """Quantum-inspired task representation."""
    id: str
    name: str
    description: str
    priority: TaskPriority
    state: TaskState = TaskState.SUPERPOSITION
    dependencies: Set[str] = field(default_factory=set)
    entangled_tasks: Set[str] = field(default_factory=set)
    solutions: List[Dict[str, Any]] = field(default_factory=list)
    selected_solution: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    success_probability: float = 0.8
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class QuantumPlan:
    """Quantum-inspired execution plan."""
    id: str
    name: str
    objective: str
    tasks: List[QuantumTask] = field(default_factory=list)
    success_criteria: Dict[str, float] = field(default_factory=dict)
    resource_limits: Dict[str, float] = field(default_factory=dict)
    execution_graph: Dict[str, List[str]] = field(default_factory=dict)
    parallel_groups: List[List[str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    estimated_completion: Optional[float] = None


class QuantumTaskPlanner:
    """Quantum-inspired autonomous task planner."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.config = config or get_config().to_dict()
        
        # Quantum parameters
        self.superposition_depth = self.config.get("quantum", {}).get("superposition_depth", 3)
        self.entanglement_strength = self.config.get("quantum", {}).get("entanglement_strength", 0.7)
        self.collapse_threshold = self.config.get("quantum", {}).get("collapse_threshold", 0.9)
        
        # Planning parameters
        self.max_parallel_tasks = self.config.get("planning", {}).get("max_parallel_tasks", 4)
        self.resource_buffer = self.config.get("planning", {}).get("resource_buffer", 0.2)
        self.success_threshold = self.config.get("planning", {}).get("success_threshold", 0.85)
        
        # State tracking
        self.active_plans: Dict[str, QuantumPlan] = {}
        self.completed_tasks: Dict[str, QuantumTask] = {}
        self.resource_usage: Dict[str, float] = {}
        
        self.logger.info("QuantumTaskPlanner initialized with quantum parameters")
    
    def create_quantum_plan(
        self,
        objective: str,
        requirements: List[str],
        constraints: Optional[Dict[str, Any]] = None
    ) -> QuantumPlan:
        """Create quantum-inspired execution plan."""
        plan_id = f"plan_{int(time.time())}"
        constraints = constraints or {}
        
        self.logger.info(f"Creating quantum plan: {objective}")
        
        # Decompose objective into quantum tasks
        tasks = self._decompose_objective(objective, requirements)
        
        # Apply quantum superposition to explore solution space
        for task in tasks:
            task.solutions = self._generate_solution_superposition(task)
        
        # Create quantum entanglement between related tasks
        self._create_task_entanglement(tasks)
        
        # Build execution graph with quantum parallelization
        execution_graph, parallel_groups = self._build_quantum_execution_graph(tasks)
        
        plan = QuantumPlan(
            id=plan_id,
            name=f"Quantum Plan: {objective}",
            objective=objective,
            tasks=tasks,
            execution_graph=execution_graph,
            parallel_groups=parallel_groups,
            resource_limits=constraints.get("resources", {}),
            success_criteria=constraints.get("success_criteria", {})
        )
        
        # Estimate completion time using quantum probability
        plan.estimated_completion = self._estimate_quantum_completion(plan)
        
        self.active_plans[plan_id] = plan
        self.logger.info(f"Quantum plan created with {len(tasks)} tasks, estimated completion: {plan.estimated_completion:.2f}s")
        
        return plan
    
    def _decompose_objective(self, objective: str, requirements: List[str]) -> List[QuantumTask]:
        """Decompose objective into quantum tasks."""
        tasks = []
        
        # SDLC-specific task decomposition patterns
        sdlc_patterns = {
            "implementation": ["analysis", "design", "coding", "testing", "integration"],
            "optimization": ["profiling", "bottleneck_identification", "algorithm_optimization", "validation"],
            "security": ["threat_modeling", "vulnerability_scanning", "penetration_testing", "remediation"],
            "deployment": ["build", "containerization", "staging", "production_deployment", "monitoring"],
            "testing": ["unit_tests", "integration_tests", "performance_tests", "security_tests"],
        }
        
        # Analyze objective for task patterns
        objective_lower = objective.lower()
        detected_patterns = []
        
        for pattern, pattern_tasks in sdlc_patterns.items():
            if pattern in objective_lower:
                detected_patterns.extend(pattern_tasks)
        
        # Create quantum tasks for detected patterns
        for i, task_name in enumerate(detected_patterns):
            task_id = f"task_{task_name}_{i}"
            
            # Determine priority based on SDLC criticality
            priority = self._determine_task_priority(task_name, requirements)
            
            # Calculate success probability based on complexity
            success_prob = self._calculate_success_probability(task_name, requirements)
            
            task = QuantumTask(
                id=task_id,
                name=task_name,
                description=f"Execute {task_name} for {objective}",
                priority=priority,
                success_probability=success_prob,
                resource_requirements=self._estimate_resource_requirements(task_name)
            )
            
            tasks.append(task)
        
        # Add dependencies based on SDLC workflow
        self._add_task_dependencies(tasks)
        
        return tasks
    
    def _generate_solution_superposition(self, task: QuantumTask) -> List[Dict[str, Any]]:
        """Generate quantum superposition of potential solutions."""
        solutions = []
        
        # Solution generation patterns based on task type
        solution_patterns = {
            "analysis": [
                {"approach": "static_analysis", "tools": ["ast", "mypy"], "complexity": 0.3},
                {"approach": "dynamic_analysis", "tools": ["profiler", "tracer"], "complexity": 0.6},
                {"approach": "ml_analysis", "tools": ["embeddings", "clustering"], "complexity": 0.8}
            ],
            "testing": [
                {"approach": "unit_testing", "framework": "pytest", "coverage": 0.85, "complexity": 0.4},
                {"approach": "integration_testing", "framework": "pytest", "coverage": 0.75, "complexity": 0.6},
                {"approach": "e2e_testing", "framework": "selenium", "coverage": 0.65, "complexity": 0.8}
            ],
            "optimization": [
                {"approach": "algorithmic", "method": "complexity_reduction", "improvement": 0.3},
                {"approach": "caching", "method": "memoization", "improvement": 0.5},
                {"approach": "parallelization", "method": "concurrent_futures", "improvement": 0.7}
            ],
            "deployment": [
                {"approach": "docker", "strategy": "containerization", "scalability": 0.8},
                {"approach": "kubernetes", "strategy": "orchestration", "scalability": 0.9},
                {"approach": "serverless", "strategy": "functions", "scalability": 0.7}
            ]
        }
        
        # Find matching patterns
        for pattern_name, pattern_solutions in solution_patterns.items():
            if pattern_name in task.name.lower():
                for solution in pattern_solutions:
                    # Add quantum properties
                    quantum_solution = solution.copy()
                    quantum_solution["quantum_probability"] = np.random.beta(2, 2)  # Quantum uncertainty
                    quantum_solution["coherence_time"] = np.random.exponential(10)  # Decoherence
                    solutions.append(quantum_solution)
                break
        
        # Generate default solutions if no pattern matches
        if not solutions:
            for i in range(self.superposition_depth):
                solutions.append({
                    "approach": f"approach_{i}",
                    "quantum_probability": np.random.beta(2, 2),
                    "coherence_time": np.random.exponential(10),
                    "complexity": np.random.uniform(0.2, 0.8)
                })
        
        self.logger.debug(f"Generated {len(solutions)} quantum solutions for task {task.name}")
        return solutions
    
    def _create_task_entanglement(self, tasks: List[QuantumTask]) -> None:
        """Create quantum entanglement between related tasks."""
        for i, task1 in enumerate(tasks):
            for j, task2 in enumerate(tasks):
                if i != j:
                    # Calculate entanglement strength based on task relationship
                    entanglement = self._calculate_entanglement_strength(task1, task2)
                    
                    if entanglement > self.entanglement_strength:
                        task1.entangled_tasks.add(task2.id)
                        task2.entangled_tasks.add(task1.id)
                        
                        self.logger.debug(f"Created quantum entanglement: {task1.name} <-> {task2.name} (strength: {entanglement:.3f})")
    
    def _calculate_entanglement_strength(self, task1: QuantumTask, task2: QuantumTask) -> float:
        """Calculate quantum entanglement strength between tasks."""
        # Resource overlap
        resource_overlap = 0.0
        for resource in task1.resource_requirements:
            if resource in task2.resource_requirements:
                resource_overlap += 1.0
        
        # Dependency relationship
        dependency_strength = 0.0
        if task2.id in task1.dependencies or task1.id in task2.dependencies:
            dependency_strength = 0.8
        
        # Task type similarity
        type_similarity = 0.0
        task1_words = set(task1.name.lower().split())
        task2_words = set(task2.name.lower().split())
        if task1_words & task2_words:
            type_similarity = len(task1_words & task2_words) / len(task1_words | task2_words)
        
        return (resource_overlap * 0.3 + dependency_strength * 0.5 + type_similarity * 0.2)
    
    def _build_quantum_execution_graph(self, tasks: List[QuantumTask]) -> Tuple[Dict[str, List[str]], List[List[str]]]:
        """Build quantum execution graph with parallel groups."""
        graph = {}
        parallel_groups = []
        
        # Build dependency graph
        for task in tasks:
            graph[task.id] = list(task.dependencies)
        
        # Find parallel execution groups using quantum entanglement
        processed_tasks = set()
        
        for task in tasks:
            if task.id not in processed_tasks:
                # Create parallel group with entangled tasks
                parallel_group = [task.id]
                processed_tasks.add(task.id)
                
                # Add entangled tasks that can run in parallel
                for entangled_id in task.entangled_tasks:
                    entangled_task = next((t for t in tasks if t.id == entangled_id), None)
                    if entangled_task and entangled_id not in processed_tasks:
                        # Check if tasks can run in parallel (no dependencies between them)
                        if entangled_id not in task.dependencies and task.id not in entangled_task.dependencies:
                            parallel_group.append(entangled_id)
                            processed_tasks.add(entangled_id)
                
                if len(parallel_group) > 1:
                    parallel_groups.append(parallel_group)
        
        self.logger.debug(f"Created quantum execution graph with {len(parallel_groups)} parallel groups")
        return graph, parallel_groups
    
    async def execute_quantum_plan(self, plan: QuantumPlan) -> Dict[str, Any]:
        """Execute quantum plan with autonomous decision-making."""
        self.logger.info(f"Executing quantum plan: {plan.name}")
        
        results = {
            "plan_id": plan.id,
            "start_time": time.time(),
            "task_results": {},
            "collapsed_solutions": {},
            "success": False,
            "completion_time": None
        }
        
        try:
            # Execute parallel groups with quantum parallelization
            for group in plan.parallel_groups:
                await self._execute_parallel_group(plan, group, results)
            
            # Execute remaining tasks sequentially
            remaining_tasks = [t for t in plan.tasks if t.id not in results["task_results"]]
            for task in remaining_tasks:
                await self._execute_quantum_task(task, results)
            
            # Calculate overall success
            success_rate = sum(1 for r in results["task_results"].values() if r.get("success", False))
            success_rate /= len(plan.tasks)
            
            results["success"] = success_rate >= self.success_threshold
            results["completion_time"] = time.time() - results["start_time"]
            
            self.logger.info(f"Quantum plan execution completed: success={results['success']}, time={results['completion_time']:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Quantum plan execution failed: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    async def _execute_parallel_group(self, plan: QuantumPlan, group: List[str], results: Dict[str, Any]) -> None:
        """Execute parallel group of quantum entangled tasks."""
        group_tasks = [t for t in plan.tasks if t.id in group]
        
        self.logger.info(f"Executing parallel group with {len(group_tasks)} entangled tasks")
        
        # Execute tasks concurrently using quantum parallelization
        async def execute_task_wrapper(task):
            return await self._execute_quantum_task(task, results)
        
        # Use asyncio.gather for concurrent execution
        await asyncio.gather(*[execute_task_wrapper(task) for task in group_tasks])
    
    async def _execute_quantum_task(self, task: QuantumTask, results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual quantum task with solution collapse."""
        self.logger.debug(f"Executing quantum task: {task.name}")
        
        task.started_at = time.time()
        task.state = TaskState.ENTANGLED
        
        task_result = {
            "task_id": task.id,
            "start_time": task.started_at,
            "success": False,
            "collapsed_solution": None,
            "execution_time": None,
            "metrics": {}
        }
        
        try:
            # Quantum solution collapse - select optimal solution
            collapsed_solution = await self._collapse_quantum_solution(task)
            task.selected_solution = collapsed_solution
            task.state = TaskState.COLLAPSED
            
            # Execute collapsed solution
            execution_result = await self._execute_collapsed_solution(task, collapsed_solution)
            
            # Update task state and metrics
            task.state = TaskState.COMPLETED if execution_result["success"] else TaskState.FAILED
            task.completed_at = time.time()
            task.execution_time = task.completed_at - task.started_at
            
            task_result.update(execution_result)
            task_result["collapsed_solution"] = collapsed_solution
            task_result["execution_time"] = task.execution_time
            
            results["task_results"][task.id] = task_result
            results["collapsed_solutions"][task.id] = collapsed_solution
            
            self.logger.debug(f"Quantum task completed: {task.name}, success={execution_result['success']}")
            
        except Exception as e:
            task.state = TaskState.FAILED
            task_result["error"] = str(e)
            results["task_results"][task.id] = task_result
            
            self.logger.error(f"Quantum task failed: {task.name}, error={str(e)}")
        
        return task_result
    
    async def _collapse_quantum_solution(self, task: QuantumTask) -> Dict[str, Any]:
        """Collapse quantum superposition to select optimal solution."""
        if not task.solutions:
            return {"approach": "default", "quantum_probability": 1.0}
        
        # Calculate collapse probabilities based on quantum mechanics
        probabilities = []
        for solution in task.solutions:
            # Quantum probability with decoherence
            base_prob = solution.get("quantum_probability", 0.5)
            coherence_factor = np.exp(-time.time() / solution.get("coherence_time", 10))
            collapse_prob = base_prob * coherence_factor
            probabilities.append(collapse_prob)
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / len(task.solutions)] * len(task.solutions)
        
        # Quantum measurement - select solution based on probability distribution
        selected_idx = np.random.choice(len(task.solutions), p=probabilities)
        collapsed_solution = task.solutions[selected_idx].copy()
        collapsed_solution["collapse_probability"] = probabilities[selected_idx]
        
        self.logger.debug(f"Quantum solution collapsed for {task.name}: {collapsed_solution['approach']}")
        return collapsed_solution
    
    async def _execute_collapsed_solution(self, task: QuantumTask, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the collapsed quantum solution."""
        # Simulate solution execution with realistic delays
        execution_delay = solution.get("complexity", 0.5) * np.random.uniform(0.1, 2.0)
        await asyncio.sleep(execution_delay)
        
        # Calculate success probability based on solution quality and task complexity
        base_success = task.success_probability
        solution_quality = solution.get("collapse_probability", 0.5)
        success_probability = base_success * solution_quality
        
        # Simulate execution result
        success = np.random.random() < success_probability
        
        result = {
            "success": success,
            "approach": solution.get("approach", "unknown"),
            "execution_delay": execution_delay,
            "success_probability": success_probability,
            "metrics": {
                "quantum_coherence": solution.get("coherence_time", 0),
                "collapse_probability": solution.get("collapse_probability", 0),
                "complexity_score": solution.get("complexity", 0)
            }
        }
        
        return result
    
    def _determine_task_priority(self, task_name: str, requirements: List[str]) -> TaskPriority:
        """Determine task priority based on SDLC criticality."""
        critical_tasks = ["security", "testing", "deployment", "monitoring"]
        high_priority_tasks = ["analysis", "design", "integration"]
        
        task_lower = task_name.lower()
        
        if any(critical in task_lower for critical in critical_tasks):
            return TaskPriority.CRITICAL
        elif any(high in task_lower for high in high_priority_tasks):
            return TaskPriority.HIGH
        else:
            return TaskPriority.MEDIUM
    
    def _calculate_success_probability(self, task_name: str, requirements: List[str]) -> float:
        """Calculate task success probability based on complexity and requirements."""
        base_probability = 0.8
        
        complexity_factors = {
            "security": -0.2,
            "optimization": -0.15,
            "integration": -0.1,
            "testing": 0.1,
            "analysis": 0.05
        }
        
        task_lower = task_name.lower()
        adjustment = 0.0
        
        for factor, value in complexity_factors.items():
            if factor in task_lower:
                adjustment += value
                break
        
        # Adjust based on requirements complexity
        req_complexity = len(requirements) * 0.02
        
        final_probability = max(0.3, min(0.95, base_probability + adjustment - req_complexity))
        return final_probability
    
    def _estimate_resource_requirements(self, task_name: str) -> Dict[str, float]:
        """Estimate resource requirements for task."""
        base_requirements = {
            "cpu": 0.3,
            "memory": 0.2,
            "storage": 0.1,
            "network": 0.1
        }
        
        # Task-specific multipliers
        multipliers = {
            "optimization": {"cpu": 2.0, "memory": 1.5},
            "testing": {"cpu": 1.5, "memory": 1.2, "storage": 1.3},
            "deployment": {"network": 2.0, "storage": 1.5},
            "security": {"cpu": 1.3, "memory": 1.3},
            "analysis": {"memory": 1.8, "storage": 1.2}
        }
        
        task_lower = task_name.lower()
        requirements = base_requirements.copy()
        
        for pattern, pattern_multipliers in multipliers.items():
            if pattern in task_lower:
                for resource, multiplier in pattern_multipliers.items():
                    requirements[resource] *= multiplier
                break
        
        return requirements
    
    def _add_task_dependencies(self, tasks: List[QuantumTask]) -> None:
        """Add SDLC dependencies between tasks."""
        task_map = {task.name: task for task in tasks}
        
        # SDLC dependency patterns
        dependencies = {
            "design": ["analysis"],
            "coding": ["design"],
            "unit_tests": ["coding"],
            "integration_tests": ["unit_tests"],
            "performance_tests": ["integration_tests"],
            "security_tests": ["integration_tests"],
            "build": ["unit_tests"],
            "containerization": ["build"],
            "staging": ["containerization"],
            "production_deployment": ["staging", "security_tests"],
            "monitoring": ["production_deployment"]
        }
        
        for task_name, task_deps in dependencies.items():
            if task_name in task_map:
                for dep_name in task_deps:
                    if dep_name in task_map:
                        task_map[task_name].dependencies.add(task_map[dep_name].id)
    
    def _estimate_quantum_completion(self, plan: QuantumPlan) -> float:
        """Estimate plan completion time using quantum probability."""
        total_time = 0.0
        
        # Sequential time for critical path
        critical_path_time = 0.0
        for task in plan.tasks:
            if task.priority == TaskPriority.CRITICAL:
                # Estimate task duration based on complexity and probability
                base_duration = task.resource_requirements.get("cpu", 0.5) * 10
                uncertainty_factor = 1.0 / task.success_probability
                critical_path_time += base_duration * uncertainty_factor
        
        # Parallel execution savings from quantum entanglement
        parallel_savings = len(plan.parallel_groups) * 0.3
        
        total_time = max(1.0, critical_path_time - parallel_savings)
        return total_time


class QuantumDecisionEngine:
    """Quantum-inspired autonomous decision engine."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.decision_history: List[Dict[str, Any]] = []
        self.learning_rate = 0.1
        self.quantum_confidence_threshold = 0.75
    
    async def make_autonomous_decision(
        self,
        context: Dict[str, Any],
        options: List[Dict[str, Any]],
        criteria: Dict[str, float]
    ) -> Dict[str, Any]:
        """Make autonomous decision using quantum-inspired algorithms."""
        self.logger.info(f"Making autonomous decision with {len(options)} options")
        
        # Create quantum superposition of options
        quantum_options = self._create_option_superposition(options, criteria)
        
        # Apply quantum interference patterns
        interference_adjusted = self._apply_quantum_interference(quantum_options, context)
        
        # Measure quantum state to collapse to decision
        decision = self._measure_quantum_decision(interference_adjusted, criteria)
        
        # Learn from decision outcome for future improvements
        self._record_decision(context, options, decision, criteria)
        
        self.logger.info(f"Autonomous decision made: {decision.get('selected_option', 'unknown')}")
        return decision
    
    def _create_option_superposition(
        self,
        options: List[Dict[str, Any]],
        criteria: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Create quantum superposition of decision options."""
        quantum_options = []
        
        for option in options:
            quantum_option = option.copy()
            
            # Calculate quantum amplitudes based on criteria
            amplitude = 0.0
            for criterion, weight in criteria.items():
                option_score = option.get(criterion, 0.0)
                amplitude += option_score * weight
            
            # Normalize amplitude
            quantum_option["quantum_amplitude"] = max(0.0, min(1.0, amplitude))
            quantum_option["quantum_phase"] = np.random.uniform(0, 2 * np.pi)
            
            quantum_options.append(quantum_option)
        
        return quantum_options
    
    def _apply_quantum_interference(
        self,
        quantum_options: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply quantum interference patterns to options."""
        # Context-based interference
        context_factors = context.get("environmental_factors", {})
        
        for option in quantum_options:
            # Constructive/destructive interference based on context
            interference_factor = 1.0
            
            for factor, factor_value in context_factors.items():
                if factor in option:
                    phase_difference = abs(option["quantum_phase"] - factor_value)
                    if phase_difference < np.pi / 2:  # Constructive interference
                        interference_factor *= 1.2
                    else:  # Destructive interference
                        interference_factor *= 0.8
            
            # Apply interference to amplitude
            option["quantum_amplitude"] *= interference_factor
            option["interference_factor"] = interference_factor
        
        return quantum_options
    
    def _measure_quantum_decision(
        self,
        quantum_options: List[Dict[str, Any]],
        criteria: Dict[str, float]
    ) -> Dict[str, Any]:
        """Measure quantum state to collapse to final decision."""
        if not quantum_options:
            return {"error": "No options available"}
        
        # Calculate measurement probabilities
        probabilities = []
        for option in quantum_options:
            # Quantum probability = |amplitude|²
            prob = option["quantum_amplitude"] ** 2
            probabilities.append(prob)
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / len(quantum_options)] * len(quantum_options)
        
        # Quantum measurement
        selected_idx = np.random.choice(len(quantum_options), p=probabilities)
        selected_option = quantum_options[selected_idx]
        
        # Calculate confidence based on quantum coherence
        max_prob = max(probabilities)
        confidence = max_prob if max_prob > self.quantum_confidence_threshold else max_prob * 0.7
        
        return {
            "selected_option": selected_option,
            "selected_index": selected_idx,
            "confidence": confidence,
            "quantum_probability": probabilities[selected_idx],
            "measurement_time": time.time(),
            "all_probabilities": probabilities
        }
    
    def _record_decision(
        self,
        context: Dict[str, Any],
        options: List[Dict[str, Any]],
        decision: Dict[str, Any],
        criteria: Dict[str, float]
    ) -> None:
        """Record decision for learning and improvement."""
        decision_record = {
            "timestamp": time.time(),
            "context": context,
            "options_count": len(options),
            "decision": decision,
            "criteria": criteria,
            "confidence": decision.get("confidence", 0.0)
        }
        
        self.decision_history.append(decision_record)
        
        # Keep only recent decisions for memory management
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-500:]
    
    def get_decision_analytics(self) -> Dict[str, Any]:
        """Get analytics on decision-making performance."""
        if not self.decision_history:
            return {"message": "No decision history available"}
        
        recent_decisions = self.decision_history[-100:]
        
        avg_confidence = np.mean([d["confidence"] for d in recent_decisions])
        decision_types = {}
        
        for decision in recent_decisions:
            option_type = decision["decision"].get("selected_option", {}).get("type", "unknown")
            decision_types[option_type] = decision_types.get(option_type, 0) + 1
        
        return {
            "total_decisions": len(self.decision_history),
            "recent_decisions": len(recent_decisions),
            "average_confidence": avg_confidence,
            "decision_types": decision_types,
            "quantum_coherence": avg_confidence > self.quantum_confidence_threshold
        }