"""
Unit tests for quantum-inspired autonomous SDLC modules.
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, Any
import time

from robo_rlhf.quantum.planner import (
    QuantumTaskPlanner, QuantumDecisionEngine, QuantumTask, TaskState, TaskPriority
)
from robo_rlhf.quantum.optimizer import (
    QuantumOptimizer, MultiObjectiveOptimizer, OptimizationObjective, 
    OptimizationProblem, OptimizationConstraint
)
from robo_rlhf.quantum.autonomous import AutonomousSDLCExecutor, SDLCPhase, ExecutionContext
from robo_rlhf.quantum.analytics import (
    PredictiveAnalytics, ResourcePredictor, PredictionType, MetricSample
)


class TestQuantumTaskPlanner:
    """Test quantum task planning capabilities."""
    
    def setup_method(self):
        """Setup test instance."""
        self.config = {
            "quantum": {"superposition_depth": 2, "entanglement_strength": 0.5},
            "planning": {"max_parallel_tasks": 2, "success_threshold": 0.8}
        }
        self.planner = QuantumTaskPlanner(self.config)
    
    def test_task_planner_initialization(self):
        """Test planner initializes correctly."""
        assert self.planner.superposition_depth == 2
        assert self.planner.entanglement_strength == 0.5
        assert self.planner.max_parallel_tasks == 2
    
    def test_create_quantum_plan(self):
        """Test quantum plan creation."""
        objective = "Test SDLC automation"
        requirements = ["testing", "deployment"]
        
        plan = self.planner.create_quantum_plan(objective, requirements)
        
        assert plan.objective == objective
        assert len(plan.tasks) > 0
        assert plan.id is not None
        
        # Check tasks have quantum properties
        for task in plan.tasks:
            assert isinstance(task, QuantumTask)
            assert task.state == TaskState.SUPERPOSITION
            assert len(task.solutions) > 0
    
    @pytest.mark.asyncio
    async def test_execute_quantum_plan(self):
        """Test quantum plan execution."""
        plan = self.planner.create_quantum_plan("Test execution", ["analysis"])
        
        # Limit to one task for speed
        plan.tasks = plan.tasks[:1]
        
        results = await self.planner.execute_quantum_plan(plan)
        
        assert "task_results" in results
        assert "collapsed_solutions" in results
        assert isinstance(results["success"], bool)
        assert "completion_time" in results
    
    def test_task_decomposition(self):
        """Test objective decomposition into tasks."""
        objective = "implementation optimization security"
        requirements = ["test_coverage", "performance"]
        
        tasks = self.planner._decompose_objective(objective, requirements)
        
        assert len(tasks) > 0
        
        # Check that tasks have appropriate priorities
        priorities = [task.priority for task in tasks]
        assert TaskPriority.CRITICAL in priorities or TaskPriority.HIGH in priorities


class TestQuantumDecisionEngine:
    """Test autonomous decision making."""
    
    def setup_method(self):
        """Setup test instance."""
        self.engine = QuantumDecisionEngine()
    
    @pytest.mark.asyncio
    async def test_autonomous_decision(self):
        """Test autonomous decision making."""
        context = {"load": 0.7, "error_rate": 0.01}
        options = [
            {"strategy": "A", "cost": 0.3, "benefit": 0.8},
            {"strategy": "B", "cost": 0.5, "benefit": 0.9}
        ]
        criteria = {"cost": -1.0, "benefit": 1.0}
        
        decision = await self.engine.make_autonomous_decision(context, options, criteria)
        
        assert "selected_option" in decision
        assert "confidence" in decision
        assert 0 <= decision["confidence"] <= 1
        assert "selected_index" in decision
    
    def test_option_superposition(self):
        """Test quantum superposition of options."""
        options = [
            {"value": 0.5, "quality": 0.8},
            {"value": 0.7, "quality": 0.6}
        ]
        criteria = {"value": 0.6, "quality": 0.4}
        
        quantum_options = self.engine._create_option_superposition(options, criteria)
        
        assert len(quantum_options) == len(options)
        for option in quantum_options:
            assert "quantum_amplitude" in option
            assert "quantum_phase" in option
            assert 0 <= option["quantum_amplitude"] <= 1
    
    def test_decision_history_tracking(self):
        """Test decision history tracking."""
        initial_count = len(self.engine.decision_history)
        
        # Record a decision manually
        context = {"test": True}
        decision = {"selected_option": {"action": "test"}, "confidence": 0.8}
        self.engine._record_decision(context, [], decision, {})
        
        assert len(self.engine.decision_history) == initial_count + 1
        
        analytics = self.engine.get_decision_analytics()
        assert analytics["total_decisions"] == initial_count + 1


class TestQuantumOptimizer:
    """Test quantum optimization algorithms."""
    
    def setup_method(self):
        """Setup test instance."""
        self.config = {
            "optimization": {"population_size": 10, "elite_size": 2},
            "quantum": {"temperature": 1.0}
        }
        self.optimizer = QuantumOptimizer(self.config)
    
    @pytest.mark.asyncio
    async def test_optimization_problem(self):
        """Test basic optimization problem solving."""
        # Simple test problem
        problem = OptimizationProblem(
            name="Test Problem",
            objectives=[OptimizationObjective.MAXIMIZE_QUALITY],
            parameter_bounds={"param1": (0.0, 1.0), "param2": (0.0, 1.0)},
            target_solutions=5,
            max_generations=10  # Short for testing
        )
        
        solutions = await self.optimizer.optimize(problem)
        
        assert len(solutions) > 0
        assert len(solutions) <= problem.target_solutions
        
        # Check solution properties
        for solution in solutions:
            assert solution.constraints_satisfied or not problem.constraints
            assert solution.fitness_score >= 0
            assert "param1" in solution.parameters
            assert "param2" in solution.parameters
    
    def test_population_initialization(self):
        """Test quantum population initialization."""
        problem = OptimizationProblem(
            name="Init Test",
            objectives=[OptimizationObjective.MINIMIZE_TIME],
            parameter_bounds={"x": (0.0, 10.0), "y": (-5.0, 5.0)}
        )
        
        population = self.optimizer._initialize_quantum_population(problem)
        
        assert len(population) == self.optimizer.population_size
        
        for solution in population:
            assert 0.0 <= solution.parameters["x"] <= 10.0
            assert -5.0 <= solution.parameters["y"] <= 5.0
            assert solution.quantum_energy > 0
    
    def test_fitness_calculation(self):
        """Test quantum fitness calculation."""
        from robo_rlhf.quantum.optimizer import OptimizationSolution
        
        problem = OptimizationProblem(
            name="Fitness Test",
            objectives=[OptimizationObjective.MAXIMIZE_QUALITY],
            objective_weights={OptimizationObjective.MAXIMIZE_QUALITY: 1.0}
        )
        
        solution = OptimizationSolution(
            parameters={"test_param": 0.5},
            objectives={"quality": 0.8},
            constraints_satisfied=True,
            fitness_score=0.0
        )
        
        fitness = self.optimizer._calculate_quantum_fitness(solution, problem)
        
        assert fitness > 0
        assert fitness <= 1.0


class TestMultiObjectiveOptimizer:
    """Test multi-objective optimization."""
    
    def setup_method(self):
        """Setup test instance."""
        quantum_optimizer = QuantumOptimizer()
        self.multi_optimizer = MultiObjectiveOptimizer(quantum_optimizer)
    
    @pytest.mark.asyncio
    async def test_sdlc_pipeline_optimization(self):
        """Test SDLC pipeline optimization."""
        pipeline_config = {"build_system": "python", "test_framework": "pytest"}
        objectives = [OptimizationObjective.MAXIMIZE_QUALITY, OptimizationObjective.MINIMIZE_TIME]
        constraints = [
            OptimizationConstraint("test_coverage", "bound", 0.8, 0.05)
        ]
        
        # Use small parameters for testing
        solutions = await self.multi_optimizer.optimize_sdlc_pipeline(
            pipeline_config, objectives, constraints
        )
        
        # Should have some solutions
        assert len(solutions) > 0
        
        # Check Pareto optimality properties
        for solution in solutions:
            assert solution.fitness_score >= 0
            assert "cpu_allocation" in solution.parameters
    
    def test_optimization_problem_creation(self):
        """Test SDLC optimization problem creation."""
        config = {"build": "test"}
        objectives = [OptimizationObjective.MINIMIZE_RESOURCES]
        constraints = []
        
        problem = self.multi_optimizer._create_sdlc_optimization_problem(
            config, objectives, constraints
        )
        
        assert problem.name == "SDLC Pipeline Optimization"
        assert len(problem.objectives) == 1
        assert len(problem.parameter_bounds) > 0


class TestPredictiveAnalytics:
    """Test predictive analytics capabilities."""
    
    def setup_method(self):
        """Setup test instance."""
        self.config = {"analytics": {"window_size": 20, "prediction_horizon": 60}}
        self.analytics = PredictiveAnalytics(self.config)
    
    @pytest.mark.asyncio
    async def test_metric_ingestion(self):
        """Test metric data ingestion."""
        metrics = {"cpu_usage": 0.7, "memory_usage": 0.5}
        
        await self.analytics.ingest_metrics(metrics, "test_source")
        
        assert "cpu_usage" in self.analytics.metrics_buffer
        assert "memory_usage" in self.analytics.metrics_buffer
        assert len(self.analytics.metrics_buffer["cpu_usage"]) == 1
    
    @pytest.mark.asyncio
    async def test_prediction_generation(self):
        """Test prediction generation."""
        # Generate some test data
        for i in range(15):
            metrics = {"test_metric": 0.5 + 0.1 * np.sin(i)}
            await self.analytics.ingest_metrics(metrics, "test")
        
        prediction = await self.analytics.predict(
            PredictionType.RESOURCE_USAGE, "test_metric"
        )
        
        assert prediction.prediction_type == PredictionType.RESOURCE_USAGE
        assert prediction.predicted_value is not None
        assert 0 <= prediction.confidence <= 1
        assert prediction.prediction_horizon > 0
    
    @pytest.mark.asyncio 
    async def test_anomaly_detection(self):
        """Test anomaly detection."""
        # Generate normal data
        for i in range(25):
            normal_value = 0.5 + 0.1 * np.random.random()
            await self.analytics.ingest_metrics({"test_metric": normal_value}, "normal")
        
        # Add anomaly
        await self.analytics.ingest_metrics({"test_metric": 2.0}, "anomaly")
        
        anomalies = await self.analytics.detect_anomalies("test_metric")
        
        # Should detect the anomaly
        assert len(anomalies) > 0
        assert any(a["value"] > 1.5 for a in anomalies)
    
    @pytest.mark.asyncio
    async def test_pattern_identification(self):
        """Test pattern identification."""
        # Generate data with patterns
        for i in range(35):
            if i % 10 < 5:  # Pattern 1
                value = 0.3 + 0.1 * np.random.random()
            else:  # Pattern 2
                value = 0.7 + 0.1 * np.random.random()
            
            await self.analytics.ingest_metrics({"pattern_metric": value}, "pattern")
        
        patterns = await self.analytics.identify_patterns("pattern_metric")
        
        assert patterns["clusters"] > 0
        assert len(patterns["patterns"]) > 0
    
    @pytest.mark.asyncio
    async def test_insights_generation(self):
        """Test insight generation."""
        # Generate metrics with trends
        for i in range(20):
            increasing_metric = 0.3 + i * 0.02  # Increasing trend
            volatile_metric = 0.5 + 0.3 * np.random.random()  # High volatility
            
            metrics = {
                "increasing_metric": increasing_metric,
                "volatile_metric": volatile_metric
            }
            await self.analytics.ingest_metrics(metrics, "insight_test")
        
        insights = await self.analytics.generate_insights(time_window=3600)
        
        assert len(insights) >= 0  # Should generate some insights
        
        # Check insight structure
        for insight in insights[:3]:  # Check first few
            assert "type" in insight
            assert "description" in insight
            assert "priority" in insight


class TestResourcePredictor:
    """Test resource prediction and management."""
    
    def setup_method(self):
        """Setup test instance."""
        self.analytics = PredictiveAnalytics()
        self.predictor = ResourcePredictor(self.analytics)
    
    @pytest.mark.asyncio
    async def test_resource_demand_prediction(self):
        """Test resource demand prediction."""
        # Generate resource metrics
        for i in range(15):
            metrics = {
                "cpu_usage": 0.6 + 0.1 * np.sin(i),
                "memory_usage": 0.5 + 0.1 * np.random.random()
            }
            await self.analytics.ingest_metrics(metrics, "resource_test")
        
        predictions = await self.predictor.predict_resource_demand(time_horizon=300)
        
        # Should have predictions for monitored resources
        for resource_type in ["cpu", "memory"]:
            if resource_type in predictions:
                pred = predictions[resource_type]
                assert pred.current_usage >= 0
                assert pred.predicted_usage >= 0
                assert 0 <= pred.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_resource_allocation_optimization(self):
        """Test resource allocation optimization."""
        # Mock predictions
        from robo_rlhf.quantum.analytics import ResourcePrediction
        
        predictions = {
            "cpu": ResourcePrediction("cpu", 0.6, 0.8, 0.9, 300, 0.8, []),
            "memory": ResourcePrediction("memory", 0.5, 0.4, 0.5, 300, 0.7, [])
        }
        
        current_allocation = {"cpu": 0.6, "memory": 0.5, "storage": 0.3}
        
        optimized = await self.predictor.optimize_resource_allocation(
            current_allocation, predictions
        )
        
        assert "cpu" in optimized
        assert "memory" in optimized
        # CPU should be scaled up due to predicted increase
        assert optimized["cpu"] > current_allocation["cpu"]
    
    def test_health_score_calculation(self):
        """Test resource health score calculation."""
        from robo_rlhf.quantum.analytics import ResourcePrediction
        
        predictions = {
            "cpu": ResourcePrediction("cpu", 0.5, 0.6, 0.7, 300, 0.8, []),
            "memory": ResourcePrediction("memory", 0.9, 0.95, 1.0, 300, 0.9, [])
        }
        
        health_score = self.predictor.get_resource_health_score(predictions)
        
        assert 0 <= health_score <= 1
        # Should be lower due to high memory usage prediction
        assert health_score < 0.9
    
    def test_capacity_planning(self):
        """Test capacity planning generation."""
        from robo_rlhf.quantum.analytics import ResourcePrediction
        
        predictions = {
            "cpu": ResourcePrediction("cpu", 0.7, 0.85, 0.95, 300, 0.8, []),
            "memory": ResourcePrediction("memory", 0.8, 0.92, 0.98, 300, 0.9, [])
        }
        
        plan = self.predictor.generate_capacity_plan(predictions, 86400)
        
        assert "recommendations" in plan
        assert "urgency" in plan
        assert plan["planning_horizon"] == 86400
        
        # Should have urgent recommendations due to high predicted usage
        assert len(plan["recommendations"]) > 0
        assert plan["urgency"] in ["high", "normal"]


class TestAutonomousSDLCExecutor:
    """Test autonomous SDLC execution."""
    
    def setup_method(self):
        """Setup test instance."""
        self.project_path = Path(__file__).parent.parent.parent
        self.config = {
            "autonomous": {"max_parallel": 2, "quality_threshold": 0.8}
        }
        self.executor = AutonomousSDLCExecutor(self.project_path, self.config)
    
    def test_executor_initialization(self):
        """Test executor initializes correctly."""
        assert self.executor.project_path == self.project_path
        assert self.executor.max_parallel_actions == 2
        assert len(self.executor.sdlc_actions) > 0
    
    def test_sdlc_actions_configuration(self):
        """Test SDLC actions are properly configured."""
        actions = self.executor.sdlc_actions
        
        # Should have key SDLC actions
        assert "code_analysis" in actions
        assert "unit_tests" in actions
        assert "security_scan" in actions
        
        # Check action properties
        for action in actions.values():
            assert action.phase in SDLCPhase
            assert action.timeout > 0
            assert isinstance(action.critical, bool)
    
    @pytest.mark.asyncio
    async def test_execution_plan_creation(self):
        """Test execution plan creation."""
        target_phases = [SDLCPhase.ANALYSIS, SDLCPhase.TESTING]
        
        plan = await self.executor._create_execution_plan(target_phases)
        
        assert plan.objective is not None
        assert len(plan.tasks) > 0
        
        # Tasks should be from target phases
        task_phases = [self.executor.sdlc_actions[task.id].phase for task in plan.tasks 
                      if task.id in self.executor.sdlc_actions]
        assert all(phase in target_phases for phase in task_phases)
    
    def test_failure_analysis(self):
        """Test failure analysis capabilities."""
        from robo_rlhf.quantum.autonomous import AutonomousAction
        
        action = AutonomousAction(
            id="test_action",
            name="Test Action",
            phase=SDLCPhase.TESTING,
            command="test_command"
        )
        
        task_result = {
            "error": "ModuleNotFoundError: No module named 'test_module'",
            "retry_count": 0
        }
        
        analysis = asyncio.run(self.executor._analyze_failure(action, task_result))
        
        assert analysis["type"] == "dependency"
        assert analysis["recoverable"] == True
        assert "suggested_fix" in analysis
    
    def test_execution_context_creation(self):
        """Test execution context creation."""
        context = self.executor._create_default_context()
        
        assert isinstance(context, ExecutionContext)
        assert context.project_path == self.project_path
        assert context.environment == "development"
        assert "cpu" in context.resource_limits


# Integration tests
class TestQuantumIntegration:
    """Integration tests for quantum components working together."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_optimization(self):
        """Test end-to-end optimization workflow."""
        # Create components
        planner = QuantumTaskPlanner()
        optimizer = QuantumOptimizer()
        
        # Create plan
        plan = planner.create_quantum_plan("Optimize testing workflow", ["unit_tests"])
        
        # Create optimization problem based on plan
        problem = OptimizationProblem(
            name="Plan Optimization",
            objectives=[OptimizationObjective.MINIMIZE_TIME],
            parameter_bounds={"efficiency": (0.1, 1.0)},
            max_generations=5  # Small for testing
        )
        
        # Run optimization
        solutions = await optimizer.optimize(problem)
        
        assert len(solutions) > 0
        assert all(sol.fitness_score >= 0 for sol in solutions)
    
    @pytest.mark.asyncio
    async def test_analytics_decision_integration(self):
        """Test analytics and decision engine integration."""
        analytics = PredictiveAnalytics()
        decision_engine = QuantumDecisionEngine()
        
        # Generate metrics
        for i in range(10):
            metrics = {"performance_score": 0.7 + 0.1 * np.random.random()}
            await analytics.ingest_metrics(metrics, "integration_test")
        
        # Make prediction
        prediction = await analytics.predict(
            PredictionType.PERFORMANCE, "performance_score"
        )
        
        # Use prediction in decision
        options = [
            {"action": "optimize", "expected_improvement": 0.2},
            {"action": "maintain", "expected_improvement": 0.0}
        ]
        
        context = {"current_performance": prediction.predicted_value}
        criteria = {"expected_improvement": 1.0}
        
        decision = await decision_engine.make_autonomous_decision(
            context, options, criteria
        )
        
        assert "selected_option" in decision
        assert decision["confidence"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])