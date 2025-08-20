"""
Comprehensive test suite for the new autonomous SDLC generation features.

Tests all three generations:
- Generation 1: RLHF Optimization, Dashboard, State Persistence
- Generation 2: Error Handling, Monitoring
- Generation 3: Performance Optimization
"""

import pytest
import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

# Generation 1 imports
from robo_rlhf.quantum.rlhf_optimizer import (
    RLHFQuantumOptimizer, RLHFPhase, RLHFOptimizationResult, TrainingPrediction
)
from robo_rlhf.web.dashboard import AutonomousSDLCDashboard, DashboardServer
from robo_rlhf.core.state_manager import (
    PersistentStateManager, StateType, StateCheckpoint, RecoveryInfo
)

# Generation 2 imports
from robo_rlhf.core.error_handling import (
    QuantumErrorRecovery, ErrorContext, RecoveryResult, ErrorSeverity, RecoveryStrategy
)
from robo_rlhf.core.monitoring import (
    QuantumMonitoringEngine, Metric, Alert, HealthCheck, MetricType, AlertSeverity
)

# Generation 3 imports
from robo_rlhf.core.performance_optimizer import (
    QuantumPerformanceOptimizer, OptimizationTarget, ScalingStrategy, 
    PerformanceMetrics, OptimizationResult, ScalingDecision
)


class TestGeneration1RLHFOptimizer:
    """Test Generation 1: RLHF-Specific Quantum Optimization."""

    @pytest.fixture
    async def rlhf_optimizer(self):
        """Create RLHF optimizer instance."""
        optimizer = RLHFQuantumOptimizer()
        yield optimizer

    @pytest.mark.asyncio
    async def test_preference_collection_optimization(self, rlhf_optimizer):
        """Test preference collection optimization."""
        current_strategy = {
            "sampling_strategy": "random",
            "pair_generation": "random",
            "pairs_per_episode": 50
        }
        
        training_data_stats = {
            "num_samples": 1000,
            "quality_score": 0.85,
            "diversity_score": 0.7
        }
        
        result = await rlhf_optimizer.optimize_preference_collection(
            current_strategy, training_data_stats
        )
        
        assert isinstance(result, RLHFOptimizationResult)
        assert result.phase == RLHFPhase.PREFERENCE_COLLECTION
        assert result.expected_improvement >= 0
        assert result.confidence > 0
        assert len(result.optimized_parameters) > 0
        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_reward_model_training_optimization(self, rlhf_optimizer):
        """Test reward model training optimization."""
        training_config = {
            "learning_rate": 1e-3,
            "batch_size": 32,
            "hidden_dim": 256
        }
        
        preference_data_stats = {
            "num_preferences": 500,
            "agreement_rate": 0.8
        }
        
        result = await rlhf_optimizer.tune_reward_model_training(
            "transformer", training_config, preference_data_stats
        )
        
        assert isinstance(result, RLHFOptimizationResult)
        assert result.phase == RLHFPhase.REWARD_MODEL_TRAINING
        assert "learning_rate" in result.optimized_parameters
        assert result.expected_improvement >= 0

    @pytest.mark.asyncio
    async def test_policy_optimization(self, rlhf_optimizer):
        """Test policy gradient optimization."""
        policy_config = {
            "algorithm": "ppo",
            "learning_rate": 3e-4,
            "clip_range": 0.2
        }
        
        environment_stats = {"episode_length": 100, "success_rate": 0.7}
        reward_model_performance = {"accuracy": 0.9, "loss": 0.1}
        
        result = await rlhf_optimizer.optimize_policy_gradient_updates(
            policy_config, environment_stats, reward_model_performance
        )
        
        assert isinstance(result, RLHFOptimizationResult)
        assert result.phase == RLHFPhase.POLICY_OPTIMIZATION
        assert len(result.optimized_parameters) > 0

    @pytest.mark.asyncio
    async def test_training_convergence_prediction(self, rlhf_optimizer):
        """Test training convergence prediction."""
        training_config = {"optimizer": "adam", "learning_rate": 1e-3}
        historical_data = [
            {"loss": 0.5, "accuracy": 0.8, "epoch": i}
            for i in range(10)
        ]
        
        prediction = await rlhf_optimizer.predict_training_convergence(
            training_config, historical_data
        )
        
        assert isinstance(prediction, TrainingPrediction)
        assert 0 <= prediction.convergence_probability <= 1
        assert prediction.estimated_epochs > 0
        assert isinstance(prediction.expected_performance, dict)
        assert isinstance(prediction.resource_requirements, dict)
        assert isinstance(prediction.confidence_interval, tuple)

    @pytest.mark.asyncio
    async def test_optimization_state_persistence(self, rlhf_optimizer):
        """Test optimization state saving and loading."""
        # Perform some optimizations first
        current_strategy = {"sampling_strategy": "diversity"}
        training_data_stats = {"num_samples": 500, "quality_score": 0.8}
        
        await rlhf_optimizer.optimize_preference_collection(
            current_strategy, training_data_stats
        )
        
        # Save state
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        await rlhf_optimizer.save_optimization_state(temp_path)
        
        # Verify file exists and contains data
        assert Path(temp_path).exists()
        
        with open(temp_path) as f:
            saved_data = json.load(f)
        
        assert "optimization_history" in saved_data
        assert "summary" in saved_data
        assert len(saved_data["optimization_history"]) > 0
        
        # Test loading
        new_optimizer = RLHFQuantumOptimizer()
        await new_optimizer.load_optimization_state(temp_path)
        
        assert len(new_optimizer.optimization_history) > 0
        
        # Cleanup
        Path(temp_path).unlink()

    def test_optimization_summary(self, rlhf_optimizer):
        """Test optimization summary generation."""
        summary = rlhf_optimizer.get_optimization_summary()
        
        assert isinstance(summary, dict)
        if summary.get("total_optimizations", 0) == 0:
            assert "message" in summary
        else:
            assert "average_improvement" in summary
            assert "phase_breakdown" in summary


class TestGeneration1Dashboard:
    """Test Generation 1: Unified Execution Dashboard."""

    @pytest.fixture
    def dashboard(self):
        """Create dashboard instance."""
        config = {
            "host": "localhost",
            "port": 8081,  # Different port for testing
            "update_interval": 1.0
        }
        return AutonomousSDLCDashboard(config)

    def test_dashboard_initialization(self, dashboard):
        """Test dashboard initialization."""
        assert dashboard.config["host"] == "localhost"
        assert dashboard.config["port"] == 8081
        assert dashboard.execution_state["current_phase"] == "idle"
        assert dashboard.metrics["system_health"] == 100.0

    @pytest.mark.asyncio
    async def test_dashboard_html_generation(self, dashboard):
        """Test dashboard HTML generation."""
        html = await dashboard.generate_dashboard_html()
        
        assert isinstance(html, str)
        assert "Autonomous SDLC Dashboard" in html
        assert "<!DOCTYPE html>" in html
        assert "vue.global.js" in html

    @pytest.mark.asyncio
    async def test_execution_state_management(self, dashboard):
        """Test execution state updates."""
        # Test state update
        initial_phase = dashboard.execution_state["current_phase"]
        
        # Simulate execution start
        dashboard.execution_state.update({
            "current_phase": "testing",
            "progress": 25.0,
            "active_tasks": [{"id": "test1", "name": "Running tests"}]
        })
        
        assert dashboard.execution_state["current_phase"] == "testing"
        assert dashboard.execution_state["progress"] == 25.0
        assert len(dashboard.execution_state["active_tasks"]) == 1

    @pytest.mark.asyncio
    async def test_metrics_update(self, dashboard):
        """Test metrics update functionality."""
        await dashboard._update_metrics()
        
        assert isinstance(dashboard.metrics["cpu_usage"], (int, float))
        assert isinstance(dashboard.metrics["memory_usage"], (int, float))
        assert isinstance(dashboard.metrics["gpu_usage"], (int, float))
        assert 0 <= dashboard.metrics["system_health"] <= 100

    @pytest.mark.asyncio
    async def test_websocket_message_handling(self, dashboard):
        """Test WebSocket message handling."""
        mock_websocket = AsyncMock()
        
        # Test ping message
        await dashboard._handle_websocket_message(mock_websocket, {"type": "ping"})
        mock_websocket.send_json.assert_called_with({"type": "pong"})

    def test_dashboard_server_creation(self):
        """Test dashboard server creation."""
        from robo_rlhf.web.dashboard import create_dashboard_server
        
        server = create_dashboard_server()
        assert isinstance(server, DashboardServer)
        assert hasattr(server, 'dashboard')


class TestGeneration1StateManager:
    """Test Generation 1: Enhanced State Persistence."""

    @pytest.fixture
    async def state_manager(self):
        """Create state manager with temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db_path = f.name
        
        manager = PersistentStateManager(db_path=temp_db_path)
        yield manager
        
        # Cleanup
        Path(temp_db_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_checkpoint_creation_and_restoration(self, state_manager):
        """Test checkpoint creation and restoration."""
        # Create checkpoint
        test_data = {
            "model_state": {"epoch": 10, "loss": 0.1},
            "optimizer_state": {"lr": 1e-3},
            "metrics": {"accuracy": 0.95}
        }
        
        checkpoint_id = await state_manager.save_execution_checkpoint(
            StateType.TRAINING_STATE,
            test_data,
            metadata={"description": "test checkpoint"}
        )
        
        assert isinstance(checkpoint_id, str)
        assert len(checkpoint_id) > 0
        
        # Restore checkpoint
        restored_data = await state_manager.restore_from_checkpoint(checkpoint_id)
        
        assert restored_data == test_data
        assert restored_data["model_state"]["epoch"] == 10
        assert restored_data["metrics"]["accuracy"] == 0.95

    @pytest.mark.asyncio
    async def test_state_transactions(self, state_manager):
        """Test atomic state transactions."""
        initial_state = {"counter": 0, "status": "idle"}
        await state_manager.update_state(StateType.EXECUTION_STATE, initial_state)
        
        # Successful transaction
        async with state_manager.state_transaction(StateType.EXECUTION_STATE) as state:
            state["counter"] = 5
            state["status"] = "active"
        
        final_state = await state_manager.get_state(StateType.EXECUTION_STATE)
        assert final_state["counter"] == 5
        assert final_state["status"] == "active"
        
        # Failed transaction (should rollback)
        try:
            async with state_manager.state_transaction(StateType.EXECUTION_STATE) as state:
                state["counter"] = 10
                raise ValueError("Transaction failed")
        except ValueError:
            pass
        
        # State should be rolled back
        current_state = await state_manager.get_state(StateType.EXECUTION_STATE)
        assert current_state["counter"] == 5  # Should not be 10

    @pytest.mark.asyncio
    async def test_checkpoint_listing_and_info(self, state_manager):
        """Test checkpoint listing and info retrieval."""
        # Create multiple checkpoints
        for i in range(3):
            await state_manager.save_execution_checkpoint(
                StateType.EXECUTION_STATE,
                {"iteration": i, "data": f"test_{i}"},
                metadata={"iteration": i}
            )
        
        # List checkpoints
        checkpoints = await state_manager.list_checkpoints(
            state_type=StateType.EXECUTION_STATE,
            limit=10
        )
        
        assert len(checkpoints) == 3
        assert all("id" in cp for cp in checkpoints)
        assert all("timestamp" in cp for cp in checkpoints)
        
        # Get specific checkpoint info
        checkpoint_info = await state_manager.get_checkpoint_info(checkpoints[0]["id"])
        assert checkpoint_info is not None
        assert "metadata" in checkpoint_info

    @pytest.mark.asyncio
    async def test_state_statistics(self, state_manager):
        """Test state statistics generation."""
        # Create some checkpoints
        await state_manager.save_execution_checkpoint(
            StateType.TRAINING_STATE,
            {"test": "data"}
        )
        
        stats = await state_manager.get_state_statistics()
        
        assert isinstance(stats, dict)
        assert "total_checkpoints" in stats
        assert "checkpoints_by_type" in stats
        assert stats["total_checkpoints"] >= 1

    @pytest.mark.asyncio
    async def test_checkpoint_cleanup(self, state_manager):
        """Test old checkpoint cleanup."""
        # Create multiple checkpoints
        for i in range(10):
            await state_manager.save_execution_checkpoint(
                StateType.EXECUTION_STATE,
                {"iteration": i}
            )
        
        # Cleanup keeping only 5
        deleted_count = await state_manager.cleanup_old_checkpoints(keep_count=5)
        
        assert deleted_count == 5
        
        # Verify only 5 remain
        remaining = await state_manager.list_checkpoints(limit=20)
        assert len(remaining) == 5


class TestGeneration2ErrorHandling:
    """Test Generation 2: Comprehensive Error Handling."""

    @pytest.fixture
    async def error_recovery(self):
        """Create error recovery system."""
        state_manager = PersistentStateManager()
        return QuantumErrorRecovery(state_manager)

    @pytest.mark.asyncio
    async def test_error_context_creation(self, error_recovery):
        """Test error context creation and analysis."""
        test_error = ValueError("Test error message")
        context = {
            "args": [1, 2, 3],
            "kwargs": {"param": "value"}
        }
        
        result = await error_recovery.handle_error(
            error=test_error,
            context=context,
            component="test_component",
            function_name="test_function"
        )
        
        assert isinstance(result, RecoveryResult)
        assert result.strategy_used in RecoveryStrategy
        assert len(result.recovery_actions) > 0
        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_recovery_strategies(self, error_recovery):
        """Test different recovery strategies."""
        error_context = ErrorContext(
            error_id="test_error",
            timestamp=datetime.now(),
            error_type="ValueError",
            error_message="Test error",
            severity=ErrorSeverity.MEDIUM,
            component="test",
            function_name="test_func",
            args={},
            kwargs={},
            stack_trace="",
            system_state={}
        )
        
        # Test retry recovery
        success, actions = await error_recovery._execute_retry_recovery(error_context)
        assert isinstance(success, bool)
        assert isinstance(actions, list)
        assert len(actions) > 0
        
        # Test rollback recovery
        success, actions = await error_recovery._execute_rollback_recovery(error_context)
        assert isinstance(success, bool)
        assert isinstance(actions, list)

    @pytest.mark.asyncio
    async def test_quantum_recovery_process(self, error_recovery):
        """Test quantum-inspired recovery process."""
        error_context = ErrorContext(
            error_id="quantum_test",
            timestamp=datetime.now(),
            error_type="ConnectionError",
            error_message="Connection failed",
            severity=ErrorSeverity.HIGH,
            component="network",
            function_name="connect",
            args={},
            kwargs={},
            stack_trace="",
            system_state={}
        )
        
        result = await error_recovery._quantum_recovery_process(error_context)
        
        assert isinstance(result, RecoveryResult)
        assert result.strategy_used in RecoveryStrategy
        assert len(result.recovery_actions) > 0

    @pytest.mark.asyncio
    async def test_error_pattern_analysis(self, error_recovery):
        """Test error pattern analysis and statistics."""
        # Generate some test errors
        for i in range(5):
            test_error = ValueError(f"Test error {i}")
            await error_recovery.handle_error(
                error=test_error,
                context={"iteration": i},
                component="test_component"
            )
        
        # Get error statistics
        stats = error_recovery.get_error_statistics()
        
        assert isinstance(stats, dict)
        assert stats["total_errors"] == 5
        assert "error_types" in stats
        assert "ValueError" in stats["error_types"]
        assert stats["error_types"]["ValueError"] == 5

    def test_error_handler_decorator(self):
        """Test error handler decorator functionality."""
        from robo_rlhf.core.error_handling import error_handler
        
        @error_handler(component="test", max_retries=2)
        async def failing_function():
            raise ValueError("Function failed")
        
        # Test that decorator is applied
        assert hasattr(failing_function, '__wrapped__')
        assert asyncio.iscoroutinefunction(failing_function)


class TestGeneration2Monitoring:
    """Test Generation 2: Advanced Monitoring System."""

    @pytest.fixture
    async def monitoring_engine(self):
        """Create monitoring engine."""
        config = {"collection_interval": 1, "alert_cooldown": 1}
        engine = QuantumMonitoringEngine(config)
        yield engine
        
        if engine.is_monitoring:
            await engine.stop_monitoring()

    @pytest.mark.asyncio
    async def test_monitoring_startup_shutdown(self, monitoring_engine):
        """Test monitoring system startup and shutdown."""
        assert not monitoring_engine.is_monitoring
        
        # Start monitoring
        await monitoring_engine.start_monitoring()
        assert monitoring_engine.is_monitoring
        assert len(monitoring_engine.monitoring_tasks) > 0
        
        # Stop monitoring
        await monitoring_engine.stop_monitoring()
        assert not monitoring_engine.is_monitoring
        assert len(monitoring_engine.monitoring_tasks) == 0

    @pytest.mark.asyncio
    async def test_metric_recording(self, monitoring_engine):
        """Test custom metric recording."""
        await monitoring_engine.record_metric(
            name="test.metric",
            value=42.5,
            metric_type=MetricType.GAUGE,
            labels={"component": "test"}
        )
        
        assert len(monitoring_engine.metrics_buffer) == 1
        metric = monitoring_engine.metrics_buffer[0]
        assert metric.name == "test.metric"
        assert metric.value == 42.5
        assert metric.metric_type == MetricType.GAUGE

    @pytest.mark.asyncio
    async def test_alert_creation_and_resolution(self, monitoring_engine):
        """Test alert creation and resolution."""
        # Create alert
        alert_id = await monitoring_engine.create_alert(
            title="Test Alert",
            description="This is a test alert",
            severity=AlertSeverity.WARNING,
            source="test_source"
        )
        
        assert isinstance(alert_id, str)
        assert len(monitoring_engine.alerts) == 1
        
        alert = monitoring_engine.alerts[0]
        assert alert.title == "Test Alert"
        assert alert.severity == AlertSeverity.WARNING
        assert not alert.resolved
        
        # Resolve alert
        success = await monitoring_engine.resolve_alert(alert_id, "Test resolution")
        assert success
        assert alert.resolved
        assert alert.resolution_time is not None

    @pytest.mark.asyncio
    async def test_health_check_registration(self, monitoring_engine):
        """Test health check registration and execution."""
        call_count = 0
        
        def test_health_check():
            nonlocal call_count
            call_count += 1
            return "healthy"
        
        await monitoring_engine.register_health_check(
            component="test_component",
            check_function=test_health_check,
            timeout=1.0
        )
        
        # Manually execute health check
        check_func = monitoring_engine._health_check_functions["test_component"]
        await check_func()
        
        assert "test_component" in monitoring_engine.health_checks
        health_check = monitoring_engine.health_checks["test_component"]
        assert health_check.status == "healthy"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_metrics_summary(self, monitoring_engine):
        """Test metrics summary generation."""
        # Record several metrics
        for i in range(10):
            await monitoring_engine.record_metric(
                f"test.metric_{i % 3}",
                float(i),
                MetricType.GAUGE
            )
        
        summary = await monitoring_engine.get_metrics_summary()
        
        assert isinstance(summary, dict)
        assert len(summary) == 3  # 3 unique metric names
        
        for metric_name, stats in summary.items():
            assert "count" in stats
            assert "average" in stats
            assert "min" in stats
            assert "max" in stats

    @pytest.mark.asyncio
    async def test_system_health_assessment(self, monitoring_engine):
        """Test system health assessment."""
        # Register a health check
        def healthy_component():
            return "OK"
        
        await monitoring_engine.register_health_check(
            "healthy_component",
            healthy_component
        )
        
        # Execute health check
        check_func = monitoring_engine._health_check_functions["healthy_component"]
        await check_func()
        
        # Get system health
        health = await monitoring_engine.get_system_health()
        
        assert isinstance(health, dict)
        assert "overall_health" in health
        assert "component_count" in health
        assert "health_distribution" in health
        assert health["overall_health"] in ["healthy", "degraded", "unhealthy", "unknown"]

    @pytest.mark.asyncio
    async def test_performance_insights(self, monitoring_engine):
        """Test performance insights generation."""
        # Record some metrics first
        for i in range(20):
            await monitoring_engine.record_metric("cpu.usage", 50 + i, MetricType.GAUGE)
            await monitoring_engine.record_metric("memory.usage", 60 + i, MetricType.GAUGE)
        
        insights = await monitoring_engine.get_performance_insights()
        
        assert isinstance(insights, dict)
        assert "trends" in insights
        assert "anomalies" in insights
        assert "predictions" in insights
        assert "correlations" in insights

    def test_monitoring_decorators(self):
        """Test monitoring decorators."""
        from robo_rlhf.core.monitoring import monitor_execution
        
        @monitor_execution("test_app")
        async def test_function():
            return "success"
        
        assert hasattr(test_function, '__wrapped__')
        assert asyncio.iscoroutinefunction(test_function)


class TestGeneration3PerformanceOptimizer:
    """Test Generation 3: Performance Optimization."""

    @pytest.fixture
    async def performance_optimizer(self):
        """Create performance optimizer."""
        config = {"optimization_interval": 1, "max_workers": 4}
        optimizer = QuantumPerformanceOptimizer(config=config)
        yield optimizer
        
        if optimizer.is_optimizing:
            await optimizer.stop_optimization()

    @pytest.mark.asyncio
    async def test_optimization_startup_shutdown(self, performance_optimizer):
        """Test performance optimization startup and shutdown."""
        assert not performance_optimizer.is_optimizing
        
        # Start optimization
        await performance_optimizer.start_optimization()
        assert performance_optimizer.is_optimizing
        assert performance_optimizer.thread_pool is not None
        assert performance_optimizer.process_pool is not None
        
        # Stop optimization
        await performance_optimizer.stop_optimization()
        assert not performance_optimizer.is_optimizing

    @pytest.mark.asyncio
    async def test_performance_metrics_capture(self, performance_optimizer):
        """Test performance metrics capture."""
        metrics = await performance_optimizer._capture_performance_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert isinstance(metrics.cpu_usage, (int, float))
        assert isinstance(metrics.memory_usage, (int, float))
        assert isinstance(metrics.latency_p95, (int, float))
        assert isinstance(metrics.throughput, (int, float))
        assert metrics.cpu_usage >= 0
        assert metrics.memory_usage >= 0

    @pytest.mark.asyncio
    async def test_optimization_for_different_targets(self, performance_optimizer):
        """Test optimization for different targets."""
        targets = [
            OptimizationTarget.LATENCY,
            OptimizationTarget.THROUGHPUT,
            OptimizationTarget.RESOURCE_EFFICIENCY,
            OptimizationTarget.BALANCED
        ]
        
        for target in targets:
            result = await performance_optimizer.optimize_for_target(target)
            
            assert isinstance(result, OptimizationResult)
            assert result.target == target
            assert result.execution_time > 0
            assert len(result.actions_taken) > 0
            assert isinstance(result.improvement_percentage, (int, float))

    @pytest.mark.asyncio
    async def test_auto_scaling_decision(self, performance_optimizer):
        """Test auto-scaling decision making."""
        # Create metrics that should trigger scaling
        high_cpu_metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=90.0,  # High CPU usage
            memory_usage=50.0,
            disk_io=0,
            network_io=0,
            latency_p95=50.0,
            throughput=100.0,
            error_rate=0.1,
            queue_length=10,
            active_connections=50,
            resource_efficiency=60.0
        )
        
        scaling_decision = await performance_optimizer.auto_scale(high_cpu_metrics)
        
        if scaling_decision:  # Scaling might not always be needed
            assert isinstance(scaling_decision, ScalingDecision)
            assert scaling_decision.strategy in ScalingStrategy
            assert scaling_decision.action in ["scale_up", "scale_down", "scale_out", "scale_in"]
            assert scaling_decision.magnitude > 0
            assert 0 <= scaling_decision.confidence <= 1

    @pytest.mark.asyncio
    async def test_resource_allocation_optimization(self, performance_optimizer):
        """Test resource allocation optimization."""
        workload_distribution = {
            "training": 0.4,
            "inference": 0.3,
            "data_processing": 0.2,
            "monitoring": 0.1
        }
        
        allocation = await performance_optimizer.optimize_resource_allocation(
            workload_distribution
        )
        
        assert isinstance(allocation, dict)
        assert len(allocation) == len(workload_distribution)
        
        # Check that allocation sums to approximately 100
        total_allocation = sum(allocation.values())
        assert 99 <= total_allocation <= 101  # Allow for small floating point errors

    @pytest.mark.asyncio
    async def test_performance_insights(self, performance_optimizer):
        """Test performance insights generation."""
        # Add some performance history
        for i in range(5):
            metrics = PerformanceMetrics(
                timestamp=datetime.now() - timedelta(minutes=i),
                cpu_usage=50 + i * 5,
                memory_usage=60 + i * 3,
                disk_io=100 + i * 10,
                network_io=200 + i * 20,
                latency_p95=50 + i * 2,
                throughput=100 - i * 5,
                error_rate=0.1 + i * 0.05,
                queue_length=10 + i * 2,
                active_connections=50 + i * 5,
                resource_efficiency=80 - i * 2
            )
            performance_optimizer.performance_history.append(metrics)
        
        insights = await performance_optimizer.get_performance_insights()
        
        assert isinstance(insights, dict)
        assert "current_performance" in insights
        assert "performance_trends" in insights
        assert "bottlenecks" in insights
        assert "optimization_opportunities" in insights
        assert "resource_utilization" in insights

    @pytest.mark.asyncio
    async def test_quantum_vs_classical_optimization(self, performance_optimizer):
        """Test quantum vs classical optimization approaches."""
        # Test quantum optimization
        performance_optimizer.quantum_optimization_enabled = True
        quantum_actions = await performance_optimizer._quantum_optimization_process(
            OptimizationTarget.LATENCY
        )
        
        # Test classical optimization
        performance_optimizer.quantum_optimization_enabled = False
        classical_actions = await performance_optimizer._classical_optimization_process(
            OptimizationTarget.LATENCY
        )
        
        assert isinstance(quantum_actions, list)
        assert isinstance(classical_actions, list)
        assert len(quantum_actions) > 0
        assert len(classical_actions) > 0
        
        # Both should produce optimization actions
        for action in quantum_actions + classical_actions:
            assert isinstance(action, dict)
            assert "type" in action

    def test_optimization_summary(self, performance_optimizer):
        """Test optimization summary generation."""
        summary = performance_optimizer.get_optimization_summary()
        
        assert isinstance(summary, dict)
        assert "total_optimizations" in summary
        assert "quantum_enabled" in summary

    @pytest.mark.asyncio
    async def test_scaling_strategy_evaluation(self, performance_optimizer):
        """Test scaling strategy evaluation."""
        metrics = await performance_optimizer._capture_performance_metrics()
        
        scaling_decision = await performance_optimizer._analyze_scaling_needs(metrics)
        
        # Scaling decision might be None if no scaling is needed
        if scaling_decision:
            assert isinstance(scaling_decision, ScalingDecision)
            assert hasattr(scaling_decision, 'strategy')
            assert hasattr(scaling_decision, 'action')
            assert hasattr(scaling_decision, 'magnitude')


class TestIntegrationAllGenerations:
    """Integration tests across all three generations."""

    @pytest.mark.asyncio
    async def test_full_autonomous_sdlc_pipeline(self):
        """Test complete autonomous SDLC pipeline integration."""
        # Initialize all components
        state_manager = PersistentStateManager()
        monitoring = QuantumMonitoringEngine()
        performance_optimizer = QuantumPerformanceOptimizer(monitoring_engine=monitoring)
        rlhf_optimizer = RLHFQuantumOptimizer()
        dashboard = AutonomousSDLCDashboard()
        
        try:
            # Start systems
            await monitoring.start_monitoring()
            await performance_optimizer.start_optimization()
            
            # Record some metrics
            await monitoring.record_metric("test.integration", 1.0, MetricType.COUNTER)
            
            # Create state checkpoint
            checkpoint_id = await state_manager.save_execution_checkpoint(
                StateType.EXECUTION_STATE,
                {"integration_test": True, "timestamp": datetime.now().isoformat()}
            )
            
            # Perform RLHF optimization
            result = await rlhf_optimizer.optimize_preference_collection(
                {"strategy": "test"}, {"samples": 100}
            )
            
            # Optimize performance
            perf_result = await performance_optimizer.optimize_for_target(
                OptimizationTarget.BALANCED
            )
            
            # Verify all components worked
            assert checkpoint_id is not None
            assert isinstance(result, RLHFOptimizationResult)
            assert isinstance(perf_result, OptimizationResult)
            assert len(monitoring.metrics_buffer) > 0
            
            # Test dashboard state
            dashboard_health = await dashboard._update_metrics()
            execution_state = dashboard.execution_state
            assert execution_state["current_phase"] == "idle"
            
        finally:
            # Cleanup
            await monitoring.stop_monitoring()
            await performance_optimizer.stop_optimization()

    @pytest.mark.asyncio
    async def test_error_recovery_with_state_rollback(self):
        """Test error recovery with state manager integration."""
        state_manager = PersistentStateManager()
        error_recovery = QuantumErrorRecovery(state_manager)
        
        # Create initial state
        initial_state = {"status": "working", "progress": 0.5}
        checkpoint_id = await state_manager.save_execution_checkpoint(
            StateType.EXECUTION_STATE,
            initial_state
        )
        
        # Simulate error that triggers rollback
        test_error = RuntimeError("Critical system failure")
        context = {"operation": "critical_task", "checkpoint_id": checkpoint_id}
        
        recovery_result = await error_recovery.handle_error(
            error=test_error,
            context=context,
            component="integration_test",
            function_name="test_critical_operation"
        )
        
        assert isinstance(recovery_result, RecoveryResult)
        assert recovery_result.strategy_used in RecoveryStrategy
        
        # Verify state can be restored
        restored_state = await state_manager.restore_from_checkpoint(checkpoint_id)
        assert restored_state == initial_state

    @pytest.mark.asyncio
    async def test_performance_monitoring_optimization_loop(self):
        """Test performance monitoring and optimization feedback loop."""
        monitoring = QuantumMonitoringEngine()
        performance_optimizer = QuantumPerformanceOptimizer(monitoring_engine=monitoring)
        
        try:
            await monitoring.start_monitoring()
            await performance_optimizer.start_optimization()
            
            # Record performance metrics that should trigger optimization
            for i in range(5):
                await monitoring.record_metric("cpu.usage", 85.0 + i, MetricType.GAUGE)
                await monitoring.record_metric("memory.usage", 80.0 + i, MetricType.GAUGE)
                await monitoring.record_metric("latency.p95", 150.0 + i * 10, MetricType.TIMER)
            
            # Trigger optimization based on metrics
            summary = await monitoring.get_metrics_summary()
            assert "cpu.usage" in summary
            
            # Verify optimization can read monitoring data
            performance_insights = await performance_optimizer.get_performance_insights()
            assert isinstance(performance_insights, dict)
            
            # Test optimization with current metrics
            result = await performance_optimizer.optimize_for_target(
                OptimizationTarget.RESOURCE_EFFICIENCY
            )
            
            assert isinstance(result, OptimizationResult)
            assert result.execution_time > 0
            
        finally:
            await monitoring.stop_monitoring()
            await performance_optimizer.stop_optimization()

    @pytest.mark.asyncio
    async def test_dashboard_real_time_updates(self):
        """Test dashboard real-time updates with all systems."""
        monitoring = QuantumMonitoringEngine()
        dashboard = AutonomousSDLCDashboard()
        
        # Connect dashboard to monitoring (in real implementation)
        dashboard.monitoring = monitoring
        
        try:
            await monitoring.start_monitoring()
            
            # Simulate execution progress
            dashboard.execution_state.update({
                "current_phase": "optimization",
                "progress": 75.0,
                "active_tasks": [
                    {"id": "opt1", "name": "RLHF optimization"},
                    {"id": "opt2", "name": "Performance tuning"}
                ]
            })
            
            # Update dashboard metrics
            await dashboard._update_metrics()
            
            # Verify dashboard state
            assert dashboard.execution_state["current_phase"] == "optimization"
            assert dashboard.execution_state["progress"] == 75.0
            assert len(dashboard.execution_state["active_tasks"]) == 2
            
            # Test dashboard API endpoints (mock)
            health = await dashboard.get_system_health()
            assert isinstance(health, dict)
            
        finally:
            await monitoring.stop_monitoring()


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks for all components."""

    @pytest.mark.asyncio
    async def test_rlhf_optimization_performance(self):
        """Benchmark RLHF optimization performance."""
        optimizer = RLHFQuantumOptimizer()
        
        start_time = time.time()
        
        result = await optimizer.optimize_preference_collection(
            {"strategy": "benchmark"}, {"samples": 1000}
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time
        assert execution_time < 10.0  # 10 seconds max
        assert isinstance(result, RLHFOptimizationResult)

    @pytest.mark.asyncio
    async def test_state_manager_throughput(self):
        """Benchmark state manager throughput."""
        state_manager = PersistentStateManager()
        
        start_time = time.time()
        checkpoint_ids = []
        
        # Create 100 checkpoints
        for i in range(100):
            checkpoint_id = await state_manager.save_execution_checkpoint(
                StateType.EXECUTION_STATE,
                {"iteration": i, "data": f"benchmark_{i}"}
            )
            checkpoint_ids.append(checkpoint_id)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle 100 checkpoints in reasonable time
        assert total_time < 30.0  # 30 seconds max
        assert len(checkpoint_ids) == 100
        
        # Test retrieval performance
        start_time = time.time()
        
        for checkpoint_id in checkpoint_ids[:10]:  # Test first 10
            data = await state_manager.restore_from_checkpoint(checkpoint_id)
            assert data is not None
        
        end_time = time.time()
        retrieval_time = end_time - start_time
        
        # Should retrieve 10 checkpoints quickly
        assert retrieval_time < 5.0  # 5 seconds max

    @pytest.mark.asyncio
    async def test_monitoring_metric_throughput(self):
        """Benchmark monitoring system metric throughput."""
        monitoring = QuantumMonitoringEngine()
        
        start_time = time.time()
        
        # Record 1000 metrics
        for i in range(1000):
            await monitoring.record_metric(
                f"benchmark.metric_{i % 10}",
                float(i),
                MetricType.GAUGE
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle 1000 metrics quickly
        assert total_time < 10.0  # 10 seconds max
        assert len(monitoring.metrics_buffer) == 1000

if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])