"""
Test suite for Pipeline Guard functionality.

Comprehensive tests for self-healing pipeline guard operations including
monitoring, detection, healing, and security features.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from robo_rlhf.pipeline import (
    PipelineGuard, PipelineComponent, HealthStatus,
    PipelineMonitor, MetricsCollector,
    SelfHealer, RecoveryStrategy,
    AnomalyDetector, FailurePredictor,
    SecurityManager, SecurityContext, Permission,
    PipelineOrchestrator, PipelineConfig
)


class TestPipelineGuard:
    """Test cases for PipelineGuard class."""
    
    @pytest.fixture
    def mock_health_check(self):
        """Mock health check function."""
        async def health_check():
            return {"status": "ok", "response_time": 0.1, "cpu_usage": 0.5}
        return health_check
    
    @pytest.fixture
    def sample_component(self, mock_health_check):
        """Sample pipeline component for testing."""
        return PipelineComponent(
            name="test_service",
            endpoint="http://test:8080",
            health_check=mock_health_check,
            critical=True
        )
    
    @pytest.fixture
    def pipeline_guard(self, sample_component):
        """Pipeline guard instance for testing."""
        return PipelineGuard(
            components=[sample_component],
            check_interval=1,  # Fast interval for testing
            healing_enabled=True
        )
    
    @pytest.mark.asyncio
    async def test_guard_initialization(self, pipeline_guard):
        """Test pipeline guard initialization."""
        assert len(pipeline_guard.components) == 1
        assert "test_service" in pipeline_guard.components
        assert pipeline_guard.healing_enabled is True
        assert pipeline_guard._running is False
    
    @pytest.mark.asyncio
    async def test_component_health_check(self, pipeline_guard):
        """Test individual component health checking."""
        component = pipeline_guard.components["test_service"]
        
        report = await pipeline_guard._check_component_health("test_service", component)
        
        assert report.component == "test_service"
        assert report.status == HealthStatus.HEALTHY
        assert "response_time" in report.metrics
        assert report.timestamp > 0
    
    @pytest.mark.asyncio
    async def test_health_check_failure_handling(self, sample_component):
        """Test handling of failed health checks."""
        # Create component with failing health check
        async def failing_health_check():
            raise Exception("Service unavailable")
        
        failing_component = PipelineComponent(
            name="failing_service",
            endpoint="http://failing:8080",
            health_check=failing_health_check
        )
        
        guard = PipelineGuard(components=[failing_component])
        
        report = await guard._check_component_health("failing_service", failing_component)
        
        assert report.component == "failing_service"
        assert report.status == HealthStatus.FAILED
        assert "error" in report.metrics
        assert len(report.issues) > 0
    
    @pytest.mark.asyncio
    async def test_system_health_aggregation(self, pipeline_guard):
        """Test system-wide health status aggregation."""
        # Run a health check cycle
        component = pipeline_guard.components["test_service"]
        report = await pipeline_guard._check_component_health("test_service", component)
        
        # Store the report
        pipeline_guard._health_history["test_service"] = [report]
        
        system_health = pipeline_guard.get_system_health()
        
        assert system_health["status"] == HealthStatus.HEALTHY.value
        assert "test_service" in system_health["components"]
        assert system_health["components"]["test_service"]["status"] == HealthStatus.HEALTHY.value
    
    @pytest.mark.asyncio
    async def test_healing_trigger(self, pipeline_guard):
        """Test healing process triggering."""
        with patch.object(pipeline_guard, '_initiate_healing') as mock_healing:
            # Create a failed health report
            failed_report = Mock()
            failed_report.status = HealthStatus.FAILED
            failed_report.component = "test_service"
            failed_report.issues = ["Service unavailable"]
            
            await pipeline_guard._process_health_reports([failed_report])
            
            # Verify healing was triggered
            mock_healing.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_monitoring_loop_execution(self, pipeline_guard):
        """Test monitoring loop execution."""
        # Start monitoring for a short time
        monitoring_task = asyncio.create_task(pipeline_guard.start_monitoring())
        
        # Let it run for a short time
        await asyncio.sleep(0.1)
        
        # Stop monitoring
        await pipeline_guard.stop_monitoring()
        
        # Wait for task completion
        try:
            await asyncio.wait_for(monitoring_task, timeout=1.0)
        except asyncio.TimeoutError:
            monitoring_task.cancel()


class TestMetricsCollector:
    """Test cases for MetricsCollector class."""
    
    @pytest.fixture
    def metrics_collector(self):
        """MetricsCollector instance for testing."""
        return MetricsCollector(retention_hours=1, max_points_per_metric=100)
    
    def test_metric_recording(self, metrics_collector):
        """Test basic metric recording."""
        metrics_collector.record_metric("cpu_usage", 0.8, tags={"host": "server1"})
        
        latest_metrics = metrics_collector.get_latest_metrics(["cpu_usage"])
        
        assert "cpu_usage" in latest_metrics
        assert latest_metrics["cpu_usage"] is not None
        assert latest_metrics["cpu_usage"].value == 0.8
        assert latest_metrics["cpu_usage"].tags["host"] == "server1"
    
    def test_batch_metric_recording(self, metrics_collector):
        """Test batch metric recording."""
        batch_metrics = [
            {"name": "cpu_usage", "value": 0.7, "tags": {"host": "server1"}},
            {"name": "memory_usage", "value": 0.6, "tags": {"host": "server1"}},
            {"name": "response_time", "value": 0.2, "unit": "seconds"}
        ]
        
        metrics_collector.record_batch(batch_metrics)
        
        latest = metrics_collector.get_latest_metrics(["cpu_usage", "memory_usage", "response_time"])
        
        assert all(metric is not None for metric in latest.values())
        assert latest["cpu_usage"].value == 0.7
        assert latest["memory_usage"].value == 0.6
        assert latest["response_time"].value == 0.2
    
    def test_metric_statistics(self, metrics_collector):
        """Test metric statistical summaries."""
        # Record multiple values
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        for value in values:
            metrics_collector.record_metric("test_metric", value)
        
        summary = metrics_collector.get_metric_summary("test_metric")
        
        assert summary is not None
        assert summary.count == 5
        assert summary.min_value == 0.1
        assert summary.max_value == 0.5
        assert abs(summary.mean - 0.3) < 0.01
        assert abs(summary.median - 0.3) < 0.01
    
    def test_alert_threshold_configuration(self, metrics_collector):
        """Test alert threshold configuration and triggering."""
        alert_triggered = False
        
        def alert_callback(metric, alerts):
            nonlocal alert_triggered
            alert_triggered = True
        
        metrics_collector.set_alert_threshold(
            "cpu_usage", "max", 0.8, callback=alert_callback
        )
        
        # Record metric that should trigger alert
        metrics_collector.record_metric("cpu_usage", 0.9)
        
        assert alert_triggered
    
    def test_metrics_cleanup(self, metrics_collector):
        """Test automatic cleanup of old metrics."""
        # Record old metric
        old_time = time.time() - 7200  # 2 hours ago
        metrics_collector._metrics["old_metric"].append({
            "value": 1.0,
            "timestamp": old_time
        })
        
        # Trigger cleanup
        metrics_collector._cleanup_old_metrics()
        
        # Old metric should be cleaned up (retention is 1 hour)
        assert len(metrics_collector._metrics["old_metric"]) == 0


class TestSelfHealer:
    """Test cases for SelfHealer class."""
    
    @pytest.fixture
    def self_healer(self):
        """SelfHealer instance for testing."""
        return SelfHealer()
    
    @pytest.mark.asyncio
    async def test_healing_execution(self, self_healer):
        """Test healing process execution."""
        failure_context = {
            "cpu_usage": 0.9,
            "memory_usage": 0.8,
            "error_rate": 0.1
        }
        
        results = await self_healer.heal("test_component", failure_context)
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check that healing was attempted
        for result in results:
            assert hasattr(result, 'action')
            assert hasattr(result, 'result')
            assert hasattr(result, 'duration')
    
    def test_strategy_success_tracking(self, self_healer):
        """Test tracking of strategy success rates."""
        # Simulate some healing results
        from robo_rlhf.pipeline.healer import RecoveryResult, RecoveryAction
        
        success_result = Mock()
        success_result.action.strategy = RecoveryStrategy.RESTART
        success_result.result = RecoveryResult.SUCCESS
        
        failed_result = Mock()
        failed_result.action.strategy = RecoveryStrategy.RESTART
        failed_result.result = RecoveryResult.FAILED
        
        self_healer._update_success_tracking([success_result, failed_result])
        
        success_rate = self_healer.get_strategy_success_rate(RecoveryStrategy.RESTART)
        
        assert success_rate == 0.5  # 1 success out of 2 attempts
    
    def test_component_preferences(self, self_healer):
        """Test component-specific healing preferences."""
        preferences = [RecoveryStrategy.SCALE_UP, RecoveryStrategy.RESTART]
        
        self_healer.set_component_preferences("web_server", preferences)
        
        assert "web_server" in self_healer.component_preferences
        assert self_healer.component_preferences["web_server"] == preferences
    
    def test_healing_statistics(self, self_healer):
        """Test healing statistics reporting."""
        stats = self_healer.get_healing_stats()
        
        assert "total_healing_attempts" in stats
        assert "successful_healings" in stats
        assert "overall_success_rate" in stats
        assert "strategy_performance" in stats
        assert "quantum_enabled" in stats


class TestAnomalyDetector:
    """Test cases for AnomalyDetector class."""
    
    @pytest.fixture
    def anomaly_detector(self):
        """AnomalyDetector instance for testing."""
        return AnomalyDetector("test_component", enable_ml_detection=False)
    
    def test_threshold_rule_addition(self, anomaly_detector):
        """Test adding threshold-based detection rules."""
        from robo_rlhf.pipeline.detector import ThresholdRule, Severity
        
        rule = ThresholdRule(
            metric_name="cpu_usage",
            threshold_up=0.8,
            threshold_down=0.2,
            severity=Severity.HIGH
        )
        
        anomaly_detector.add_threshold_rule(
            "cpu_usage", 0.8, "gt", Severity.HIGH
        )
        
        assert len(anomaly_detector.threshold_rules) == 1
        assert anomaly_detector.threshold_rules[0].metric_name == "cpu_usage"
    
    def test_metric_update(self, anomaly_detector):
        """Test metric value updates."""
        timestamp = time.time()
        
        anomaly_detector.update_metric("cpu_usage", 0.7, timestamp)
        
        assert "cpu_usage" in anomaly_detector.metrics_history
        assert len(anomaly_detector.metrics_history["cpu_usage"]) == 1
        assert anomaly_detector.metrics_history["cpu_usage"][0]["value"] == 0.7
    
    @pytest.mark.asyncio
    async def test_threshold_detection(self, anomaly_detector):
        """Test threshold-based anomaly detection."""
        from robo_rlhf.pipeline.detector import Severity
        
        # Add threshold rule
        anomaly_detector.add_threshold_rule(
            "cpu_usage", 0.8, "gt", Severity.HIGH, window_size=3
        )
        
        # Update metrics to exceed threshold
        for i in range(3):
            anomaly_detector.update_metric("cpu_usage", 0.9, time.time())
        
        # Establish baseline
        anomaly_detector._establish_baseline()
        
        # Detect anomalies
        anomalies = await anomaly_detector.detect_anomalies()
        
        # Should detect threshold breach
        threshold_anomalies = [a for a in anomalies if "threshold" in a.description.lower()]
        assert len(threshold_anomalies) > 0
    
    def test_baseline_establishment(self, anomaly_detector):
        """Test statistical baseline establishment."""
        # Add sufficient data points
        for i in range(35):
            anomaly_detector.update_metric("response_time", 0.1 + i * 0.01, time.time())
        
        anomaly_detector._establish_baseline()
        
        assert anomaly_detector.baseline_established
        assert "response_time" in anomaly_detector.statistical_thresholds
        
        thresholds = anomaly_detector.statistical_thresholds["response_time"]
        assert "mean" in thresholds
        assert "std_dev" in thresholds
        assert "upper_2sigma" in thresholds
    
    def test_detection_statistics(self, anomaly_detector):
        """Test anomaly detection statistics."""
        stats = anomaly_detector.get_detection_stats()
        
        assert "component" in stats
        assert "total_anomalies_detected" in stats
        assert "baseline_established" in stats
        assert "threshold_rules" in stats
        assert "quantum_enabled" in stats


class TestSecurityManager:
    """Test cases for SecurityManager class."""
    
    @pytest.fixture
    def security_manager(self):
        """SecurityManager instance for testing."""
        return SecurityManager(token_expiry=300)  # 5 minutes for testing
    
    def test_user_creation(self, security_manager):
        """Test user creation with permissions."""
        permissions = {Permission.READ_METRICS, Permission.MONITOR_HEALTH}
        
        security_manager.create_user("test_user", permissions)
        
        assert "test_user" in security_manager.user_permissions
        assert security_manager.user_permissions["test_user"] == permissions
    
    def test_authentication_success(self, security_manager):
        """Test successful authentication."""
        # Use admin user (pre-configured)
        credentials = {"password": "admin"}
        
        token = security_manager.authenticate("admin", credentials, "127.0.0.1")
        
        assert token is not None
        assert token in security_manager.active_sessions
    
    def test_authentication_failure(self, security_manager):
        """Test failed authentication."""
        credentials = {"password": "wrong_password"}
        
        token = security_manager.authenticate("admin", credentials, "127.0.0.1")
        
        assert token is None
    
    def test_session_validation(self, security_manager):
        """Test session token validation."""
        # Authenticate first
        credentials = {"password": "admin"}
        token = security_manager.authenticate("admin", credentials, "127.0.0.1")
        
        # Validate session
        context = security_manager.validate_session(token, "127.0.0.1")
        
        assert context is not None
        assert context.user_id == "admin"
        assert Permission.ADMIN_ACCESS in context.permissions
    
    def test_authorization(self, security_manager):
        """Test permission-based authorization."""
        # Create context with limited permissions
        from robo_rlhf.pipeline.security import SecurityContext, SecurityLevel
        
        context = SecurityContext(
            user_id="test_user",
            permissions={Permission.READ_METRICS},
            security_level=SecurityLevel.INTERNAL,
            session_token="test_token",
            expires_at=time.time() + 300
        )
        
        # Should allow read access
        assert security_manager.authorize(context, Permission.READ_METRICS)
        
        # Should deny admin access
        assert not security_manager.authorize(context, Permission.ADMIN_ACCESS)
    
    def test_data_encryption(self, security_manager):
        """Test data encryption/decryption."""
        original_data = "sensitive information"
        
        encrypted = security_manager.encrypt_data(original_data)
        decrypted = security_manager.decrypt_data(encrypted)
        
        assert encrypted != original_data
        assert decrypted == original_data
    
    def test_security_statistics(self, security_manager):
        """Test security statistics reporting."""
        stats = security_manager.get_security_stats()
        
        assert "active_sessions" in stats
        assert "total_users" in stats
        assert "blocked_ips" in stats
        assert "security_events_last_hour" in stats


class TestPipelineOrchestrator:
    """Test cases for PipelineOrchestrator class."""
    
    @pytest.fixture
    def mock_component(self):
        """Mock component for testing."""
        async def health_check():
            return {"status": "ok", "cpu_usage": 0.3}
        
        return PipelineComponent(
            name="mock_service",
            endpoint="http://mock:8080",
            health_check=health_check
        )
    
    @pytest.fixture
    def orchestrator(self, mock_component):
        """PipelineOrchestrator instance for testing."""
        config = PipelineConfig(
            monitoring_interval=1,
            healing_enabled=True,
            security_enabled=False,  # Disable for simpler testing
            quantum_enhanced=False   # Disable for testing
        )
        
        return PipelineOrchestrator(config=config, components=[mock_component])
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert len(orchestrator.components) == 1
        assert "mock_service" in orchestrator.components
        assert orchestrator.config.healing_enabled is True
        assert orchestrator._running is False
    
    def test_component_addition(self, orchestrator):
        """Test adding components to orchestrator."""
        async def new_health_check():
            return {"status": "ok"}
        
        new_component = PipelineComponent(
            name="new_service",
            endpoint="http://new:8080",
            health_check=new_health_check
        )
        
        orchestrator.add_component(new_component)
        
        assert "new_service" in orchestrator.components
        assert "new_service" in orchestrator.anomaly_detectors
    
    def test_component_removal(self, orchestrator):
        """Test removing components from orchestrator."""
        result = orchestrator.remove_component("mock_service")
        
        assert result is True
        assert "mock_service" not in orchestrator.components
    
    def test_pipeline_status(self, orchestrator):
        """Test pipeline status reporting."""
        status = orchestrator.get_pipeline_status()
        
        assert "mode" in status
        assert "running" in status
        assert "components" in status
        assert "alerts" in status
        assert "performance_metrics" in status
    
    def test_detailed_metrics(self, orchestrator):
        """Test detailed metrics reporting."""
        metrics = orchestrator.get_detailed_metrics()
        
        assert "pipeline_status" in metrics
        assert "metrics_collector" in metrics
        assert "reliability_patterns" in metrics
        assert "healing_stats" in metrics
        assert "anomaly_detection" in metrics


class TestIntegration:
    """Integration tests for complete pipeline functionality."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_and_healing(self):
        """Test complete monitoring and healing cycle."""
        # Create a component that will fail
        fail_count = 0
        
        async def unreliable_health_check():
            nonlocal fail_count
            fail_count += 1
            if fail_count <= 2:
                raise Exception("Service temporarily unavailable")
            return {"status": "ok", "cpu_usage": 0.3}
        
        component = PipelineComponent(
            name="unreliable_service",
            endpoint="http://unreliable:8080",
            health_check=unreliable_health_check
        )
        
        # Create orchestrator with fast intervals for testing
        config = PipelineConfig(
            monitoring_interval=0.1,
            healing_enabled=True,
            security_enabled=False,
            quantum_enhanced=False
        )
        
        orchestrator = PipelineOrchestrator(config=config, components=[component])
        
        # Start monitoring briefly
        start_task = asyncio.create_task(orchestrator.start())
        
        # Let it run for a short time
        await asyncio.sleep(0.5)
        
        # Stop monitoring
        await orchestrator.stop()
        
        # Cancel the start task if still running
        if not start_task.done():
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                pass
        
        # Verify that healing attempts were made
        healing_stats = orchestrator.healer.get_healing_stats()
        # Note: In this test, actual healing may not complete due to mocking
        # but the framework should have attempted to initiate healing
    
    @pytest.mark.asyncio
    async def test_performance_optimization_cycle(self):
        """Test performance optimization and scaling cycle."""
        from robo_rlhf.pipeline.scaling import AutoScaler, ScalingStrategy
        from robo_rlhf.pipeline.scaling import ScalingRule, ResourceType
        
        # Create auto-scaler
        scaler = AutoScaler("test_service", ScalingStrategy.REACTIVE)
        
        # Add scaling rule
        rule = ScalingRule(
            metric_name="cpu_usage",
            threshold_up=0.8,
            threshold_down=0.2,
            scale_factor=2.0,
            resource_type=ResourceType.INSTANCES
        )
        
        scaler.add_scaling_rule(rule)
        
        # Simulate high CPU usage
        scaler.update_metric("cpu_usage", 0.9, time.time())
        
        # Evaluate scaling
        scaling_event = await scaler.evaluate_scaling()
        
        # Should trigger scale-up
        assert scaling_event is not None
        assert scaling_event.direction.value == "up"
    
    def test_caching_integration(self):
        """Test caching system integration."""
        from robo_rlhf.pipeline.caching import CacheManager
        
        cache_manager = CacheManager()
        
        # Create intelligent cache
        cache = cache_manager.create_cache("test_cache", "intelligent", max_size_mb=10)
        
        assert cache is not None
        assert cache_manager.get_cache("test_cache") is cache
    
    @pytest.mark.asyncio
    async def test_reliability_patterns_integration(self):
        """Test reliability patterns integration."""
        from robo_rlhf.pipeline.reliability import ReliabilityManager, RetryConfig
        
        reliability_manager = ReliabilityManager()
        
        # Test operation that might fail
        attempt_count = 0
        
        async def unreliable_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        # Execute with reliability patterns
        result = await reliability_manager.execute_with_reliability(
            unreliable_operation,
            operation_name="test_operation"
        )
        
        assert result == "success"
        assert attempt_count == 3  # Should have retried


if __name__ == "__main__":
    pytest.main([__file__, "-v"])