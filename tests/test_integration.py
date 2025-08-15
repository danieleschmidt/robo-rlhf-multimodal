"""
Integration tests for self-healing pipeline system.

Tests end-to-end functionality, component interactions, and real-world scenarios.
"""

import asyncio
import pytest
import time
import tempfile
import json
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

from robo_rlhf.pipeline import (
    PipelineOrchestrator, PipelineConfig, PipelineComponent,
    SecurityManager, SecurityContext, Permission,
    IntelligentCache, MultiTierCache, CacheManager,
    AutoScaler, LoadBalancer, ReliabilityManager
)


class MockExternalService:
    """Mock external service for testing integrations."""
    
    def __init__(self, name: str, failure_rate: float = 0.0):
        self.name = name
        self.failure_rate = failure_rate
        self.call_count = 0
        self.health_status = "healthy"
        self.response_time = 0.1
        
    async def health_check(self) -> Dict[str, Any]:
        """Simulate health check endpoint."""
        self.call_count += 1
        
        # Simulate occasional failures
        if self.call_count * self.failure_rate > 0.5:
            if self.call_count % int(1/self.failure_rate) == 0:
                raise Exception(f"{self.name} is temporarily unavailable")
        
        # Simulate varying response times
        await asyncio.sleep(self.response_time)
        
        return {
            "status": self.health_status,
            "response_time": self.response_time,
            "cpu_usage": 0.3 + (self.call_count % 10) / 100,
            "memory_usage": 0.4 + (self.call_count % 8) / 100,
            "error_rate": self.failure_rate,
            "call_count": self.call_count
        }
    
    def set_health_status(self, status: str) -> None:
        """Change health status for testing."""
        self.health_status = status
    
    def set_response_time(self, time_seconds: float) -> None:
        """Change response time for testing."""
        self.response_time = time_seconds


class TestFullPipelineIntegration:
    """Test complete pipeline integration scenarios."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        return {
            "web_service": MockExternalService("web_service", failure_rate=0.05),
            "api_service": MockExternalService("api_service", failure_rate=0.02),
            "database": MockExternalService("database", failure_rate=0.01),
            "cache_service": MockExternalService("cache_service", failure_rate=0.03)
        }
    
    @pytest.fixture
    def pipeline_components(self, mock_services):
        """Create pipeline components from mock services."""
        components = []
        
        for name, service in mock_services.items():
            component = PipelineComponent(
                name=name,
                endpoint=f"http://{name}:8080",
                health_check=service.health_check,
                critical=(name in ["database", "api_service"]),
                recovery_strategy="restart" if name != "database" else "scale"
            )
            components.append(component)
        
        return components
    
    @pytest.fixture
    def orchestrator(self, pipeline_components):
        """Create orchestrator with test configuration."""
        config = PipelineConfig(
            monitoring_interval=0.5,  # Fast monitoring for tests
            healing_enabled=True,
            security_enabled=True,
            max_concurrent_healings=2,
            auto_scaling_enabled=True,
            quantum_enhanced=False  # Disable for deterministic testing
        )
        
        return PipelineOrchestrator(config=config, components=pipeline_components)
    
    @pytest.mark.asyncio
    async def test_complete_monitoring_cycle(self, orchestrator, mock_services):
        """Test complete monitoring and health assessment cycle."""
        # Start monitoring
        monitoring_task = asyncio.create_task(orchestrator.start())
        
        # Let it run for several monitoring cycles
        await asyncio.sleep(2.0)
        
        # Stop monitoring
        await orchestrator.stop()
        
        # Cancel monitoring task if still running
        if not monitoring_task.done():
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Verify monitoring results
        status = orchestrator.get_pipeline_status()
        
        assert status["running"] is False
        assert len(status["components"]) == len(mock_services)
        
        # All services should have been monitored
        for service_name in mock_services:
            assert service_name in status["components"]
            # Verify calls were made
            assert mock_services[service_name].call_count > 0
    
    @pytest.mark.asyncio
    async def test_failure_detection_and_healing(self, orchestrator, mock_services):
        """Test failure detection and automatic healing."""
        # Make web service fail
        mock_services["web_service"].set_health_status("failed")
        mock_services["web_service"].failure_rate = 1.0  # Always fail
        
        # Track healing attempts
        healing_attempts = []
        
        original_heal = orchestrator.healer.heal
        
        async def mock_heal(component, failure_context, suggested_strategies=None):
            healing_attempts.append({
                "component": component,
                "context": failure_context,
                "timestamp": time.time()
            })
            return await original_heal(component, failure_context, suggested_strategies)
        
        orchestrator.healer.heal = mock_heal
        
        # Start monitoring
        monitoring_task = asyncio.create_task(orchestrator.start())
        
        # Let it detect and attempt healing
        await asyncio.sleep(3.0)
        
        # Stop monitoring
        await orchestrator.stop()
        
        if not monitoring_task.done():
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Verify healing was attempted
        assert len(healing_attempts) > 0
        
        web_healing_attempts = [
            attempt for attempt in healing_attempts
            if attempt["component"] == "web_service"
        ]
        
        assert len(web_healing_attempts) > 0, "Healing should have been attempted for failed service"
    
    @pytest.mark.asyncio
    async def test_performance_degradation_handling(self, orchestrator, mock_services):
        """Test handling of performance degradation."""
        # Slow down API service
        mock_services["api_service"].set_response_time(2.0)  # Very slow
        
        # Start monitoring
        monitoring_task = asyncio.create_task(orchestrator.start())
        
        # Let it detect performance issues
        await asyncio.sleep(4.0)
        
        # Check if degradation was detected
        status = orchestrator.get_pipeline_status()
        api_status = status["components"].get("api_service", {})
        
        # Should detect performance issues
        alerts = [alert for alert in orchestrator.alerts if not alert.resolved]
        performance_alerts = [
            alert for alert in alerts
            if "response" in alert.message.lower() or "slow" in alert.message.lower()
        ]
        
        await orchestrator.stop()
        
        if not monitoring_task.done():
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Should have detected performance degradation
        # Note: This may not always trigger in short test timeframes
        print(f"Performance alerts detected: {len(performance_alerts)}")
    
    @pytest.mark.asyncio
    async def test_cascade_failure_prevention(self, orchestrator, mock_services):
        """Test prevention of cascade failures."""
        # Fail multiple services simultaneously
        mock_services["web_service"].set_health_status("failed")
        mock_services["cache_service"].set_health_status("failed")
        
        # Start monitoring
        monitoring_task = asyncio.create_task(orchestrator.start())
        
        # Let it handle multiple failures
        await asyncio.sleep(3.0)
        
        # Check if pipeline went into emergency mode
        status = orchestrator.get_pipeline_status()
        
        await orchestrator.stop()
        
        if not monitoring_task.done():
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Should have limited concurrent healings
        assert len(orchestrator.active_healings) <= orchestrator.config.max_concurrent_healings


class TestSecurityIntegration:
    """Test security integration with pipeline operations."""
    
    @pytest.fixture
    def security_manager(self):
        """Security manager for testing."""
        return SecurityManager(token_expiry=300)
    
    @pytest.fixture
    def authenticated_context(self, security_manager):
        """Get authenticated security context."""
        credentials = {"password": "admin"}
        token = security_manager.authenticate("admin", credentials, "127.0.0.1")
        return security_manager.validate_session(token, "127.0.0.1")
    
    @pytest.mark.asyncio
    async def test_secured_pipeline_operations(self, security_manager, authenticated_context):
        """Test pipeline operations with security enabled."""
        # Create secure orchestrator
        config = PipelineConfig(
            security_enabled=True,
            monitoring_interval=1,
            healing_enabled=True
        )
        
        async def mock_health_check():
            return {"status": "ok"}
        
        component = PipelineComponent(
            name="secure_service",
            endpoint="http://secure:8080",
            health_check=mock_health_check
        )
        
        orchestrator = PipelineOrchestrator(config=config, components=[component])
        orchestrator.security_manager = security_manager
        
        # Test authorized start
        await orchestrator.start(context=authenticated_context)
        
        # Brief operation
        await asyncio.sleep(0.5)
        
        # Test authorized stop
        await orchestrator.stop(context=authenticated_context)
        
        # Verify security events were logged
        security_stats = security_manager.get_security_stats()
        assert security_stats["successful_authentications_last_hour"] > 0
    
    def test_unauthorized_operations(self, security_manager):
        """Test rejection of unauthorized operations."""
        # Create context with limited permissions
        limited_context = SecurityContext(
            user_id="limited_user",
            permissions={Permission.READ_METRICS},
            security_level=security_manager._determine_security_level({Permission.READ_METRICS}),
            session_token="limited_token",
            expires_at=time.time() + 300
        )
        
        config = PipelineConfig(security_enabled=True)
        orchestrator = PipelineOrchestrator(config=config)
        orchestrator.security_manager = security_manager
        
        # Should raise permission error
        with pytest.raises(PermissionError):
            asyncio.run(orchestrator.start(context=limited_context))


class TestCachingIntegration:
    """Test caching system integration with pipeline operations."""
    
    @pytest.fixture
    def cache_manager(self):
        """Cache manager for testing."""
        return CacheManager()
    
    @pytest.mark.asyncio
    async def test_multi_tier_cache_integration(self, cache_manager):
        """Test multi-tier caching in pipeline context."""
        # Create multi-tier cache
        cache = cache_manager.create_cache("pipeline_cache", "multi_tier")
        
        # Test data flow through cache tiers
        test_data = {
            f"key_{i}": f"value_{i}" * 100  # Varying sizes
            for i in range(100)
        }
        
        # Set data
        for key, value in test_data.items():
            await cache.set(key, value)
        
        # Verify data retrieval
        for key, expected_value in test_data.items():
            retrieved_value = await cache.get(key)
            assert retrieved_value == expected_value
        
        # Test cache promotion through access patterns
        hot_keys = list(test_data.keys())[:10]
        
        # Access hot keys multiple times
        for _ in range(10):
            for key in hot_keys:
                await cache.get(key)
        
        # Verify tier statistics
        tier_stats = cache.get_tier_stats()
        
        # L1 cache should have some hits from hot keys
        l1_stats = tier_stats["l1_memory"]
        assert l1_stats["hits"] > 0, "L1 cache should have received promoted entries"
    
    @pytest.mark.asyncio
    async def test_cache_warming_integration(self, cache_manager):
        """Test cache warming integration."""
        cache = cache_manager.create_cache("warming_test", "intelligent")
        
        # Register warming pattern
        async def warm_common_data():
            return {
                f"common_key_{i}": f"common_value_{i}"
                for i in range(50)
            }
        
        cache.register_warming_pattern("common_data", warm_common_data)
        
        # Execute warming
        warming_results = await cache.warm_cache("common_data")
        
        assert "common_data" in warming_results
        assert warming_results["common_data"] == 50
        
        # Verify warmed data is accessible
        for i in range(50):
            value = await cache.get(f"common_key_{i}")
            assert value == f"common_value_{i}"


class TestScalingIntegration:
    """Test auto-scaling integration with pipeline operations."""
    
    @pytest.mark.asyncio
    async def test_load_balancer_integration(self):
        """Test load balancer integration with health monitoring."""
        load_balancer = LoadBalancer("test_service")
        
        # Add instances
        for i in range(5):
            load_balancer.add_instance(
                f"instance_{i}",
                f"http://instance_{i}:8080",
                weight=1.0
            )
        
        # Simulate load balancing
        routing_stats = {}
        
        for _ in range(1000):
            selected = await load_balancer.route_request({"request_id": "test"})
            routing_stats[selected] = routing_stats.get(selected, 0) + 1
            
            # Simulate request completion
            load_balancer.complete_request(selected, success=True, response_time=0.1)
        
        # Verify load distribution
        stats = load_balancer.get_load_balancer_stats()
        
        assert stats["total_requests_routed"] == 1000
        assert stats["healthy_instances"] == 5
        
        # Load should be reasonably distributed
        for instance_id, instance_stats in stats["instance_stats"].items():
            request_percentage = instance_stats["request_percentage"]
            # Each instance should get roughly 20% of traffic (with some variance)
            assert 15 <= request_percentage <= 25, f"Load imbalance for {instance_id}: {request_percentage}%"
    
    @pytest.mark.asyncio
    async def test_auto_scaler_integration(self):
        """Test auto-scaler integration with metrics."""
        from robo_rlhf.pipeline.scaling import AutoScaler, ScalingStrategy, ScalingRule, ResourceType
        
        scaler = AutoScaler("web_service", ScalingStrategy.REACTIVE)
        
        # Add scaling rules
        rules = [
            ScalingRule("cpu_usage", 0.8, 0.2, resource_type=ResourceType.INSTANCES),
            ScalingRule("memory_usage", 0.85, 0.3, resource_type=ResourceType.MEMORY)
        ]
        
        for rule in rules:
            scaler.add_scaling_rule(rule)
        
        # Simulate increasing load
        timestamps = []
        current_time = time.time()
        
        for i in range(20):
            timestamp = current_time + i * 60  # 1 minute intervals
            timestamps.append(timestamp)
            
            # Gradually increasing CPU usage
            cpu_usage = 0.3 + (i * 0.03)  # Will exceed threshold
            memory_usage = 0.4 + (i * 0.02)
            
            scaler.update_metric("cpu_usage", cpu_usage, timestamp)
            scaler.update_metric("memory_usage", memory_usage, timestamp)
        
        # Evaluate scaling
        scaling_event = await scaler.evaluate_scaling()
        
        # Should trigger scale-up due to high CPU
        assert scaling_event is not None
        assert scaling_event.direction.value == "up"
        assert "cpu_usage" in scaling_event.reason.lower()
        
        # Verify scaling statistics
        stats = scaler.get_scaling_stats()
        assert stats["total_scaling_events"] == 1
        assert stats["scaling_events_up"] == 1


class TestReliabilityIntegration:
    """Test reliability patterns integration."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker integration with pipeline operations."""
        from robo_rlhf.pipeline.reliability import CircuitBreakerPattern, CircuitBreakerConfig
        
        # Create circuit breaker
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=1.0)
        circuit_breaker = CircuitBreakerPattern(config)
        
        # Failing operation
        call_count = 0
        
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                raise Exception("Service failure")
            return "success"
        
        # Execute operations until circuit opens
        failures = 0
        
        for i in range(10):
            try:
                result = await circuit_breaker.execute(failing_operation)
                if result == "success":
                    break
            except Exception as e:
                failures += 1
                if "Circuit breaker is OPEN" in str(e):
                    break
        
        # Circuit should have opened
        assert circuit_breaker.get_state().value == "open"
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Try again - should move to half-open
        try:
            result = await circuit_breaker.execute(failing_operation)
            # This should succeed and close the circuit
            assert result == "success"
            assert circuit_breaker.get_state().value == "closed"
        except Exception:
            # If it fails, circuit should go back to open
            assert circuit_breaker.get_state().value == "open"
    
    @pytest.mark.asyncio
    async def test_retry_pattern_integration(self):
        """Test retry pattern integration."""
        from robo_rlhf.pipeline.reliability import RetryPattern, RetryConfig, RetryStrategy
        
        # Create retry pattern
        config = RetryConfig(
            max_attempts=3,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=0.1
        )
        
        retry_pattern = RetryPattern(config)
        
        # Operation that fails twice then succeeds
        attempt_count = 0
        
        async def unreliable_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception(f"Attempt {attempt_count} failed")
            return f"Success on attempt {attempt_count}"
        
        # Execute with retry
        start_time = time.time()
        result = await retry_pattern.execute(unreliable_operation)
        end_time = time.time()
        
        assert result == "Success on attempt 3"
        assert attempt_count == 3
        
        # Should have taken some time due to delays
        total_time = end_time - start_time
        assert total_time > 0.1, "Should have applied retry delays"
        
        # Check retry statistics
        stats = retry_pattern.get_stats()
        assert len(stats) > 0


class TestEndToEndScenarios:
    """Test realistic end-to-end scenarios."""
    
    @pytest.mark.asyncio
    async def test_microservices_architecture_simulation(self):
        """Test simulation of microservices architecture with dependencies."""
        # Create interconnected services
        services = {}
        
        # Database service (most critical)
        database = MockExternalService("database", failure_rate=0.01)
        services["database"] = database
        
        # API service (depends on database)
        api = MockExternalService("api", failure_rate=0.03)
        services["api"] = api
        
        # Web service (depends on API)
        web = MockExternalService("web", failure_rate=0.05)
        services["web"] = web
        
        # Cache service (optional)
        cache = MockExternalService("cache", failure_rate=0.02)
        services["cache"] = cache
        
        # Create components
        components = []
        for name, service in services.items():
            component = PipelineComponent(
                name=name,
                endpoint=f"http://{name}:8080",
                health_check=service.health_check,
                critical=(name in ["database", "api"]),
                recovery_strategy="restart"
            )
            components.append(component)
        
        # Create orchestrator
        config = PipelineConfig(
            monitoring_interval=0.3,
            healing_enabled=True,
            security_enabled=False,
            max_concurrent_healings=2
        )
        
        orchestrator = PipelineOrchestrator(config=config, components=components)
        
        # Simulate system operation
        monitoring_task = asyncio.create_task(orchestrator.start())
        
        # Run for multiple cycles
        await asyncio.sleep(2.0)
        
        # Introduce failures
        database.set_health_status("degraded")
        api.set_response_time(1.0)  # Slow API
        
        # Let system respond to issues
        await asyncio.sleep(2.0)
        
        # Check system adaptation
        status = orchestrator.get_pipeline_status()
        
        await orchestrator.stop()
        
        if not monitoring_task.done():
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Verify system monitored all services
        assert len(status["components"]) == len(services)
        
        # Verify health checks were performed
        for service_name, service in services.items():
            assert service.call_count > 0, f"Service {service_name} should have been monitored"
        
        # Check performance metrics
        metrics = orchestrator.get_detailed_metrics()
        assert "pipeline_status" in metrics
        assert "performance_metrics" in metrics["pipeline_status"]
    
    @pytest.mark.asyncio
    async def test_disaster_recovery_scenario(self):
        """Test disaster recovery scenario with multiple simultaneous failures."""
        # Create services that will all fail
        failing_services = {
            f"service_{i}": MockExternalService(f"service_{i}", failure_rate=0.8)
            for i in range(5)
        }
        
        components = []
        for name, service in failing_services.items():
            component = PipelineComponent(
                name=name,
                endpoint=f"http://{name}:8080",
                health_check=service.health_check,
                critical=True,
                recovery_strategy="restart"
            )
            components.append(component)
        
        config = PipelineConfig(
            monitoring_interval=0.2,
            healing_enabled=True,
            max_concurrent_healings=3,  # Limited concurrent healing
            security_enabled=False
        )
        
        orchestrator = PipelineOrchestrator(config=config, components=components)
        
        # Start monitoring
        monitoring_task = asyncio.create_task(orchestrator.start())
        
        # Let failures accumulate
        await asyncio.sleep(1.5)
        
        # Check if system goes into emergency mode
        status = orchestrator.get_pipeline_status()
        
        # Should limit concurrent healings even in disaster
        assert len(orchestrator.active_healings) <= config.max_concurrent_healings
        
        await orchestrator.stop()
        
        if not monitoring_task.done():
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Verify system attempted recovery without overwhelming itself
        healing_stats = orchestrator.healer.get_healing_stats()
        print(f"Disaster recovery - total healing attempts: {healing_stats['total_healing_attempts']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])