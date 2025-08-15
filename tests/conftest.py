"""
Pytest configuration and shared fixtures for pipeline tests.

Provides common test fixtures and configuration for the self-healing pipeline test suite.
"""

import asyncio
import pytest
import tempfile
import os
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock

# Mock the quantum modules to avoid import errors in testing
import sys
from unittest.mock import MagicMock

# Create mock quantum modules
mock_quantum = MagicMock()
mock_quantum.QuantumTaskPlanner = MagicMock
mock_quantum.QuantumDecisionEngine = MagicMock
mock_quantum.QuantumOptimizer = MagicMock
mock_quantum.MultiObjectiveOptimizer = MagicMock
mock_quantum.AutonomousSDLCExecutor = MagicMock
mock_quantum.PredictiveAnalytics = MagicMock
mock_quantum.ResourcePredictor = MagicMock

sys.modules['robo_rlhf.quantum'] = mock_quantum

# Mock external dependencies that might not be available
external_mocks = {
    'jwt': MagicMock(),
    'cryptography.fernet': MagicMock(),
    'cryptography.hazmat.primitives': MagicMock(),
    'cryptography.hazmat.primitives.kdf.pbkdf2': MagicMock(),
    'sklearn.ensemble': MagicMock(),
    'psutil': MagicMock(),
}

for module_name, mock_module in external_mocks.items():
    if module_name not in sys.modules:
        sys.modules[module_name] = mock_module


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_health_check():
    """Standard mock health check function."""
    async def health_check():
        return {
            "status": "ok",
            "response_time": 0.1,
            "cpu_usage": 0.3,
            "memory_usage": 0.4,
            "error_rate": 0.0
        }
    return health_check


@pytest.fixture
async def failing_health_check():
    """Mock health check that always fails."""
    async def health_check():
        raise Exception("Service unavailable")
    return health_check


@pytest.fixture
async def slow_health_check():
    """Mock health check with slow response."""
    async def health_check():
        await asyncio.sleep(0.5)  # Simulate slow response
        return {
            "status": "ok",
            "response_time": 0.5,
            "cpu_usage": 0.8,  # High CPU
            "memory_usage": 0.7  # High memory
        }
    return health_check


@pytest.fixture
def sample_metrics_data():
    """Sample metrics data for testing."""
    return {
        "cpu_usage": [0.3, 0.4, 0.5, 0.6, 0.7],
        "memory_usage": [0.2, 0.3, 0.4, 0.5, 0.6],
        "response_time": [0.1, 0.15, 0.2, 0.25, 0.3],
        "error_rate": [0.01, 0.02, 0.01, 0.03, 0.02]
    }


@pytest.fixture
def temp_directory():
    """Temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


# Helper functions for tests
def create_mock_component(name: str, health_check_func=None):
    """Create a mock pipeline component."""
    from robo_rlhf.pipeline.guard import PipelineComponent
    
    if health_check_func is None:
        async def default_health_check():
            return {"status": "ok", "response_time": 0.1}
        health_check_func = default_health_check
    
    return PipelineComponent(
        name=name,
        endpoint=f"http://{name}:8080",
        health_check=health_check_func,
        critical=False
    )


def assert_metrics_recorded(metrics_collector, metric_names: List[str]):
    """Assert that specific metrics were recorded."""
    latest_metrics = metrics_collector.get_latest_metrics(metric_names)
    
    for metric_name in metric_names:
        assert metric_name in latest_metrics
        assert latest_metrics[metric_name] is not None


def assert_health_status(health_report, expected_status: str):
    """Assert health report has expected status."""
    assert health_report.status.value == expected_status


# Test data factories
class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_metric_batch(count: int = 10):
        """Create a batch of test metrics."""
        return [
            {
                "name": f"test_metric_{i}",
                "value": float(i),
                "tags": {"index": str(i), "batch": "test"}
            }
            for i in range(count)
        ]
    
    @staticmethod
    def create_anomaly_data():
        """Create test data for anomaly detection."""
        return {
            "normal_values": [0.5, 0.52, 0.48, 0.51, 0.49, 0.53, 0.47],
            "anomalous_values": [0.9, 0.95, 0.88, 0.92],
            "timestamps": [i * 60 for i in range(11)]  # 1 minute intervals
        }


@pytest.fixture
def test_data_factory():
    """Test data factory fixture."""
    return TestDataFactory