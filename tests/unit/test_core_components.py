"""
Unit tests for core components.
"""

import pytest
import tempfile
import time
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from robo_rlhf.core.exceptions import (
    ValidationError, SecurityError, ConfigurationError, DataCollectionError
)
from robo_rlhf.core.validators import (
    validate_observations, validate_actions, validate_preferences,
    validate_episode_id, validate_path, validate_numeric
)
from robo_rlhf.core.security import (
    sanitize_input, check_file_safety, generate_file_hash,
    verify_file_integrity, sanitize_filename, RateLimiter
)
from robo_rlhf.core.config import Config, DataCollectionConfig, PreferenceConfig
from robo_rlhf.core.performance import (
    PerformanceMonitor, LRUCache, cached, timer, measure_time,
    ThreadPool, ResourcePool, optimize_memory
)


class TestValidation:
    """Test validation utilities."""
    
    def test_validate_observations_valid(self):
        """Test valid observations pass validation."""
        observations = {
            'rgb': np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8),
            'depth': np.random.rand(10, 64, 64, 1),
            'proprioception': np.random.randn(10, 7)
        }
        
        result = validate_observations(
            observations,
            required_modalities=['rgb', 'proprioception'],
            image_modalities=['rgb', 'depth']
        )
        
        assert result == observations
    
    def test_validate_observations_missing_modality(self):
        """Test validation fails for missing required modality."""
        observations = {
            'rgb': np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_observations(
                observations,
                required_modalities=['rgb', 'proprioception']
            )
        
        assert "Missing required modalities" in str(exc_info.value)
        assert "proprioception" in str(exc_info.value)
    
    def test_validate_observations_inconsistent_length(self):
        """Test validation fails for inconsistent sequence lengths."""
        observations = {
            'rgb': np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8),
            'proprioception': np.random.randn(5, 7)  # Different length
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_observations(observations)
        
        assert "Inconsistent sequence lengths" in str(exc_info.value)
    
    def test_validate_actions_valid(self):
        """Test valid actions pass validation."""
        actions = np.random.uniform(-1, 1, (50, 7))
        
        result = validate_actions(
            actions,
            expected_dim=7,
            action_bounds=(-2.0, 2.0)
        )
        
        np.testing.assert_array_equal(result, actions)
    
    def test_validate_actions_wrong_dimension(self):
        """Test validation fails for wrong action dimension."""
        actions = np.random.randn(50, 5)  # Wrong dimension
        
        with pytest.raises(ValidationError) as exc_info:
            validate_actions(actions, expected_dim=7)
        
        assert "Wrong action dimension" in str(exc_info.value)
    
    def test_validate_actions_out_of_bounds(self):
        """Test validation fails for actions outside bounds."""
        actions = np.random.uniform(-5, 5, (10, 7))  # Outside bounds
        
        with pytest.raises(ValidationError) as exc_info:
            validate_actions(actions, action_bounds=(-2.0, 2.0))
        
        assert "Actions outside bounds" in str(exc_info.value)
    
    def test_validate_preferences_valid(self):
        """Test valid preferences pass validation."""
        preferences = [
            {"annotator_id": "expert1", "choice": "a", "confidence": 0.9},
            {"annotator_id": "expert2", "choice": "b", "confidence": 0.8}
        ]
        
        result = validate_preferences(preferences)
        assert result == preferences
    
    def test_validate_preferences_invalid_choice(self):
        """Test validation fails for invalid choice."""
        preferences = [
            {"annotator_id": "expert1", "choice": "invalid", "confidence": 0.9}
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            validate_preferences(preferences)
        
        assert "Invalid choice" in str(exc_info.value)
    
    def test_validate_episode_id_valid(self):
        """Test valid episode ID passes validation."""
        episode_id = "episode_20250101_001"
        result = validate_episode_id(episode_id)
        assert result == episode_id
    
    def test_validate_episode_id_invalid_chars(self):
        """Test validation fails for invalid characters."""
        episode_id = "episode@invalid!"
        
        with pytest.raises(ValidationError) as exc_info:
            validate_episode_id(episode_id)
        
        assert "alphanumeric characters" in str(exc_info.value)
    
    def test_validate_numeric_valid(self):
        """Test valid numeric values pass validation."""
        assert validate_numeric(5.5, min_value=0, max_value=10) == 5.5
        assert validate_numeric(42, must_be_integer=True) == 42
        assert validate_numeric(3.14, must_be_positive=True) == 3.14
    
    def test_validate_numeric_out_of_range(self):
        """Test validation fails for out of range values."""
        with pytest.raises(ValidationError):
            validate_numeric(15, min_value=0, max_value=10)
        
        with pytest.raises(ValidationError):
            validate_numeric(-5, must_be_positive=True)


class TestSecurity:
    """Test security utilities."""
    
    def test_sanitize_input_safe(self):
        """Test safe input passes through."""
        safe_input = "Hello world 123"
        result = sanitize_input(safe_input)
        assert result == safe_input
    
    def test_sanitize_input_dangerous(self):
        """Test dangerous input is blocked."""
        dangerous_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('test')",
            "eval('malicious code')",
            "import os; os.system('rm -rf /')"
        ]
        
        for dangerous_input in dangerous_inputs:
            with pytest.raises(SecurityError):
                sanitize_input(dangerous_input)
    
    def test_sanitize_input_max_length(self):
        """Test input length validation."""
        long_input = "x" * 1000
        
        with pytest.raises(SecurityError) as exc_info:
            sanitize_input(long_input, max_length=100)
        
        assert "too long" in str(exc_info.value)
    
    def test_check_file_safety_safe_file(self):
        """Test safe file passes security check."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            f.write(b'{"safe": "content"}')
            temp_path = Path(f.name)
        
        try:
            safety_info = check_file_safety(
                temp_path,
                allowed_extensions=['.json'],
                max_size=1024
            )
            
            assert safety_info['is_safe'] is True
            assert safety_info['extension'] == '.json'
        
        finally:
            temp_path.unlink()
    
    def test_check_file_safety_dangerous_extension(self):
        """Test dangerous file extension is blocked."""
        with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(SecurityError) as exc_info:
                check_file_safety(temp_path)
            
            assert "dangerous file extension" in str(exc_info.value).lower()
        
        finally:
            temp_path.unlink()
    
    def test_generate_file_hash(self):
        """Test file hash generation."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = Path(f.name)
        
        try:
            hash_value = generate_file_hash(temp_path)
            assert len(hash_value) == 64  # SHA256 hex length
            assert isinstance(hash_value, str)
        
        finally:
            temp_path.unlink()
    
    def test_verify_file_integrity(self):
        """Test file integrity verification."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = Path(f.name)
        
        try:
            # Generate hash
            expected_hash = generate_file_hash(temp_path)
            
            # Verify integrity
            assert verify_file_integrity(temp_path, expected_hash) is True
            
            # Test with wrong hash
            with pytest.raises(SecurityError):
                verify_file_integrity(temp_path, "wrong_hash")
        
        finally:
            temp_path.unlink()
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        dangerous_filename = "../../../etc/passwd"
        safe_filename = sanitize_filename(dangerous_filename)
        
        assert ".." not in safe_filename
        assert "/" not in safe_filename
        assert safe_filename == "passwd"
    
    def test_rate_limiter(self):
        """Test rate limiting functionality."""
        rate_limiter = RateLimiter(max_requests=2, time_window=1)
        
        # First two requests should be allowed
        assert rate_limiter.is_allowed("client1") is True
        assert rate_limiter.is_allowed("client1") is True
        
        # Third request should be blocked
        assert rate_limiter.is_allowed("client1") is False
        
        # Different client should be allowed
        assert rate_limiter.is_allowed("client2") is True


class TestConfiguration:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = Config()
        
        assert config.environment == "development"
        assert config.debug is False
        assert config.data_collection.output_dir == "data/demonstrations"
        assert config.security.enable_input_validation is True
    
    def test_config_from_dict(self):
        """Test configuration from dictionary."""
        config_dict = {
            "debug": True,
            "environment": "testing",
            "data_collection": {
                "output_dir": "test_data",
                "recording_fps": 60
            },
            "models": {
                "batch_size": 64
            }
        }
        
        config = Config.from_dict(config_dict)
        
        assert config.debug is True
        assert config.environment == "testing"
        assert config.data_collection.output_dir == "test_data"
        assert config.data_collection.recording_fps == 60
        assert config.models.batch_size == 64
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid environment
        with pytest.raises(ConfigurationError):
            Config(environment="invalid")
        
        # Test invalid recording FPS
        with pytest.raises(ConfigurationError):
            DataCollectionConfig(recording_fps=-1)
        
        # Test invalid consensus threshold
        with pytest.raises(ConfigurationError):
            PreferenceConfig(consensus_threshold=1.5)
    
    def test_config_save_load(self):
        """Test configuration save and load."""
        config = Config(debug=True, environment="testing")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.yaml"
            
            # Save configuration
            config.save(config_file)
            assert config_file.exists()
            
            # Load configuration
            loaded_config = Config.from_file(config_file)
            assert loaded_config.debug is True
            assert loaded_config.environment == "testing"


class TestPerformance:
    """Test performance utilities."""
    
    def test_performance_monitor(self):
        """Test performance monitoring."""
        monitor = PerformanceMonitor()
        
        # Test timer
        monitor.start_timer("test_op")
        time.sleep(0.01)
        duration = monitor.stop_timer("test_op")
        
        assert duration > 0
        assert duration < 1.0  # Should be around 0.01s
        
        # Test counter
        monitor.increment_counter("test_counter", 5)
        monitor.increment_counter("test_counter", 3)
        
        metrics = monitor.get_metrics()
        assert metrics["test_counter"] == 8
        assert "test_op_avg" in metrics
    
    def test_timer_context_manager(self):
        """Test timer context manager."""
        monitor = PerformanceMonitor()
        
        with timer("context_test"):
            time.sleep(0.01)
        
        metrics = monitor.get_metrics()
        assert "context_test_avg" in metrics
        assert metrics["context_test_avg"] > 0
    
    def test_measure_time_decorator(self):
        """Test measure_time decorator."""
        @measure_time
        def test_function():
            time.sleep(0.01)
            return "result"
        
        result = test_function()
        assert result == "result"
        
        # Check that timing was recorded
        monitor = PerformanceMonitor()
        metrics = monitor.get_metrics()
        # Should have timing for the function
        function_metrics = [k for k in metrics.keys() if "test_function" in k]
        assert len(function_metrics) > 0
    
    def test_lru_cache(self):
        """Test LRU cache implementation."""
        cache = LRUCache(maxsize=3)
        
        # Test cache miss
        value, hit = cache.get("key1")
        assert hit is False
        assert value is None
        
        # Test cache put and hit
        cache.put("key1", "value1")
        value, hit = cache.get("key1")
        assert hit is True
        assert value == "value1"
        
        # Test cache eviction
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        cache.put("key4", "value4")  # Should evict key1
        
        value, hit = cache.get("key1")
        assert hit is False  # key1 should be evicted
        
        # Test statistics
        stats = cache.stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
    
    def test_cached_decorator(self):
        """Test cached decorator."""
        call_count = 0
        
        @cached(maxsize=2)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call - cache miss
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call with same args - cache hit
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Should not increment
        
        # Call with different args - cache miss
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2
        
        # Check cache stats
        stats = expensive_function.cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
    
    def test_thread_pool(self):
        """Test thread pool functionality."""
        def simple_task(x):
            time.sleep(0.01)
            return x * 2
        
        with ThreadPool(max_workers=2) as pool:
            # Submit tasks
            futures = [pool.submit(simple_task, i) for i in range(5)]
            
            # Collect results
            results = [future.result() for future in futures]
            
            expected = [i * 2 for i in range(5)]
            assert results == expected
    
    def test_resource_pool(self):
        """Test resource pool functionality."""
        creation_count = 0
        
        def create_resource():
            nonlocal creation_count
            creation_count += 1
            return f"resource_{creation_count}"
        
        def reset_resource(resource):
            pass  # No-op for testing
        
        pool = ResourcePool(
            create_func=create_resource,
            reset_func=reset_resource,
            max_size=2
        )
        
        # Test resource acquisition and release
        with pool.get_resource() as resource:
            assert resource == "resource_1"
            assert creation_count == 1
        
        # Test resource reuse
        with pool.get_resource() as resource:
            assert resource == "resource_1"  # Should reuse
            assert creation_count == 1  # Should not create new
        
        # Test pool statistics
        stats = pool.stats()
        assert stats["created_count"] == 1
        assert stats["max_size"] == 2
    
    def test_optimize_memory(self):
        """Test memory optimization."""
        # Create some objects to garbage collect
        large_list = [list(range(1000)) for _ in range(100)]
        del large_list
        
        result = optimize_memory()
        
        assert isinstance(result, dict)
        assert "collected_objects" in result
        assert "memory_mb" in result
        assert result["collected_objects"] >= 0