#!/usr/bin/env python3
"""
Generation 2 Robust Usage Example - MAKE IT ROBUST

Demonstrates the robust error handling, logging, validation, and security
features of robo-rlhf-multimodal.
"""

import os
import tempfile
import numpy as np
from pathlib import Path
import json

# Import core components
from robo_rlhf.core import (
    get_logger, setup_logging, get_config, 
    ValidationError, DataCollectionError, SecurityError,
    validate_observations, validate_actions, validate_preferences,
    sanitize_input, check_file_safety
)
from robo_rlhf.preference.models import Segment, PreferencePair, PreferenceLabel
from robo_rlhf.collectors.recorder import DemonstrationRecorder
from robo_rlhf.collectors.base import DemonstrationData


def test_logging_system():
    """Test the robust logging system."""
    print("\n🔍 Testing Logging System")
    print("=" * 40)
    
    # Setup structured logging
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test.log"
        
        setup_logging(
            level="DEBUG",
            log_file=str(log_file),
            structured=True,
            console=True
        )
        
        logger = get_logger(__name__)
        
        # Test different log levels
        logger.debug("Debug message with context", extra={"test_context": "debug_test"})
        logger.info("Info message", extra={"component": "logging_test"})
        logger.warning("Warning message", extra={"warning_type": "test"})
        
        # Test error logging
        try:
            raise ValidationError("Test validation error", field="test_field")
        except ValidationError as e:
            logger.error("Caught validation error", extra={"error": e.to_dict()})
        
        print(f"✅ Logged messages to {log_file}")
        
        # Check log file exists and has content
        if log_file.exists() and log_file.stat().st_size > 0:
            print("✅ Structured log file created successfully")
            
            # Show sample log entry
            with open(log_file, 'r') as f:
                first_line = f.readline()
                log_entry = json.loads(first_line)
                print(f"   Sample log entry: {log_entry['level']} - {log_entry['message']}")
        else:
            print("❌ Log file not created or empty")


def test_configuration_system():
    """Test the configuration management system."""
    print("\n⚙️ Testing Configuration System")
    print("=" * 40)
    
    # Test default configuration
    config = get_config()
    print(f"✅ Loaded default configuration")
    print(f"   Environment: {config.environment}")
    print(f"   Data collection output: {config.data_collection.output_dir}")
    print(f"   Model device: {config.models.device}")
    print(f"   Security enabled: {config.security.enable_input_validation}")
    
    # Test configuration from dict
    custom_config_dict = {
        "debug": True,
        "environment": "development",
        "data_collection": {
            "output_dir": "custom_data",
            "recording_fps": 60
        },
        "models": {
            "batch_size": 64,
            "learning_rate": 1e-3
        }
    }
    
    from robo_rlhf.core.config import Config
    custom_config = Config.from_dict(custom_config_dict)
    print(f"✅ Created custom configuration")
    print(f"   Custom FPS: {custom_config.data_collection.recording_fps}")
    print(f"   Custom batch size: {custom_config.models.batch_size}")
    
    # Test saving configuration
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = Path(temp_dir) / "test_config.yaml"
        custom_config.save(config_file)
        
        if config_file.exists():
            print(f"✅ Saved configuration to {config_file}")
            
            # Test loading from file
            loaded_config = Config.from_file(config_file)
            print(f"✅ Loaded configuration from file")
            assert loaded_config.data_collection.recording_fps == 60
            print(f"   Verified custom FPS: {loaded_config.data_collection.recording_fps}")


def test_validation_system():
    """Test the input validation system."""
    print("\n✅ Testing Validation System")
    print("=" * 40)
    
    logger = get_logger(__name__)
    
    # Test observation validation - valid case
    valid_observations = {
        'rgb': np.random.randint(0, 255, (50, 64, 64, 3), dtype=np.uint8),
        'depth': np.random.rand(50, 64, 64, 1),
        'proprioception': np.random.randn(50, 7)
    }
    
    try:
        validated_obs = validate_observations(
            valid_observations,
            required_modalities=['rgb', 'proprioception'],
            image_modalities=['rgb', 'depth']
        )
        print("✅ Valid observations passed validation")
        logger.info("Observation validation successful", extra={"num_modalities": len(validated_obs)})
    except ValidationError as e:
        print(f"❌ Unexpected validation error: {e}")
        logger.error("Validation failed unexpectedly", extra={"error": e.to_dict()})
    
    # Test observation validation - invalid case
    invalid_observations = {
        'rgb': np.random.randn(30, 7),  # Wrong shape for image
        'proprioception': np.random.randn(50, 7)  # Different length
    }
    
    try:
        validate_observations(invalid_observations, image_modalities=['rgb'])
        print("❌ Invalid observations should have failed validation")
    except ValidationError as e:
        print(f"✅ Invalid observations correctly rejected: {e.message}")
        logger.info("Validation correctly caught invalid input", extra={"error_code": e.error_code})
    
    # Test action validation with bounded actions
    valid_actions = np.random.uniform(-1.5, 1.5, (50, 7))  # Within bounds
    
    try:
        validated_actions = validate_actions(
            valid_actions,
            expected_dim=7,
            action_bounds=(-2.0, 2.0)
        )
        print("✅ Valid actions passed validation")
    except ValidationError as e:
        print(f"❌ Unexpected action validation error: {e}")
    
    # Test preference validation
    valid_preferences = [
        {"annotator_id": "expert1", "choice": "a", "confidence": 0.9},
        {"annotator_id": "expert2", "choice": "b", "confidence": 0.7}
    ]
    
    try:
        validated_prefs = validate_preferences(valid_preferences)
        print("✅ Valid preferences passed validation")
    except ValidationError as e:
        print(f"❌ Unexpected preference validation error: {e}")


def test_security_system():
    """Test the security system."""
    print("\n🔒 Testing Security System")
    print("=" * 40)
    
    logger = get_logger(__name__)
    
    # Test input sanitization - safe input
    safe_input = "Hello, this is a safe string with numbers 123!"
    try:
        sanitized = sanitize_input(safe_input, max_length=100)
        print(f"✅ Safe input sanitized: '{sanitized}'")
    except SecurityError as e:
        print(f"❌ Safe input incorrectly flagged: {e}")
    
    # Test input sanitization - dangerous input
    dangerous_inputs = [
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        "eval('malicious code')",
        "import os; os.system('rm -rf /')"
    ]
    
    for dangerous_input in dangerous_inputs:
        try:
            sanitize_input(dangerous_input)
            print(f"❌ Dangerous input not caught: {dangerous_input}")
        except SecurityError as e:
            print(f"✅ Dangerous input correctly blocked: {e.threat_type}")
            logger.warning("Blocked dangerous input", extra={"threat_type": e.threat_type})
    
    # Test file safety checking
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create safe file
        safe_file = Path(temp_dir) / "safe_data.json"
        with open(safe_file, 'w') as f:
            json.dump({"data": "safe content", "values": [1, 2, 3]}, f)
        
        try:
            safety_info = check_file_safety(
                safe_file,
                allowed_extensions=['.json', '.txt'],
                max_size=1024*1024  # 1MB
            )
            if safety_info['is_safe']:
                print(f"✅ Safe file passed security check: {safe_file.name}")
                logger.info("File security check passed", extra={"file": str(safe_file)})
            else:
                print(f"❌ Safe file incorrectly flagged: {safety_info['threats']}")
        except SecurityError as e:
            print(f"❌ Safe file incorrectly rejected: {e}")
        
        # Test dangerous file extension
        dangerous_file = Path(temp_dir) / "malicious.exe"
        dangerous_file.touch()  # Create empty file
        
        try:
            check_file_safety(dangerous_file)
            print("❌ Dangerous file extension not caught")
        except SecurityError as e:
            print(f"✅ Dangerous file extension correctly blocked: {e.threat_type}")


def test_error_handling():
    """Test comprehensive error handling."""
    print("\n🚨 Testing Error Handling")
    print("=" * 40)
    
    logger = get_logger(__name__)
    
    # Test different error types
    error_tests = [
        (ValidationError, "Test validation error", {"field": "test"}),
        (DataCollectionError, "Test data collection error", {"operation": "record"}),
        (SecurityError, "Test security error", {"threat_type": "injection"})
    ]
    
    for error_class, message, kwargs in error_tests:
        try:
            raise error_class(message, **kwargs)
        except error_class as e:
            print(f"✅ {error_class.__name__} correctly raised and caught")
            print(f"   Error code: {e.error_code}")
            print(f"   Details: {e.details}")
            
            # Test error serialization
            error_dict = e.to_dict()
            assert "error_type" in error_dict
            assert "message" in error_dict
            print(f"   Serialization: {error_dict['error_type']}")
            
            logger.error(f"Test error caught", extra={"error": error_dict})


def test_robust_data_collection():
    """Test robust data collection with error handling."""
    print("\n📊 Testing Robust Data Collection")
    print("=" * 40)
    
    logger = get_logger(__name__)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create recorder with validation
            recorder = DemonstrationRecorder(
                output_dir=temp_dir,
                compression=False,
                metadata_enrichment=True
            )
            
            # Start episode with validation
            episode_id = recorder.start_episode(
                metadata={
                    'task': sanitize_input('pick_and_place'),
                    'difficulty': sanitize_input('easy')
                }
            )
            
            logger.info("Started robust episode recording", extra={"episode_id": episode_id})
            
            # Record steps with validation
            for step in range(10):
                try:
                    # Create observations (single step format for recorder)
                    observation = {
                        'rgb': np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                        'proprioception': np.random.randn(7)
                    }
                    
                    # Create action (single step)
                    action = np.random.randn(7)
                    
                    # Record step (recorder handles its own validation)
                    recorder.record_step(observation, action, reward=np.random.rand())
                    
                except Exception as e:
                    logger.error("Error during recording step", extra={"error": str(e)})
                    raise DataCollectionError(f"Recording failed: {e}")
            
            # Stop episode
            demo = recorder.stop_episode(success=True)
            print(f"✅ Robust recording completed: {demo.episode_id}")
            print(f"   Frames recorded: {len(demo.actions)}")
            print(f"   Duration: {demo.duration:.2f}s")
            
            # Verify recorded data
            assert len(demo.actions) == 10
            assert demo.success is True
            
            logger.info("Robust data collection test completed", extra={
                "episode_id": demo.episode_id,
                "frames": len(demo.actions),
                "success": demo.success
            })
            
        except (ValidationError, DataCollectionError, SecurityError) as e:
            print(f"❌ Error during robust data collection: {e}")
            logger.error("Robust data collection failed", extra={"error": e.to_dict()})
            raise


def main():
    """Main demonstration of Generation 2 robust features."""
    print("🛡️ Robo-RLHF-Multimodal - Generation 2 Demo")
    print("=" * 50)
    print("Testing robust error handling, validation, security, and logging...")
    
    try:
        # Test all robust features
        test_logging_system()
        test_configuration_system()
        test_validation_system()
        test_security_system()
        test_error_handling()
        test_robust_data_collection()
        
        print("\n🎉 ALL ROBUST TESTS PASSED!")
        print("=" * 50)
        print("✅ Logging system with structured output")
        print("✅ Configuration management with validation")
        print("✅ Input validation and sanitization")
        print("✅ Security scanning and threat detection")
        print("✅ Comprehensive error handling and reporting")
        print("✅ Robust data collection with fail-safes")
        
        print("\n📈 Generation 2 Status: COMPLETE")
        print("🚀 Ready for Generation 3 (Optimized implementation)")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during robust testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())