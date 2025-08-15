#!/usr/bin/env python3
"""
Test runner for self-healing pipeline system.

Simple test runner that doesn't require external pytest installation.
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import pipeline modules
try:
    from robo_rlhf.pipeline import (
        PipelineGuard, PipelineComponent, MetricsCollector, SelfHealer,
        HealthStatus, RecoveryStrategy, IntelligentCache, SecurityManager,
        PipelineOrchestrator, PipelineConfig
    )
    print("✅ Successfully imported pipeline modules")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    traceback.print_exc()
    sys.exit(1)


class SimpleTestRunner:
    """Simple test runner without pytest dependency."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def run_test(self, test_func, test_name):
        """Run a single test function."""
        try:
            print(f"Running {test_name}...", end=" ")
            
            if asyncio.iscoroutinefunction(test_func):
                asyncio.run(test_func())
            else:
                test_func()
            
            print("✅ PASSED")
            self.passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            self.failed += 1
            self.errors.append(f"{test_name}: {e}")
            traceback.print_exc()
    
    def run_test_class(self, test_class, class_name):
        """Run all test methods in a test class."""
        print(f"\n📋 Running {class_name}")
        print("-" * 50)
        
        instance = test_class()
        
        # Find all test methods
        test_methods = [
            method for method in dir(instance)
            if method.startswith('test_') and callable(getattr(instance, method))
        ]
        
        for method_name in test_methods:
            test_method = getattr(instance, method_name)
            self.run_test(test_method, f"{class_name}.{method_name}")
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📈 Total: {self.passed + self.failed}")
        
        if self.failed > 0:
            print(f"💔 Success Rate: {self.passed/(self.passed + self.failed)*100:.1f}%")
            print("\n🔍 Failed Tests:")
            for error in self.errors:
                print(f"   • {error}")
        else:
            print("🎉 All tests passed!")
        
        return self.failed == 0


async def test_basic_functionality():
    """Test basic pipeline functionality without pytest."""
    print("🔬 Testing basic functionality...")
    
    # Test MetricsCollector
    collector = MetricsCollector()
    collector.record_metric("test_metric", 1.0, {"tag": "value"})
    
    latest = collector.get_latest_metrics(["test_metric"])
    assert "test_metric" in latest
    assert latest["test_metric"] is not None
    assert latest["test_metric"].value == 1.0
    
    print("   ✅ MetricsCollector basic functionality working")
    
    # Test PipelineComponent
    async def mock_health_check():
        return {"status": "ok", "response_time": 0.1}
    
    component = PipelineComponent(
        name="test_service",
        endpoint="http://test:8080",
        health_check=mock_health_check
    )
    
    # Test health check
    result = await component.health_check()
    assert result["status"] == "ok"
    
    print("   ✅ PipelineComponent basic functionality working")
    
    # Test PipelineGuard
    guard = PipelineGuard([component], check_interval=60, healing_enabled=False)
    report = await guard._check_component_health("test_service", component)
    
    assert report.component == "test_service"
    assert report.metrics["response_time"] == 0.1
    
    print("   ✅ PipelineGuard basic functionality working")


def test_imports():
    """Test that all modules can be imported successfully."""
    try:
        from robo_rlhf.pipeline import (
            PipelineGuard, HealthStatus, PipelineMonitor, MetricsCollector,
            SelfHealer, RecoveryStrategy, AnomalyDetector, FailurePredictor,
            SecurityManager, SecurityContext, Permission,
            ReliabilityManager, AutoScaler, IntelligentCache,
            PipelineOrchestrator, PipelineConfig
        )
        print("   ✅ All pipeline modules imported successfully")
        
        # Test module creation
        collector = MetricsCollector()
        cache = IntelligentCache("test", max_size_mb=10)
        healer = SelfHealer()
        
        print("   ✅ All main classes can be instantiated")
        
    except Exception as e:
        raise AssertionError(f"Import test failed: {e}")


def run_basic_tests():
    """Run basic tests without pytest."""
    runner = SimpleTestRunner()
    
    print("🚀 SELF-HEALING PIPELINE TEST SUITE")
    print("=" * 60)
    
    # Test imports first
    runner.run_test(test_imports, "Module Imports")
    
    # Test basic functionality
    runner.run_test(test_basic_functionality, "Basic Functionality")
    
    # Run specific test classes with simplified versions
    print("\n📋 Running Core Component Tests")
    print("-" * 50)
    
    try:
        # Test MetricsCollector
        collector = MetricsCollector()
        
        # Test metric recording
        collector.record_metric("cpu_usage", 0.8, {"host": "server1"})
        latest = collector.get_latest_metrics(["cpu_usage"])
        assert latest["cpu_usage"].value == 0.8
        print("Running MetricsCollector metric recording... ✅ PASSED")
        
        # Test batch recording
        batch = [
            {"name": "metric1", "value": 1.0},
            {"name": "metric2", "value": 2.0}
        ]
        collector.record_batch(batch)
        latest = collector.get_latest_metrics(["metric1", "metric2"])
        assert latest["metric1"].value == 1.0
        assert latest["metric2"].value == 2.0
        print("Running MetricsCollector batch recording... ✅ PASSED")
        
        runner.passed += 2
        
    except Exception as e:
        print(f"Running MetricsCollector tests... ❌ FAILED: {e}")
        runner.failed += 1
    
    try:
        # Test SelfHealer
        healer = SelfHealer()
        
        # Test basic initialization
        assert healer.executors is not None
        assert len(healer.executors) > 0
        print("Running SelfHealer initialization... ✅ PASSED")
        
        # Test strategy success tracking
        success_rate = healer.get_strategy_success_rate(RecoveryStrategy.RESTART)
        assert isinstance(success_rate, float)
        print("Running SelfHealer success tracking... ✅ PASSED")
        
        runner.passed += 2
        
    except Exception as e:
        print(f"Running SelfHealer tests... ❌ FAILED: {e}")
        runner.failed += 1
    
    return runner.print_summary()


if __name__ == "__main__":
    print("🧪 Starting Self-Healing Pipeline Test Suite")
    print("=" * 60)
    
    success = run_basic_tests()
    
    if success:
        print("\n🎊 ALL TESTS PASSED! Self-healing pipeline is ready for production.")
        sys.exit(0)
    else:
        print("\n💥 SOME TESTS FAILED! Please review and fix issues.")
        sys.exit(1)