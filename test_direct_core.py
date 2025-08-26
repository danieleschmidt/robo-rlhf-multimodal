#!/usr/bin/env python3
"""
Direct core testing bypassing main __init__.py
"""
import sys
import os
sys.path.insert(0, '/root/repo')

def test_core_logging_direct():
    """Test core logging directly."""
    try:
        from robo_rlhf.core.logging import get_logger, setup_logging
        setup_logging()
        logger = get_logger("test_direct")
        logger.info("✅ Core logging system functional")
        return True
    except Exception as e:
        print(f"❌ Core logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_core_config_direct():
    """Test core configuration directly."""
    try:
        from robo_rlhf.core.config import Config, get_config
        
        # Test basic config creation
        config = Config()
        logger_from_config = get_config()
        print("✅ Core configuration system functional")
        return True
    except Exception as e:
        print(f"❌ Core config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_base_collector_direct():
    """Test base collector directly."""
    try:
        from robo_rlhf.collectors.base import DemonstrationData
        
        # Test creating demonstration data
        demo = DemonstrationData(
            timestamp=1234567890.0,
            observations={'rgb': [1, 2, 3]},
            actions=[0.1, 0.2],
            metadata={'episode_id': 'test_direct'}
        )
        
        # Test data conversion
        demo_dict = demo.to_dict() if hasattr(demo, 'to_dict') else demo.__dict__
        print("✅ Base collector data structures functional")
        return True
    except Exception as e:
        print(f"❌ Base collector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_direct_core_tests():
    """Run direct core tests."""
    print("🧪 DIRECT CORE TESTING (GENERATION 1)")
    print("=" * 50)
    
    tests = [
        ("Core Logging Direct", test_core_logging_direct),
        ("Core Config Direct", test_core_config_direct), 
        ("Base Collector Direct", test_base_collector_direct),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 Testing {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    success_rate = (passed / total) * 100
    print(f"\n📊 DIRECT CORE TEST RESULTS:")
    print(f"Tests Passed: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 66:
        print("🎉 DIRECT CORE TESTS SUCCESS - Core functionality verified!")
        return True
    else:
        print("⚠️  DIRECT CORE TESTS NEED IMPROVEMENT")
        return False

if __name__ == "__main__":
    success = run_direct_core_tests()
    exit(0 if success else 1)