#!/usr/bin/env python3
"""
Test core functionality without heavy dependencies.
"""

def test_basic_imports():
    """Test basic package structure."""
    try:
        # Test core utilities
        from robo_rlhf.core.logging import get_logger
        from robo_rlhf.core.config import get_config
        logger = get_logger("test")
        logger.info("✅ Core logging works")
        
        # Test configuration
        config = get_config()
        logger.info("✅ Configuration system works")
        
        # Test data structures
        from robo_rlhf.collectors.base import DemonstrationData
        demo_data = DemonstrationData(
            timestamp=1234567890,
            observations={'test': [1, 2, 3]},
            actions=[0.1, 0.2],
            metadata={'episode_id': 'test_001'}
        )
        logger.info("✅ Data structures work")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_quantum_planning():
    """Test quantum planning without full torch dependencies."""
    try:
        # Test basic quantum planning structure
        from robo_rlhf.quantum.planner import QuantumTaskPlanner, TaskNode, ExecutionPlan
        
        planner = QuantumTaskPlanner()
        
        # Create a simple execution plan
        plan = ExecutionPlan(
            objective="Test quantum execution",
            requirements=["basic_test", "functionality_test"],
            optimizations=["speed", "quality"]
        )
        
        # Test task generation
        tasks = planner._generate_task_superposition(plan.objective, plan.requirements)
        
        print(f"✅ Generated {len(tasks)} quantum task nodes")
        print(f"✅ Quantum planning system functional")
        
        return True
    except Exception as e:
        print(f"❌ Quantum planning test failed: {e}")
        return False

def run_generation1_tests():
    """Run Generation 1 basic functionality tests."""
    print("🧪 GENERATION 1: MAKE IT WORK - TESTING")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Quantum Planning", test_quantum_planning),
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
    print(f"\n📊 GENERATION 1 RESULTS:")
    print(f"Tests Passed: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 GENERATION 1 SUCCESSFUL - Core functionality works!")
        return True
    else:
        print("⚠️  GENERATION 1 NEEDS IMPROVEMENT")
        return False

if __name__ == "__main__":
    success = run_generation1_tests()
    exit(0 if success else 1)