"""
Command-line interface for quantum autonomous SDLC execution.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional
import json

from robo_rlhf.quantum.autonomous import AutonomousSDLCExecutor, SDLCPhase, ExecutionContext
from robo_rlhf.quantum.optimizer import OptimizationObjective
from robo_rlhf.core import get_logger, setup_logging


async def run_quantum_demo():
    """Run the comprehensive quantum demo."""
    from examples.quantum_autonomous_sdlc_demo import QuantumSDLCDemo
    
    demo = QuantumSDLCDemo()
    results = await demo.run_complete_demo()
    
    print("\n🎉 Quantum Demo Results:")
    print(f"✅ Success: {results['overall_success']}")
    print(f"⏱️  Time: {results['execution_time']:.2f}s")
    print(f"📊 Phases: {len(results['phases_completed'])}")
    print(f"🌟 Features: {len(results['quantum_features_demonstrated'])}")
    
    return results


async def run_autonomous_sdlc(args):
    """Run autonomous SDLC execution."""
    setup_logging(level="INFO")
    logger = get_logger(__name__)
    
    project_path = Path(args.project).resolve()
    
    # Load configuration
    config = {}
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    import yaml
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
    
    # Initialize autonomous executor
    executor = AutonomousSDLCExecutor(project_path, config)
    
    # Parse phases
    target_phases = []
    if args.phases:
        phase_map = {
            "analysis": SDLCPhase.ANALYSIS,
            "design": SDLCPhase.DESIGN,
            "implementation": SDLCPhase.IMPLEMENTATION,
            "testing": SDLCPhase.TESTING,
            "integration": SDLCPhase.INTEGRATION,
            "deployment": SDLCPhase.DEPLOYMENT,
            "monitoring": SDLCPhase.MONITORING,
            "optimization": SDLCPhase.OPTIMIZATION
        }
        target_phases = [phase_map[phase] for phase in args.phases if phase in phase_map]
    
    # Create execution context
    context = ExecutionContext(
        project_path=project_path,
        environment="production" if not args.demo else "demo",
        configuration=config,
        resource_limits={"cpu": 1.0, "memory": 2.0, "storage": 10.0},
        quality_gates={"test_coverage": 0.8, "success_rate": 0.9},
        monitoring_config={"enabled": True, "interval": 60}
    )
    
    logger.info(f"🚀 Starting autonomous SDLC execution for: {project_path}")
    if target_phases:
        logger.info(f"📋 Target phases: {[phase.value for phase in target_phases]}")
    
    # Execute autonomous SDLC
    try:
        results = await executor.execute_autonomous_sdlc(target_phases, context)
        
        # Print results
        print("\n🎯 Autonomous SDLC Execution Results:")
        print(f"✅ Overall Success: {results['overall_success']}")
        print(f"📊 Total Actions: {results['total_actions']}")
        print(f"🎯 Successful Actions: {results['successful_actions']}")
        print(f"❌ Failed Actions: {results['failed_actions']}")
        print(f"🔧 Optimizations Applied: {results['optimizations_applied']}")
        print(f"🔄 Rollbacks Performed: {results['rollbacks_performed']}")
        print(f"📈 Quality Score: {results['quality_score']:.2f}")
        print(f"⏱️  Execution Time: {results['execution_time']:.2f}s")
        
        if results['overall_success']:
            print("\n🎊 Autonomous SDLC execution completed successfully!")
        else:
            print("\n⚠️  Autonomous execution completed with some issues.")
            if 'error' in results:
                print(f"Error: {results['error']}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Autonomous SDLC execution failed: {str(e)}")
        print(f"\n❌ Execution failed: {str(e)}")
        return {"success": False, "error": str(e)}


def main(args) -> None:
    """Main quantum CLI entry point."""
    if args.demo:
        # Run comprehensive quantum demo
        results = asyncio.run(run_quantum_demo())
        
        if results['overall_success']:
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        # Run autonomous SDLC execution
        results = asyncio.run(run_autonomous_sdlc(args))
        
        if results.get('overall_success', False):
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--project", default=".")
    parser.add_argument("--phases", nargs="+")
    parser.add_argument("--config")
    parser.add_argument("--optimization-target", default="quality")
    parser.add_argument("--auto-approve", action="store_true")
    
    args = parser.parse_args()
    main(args)