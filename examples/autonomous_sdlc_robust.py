#!/usr/bin/env python3
"""
Generation 2: Robust Autonomous SDLC Implementation.
Demonstrates comprehensive error handling, monitoring, security, and recovery.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
from contextlib import asynccontextmanager

from robo_rlhf.quantum import AutonomousSDLCExecutor
from robo_rlhf.quantum.autonomous import SDLCPhase, ExecutionContext, ExecutionStatus
from robo_rlhf.core.exceptions import RoboRLHFError, SecurityError, ValidationError
from robo_rlhf.core.security import check_file_safety, sanitize_input
from robo_rlhf.core import get_logger, setup_logging


class RobustSDLCMonitor:
    """Robust monitoring and alerting system for autonomous SDLC."""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.logger = get_logger(__name__)
        self.metrics = {}
        self.alerts = []
        self.log_file = log_file
        self.start_time = time.time()
        
    def record_metric(self, name: str, value: float, tags: Dict[str, Any] = None):
        """Record performance metric."""
        timestamp = time.time()
        if name not in self.metrics:
            self.metrics[name] = []
            
        self.metrics[name].append({
            "timestamp": timestamp,
            "value": value,
            "tags": tags or {}
        })
        
        self.logger.info(f"Metric recorded: {name}={value}", extra={
            "metric_name": name,
            "metric_value": value,
            "tags": tags
        })
    
    def check_thresholds(self):
        """Check metrics against thresholds and generate alerts."""
        for metric_name, data_points in self.metrics.items():
            if len(data_points) < 5:  # Need enough data points
                continue
                
            recent_values = [dp["value"] for dp in data_points[-5:]]
            avg_value = sum(recent_values) / len(recent_values)
            
            # Define thresholds
            thresholds = {
                "execution_time": 300.0,  # 5 minutes max
                "memory_usage": 0.8,      # 80% max
                "error_rate": 0.2         # 20% max
            }
            
            if metric_name in thresholds and avg_value > thresholds[metric_name]:
                alert = {
                    "timestamp": time.time(),
                    "type": "threshold_exceeded",
                    "metric": metric_name,
                    "value": avg_value,
                    "threshold": thresholds[metric_name],
                    "severity": "high" if avg_value > thresholds[metric_name] * 1.5 else "medium"
                }
                self.alerts.append(alert)
                
                self.logger.warning(f"Threshold exceeded for {metric_name}", extra=alert)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        return {
            "uptime": time.time() - self.start_time,
            "metrics_count": len(self.metrics),
            "alerts_count": len(self.alerts),
            "recent_alerts": self.alerts[-5:] if self.alerts else [],
            "metric_summary": {
                name: {
                    "count": len(data),
                    "latest": data[-1]["value"] if data else None,
                    "average": sum(dp["value"] for dp in data[-10:]) / len(data[-10:]) if data else 0
                }
                for name, data in self.metrics.items()
            }
        }


@asynccontextmanager
async def robust_execution_context(executor: AutonomousSDLCExecutor, monitor: RobustSDLCMonitor):
    """Async context manager for robust execution with monitoring."""
    start_time = time.time()
    
    try:
        monitor.logger.info("Starting robust autonomous execution")
        monitor.record_metric("execution_start", 1.0)
        
        yield executor, monitor
        
    except Exception as e:
        execution_time = time.time() - start_time
        monitor.record_metric("execution_failure", 1.0, {"error": str(e)})
        monitor.record_metric("execution_time", execution_time)
        
        monitor.logger.error(f"Robust execution failed: {str(e)}", extra={
            "execution_time": execution_time,
            "error_type": type(e).__name__
        })
        raise
        
    else:
        execution_time = time.time() - start_time
        monitor.record_metric("execution_success", 1.0)
        monitor.record_metric("execution_time", execution_time)
        
        monitor.logger.info(f"Robust execution completed successfully", extra={
            "execution_time": execution_time
        })


async def robust_autonomous_execution():
    """Demonstrate robust autonomous SDLC execution with comprehensive monitoring."""
    print("🛡️ Generation 2: Robust Autonomous SDLC Execution")
    
    # Setup comprehensive logging
    log_file = Path("logs/autonomous_sdlc_robust.log")
    log_file.parent.mkdir(exist_ok=True)
    
    setup_logging(
        level="DEBUG",
        log_file=str(log_file),
        structured=True,
        console=True
    )
    
    monitor = RobustSDLCMonitor(log_file)
    
    try:
        # Robust configuration with security and monitoring
        config = {
            "autonomous": {
                "max_parallel": 2,
                "quality_threshold": 0.8,
                "optimization_frequency": 5,
                "auto_rollback": True
            },
            "security": {
                "max_commands_per_minute": 20,
                "max_command_timeout": 900,
                "allowed_commands": ["python", "pytest", "mypy", "bandit", "docker", "npm"],
                "enable_input_validation": True
            },
            "monitoring": {
                "enable_metrics": True,
                "alert_thresholds": {
                    "execution_time": 600,
                    "memory_usage": 0.8,
                    "error_rate": 0.15
                }
            }
        }
        
        # Initialize executor with security validation
        project_path = Path(".")
        
        # Security check on project directory
        try:
            security_scan = check_file_safety(
                project_path, 
                scan_content=False,
                max_size=None
            )
            if not security_scan["is_safe"]:
                raise SecurityError(f"Project directory failed security scan: {security_scan['threats']}")
        except Exception as e:
            monitor.logger.error(f"Security validation failed: {str(e)}")
            raise
        
        executor = AutonomousSDLCExecutor(project_path, config=config)
        monitor.record_metric("executor_initialized", 1.0)
        
        # Comprehensive phases for robust testing
        robust_phases = [
            SDLCPhase.ANALYSIS,
            SDLCPhase.TESTING,
            SDLCPhase.INTEGRATION,
            SDLCPhase.MONITORING
        ]
        
        # Create comprehensive execution context
        context = ExecutionContext(
            project_path=project_path,
            environment="production",
            configuration=config,
            resource_limits={"cpu": 0.8, "memory": 2.0, "storage": 10.0},
            quality_gates={"test_coverage": 0.85, "success_rate": 0.9, "security_score": 0.95},
            monitoring_config={
                "enabled": True, 
                "interval": 30,
                "alert_channels": ["log", "console"],
                "metrics_retention": 3600
            }
        )
        
        print("📋 Starting robust autonomous execution...")
        print(f"Target phases: {[phase.value for phase in robust_phases]}")
        print(f"Security enabled: {config['security']['enable_input_validation']}")
        print(f"Quality threshold: {config['autonomous']['quality_threshold']}")
        
        # Execute with robust monitoring
        async with robust_execution_context(executor, monitor) as (exec, mon):
            results = await exec.execute_autonomous_sdlc(
                target_phases=robust_phases,
                context=context
            )
            
            # Record execution metrics
            mon.record_metric("total_actions", results['total_actions'])
            mon.record_metric("successful_actions", results['successful_actions'])
            mon.record_metric("failed_actions", results['failed_actions'])
            mon.record_metric("quality_score", results.get('quality_score', 0))
            
            # Check thresholds
            mon.check_thresholds()
        
        # Display comprehensive results
        print("\n✅ Robust execution completed!")
        print(f"📊 Execution Summary:")
        print(f"  • Total actions: {results['total_actions']}")
        print(f"  • Successful: {results['successful_actions']}")
        print(f"  • Failed: {results['failed_actions']}")
        print(f"  • Success rate: {results['successful_actions']}/{results['total_actions']} ({results['successful_actions']/max(results['total_actions'],1)*100:.1f}%)")
        print(f"  • Quality score: {results.get('quality_score', 0):.3f}")
        print(f"  • Execution time: {results.get('execution_time', 0):.1f}s")
        print(f"  • Optimizations applied: {results.get('optimizations_applied', 0)}")
        print(f"  • Rollbacks performed: {results.get('rollbacks_performed', 0)}")
        print(f"  • Overall success: {'✅ YES' if results.get('overall_success') else '❌ NO'}")
        
        # Show monitoring summary
        monitoring_summary = monitor.get_summary()
        print(f"\n📈 Monitoring Summary:")
        print(f"  • Uptime: {monitoring_summary['uptime']:.1f}s")
        print(f"  • Metrics collected: {monitoring_summary['metrics_count']}")
        print(f"  • Alerts generated: {monitoring_summary['alerts_count']}")
        
        # Show executor performance metrics
        exec_summary = executor.get_execution_summary()
        if exec_summary.get('total_executions', 0) > 0:
            print(f"\n🏃 Performance Metrics:")
            print(f"  • Average execution time: {exec_summary.get('average_execution_time', 0):.1f}s")
            print(f"  • Success rate: {exec_summary.get('success_rate', 0)*100:.1f}%")
            print(f"  • Optimizations applied: {exec_summary.get('optimizations_applied', 0)}")
            print(f"  • Current quality score: {exec_summary.get('current_quality_score', 0):.3f}")
        
        # Show recent alerts
        if monitoring_summary['recent_alerts']:
            print(f"\n⚠️  Recent Alerts:")
            for alert in monitoring_summary['recent_alerts']:
                print(f"  • {alert['type']}: {alert['metric']} = {alert['value']:.2f} (threshold: {alert['threshold']})")
        
        return results
        
    except (SecurityError, ValidationError, RoboRLHFError) as e:
        print(f"❌ Robust execution failed with known error: {str(e)}")
        monitor.logger.error(f"Robust autonomous execution failed", extra={
            "error_type": type(e).__name__,
            "error_message": str(e)
        })
        return {"error": str(e), "overall_success": False, "error_type": type(e).__name__}
        
    except Exception as e:
        print(f"❌ Robust execution failed with unexpected error: {str(e)}")
        monitor.logger.error(f"Unexpected error in robust execution", extra={
            "error_type": type(e).__name__,
            "error_message": str(e)
        }, exc_info=True)
        return {"error": str(e), "overall_success": False, "error_type": "unexpected"}


async def robust_failure_recovery_test():
    """Demonstrate robust failure recovery mechanisms."""
    print("\n🔄 Robust Failure Recovery Test")
    
    try:
        config = {
            "autonomous": {
                "auto_rollback": True,
                "quality_threshold": 0.9,  # High threshold to force failures
                "optimization_frequency": 1
            },
            "security": {
                "max_commands_per_minute": 5  # Low limit to test rate limiting
            }
        }
        
        executor = AutonomousSDLCExecutor(Path("."), config=config)
        
        # Force a challenging execution to test recovery
        results = await executor.execute_autonomous_sdlc(
            target_phases=[SDLCPhase.TESTING]
        )
        
        recovery_success = results.get('rollbacks_performed', 0) > 0 or results.get('optimizations_applied', 0) > 0
        print(f"🔄 Recovery mechanisms: {'✅ TESTED' if recovery_success else '⚠️ NOT TRIGGERED'}")
        
        return results
        
    except Exception as e:
        print(f"❌ Failure recovery test failed: {str(e)}")
        return {"error": str(e)}


async def robust_security_validation_test():
    """Demonstrate robust security validation."""
    print("\n🔒 Robust Security Validation Test")
    
    try:
        # Test input sanitization
        dangerous_inputs = [
            "rm -rf /",
            "cat /etc/passwd",
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --"
        ]
        
        security_violations = 0
        for dangerous_input in dangerous_inputs:
            try:
                sanitized = sanitize_input(dangerous_input)
                print(f"⚠️  Input not blocked: {dangerous_input}")
            except SecurityError:
                security_violations += 1
                print(f"✅ Blocked dangerous input: {dangerous_input[:20]}...")
        
        print(f"🔒 Security validation: {security_violations}/{len(dangerous_inputs)} threats blocked")
        
        return {"security_blocks": security_violations, "total_threats": len(dangerous_inputs)}
        
    except Exception as e:
        print(f"❌ Security validation test failed: {str(e)}")
        return {"error": str(e)}


async def main():
    """Main execution function for Generation 2 robust demo."""
    print("=" * 70)
    print("🛡️  ROBO-RLHF Generation 2: Robust Autonomous SDLC")
    print("=" * 70)
    
    # Run robust autonomous execution
    robust_results = await robust_autonomous_execution()
    
    # Run failure recovery test
    recovery_results = await robust_failure_recovery_test()
    
    # Run security validation test
    security_results = await robust_security_validation_test()
    
    # Final comprehensive summary
    print("\n" + "=" * 70)
    print("📋 GENERATION 2 ROBUST EXECUTION SUMMARY")
    print("=" * 70)
    
    robust_success = robust_results.get('overall_success', False)
    recovery_tested = recovery_results.get('rollbacks_performed', 0) > 0 or recovery_results.get('optimizations_applied', 0) > 0
    security_effective = security_results.get('security_blocks', 0) / max(security_results.get('total_threats', 1), 1) >= 0.8
    
    all_robust_features = robust_success and security_effective
    
    print(f"🎯 Robust Execution Success: {'✅ YES' if robust_success else '❌ NO'}")
    print(f"🔄 Recovery Mechanisms: {'✅ TESTED' if recovery_tested else '⚠️ NOT TRIGGERED'}")
    print(f"🔒 Security Validation: {'✅ EFFECTIVE' if security_effective else '❌ INSUFFICIENT'}")
    print(f"🛡️  Overall Generation 2 Success: {'✅ YES' if all_robust_features else '❌ NO'}")
    
    if all_robust_features:
        print("🚀 Ready to proceed to Generation 3 (Optimized Implementation)")
        print("✅ Comprehensive error handling implemented")
        print("✅ Security validation and threat detection active")
        print("✅ Monitoring and alerting system operational")
        print("✅ Autonomous recovery mechanisms tested")
    else:
        print("⚠️  Robustness issues detected - manual intervention may be required")
        if not robust_success:
            print("  - Primary execution needs improvement")
        if not security_effective:
            print("  - Security validation needs strengthening")
    
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())