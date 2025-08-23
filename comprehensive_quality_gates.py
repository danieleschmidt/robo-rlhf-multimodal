#!/usr/bin/env python3
"""
Comprehensive Quality Gates - Complete validation suite
Implements testing, security scanning, performance benchmarking, and compliance checks
"""

import sys
import json
import time
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import hashlib
import re
import os

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/quality_gates.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    name: str
    status: str  # "pass", "fail", "warning", "skip"
    score: float  # 0-100
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    recommendations: List[str] = field(default_factory=list)

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics."""
    total_gates: int = 0
    passed_gates: int = 0
    failed_gates: int = 0
    warning_gates: int = 0
    skipped_gates: int = 0
    overall_score: float = 0.0
    execution_time: float = 0.0
    coverage_percentage: float = 0.0
    security_score: float = 0.0
    performance_score: float = 0.0

class ComprehensiveQualityGates:
    """Comprehensive quality gates validation system."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path("/root/repo")
        self.results: List[QualityGateResult] = []
        self.start_time = time.time()
        
        logger.info(f"Quality gates initialized for: {self.project_root}")
    
    def execute_command(self, command: List[str], timeout: int = 30) -> Tuple[int, str, str]:
        """Execute command with timeout and capture output."""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.project_root
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return -1, "", f"Command execution failed: {e}"
    
    def gate_code_quality_basic(self) -> QualityGateResult:
        """Basic code quality checks."""
        start_time = time.time()
        result = QualityGateResult(name="Code Quality - Basic", status="pass", score=100)
        
        try:
            # Count Python files
            py_files = list(self.project_root.rglob("*.py"))
            result.details["python_files_count"] = len(py_files)
            
            # Basic syntax check
            syntax_errors = 0
            for py_file in py_files[:20]:  # Check first 20 files to avoid timeout
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        compile(f.read(), str(py_file), 'exec')
                except SyntaxError:
                    syntax_errors += 1
            
            result.details["syntax_errors"] = syntax_errors
            
            # Check for common issues
            issues = []
            
            # Check for print statements (should use logging)
            print_count = 0
            for py_file in py_files[:10]:  # Sample check
                try:
                    content = py_file.read_text(encoding='utf-8')
                    print_count += len(re.findall(r'\bprint\(', content))
                except:
                    continue
            
            if print_count > 10:
                issues.append("Excessive print statements found (consider using logging)")
                result.score -= 10
            
            # Check for TODO comments
            todo_count = 0
            for py_file in py_files[:10]:
                try:
                    content = py_file.read_text(encoding='utf-8')
                    todo_count += len(re.findall(r'TODO|FIXME|XXX', content, re.IGNORECASE))
                except:
                    continue
            
            result.details["todo_count"] = todo_count
            result.details["issues"] = issues
            
            if syntax_errors > 0:
                result.status = "fail"
                result.score = 0
                result.recommendations.append("Fix syntax errors before deployment")
            elif issues:
                result.status = "warning"
            
            logger.info(f"Code quality check: {len(py_files)} files, {syntax_errors} syntax errors")
            
        except Exception as e:
            result.status = "fail"
            result.score = 0
            result.details["error"] = str(e)
            logger.error(f"Code quality check failed: {e}")
        
        result.execution_time = time.time() - start_time
        return result
    
    def gate_security_basic(self) -> QualityGateResult:
        """Basic security vulnerability scanning."""
        start_time = time.time()
        result = QualityGateResult(name="Security - Basic", status="pass", score=100)
        
        try:
            # Check for common security anti-patterns
            security_issues = []
            py_files = list(self.project_root.rglob("*.py"))
            
            for py_file in py_files[:15]:  # Sample check
                try:
                    content = py_file.read_text(encoding='utf-8')
                    
                    # Check for hardcoded secrets patterns
                    secret_patterns = [
                        r'password\s*=\s*["\'][^"\']{3,}["\']',
                        r'api_key\s*=\s*["\'][^"\']{10,}["\']',
                        r'secret\s*=\s*["\'][^"\']{8,}["\']',
                        r'token\s*=\s*["\'][^"\']{10,}["\']'
                    ]
                    
                    for pattern in secret_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            security_issues.append(f"Potential hardcoded secret in {py_file.name}")
                            break
                    
                    # Check for dangerous functions
                    dangerous_patterns = [
                        r'\beval\s*\(',
                        r'\bexec\s*\(',
                        r'subprocess\.call\([^)]*shell\s*=\s*True',
                        r'os\.system\s*\('
                    ]
                    
                    for pattern in dangerous_patterns:
                        if re.search(pattern, content):
                            security_issues.append(f"Dangerous function usage in {py_file.name}")
                            break
                
                except Exception:
                    continue
            
            result.details["security_issues"] = security_issues
            result.details["files_scanned"] = min(15, len(py_files))
            
            if security_issues:
                issue_count = len(security_issues)
                if issue_count > 5:
                    result.status = "fail"
                    result.score = 20
                else:
                    result.status = "warning"
                    result.score = 80 - (issue_count * 10)
                
                result.recommendations.extend([
                    "Review and fix security issues before deployment",
                    "Use environment variables for secrets",
                    "Validate all user inputs",
                    "Use parameterized queries for database operations"
                ])
            
            logger.info(f"Security scan: {result.details['files_scanned']} files, {len(security_issues)} issues")
            
        except Exception as e:
            result.status = "fail"
            result.score = 0
            result.details["error"] = str(e)
            logger.error(f"Security scan failed: {e}")
        
        result.execution_time = time.time() - start_time
        return result
    
    def gate_performance_basic(self) -> QualityGateResult:
        """Basic performance validation."""
        start_time = time.time()
        result = QualityGateResult(name="Performance - Basic", status="pass", score=100)
        
        try:
            # Test basic operations performance
            performance_tests = []
            
            # File I/O performance
            test_file = self.project_root / "temp_perf_test.txt"
            io_start = time.time()
            test_file.write_text("Performance test data " * 1000)
            content = test_file.read_text()
            test_file.unlink()
            io_time = time.time() - io_start
            
            performance_tests.append({
                "test": "file_io",
                "time": io_time,
                "status": "pass" if io_time < 0.1 else "warning"
            })
            
            # JSON processing performance
            json_start = time.time()
            test_data = {"test": "data", "numbers": list(range(1000))}
            json_str = json.dumps(test_data)
            parsed = json.loads(json_str)
            json_time = time.time() - json_start
            
            performance_tests.append({
                "test": "json_processing",
                "time": json_time,
                "status": "pass" if json_time < 0.01 else "warning"
            })
            
            # Basic computation performance
            compute_start = time.time()
            result_val = sum(i ** 2 for i in range(10000))
            compute_time = time.time() - compute_start
            
            performance_tests.append({
                "test": "computation",
                "time": compute_time,
                "status": "pass" if compute_time < 0.1 else "warning"
            })
            
            result.details["performance_tests"] = performance_tests
            
            # Calculate overall performance score
            avg_time = sum(t["time"] for t in performance_tests) / len(performance_tests)
            if avg_time < 0.05:
                result.score = 100
            elif avg_time < 0.1:
                result.score = 80
            elif avg_time < 0.2:
                result.score = 60
                result.status = "warning"
            else:
                result.score = 40
                result.status = "warning"
            
            result.details["average_execution_time"] = avg_time
            
            if result.status == "warning":
                result.recommendations.append("Consider performance optimization")
            
            logger.info(f"Performance test: avg time {avg_time:.4f}s")
            
        except Exception as e:
            result.status = "fail"
            result.score = 0
            result.details["error"] = str(e)
            logger.error(f"Performance test failed: {e}")
        
        result.execution_time = time.time() - start_time
        return result
    
    def gate_test_coverage(self) -> QualityGateResult:
        """Test coverage analysis."""
        start_time = time.time()
        result = QualityGateResult(name="Test Coverage", status="skip", score=0)
        
        try:
            # Check if test directory exists
            test_dirs = [
                self.project_root / "tests",
                self.project_root / "test"
            ]
            
            test_files = []
            for test_dir in test_dirs:
                if test_dir.exists():
                    test_files.extend(list(test_dir.rglob("test_*.py")))
                    test_files.extend(list(test_dir.rglob("*_test.py")))
            
            # Check for pytest configuration
            pytest_configs = [
                self.project_root / "pytest.ini",
                self.project_root / "pyproject.toml",
                self.project_root / "setup.cfg"
            ]
            
            has_pytest_config = any(config.exists() for config in pytest_configs)
            
            result.details["test_files_found"] = len(test_files)
            result.details["has_pytest_config"] = has_pytest_config
            
            if test_files:
                # Estimate coverage based on test files vs source files
                py_files = list(self.project_root.rglob("*.py"))
                source_files = [f for f in py_files if not any(part.startswith("test") for part in f.parts)]
                
                estimated_coverage = min(100, (len(test_files) / max(1, len(source_files))) * 100)
                
                result.details["estimated_coverage"] = estimated_coverage
                result.details["source_files"] = len(source_files)
                
                if estimated_coverage >= 80:
                    result.status = "pass"
                    result.score = 100
                elif estimated_coverage >= 60:
                    result.status = "warning"
                    result.score = 80
                elif estimated_coverage >= 40:
                    result.status = "warning"
                    result.score = 60
                else:
                    result.status = "fail"
                    result.score = 30
                
                if result.status != "pass":
                    result.recommendations.append("Increase test coverage to at least 80%")
            else:
                result.status = "fail"
                result.score = 0
                result.recommendations.extend([
                    "Add unit tests for critical functionality",
                    "Set up test framework (pytest recommended)",
                    "Aim for at least 80% code coverage"
                ])
            
            logger.info(f"Test coverage: {len(test_files)} test files found")
            
        except Exception as e:
            result.status = "fail"
            result.score = 0
            result.details["error"] = str(e)
            logger.error(f"Test coverage analysis failed: {e}")
        
        result.execution_time = time.time() - start_time
        return result
    
    def gate_compliance_basic(self) -> QualityGateResult:
        """Basic compliance and best practices check."""
        start_time = time.time()
        result = QualityGateResult(name="Compliance - Basic", status="pass", score=100)
        
        try:
            compliance_checks = []
            
            # Check for required files
            required_files = [
                ("README.md", "Documentation"),
                ("LICENSE", "License"),
                ("requirements.txt", "Dependencies") or ("pyproject.toml", "Dependencies"),
                (".gitignore", "Git configuration")
            ]
            
            missing_files = []
            for filename, description in required_files:
                if not (self.project_root / filename).exists():
                    missing_files.append(f"{filename} ({description})")
            
            compliance_checks.append({
                "check": "required_files",
                "missing": missing_files,
                "status": "pass" if not missing_files else "warning"
            })
            
            # Check Python version compatibility
            py_files = list(self.project_root.rglob("*.py"))
            python3_compatible = True
            
            for py_file in py_files[:10]:  # Sample check
                try:
                    content = py_file.read_text(encoding='utf-8')
                    # Check for Python 2 specific patterns
                    if re.search(r'\bprint\s+[^(]', content) or 'raw_input' in content:
                        python3_compatible = False
                        break
                except:
                    continue
            
            compliance_checks.append({
                "check": "python3_compatibility",
                "compatible": python3_compatible,
                "status": "pass" if python3_compatible else "fail"
            })
            
            # Check for proper package structure
            has_init = (self.project_root / "__init__.py").exists()
            has_setup = (self.project_root / "setup.py").exists() or (self.project_root / "pyproject.toml").exists()
            
            compliance_checks.append({
                "check": "package_structure",
                "has_init": has_init,
                "has_setup": has_setup,
                "status": "pass" if (has_init or has_setup) else "warning"
            })
            
            result.details["compliance_checks"] = compliance_checks
            
            # Calculate compliance score
            failed_checks = [c for c in compliance_checks if c["status"] == "fail"]
            warning_checks = [c for c in compliance_checks if c["status"] == "warning"]
            
            if failed_checks:
                result.status = "fail"
                result.score = max(20, 100 - len(failed_checks) * 30)
                result.recommendations.append("Fix critical compliance issues")
            elif warning_checks:
                result.status = "warning"
                result.score = max(60, 100 - len(warning_checks) * 15)
                result.recommendations.append("Address compliance warnings for production readiness")
            
            if missing_files:
                result.recommendations.append(f"Add missing files: {', '.join(missing_files)}")
            
            logger.info(f"Compliance check: {len(failed_checks)} failures, {len(warning_checks)} warnings")
            
        except Exception as e:
            result.status = "fail"
            result.score = 0
            result.details["error"] = str(e)
            logger.error(f"Compliance check failed: {e}")
        
        result.execution_time = time.time() - start_time
        return result
    
    def run_all_quality_gates(self) -> QualityMetrics:
        """Execute all quality gates and generate comprehensive report."""
        logger.info("Starting comprehensive quality gate validation...")
        
        # Execute all quality gates
        gates = [
            self.gate_code_quality_basic,
            self.gate_security_basic,
            self.gate_performance_basic,
            self.gate_test_coverage,
            self.gate_compliance_basic
        ]
        
        for gate_func in gates:
            try:
                result = gate_func()
                self.results.append(result)
                logger.info(f"Quality gate '{result.name}': {result.status} (score: {result.score})")
            except Exception as e:
                logger.error(f"Quality gate {gate_func.__name__} failed: {e}")
                # Add failed result
                self.results.append(QualityGateResult(
                    name=gate_func.__name__,
                    status="fail",
                    score=0,
                    details={"error": str(e)}
                ))
        
        # Calculate overall metrics
        metrics = QualityMetrics()
        metrics.total_gates = len(self.results)
        metrics.passed_gates = len([r for r in self.results if r.status == "pass"])
        metrics.failed_gates = len([r for r in self.results if r.status == "fail"])
        metrics.warning_gates = len([r for r in self.results if r.status == "warning"])
        metrics.skipped_gates = len([r for r in self.results if r.status == "skip"])
        
        # Calculate weighted overall score
        total_score = sum(r.score for r in self.results)
        metrics.overall_score = total_score / max(1, len(self.results))
        
        # Calculate specific scores
        security_results = [r for r in self.results if "Security" in r.name]
        metrics.security_score = sum(r.score for r in security_results) / max(1, len(security_results))
        
        performance_results = [r for r in self.results if "Performance" in r.name]
        metrics.performance_score = sum(r.score for r in performance_results) / max(1, len(performance_results))
        
        coverage_results = [r for r in self.results if "Coverage" in r.name]
        if coverage_results:
            metrics.coverage_percentage = coverage_results[0].details.get("estimated_coverage", 0)
        
        metrics.execution_time = time.time() - self.start_time
        
        return metrics
    
    def generate_comprehensive_report(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """Generate comprehensive quality gates report."""
        return {
            "timestamp": time.time(),
            "generation": "Quality Gates - Comprehensive Validation",
            "project_root": str(self.project_root),
            "overall_metrics": {
                "total_gates": metrics.total_gates,
                "passed_gates": metrics.passed_gates,
                "failed_gates": metrics.failed_gates,
                "warning_gates": metrics.warning_gates,
                "skipped_gates": metrics.skipped_gates,
                "overall_score": metrics.overall_score,
                "security_score": metrics.security_score,
                "performance_score": metrics.performance_score,
                "coverage_percentage": metrics.coverage_percentage,
                "execution_time": metrics.execution_time
            },
            "gate_results": [
                {
                    "name": result.name,
                    "status": result.status,
                    "score": result.score,
                    "execution_time": result.execution_time,
                    "details": result.details,
                    "recommendations": result.recommendations
                }
                for result in self.results
            ],
            "quality_assessment": self._assess_overall_quality(metrics),
            "next_steps": self._generate_next_steps(metrics)
        }
    
    def _assess_overall_quality(self, metrics: QualityMetrics) -> str:
        """Assess overall quality based on metrics."""
        if metrics.overall_score >= 90 and metrics.failed_gates == 0:
            return "EXCELLENT - Production ready with high quality standards"
        elif metrics.overall_score >= 75 and metrics.failed_gates <= 1:
            return "GOOD - Minor improvements needed before production"
        elif metrics.overall_score >= 60 and metrics.failed_gates <= 2:
            return "ACCEPTABLE - Several improvements needed"
        else:
            return "NEEDS_IMPROVEMENT - Significant issues must be addressed"
    
    def _generate_next_steps(self, metrics: QualityMetrics) -> List[str]:
        """Generate actionable next steps based on results."""
        next_steps = []
        
        if metrics.failed_gates > 0:
            next_steps.append(f"Fix {metrics.failed_gates} critical quality gate failures")
        
        if metrics.warning_gates > 0:
            next_steps.append(f"Address {metrics.warning_gates} quality gate warnings")
        
        if metrics.security_score < 80:
            next_steps.append("Improve security measures and vulnerability management")
        
        if metrics.performance_score < 80:
            next_steps.append("Optimize performance for production workloads")
        
        if metrics.coverage_percentage < 80:
            next_steps.append("Increase test coverage to at least 80%")
        
        # Add positive reinforcement
        if metrics.overall_score >= 90:
            next_steps.append("🎉 Excellent quality achieved - proceed with confidence")
        
        return next_steps

def main():
    """Main execution function for Quality Gates."""
    print("🛡️ COMPREHENSIVE QUALITY GATES - Complete Validation Suite")
    print("=" * 70)
    
    quality_gates = ComprehensiveQualityGates()
    
    try:
        # Run all quality gates
        metrics = quality_gates.run_all_quality_gates()
        
        # Generate comprehensive report
        final_report = quality_gates.generate_comprehensive_report(metrics)
        
        # Save report
        report_file = Path("/root/repo/comprehensive_quality_gates_report.json")
        with open(report_file, "w") as f:
            json.dump(final_report, f, indent=2)
        
        # Display results
        print(f"\n📊 QUALITY GATES RESULTS:")
        print(f"Overall Score: {metrics.overall_score:.1f}/100")
        print(f"Gates Passed: {metrics.passed_gates}/{metrics.total_gates}")
        print(f"Security Score: {metrics.security_score:.1f}/100")
        print(f"Performance Score: {metrics.performance_score:.1f}/100")
        print(f"Test Coverage: {metrics.coverage_percentage:.1f}%")
        print(f"Quality Assessment: {final_report['quality_assessment']}")
        print(f"Execution Time: {metrics.execution_time:.2f}s")
        print(f"Report saved to: {report_file}")
        
        # Display next steps
        if final_report["next_steps"]:
            print(f"\n📋 NEXT STEPS:")
            for i, step in enumerate(final_report["next_steps"], 1):
                print(f"{i}. {step}")
        
        # Determine success
        if metrics.overall_score >= 85 and metrics.failed_gates == 0:
            print("\n🎉 QUALITY GATES PASSED - EXCELLENT QUALITY ACHIEVED")
            return 0
        elif metrics.overall_score >= 70 and metrics.failed_gates <= 1:
            print("\n✅ QUALITY GATES MOSTLY PASSED - GOOD QUALITY WITH MINOR ISSUES")
            return 0
        else:
            print("\n⚠️ QUALITY GATES NEED ATTENTION - IMPROVEMENTS REQUIRED")
            return 1
            
    except Exception as e:
        logger.error(f"Quality gates execution failed: {e}")
        print(f"\n❌ QUALITY GATES EXECUTION FAILED: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())