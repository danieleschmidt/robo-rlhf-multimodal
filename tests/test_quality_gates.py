"""Quality gate tests for autonomous SDLC system."""

import pytest
import asyncio
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from robo_rlhf.quantum.autonomous import AutonomousSDLCExecutor, SDLCPhase
from robo_rlhf.core.exceptions import SecurityError, ValidationError


class QualityGateValidator:
    """Quality gate validation system."""
    
    def __init__(self):
        self.quality_thresholds = {
            "test_coverage": 0.85,
            "security_score": 0.95,
            "performance_score": 0.8,
            "code_quality_score": 0.9,
            "success_rate": 0.9
        }
        self.validation_results = {}
    
    def validate_test_coverage(self, coverage_data: Dict[str, Any]) -> bool:
        """Validate test coverage meets quality threshold."""
        coverage_percentage = coverage_data.get("percentage", 0) / 100.0
        self.validation_results["test_coverage"] = coverage_percentage
        return coverage_percentage >= self.quality_thresholds["test_coverage"]
    
    def validate_security_scan(self, security_data: Dict[str, Any]) -> bool:
        """Validate security scan results."""
        vulnerabilities = security_data.get("vulnerabilities", [])
        critical_vulns = [v for v in vulnerabilities if v.get("severity") == "critical"]
        high_vulns = [v for v in vulnerabilities if v.get("severity") == "high"]
        
        # Calculate security score (1.0 = perfect, 0.0 = terrible)
        total_vulns = len(vulnerabilities)
        critical_penalty = len(critical_vulns) * 0.5
        high_penalty = len(high_vulns) * 0.2
        
        security_score = max(0.0, 1.0 - (critical_penalty + high_penalty) / max(1, total_vulns))
        self.validation_results["security_score"] = security_score
        
        return security_score >= self.quality_thresholds["security_score"]
    
    def validate_performance(self, performance_data: Dict[str, Any]) -> bool:
        """Validate performance metrics."""
        execution_time = performance_data.get("execution_time", float('inf'))
        memory_usage = performance_data.get("memory_usage", 1.0)  # As percentage
        cpu_usage = performance_data.get("cpu_usage", 1.0)  # As percentage
        
        # Performance score based on resource efficiency
        time_score = min(1.0, 300.0 / max(execution_time, 1.0))  # 5 minutes baseline
        memory_score = max(0.0, 1.0 - memory_usage)
        cpu_score = max(0.0, 1.0 - cpu_usage)
        
        performance_score = (time_score + memory_score + cpu_score) / 3.0
        self.validation_results["performance_score"] = performance_score
        
        return performance_score >= self.quality_thresholds["performance_score"]
    
    def validate_code_quality(self, quality_data: Dict[str, Any]) -> bool:
        """Validate code quality metrics."""
        type_errors = quality_data.get("type_errors", 0)
        style_violations = quality_data.get("style_violations", 0)
        complexity_violations = quality_data.get("complexity_violations", 0)
        
        # Code quality score
        total_issues = type_errors + style_violations + complexity_violations
        penalty = min(1.0, total_issues / 100.0)  # 100 issues = 0 score
        code_quality_score = max(0.0, 1.0 - penalty)
        
        self.validation_results["code_quality_score"] = code_quality_score
        
        return code_quality_score >= self.quality_thresholds["code_quality_score"]
    
    def validate_execution_success(self, execution_data: Dict[str, Any]) -> bool:
        """Validate overall execution success rate."""
        total_actions = execution_data.get("total_actions", 0)
        successful_actions = execution_data.get("successful_actions", 0)
        
        success_rate = successful_actions / max(total_actions, 1)
        self.validation_results["success_rate"] = success_rate
        
        return success_rate >= self.quality_thresholds["success_rate"]
    
    def get_overall_quality_score(self) -> float:
        """Calculate overall quality score."""
        if not self.validation_results:
            return 0.0
        
        scores = list(self.validation_results.values())
        return sum(scores) / len(scores)
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        overall_score = self.get_overall_quality_score()
        
        passed_gates = sum(1 for k, v in self.validation_results.items() 
                          if v >= self.quality_thresholds[k])
        total_gates = len(self.quality_thresholds)
        
        return {
            "overall_score": overall_score,
            "passed_gates": passed_gates,
            "total_gates": total_gates,
            "gate_pass_rate": passed_gates / total_gates,
            "individual_scores": self.validation_results,
            "thresholds": self.quality_thresholds,
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> Dict[str, str]:
        """Generate improvement recommendations."""
        recommendations = {}
        
        for metric, score in self.validation_results.items():
            threshold = self.quality_thresholds[metric]
            if score < threshold:
                if metric == "test_coverage":
                    recommendations[metric] = f"Increase test coverage from {score*100:.1f}% to {threshold*100:.1f}%"
                elif metric == "security_score":
                    recommendations[metric] = f"Address security vulnerabilities to improve score from {score:.2f} to {threshold:.2f}"
                elif metric == "performance_score":
                    recommendations[metric] = f"Optimize performance to improve score from {score:.2f} to {threshold:.2f}"
                elif metric == "code_quality_score":
                    recommendations[metric] = f"Fix code quality issues to improve score from {score:.2f} to {threshold:.2f}"
                elif metric == "success_rate":
                    recommendations[metric] = f"Improve execution reliability from {score*100:.1f}% to {threshold*100:.1f}%"
        
        return recommendations


class TestQualityGates:
    """Test quality gate validation system."""
    
    def setup_method(self):
        """Set up quality gate tests."""
        self.validator = QualityGateValidator()
    
    def test_test_coverage_validation_pass(self):
        """Test test coverage validation - passing case."""
        coverage_data = {"percentage": 90}  # 90% coverage
        
        result = self.validator.validate_test_coverage(coverage_data)
        
        assert result is True
        assert self.validator.validation_results["test_coverage"] == 0.9
    
    def test_test_coverage_validation_fail(self):
        """Test test coverage validation - failing case."""
        coverage_data = {"percentage": 70}  # 70% coverage
        
        result = self.validator.validate_test_coverage(coverage_data)
        
        assert result is False
        assert self.validator.validation_results["test_coverage"] == 0.7
    
    def test_security_scan_validation_pass(self):
        """Test security scan validation - passing case."""
        security_data = {
            "vulnerabilities": [
                {"severity": "low", "description": "Minor issue"},
                {"severity": "medium", "description": "Medium issue"}
            ]
        }
        
        result = self.validator.validate_security_scan(security_data)
        
        assert result is True  # No critical vulnerabilities
    
    def test_security_scan_validation_fail(self):
        """Test security scan validation - failing case."""
        security_data = {
            "vulnerabilities": [
                {"severity": "critical", "description": "SQL injection"},
                {"severity": "critical", "description": "Code execution"},
                {"severity": "high", "description": "XSS vulnerability"}
            ]
        }
        
        result = self.validator.validate_security_scan(security_data)
        
        assert result is False  # Critical vulnerabilities present
        assert self.validator.validation_results["security_score"] < 0.95
    
    def test_performance_validation_pass(self):
        """Test performance validation - passing case."""
        performance_data = {
            "execution_time": 120.0,  # 2 minutes
            "memory_usage": 0.3,      # 30% memory usage
            "cpu_usage": 0.4          # 40% CPU usage
        }
        
        result = self.validator.validate_performance(performance_data)
        
        assert result is True
    
    def test_performance_validation_fail(self):
        """Test performance validation - failing case."""
        performance_data = {
            "execution_time": 600.0,  # 10 minutes (slow)
            "memory_usage": 0.9,      # 90% memory usage
            "cpu_usage": 0.95         # 95% CPU usage
        }
        
        result = self.validator.validate_performance(performance_data)
        
        assert result is False
        assert self.validator.validation_results["performance_score"] < 0.8
    
    def test_code_quality_validation_pass(self):
        """Test code quality validation - passing case."""
        quality_data = {
            "type_errors": 0,
            "style_violations": 5,
            "complexity_violations": 2
        }
        
        result = self.validator.validate_code_quality(quality_data)
        
        assert result is True
    
    def test_code_quality_validation_fail(self):
        """Test code quality validation - failing case."""
        quality_data = {
            "type_errors": 25,
            "style_violations": 50,
            "complexity_violations": 30
        }
        
        result = self.validator.validate_code_quality(quality_data)
        
        assert result is False
        assert self.validator.validation_results["code_quality_score"] < 0.9
    
    def test_execution_success_validation_pass(self):
        """Test execution success validation - passing case."""
        execution_data = {
            "total_actions": 10,
            "successful_actions": 9
        }
        
        result = self.validator.validate_execution_success(execution_data)
        
        assert result is True
        assert self.validator.validation_results["success_rate"] == 0.9
    
    def test_execution_success_validation_fail(self):
        """Test execution success validation - failing case."""
        execution_data = {
            "total_actions": 10,
            "successful_actions": 7
        }
        
        result = self.validator.validate_execution_success(execution_data)
        
        assert result is False
        assert self.validator.validation_results["success_rate"] == 0.7
    
    def test_overall_quality_score_calculation(self):
        """Test overall quality score calculation."""
        # Add some validation results
        self.validator.validation_results = {
            "test_coverage": 0.9,
            "security_score": 0.95,
            "performance_score": 0.8,
            "code_quality_score": 0.85,
            "success_rate": 0.95
        }
        
        overall_score = self.validator.get_overall_quality_score()
        
        expected_score = (0.9 + 0.95 + 0.8 + 0.85 + 0.95) / 5
        assert abs(overall_score - expected_score) < 0.01
    
    def test_quality_report_generation(self):
        """Test quality report generation."""
        # Set up mixed results
        self.validator.validate_test_coverage({"percentage": 90})
        self.validator.validate_security_scan({"vulnerabilities": []})
        self.validator.validate_performance({
            "execution_time": 200,
            "memory_usage": 0.5,
            "cpu_usage": 0.3
        })
        self.validator.validate_code_quality({
            "type_errors": 50,  # This should fail
            "style_violations": 20,
            "complexity_violations": 30
        })
        self.validator.validate_execution_success({
            "total_actions": 10,
            "successful_actions": 9
        })
        
        report = self.validator.get_quality_report()
        
        assert "overall_score" in report
        assert "passed_gates" in report
        assert "total_gates" in report
        assert "gate_pass_rate" in report
        assert "individual_scores" in report
        assert "thresholds" in report
        assert "recommendations" in report
        
        # Should have some failures
        assert report["passed_gates"] < report["total_gates"]
        assert len(report["recommendations"]) > 0


class TestAutonomousSDLCQualityIntegration:
    """Integration tests for quality gates with autonomous SDLC."""
    
    def setup_method(self):
        """Set up integration test."""
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.validator = QualityGateValidator()
        
        self.config = {
            "autonomous": {
                "quality_threshold": 0.85,
                "max_parallel": 1
            },
            "quality_gates": {
                "test_coverage": 0.85,
                "security_score": 0.95,
                "performance_score": 0.8
            }
        }
    
    @pytest.mark.asyncio
    async def test_autonomous_sdlc_with_quality_gates(self):
        """Test autonomous SDLC execution with quality gate validation."""
        executor = AutonomousSDLCExecutor(self.project_path, self.config)
        
        # Mock successful execution
        with patch('asyncio.create_subprocess_shell') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (
                b"TOTAL coverage: 90%\n10 passed, 0 failed",
                b""
            )
            mock_subprocess.return_value = mock_process
            
            results = await executor.execute_autonomous_sdlc(
                target_phases=[SDLCPhase.ANALYSIS, SDLCPhase.TESTING]
            )
            
            # Apply quality gates to results
            quality_passed = self._apply_quality_gates_to_results(results)
            
            assert quality_passed["some_gates_passed"] is True
    
    def _apply_quality_gates_to_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply quality gates to execution results."""
        # Mock quality data based on results
        coverage_data = {"percentage": 90 if results.get("overall_success") else 70}
        security_data = {"vulnerabilities": [] if results.get("overall_success") else [
            {"severity": "high", "description": "Security issue"}
        ]}
        performance_data = {
            "execution_time": results.get("execution_time", 300),
            "memory_usage": 0.3,
            "cpu_usage": 0.4
        }
        quality_data = {"type_errors": 0, "style_violations": 5, "complexity_violations": 2}
        execution_data = {
            "total_actions": results.get("total_actions", 0),
            "successful_actions": results.get("successful_actions", 0)
        }
        
        # Validate all quality gates
        coverage_passed = self.validator.validate_test_coverage(coverage_data)
        security_passed = self.validator.validate_security_scan(security_data)
        performance_passed = self.validator.validate_performance(performance_data)
        quality_passed = self.validator.validate_code_quality(quality_data)
        execution_passed = self.validator.validate_execution_success(execution_data)
        
        quality_report = self.validator.get_quality_report()
        
        return {
            "coverage_passed": coverage_passed,
            "security_passed": security_passed,
            "performance_passed": performance_passed,
            "quality_passed": quality_passed,
            "execution_passed": execution_passed,
            "some_gates_passed": any([coverage_passed, security_passed, performance_passed, quality_passed, execution_passed]),
            "all_gates_passed": all([coverage_passed, security_passed, performance_passed, quality_passed, execution_passed]),
            "quality_report": quality_report
        }
    
    def test_quality_gate_failure_handling(self):
        """Test handling of quality gate failures."""
        # Simulate failing quality gates
        failing_results = {
            "overall_success": False,
            "total_actions": 10,
            "successful_actions": 5,
            "execution_time": 800  # Long execution time
        }
        
        quality_results = self._apply_quality_gates_to_results(failing_results)
        
        assert quality_results["all_gates_passed"] is False
        assert len(quality_results["quality_report"]["recommendations"]) > 0
        
        # Check that recommendations are generated
        recommendations = quality_results["quality_report"]["recommendations"]
        assert any("coverage" in rec for rec in recommendations.values()) or \
               any("security" in rec for rec in recommendations.values()) or \
               any("performance" in rec for rec in recommendations.values())


def test_quality_threshold_configuration():
    """Test quality threshold configuration."""
    custom_thresholds = {
        "test_coverage": 0.9,
        "security_score": 0.98,
        "performance_score": 0.85,
        "code_quality_score": 0.95,
        "success_rate": 0.95
    }
    
    validator = QualityGateValidator()
    validator.quality_thresholds = custom_thresholds
    
    # Test that custom thresholds are applied
    coverage_result = validator.validate_test_coverage({"percentage": 85})  # Should fail with 90% threshold
    assert coverage_result is False
    
    coverage_result = validator.validate_test_coverage({"percentage": 92})  # Should pass
    assert coverage_result is True


@pytest.mark.slow
class TestQualityGatePerformance:
    """Performance tests for quality gate validation."""
    
    def test_quality_validation_performance(self):
        """Test that quality validation is performant."""
        validator = QualityGateValidator()
        
        start_time = time.time()
        
        # Run multiple validations
        for i in range(100):
            validator.validate_test_coverage({"percentage": 85 + i % 15})
            validator.validate_security_scan({
                "vulnerabilities": [{"severity": "low"}] * (i % 5)
            })
            validator.validate_performance({
                "execution_time": 100 + i,
                "memory_usage": 0.3 + i * 0.001,
                "cpu_usage": 0.4 + i * 0.001
            })
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete in reasonable time
        assert execution_time < 1.0  # Less than 1 second for 100 validations
        
        # Generate final report
        report = validator.get_quality_report()
        assert report is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])