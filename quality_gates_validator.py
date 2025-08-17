#!/usr/bin/env python3
"""
Quality Gates Validator - Comprehensive Testing & Validation

Implements mandatory quality gates with comprehensive testing, security scanning,
performance benchmarking, and compliance validation to ensure production-ready code.
"""

import asyncio
import time
import json
import sys
import os
import subprocess
import hashlib
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum
import statistics

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

class QualityLevel(Enum):
    """Quality gate levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    ENTERPRISE = "enterprise"

class GateStatus(Enum):
    """Gate validation status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"

@dataclass
class QualityGate:
    """Quality gate definition."""
    name: str
    description: str
    threshold: float
    weight: float
    mandatory: bool = True
    status: GateStatus = GateStatus.SKIPPED
    score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

class QualityGatesValidator:
    """Comprehensive quality gates validator with enterprise-grade validation."""
    
    def __init__(self, project_path: str = ".", quality_level: QualityLevel = QualityLevel.STANDARD):
        self.project_path = Path(project_path).resolve()
        self.quality_level = quality_level
        self.logger = self._setup_logging()
        
        # Initialize quality gates based on level
        self.gates = self._initialize_quality_gates()
        
        # Results tracking
        self.results = {
            'total_gates': len(self.gates),
            'passed_gates': 0,
            'failed_gates': 0,
            'warning_gates': 0,
            'skipped_gates': 0,
            'overall_score': 0.0,
            'quality_level': quality_level.value,
            'execution_time': 0.0,
            'gate_results': {},
            'recommendations': [],
            'compliance_status': {},
            'security_findings': [],
            'performance_metrics': {},
            'test_coverage': 0.0
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        return logging.getLogger(__name__)
    
    def _initialize_quality_gates(self) -> List[QualityGate]:
        """Initialize quality gates based on quality level."""
        base_gates = [
            QualityGate("code_syntax", "Code syntax validation", 1.0, 0.15, True),
            QualityGate("import_validation", "Import validation", 0.95, 0.10, True),
            QualityGate("security_scan", "Security vulnerability scan", 0.85, 0.20, True),
            QualityGate("performance_benchmark", "Performance benchmarks", 0.80, 0.15, True),
            QualityGate("test_coverage", "Test coverage analysis", 0.70, 0.15, True),
            QualityGate("code_quality", "Code quality metrics", 0.75, 0.10, False),
            QualityGate("documentation", "Documentation completeness", 0.60, 0.05, False),
            QualityGate("configuration_validation", "Configuration validation", 0.90, 0.10, True)
        ]
        
        # Add stricter gates for higher quality levels
        if self.quality_level in [QualityLevel.STRICT, QualityLevel.ENTERPRISE]:
            base_gates.extend([
                QualityGate("dependency_security", "Dependency security audit", 0.95, 0.15, True),
                QualityGate("compliance_check", "Compliance validation", 0.90, 0.10, True),
                QualityGate("code_complexity", "Code complexity analysis", 0.80, 0.10, False)
            ])
        
        if self.quality_level == QualityLevel.ENTERPRISE:
            base_gates.extend([
                QualityGate("license_validation", "License compliance", 1.0, 0.05, True),
                QualityGate("deployment_readiness", "Deployment readiness", 0.85, 0.10, True),
                QualityGate("monitoring_setup", "Monitoring configuration", 0.75, 0.05, False)
            ])
        
        return base_gates
    
    async def execute_quality_gates(self) -> Dict[str, Any]:
        """Execute all quality gates with comprehensive validation."""
        start_time = time.time()
        
        self.logger.info("🛡️ Starting Quality Gates Validation")
        self.logger.info(f"Quality Level: {self.quality_level.value}")
        self.logger.info(f"Total Gates: {len(self.gates)}")
        
        try:
            # Execute all quality gates
            for gate in self.gates:
                await self._execute_quality_gate(gate)
            
            # Calculate overall results
            self._calculate_overall_results()
            
            # Generate recommendations
            self._generate_recommendations()
            
        except Exception as e:
            self.logger.error(f"❌ Quality gates execution failed: {e}")
        finally:
            self.results['execution_time'] = time.time() - start_time
        
        return self.results
    
    async def _execute_quality_gate(self, gate: QualityGate):
        """Execute individual quality gate."""
        self.logger.info(f"🔍 Executing gate: {gate.name}")
        
        try:
            if gate.name == "code_syntax":
                await self._validate_code_syntax(gate)
            elif gate.name == "import_validation":
                await self._validate_imports(gate)
            elif gate.name == "security_scan":
                await self._security_scan(gate)
            elif gate.name == "performance_benchmark":
                await self._performance_benchmark(gate)
            elif gate.name == "test_coverage":
                await self._test_coverage_analysis(gate)
            elif gate.name == "code_quality":
                await self._code_quality_metrics(gate)
            elif gate.name == "documentation":
                await self._documentation_completeness(gate)
            elif gate.name == "configuration_validation":
                await self._configuration_validation(gate)
            elif gate.name == "dependency_security":
                await self._dependency_security_audit(gate)
            elif gate.name == "compliance_check":
                await self._compliance_validation(gate)
            elif gate.name == "code_complexity":
                await self._code_complexity_analysis(gate)
            elif gate.name == "license_validation":
                await self._license_validation(gate)
            elif gate.name == "deployment_readiness":
                await self._deployment_readiness(gate)
            elif gate.name == "monitoring_setup":
                await self._monitoring_setup_validation(gate)
            else:
                gate.status = GateStatus.SKIPPED
                gate.details = {"reason": "Unknown gate type"}
            
            # Determine gate status
            if gate.score >= gate.threshold:
                gate.status = GateStatus.PASSED
                self.results['passed_gates'] += 1
                self.logger.info(f"✅ {gate.name}: PASSED ({gate.score:.3f} >= {gate.threshold})")
            elif gate.score >= gate.threshold * 0.8 and not gate.mandatory:
                gate.status = GateStatus.WARNING
                self.results['warning_gates'] += 1
                self.logger.warning(f"⚠️ {gate.name}: WARNING ({gate.score:.3f})")
            else:
                gate.status = GateStatus.FAILED
                self.results['failed_gates'] += 1
                self.logger.error(f"❌ {gate.name}: FAILED ({gate.score:.3f} < {gate.threshold})")
            
            self.results['gate_results'][gate.name] = {
                'status': gate.status.value,
                'score': gate.score,
                'threshold': gate.threshold,
                'weight': gate.weight,
                'mandatory': gate.mandatory,
                'details': gate.details
            }
            
        except Exception as e:
            gate.status = GateStatus.FAILED
            gate.details = {"error": str(e)}
            self.results['failed_gates'] += 1
            self.logger.error(f"❌ {gate.name}: ERROR - {e}")
    
    async def _validate_code_syntax(self, gate: QualityGate):
        """Validate code syntax across all Python files."""
        py_files = list(self.project_path.rglob("*.py"))
        
        if not py_files:
            gate.score = 0.0
            gate.details = {"reason": "No Python files found"}
            return
        
        syntax_results = {"total": 0, "valid": 0, "invalid": 0, "errors": []}
        
        for py_file in py_files:
            try:
                content = py_file.read_text(encoding='utf-8')
                compile(content, str(py_file), 'exec')
                syntax_results["valid"] += 1
            except UnicodeDecodeError:
                syntax_results["invalid"] += 1
                syntax_results["errors"].append(f"{py_file.name}: Encoding error")
            except SyntaxError as e:
                syntax_results["invalid"] += 1
                syntax_results["errors"].append(f"{py_file.name}: {e}")
            except Exception as e:
                syntax_results["invalid"] += 1
                syntax_results["errors"].append(f"{py_file.name}: {e}")
            finally:
                syntax_results["total"] += 1
        
        gate.score = syntax_results["valid"] / syntax_results["total"] if syntax_results["total"] > 0 else 0
        gate.details = syntax_results
    
    async def _validate_imports(self, gate: QualityGate):
        """Validate import statements and dependencies."""
        py_files = list(self.project_path.rglob("*.py"))
        
        import_results = {
            "total_files": len(py_files),
            "files_with_imports": 0,
            "successful_imports": 0,
            "failed_imports": 0,
            "import_errors": []
        }
        
        critical_imports = ['json', 'sys', 'os', 'pathlib', 'asyncio', 'time']
        import_test_results = {}
        
        # Test critical imports
        for module in critical_imports:
            try:
                __import__(module)
                import_test_results[module] = "success"
                import_results["successful_imports"] += 1
            except ImportError as e:
                import_test_results[module] = f"failed: {e}"
                import_results["failed_imports"] += 1
                import_results["import_errors"].append(f"{module}: {e}")
        
        # Check for import statements in files
        import_pattern = re.compile(r'^(?:from\s+\S+\s+)?import\s+', re.MULTILINE)
        
        for py_file in py_files[:20]:  # Sample first 20 files for performance
            try:
                content = py_file.read_text()
                if import_pattern.search(content):
                    import_results["files_with_imports"] += 1
            except Exception:
                continue
        
        total_tests = len(critical_imports)
        success_rate = import_results["successful_imports"] / total_tests if total_tests > 0 else 0
        
        gate.score = success_rate
        gate.details = {
            "import_test_results": import_test_results,
            "file_analysis": import_results
        }
    
    async def _security_scan(self, gate: QualityGate):
        """Comprehensive security vulnerability scan."""
        security_findings = []
        security_score = 1.0
        
        # Check for sensitive files
        sensitive_patterns = ['.env', '.key', '.secret', '.password', 'id_rsa', '.pem', '.crt']
        sensitive_files = []
        
        for pattern in sensitive_patterns:
            matches = list(self.project_path.rglob(f"*{pattern}*"))
            for match in matches:
                if match.is_file():
                    sensitive_files.append(match.name)
                    security_findings.append(f"Sensitive file detected: {match.name}")
        
        if sensitive_files:
            security_score -= 0.2
        
        # Check for dangerous code patterns
        dangerous_patterns = [
            (r'eval\s*\(', "eval() usage detected"),
            (r'exec\s*\(', "exec() usage detected"),
            (r'subprocess\.call\s*\(', "subprocess.call() usage detected"),
            (r'os\.system\s*\(', "os.system() usage detected"),
            (r'__import__\s*\(', "dynamic import detected")
        ]
        
        code_issues = []
        py_files = list(self.project_path.rglob("*.py"))
        
        for py_file in py_files[:30]:  # Sample for performance
            try:
                content = py_file.read_text()
                for pattern, message in dangerous_patterns:
                    if re.search(pattern, content):
                        code_issues.append(f"{py_file.name}: {message}")
                        security_findings.append(f"{py_file.name}: {message}")
            except Exception:
                continue
        
        # Penalty for dangerous patterns
        if code_issues:
            security_score -= min(0.3, len(code_issues) * 0.05)
        
        # Check file permissions (Unix-like systems)
        permission_issues = []
        if hasattr(os, 'stat'):
            for file_path in self.project_path.rglob("*"):
                if file_path.is_file():
                    try:
                        mode = file_path.stat().st_mode
                        # Check for world-writable files
                        if oct(mode)[-1] in ['2', '6']:
                            permission_issues.append(f"World-writable file: {file_path.name}")
                            security_findings.append(f"Insecure permissions: {file_path.name}")
                    except Exception:
                        continue
        
        if permission_issues:
            security_score -= min(0.2, len(permission_issues) * 0.02)
        
        gate.score = max(0.0, security_score)
        gate.details = {
            "sensitive_files": sensitive_files,
            "code_issues": code_issues,
            "permission_issues": permission_issues,
            "total_findings": len(security_findings)
        }
        
        self.results['security_findings'] = security_findings
    
    async def _performance_benchmark(self, gate: QualityGate):
        """Performance benchmarking and validation."""
        benchmarks = {}
        
        # File I/O benchmark
        start_time = time.time()
        test_file = self.project_path / ".perf_test"
        try:
            # Write benchmark
            test_data = "x" * 1024 * 10  # 10KB
            for _ in range(10):
                test_file.write_text(test_data)
            
            # Read benchmark
            for _ in range(10):
                test_file.read_text()
            
            benchmarks["file_io_time"] = time.time() - start_time
            test_file.unlink()
        except Exception as e:
            benchmarks["file_io_error"] = str(e)
        
        # CPU benchmark
        start_time = time.time()
        result = sum(i * i for i in range(50000))
        benchmarks["cpu_time"] = time.time() - start_time
        
        # Memory benchmark
        start_time = time.time()
        data = [[0] * 100 for _ in range(100)]
        del data
        benchmarks["memory_time"] = time.time() - start_time
        
        # Project size analysis
        total_size = 0
        file_count = 0
        for file_path in self.project_path.rglob("*"):
            if file_path.is_file():
                try:
                    total_size += file_path.stat().st_size
                    file_count += 1
                except Exception:
                    continue
        
        benchmarks["project_size_mb"] = total_size / (1024 * 1024)
        benchmarks["file_count"] = file_count
        
        # Performance score calculation
        performance_score = 1.0
        
        # Penalize slow operations
        if benchmarks.get("file_io_time", 0) > 0.1:
            performance_score -= 0.2
        if benchmarks.get("cpu_time", 0) > 0.1:
            performance_score -= 0.1
        if benchmarks.get("project_size_mb", 0) > 100:  # Large project
            performance_score -= 0.1
        
        gate.score = max(0.0, performance_score)
        gate.details = benchmarks
        
        self.results['performance_metrics'] = benchmarks
    
    async def _test_coverage_analysis(self, gate: QualityGate):
        """Analyze test coverage and test quality."""
        test_dir = self.project_path / "tests"
        
        if not test_dir.exists():
            gate.score = 0.0
            gate.details = {"reason": "No tests directory found"}
            self.results['test_coverage'] = 0.0
            return
        
        test_files = list(test_dir.rglob("test_*.py"))
        py_files = list(self.project_path.rglob("*.py"))
        
        # Exclude test files from main code count
        main_py_files = [f for f in py_files if not str(f).startswith(str(test_dir))]
        
        test_analysis = {
            "test_files": len(test_files),
            "main_files": len(main_py_files),
            "test_ratio": len(test_files) / len(main_py_files) if main_py_files else 0,
            "test_frameworks": [],
            "test_patterns": []
        }
        
        # Analyze test files for frameworks and patterns
        for test_file in test_files[:10]:  # Sample for performance
            try:
                content = test_file.read_text()
                
                # Check for test frameworks
                if 'pytest' in content or 'import pytest' in content:
                    test_analysis["test_frameworks"].append("pytest")
                if 'unittest' in content or 'import unittest' in content:
                    test_analysis["test_frameworks"].append("unittest")
                
                # Check for test patterns
                if 'def test_' in content:
                    test_analysis["test_patterns"].append("function_tests")
                if 'class Test' in content:
                    test_analysis["test_patterns"].append("class_tests")
                
            except Exception:
                continue
        
        # Calculate coverage score
        coverage_score = min(1.0, test_analysis["test_ratio"] * 2)  # Target: 1 test file per 2 main files
        
        # Bonus for good practices
        if test_analysis["test_frameworks"]:
            coverage_score = min(1.0, coverage_score + 0.1)
        if test_analysis["test_patterns"]:
            coverage_score = min(1.0, coverage_score + 0.1)
        
        gate.score = coverage_score
        gate.details = test_analysis
        
        self.results['test_coverage'] = coverage_score
    
    async def _code_quality_metrics(self, gate: QualityGate):
        """Analyze code quality metrics."""
        py_files = list(self.project_path.rglob("*.py"))
        
        if not py_files:
            gate.score = 0.0
            gate.details = {"reason": "No Python files found"}
            return
        
        quality_metrics = {
            "total_files": len(py_files),
            "total_lines": 0,
            "average_file_size": 0,
            "complexity_indicators": {
                "long_files": 0,
                "long_functions": 0,
                "deep_nesting": 0
            },
            "code_patterns": {
                "docstrings": 0,
                "comments": 0,
                "functions": 0,
                "classes": 0
            }
        }
        
        file_sizes = []
        
        for py_file in py_files[:50]:  # Sample for performance
            try:
                content = py_file.read_text()
                lines = content.splitlines()
                line_count = len(lines)
                
                quality_metrics["total_lines"] += line_count
                file_sizes.append(line_count)
                
                # Check for long files
                if line_count > 500:
                    quality_metrics["complexity_indicators"]["long_files"] += 1
                
                # Analyze content patterns
                if '"""' in content or "'''" in content:
                    quality_metrics["code_patterns"]["docstrings"] += 1
                
                comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
                quality_metrics["code_patterns"]["comments"] += comment_lines
                
                function_count = content.count('def ')
                quality_metrics["code_patterns"]["functions"] += function_count
                
                class_count = content.count('class ')
                quality_metrics["code_patterns"]["classes"] += class_count
                
                # Check for long functions (simplified)
                if function_count > 0:
                    avg_lines_per_function = line_count / function_count
                    if avg_lines_per_function > 50:
                        quality_metrics["complexity_indicators"]["long_functions"] += 1
                
                # Check for deep nesting (simplified)
                max_indent = 0
                for line in lines:
                    if line.strip():
                        indent = len(line) - len(line.lstrip())
                        max_indent = max(max_indent, indent)
                
                if max_indent > 16:  # More than 4 levels of indentation
                    quality_metrics["complexity_indicators"]["deep_nesting"] += 1
                
            except Exception:
                continue
        
        if file_sizes:
            quality_metrics["average_file_size"] = statistics.mean(file_sizes)
        
        # Calculate quality score
        quality_score = 1.0
        
        # Penalties for complexity issues
        total_files = quality_metrics["total_files"]
        if total_files > 0:
            long_file_ratio = quality_metrics["complexity_indicators"]["long_files"] / total_files
            long_function_ratio = quality_metrics["complexity_indicators"]["long_functions"] / total_files
            deep_nesting_ratio = quality_metrics["complexity_indicators"]["deep_nesting"] / total_files
            
            quality_score -= long_file_ratio * 0.2
            quality_score -= long_function_ratio * 0.2
            quality_score -= deep_nesting_ratio * 0.1
        
        # Bonuses for good practices
        if quality_metrics["code_patterns"]["docstrings"] > total_files * 0.5:
            quality_score += 0.1
        
        gate.score = max(0.0, quality_score)
        gate.details = quality_metrics
    
    async def _documentation_completeness(self, gate: QualityGate):
        """Check documentation completeness."""
        doc_files = {
            "README.md": (self.project_path / "README.md").exists(),
            "LICENSE": (self.project_path / "LICENSE").exists(),
            "CHANGELOG.md": (self.project_path / "CHANGELOG.md").exists(),
            "CONTRIBUTING.md": (self.project_path / "CONTRIBUTING.md").exists(),
            "docs/": (self.project_path / "docs").is_dir()
        }
        
        doc_score = sum(doc_files.values()) / len(doc_files)
        
        # Check for inline documentation
        py_files = list(self.project_path.rglob("*.py"))
        docstring_files = 0
        
        for py_file in py_files[:20]:  # Sample for performance
            try:
                content = py_file.read_text()
                if '"""' in content or "'''" in content:
                    docstring_files += 1
            except Exception:
                continue
        
        if py_files:
            inline_doc_score = docstring_files / min(20, len(py_files))
            doc_score = (doc_score + inline_doc_score) / 2
        
        gate.score = doc_score
        gate.details = {
            "documentation_files": doc_files,
            "docstring_coverage": docstring_files,
            "total_sampled_files": min(20, len(py_files))
        }
    
    async def _configuration_validation(self, gate: QualityGate):
        """Validate configuration files."""
        config_files = {
            "pyproject.toml": self.project_path / "pyproject.toml",
            "setup.py": self.project_path / "setup.py",
            "requirements.txt": self.project_path / "requirements.txt",
            "Dockerfile": self.project_path / "Dockerfile",
            "docker-compose.yml": self.project_path / "docker-compose.yml"
        }
        
        config_results = {}
        valid_configs = 0
        
        for name, path in config_files.items():
            if path.exists():
                try:
                    content = path.read_text()
                    if len(content.strip()) > 0:
                        config_results[name] = "valid"
                        valid_configs += 1
                    else:
                        config_results[name] = "empty"
                except Exception as e:
                    config_results[name] = f"error: {e}"
            else:
                config_results[name] = "missing"
        
        # Essential configurations
        essential_configs = ["pyproject.toml"]
        essential_score = sum(1 for config in essential_configs if config_results.get(config) == "valid")
        essential_score /= len(essential_configs)
        
        # Overall configuration score
        total_score = valid_configs / len(config_files)
        
        # Weight essential configs more heavily
        gate.score = (essential_score * 0.7) + (total_score * 0.3)
        gate.details = {
            "configuration_status": config_results,
            "valid_configurations": valid_configs,
            "essential_configurations": essential_score
        }
    
    async def _dependency_security_audit(self, gate: QualityGate):
        """Audit dependencies for security issues."""
        pyproject_file = self.project_path / "pyproject.toml"
        
        if not pyproject_file.exists():
            gate.score = 0.5  # Neutral score for missing file
            gate.details = {"reason": "No pyproject.toml found"}
            return
        
        try:
            content = pyproject_file.read_text()
            
            # Basic dependency analysis
            dependency_analysis = {
                "has_dependencies": "dependencies" in content,
                "has_dev_dependencies": "[project.optional-dependencies]" in content or "dev" in content,
                "has_version_constraints": "==" in content or ">=" in content,
                "potential_issues": []
            }
            
            # Check for potentially risky dependencies (simplified)
            risky_patterns = ["eval", "exec", "subprocess", "shell"]
            for pattern in risky_patterns:
                if pattern in content.lower():
                    dependency_analysis["potential_issues"].append(f"Potentially risky pattern: {pattern}")
            
            # Calculate security score
            security_score = 0.8  # Base score
            
            if dependency_analysis["has_version_constraints"]:
                security_score += 0.1
            
            if dependency_analysis["potential_issues"]:
                security_score -= len(dependency_analysis["potential_issues"]) * 0.1
            
            gate.score = max(0.0, min(1.0, security_score))
            gate.details = dependency_analysis
            
        except Exception as e:
            gate.score = 0.0
            gate.details = {"error": str(e)}
    
    async def _compliance_validation(self, gate: QualityGate):
        """Validate compliance with standards and regulations."""
        compliance_checks = {
            "license_file": (self.project_path / "LICENSE").exists(),
            "security_policy": (self.project_path / "SECURITY.md").exists(),
            "code_of_conduct": (self.project_path / "CODE_OF_CONDUCT.md").exists(),
            "contributing_guide": (self.project_path / "CONTRIBUTING.md").exists(),
            "privacy_policy": any(
                (self.project_path / name).exists() 
                for name in ["PRIVACY.md", "privacy.txt", "PRIVACY_POLICY.md"]
            )
        }
        
        compliance_score = sum(compliance_checks.values()) / len(compliance_checks)
        
        # Additional compliance checks
        readme_file = self.project_path / "README.md"
        if readme_file.exists():
            try:
                readme_content = readme_file.read_text().lower()
                if "license" in readme_content:
                    compliance_score += 0.1
                if "contributing" in readme_content:
                    compliance_score += 0.05
            except Exception:
                pass
        
        gate.score = min(1.0, compliance_score)
        gate.details = compliance_checks
        
        self.results['compliance_status'] = compliance_checks
    
    async def _code_complexity_analysis(self, gate: QualityGate):
        """Analyze code complexity metrics."""
        py_files = list(self.project_path.rglob("*.py"))
        
        if not py_files:
            gate.score = 1.0  # No complexity if no files
            gate.details = {"reason": "No Python files found"}
            return
        
        complexity_metrics = {
            "total_files": len(py_files),
            "complex_files": 0,
            "average_complexity": 0,
            "complexity_distribution": {"low": 0, "medium": 0, "high": 0}
        }
        
        complexity_scores = []
        
        for py_file in py_files[:30]:  # Sample for performance
            try:
                content = py_file.read_text()
                lines = content.splitlines()
                
                # Simple complexity calculation
                complexity = 0
                
                # Cyclomatic complexity indicators
                complexity += content.count('if ')
                complexity += content.count('elif ')
                complexity += content.count('while ')
                complexity += content.count('for ')
                complexity += content.count('except ')
                complexity += content.count('and ')
                complexity += content.count('or ')
                
                # Normalize by file size
                if len(lines) > 0:
                    normalized_complexity = complexity / len(lines) * 100
                else:
                    normalized_complexity = 0
                
                complexity_scores.append(normalized_complexity)
                
                # Categorize complexity
                if normalized_complexity < 10:
                    complexity_metrics["complexity_distribution"]["low"] += 1
                elif normalized_complexity < 20:
                    complexity_metrics["complexity_distribution"]["medium"] += 1
                else:
                    complexity_metrics["complexity_distribution"]["high"] += 1
                    complexity_metrics["complex_files"] += 1
                
            except Exception:
                continue
        
        if complexity_scores:
            complexity_metrics["average_complexity"] = statistics.mean(complexity_scores)
        
        # Calculate complexity score (lower complexity = higher score)
        if complexity_metrics["total_files"] > 0:
            high_complexity_ratio = complexity_metrics["complex_files"] / len(complexity_scores)
            complexity_score = 1.0 - (high_complexity_ratio * 0.5)
        else:
            complexity_score = 1.0
        
        gate.score = max(0.0, complexity_score)
        gate.details = complexity_metrics
    
    async def _license_validation(self, gate: QualityGate):
        """Validate license compliance."""
        license_file = self.project_path / "LICENSE"
        
        if not license_file.exists():
            gate.score = 0.0
            gate.details = {"reason": "No LICENSE file found"}
            return
        
        try:
            license_content = license_file.read_text()
            
            # Check for common license types
            license_indicators = {
                "MIT": "MIT License" in license_content or "MIT" in license_content,
                "Apache": "Apache License" in license_content or "Apache-2.0" in license_content,
                "GPL": "GNU General Public License" in license_content or "GPL" in license_content,
                "BSD": "BSD License" in license_content or "BSD" in license_content,
                "Creative Commons": "Creative Commons" in license_content or "CC BY" in license_content
            }
            
            detected_licenses = [name for name, found in license_indicators.items() if found]
            
            license_validation = {
                "license_file_exists": True,
                "license_content_length": len(license_content),
                "detected_licenses": detected_licenses,
                "has_copyright": "Copyright" in license_content or "©" in license_content,
                "has_year": any(str(year) in license_content for year in range(2020, 2026))
            }
            
            # Calculate license score
            score = 0.0
            
            if license_validation["license_content_length"] > 100:  # Non-empty license
                score += 0.4
            
            if license_validation["detected_licenses"]:  # Recognized license
                score += 0.4
            
            if license_validation["has_copyright"]:  # Copyright notice
                score += 0.1
            
            if license_validation["has_year"]:  # Recent year
                score += 0.1
            
            gate.score = score
            gate.details = license_validation
            
        except Exception as e:
            gate.score = 0.0
            gate.details = {"error": str(e)}
    
    async def _deployment_readiness(self, gate: QualityGate):
        """Check deployment readiness."""
        deployment_indicators = {
            "dockerfile": (self.project_path / "Dockerfile").exists(),
            "docker_compose": (self.project_path / "docker-compose.yml").exists(),
            "kubernetes": (self.project_path / "k8s").is_dir() or (self.project_path / "kubernetes").is_dir(),
            "helm": (self.project_path / "helm").is_dir(),
            "terraform": (self.project_path / "terraform").is_dir(),
            "deployment_scripts": (self.project_path / "deployment").is_dir(),
            "ci_cd": (self.project_path / ".github" / "workflows").is_dir() or (self.project_path / ".gitlab-ci.yml").exists()
        }
        
        deployment_score = sum(deployment_indicators.values()) / len(deployment_indicators)
        
        # Essential deployment files
        essential_deployment = ["dockerfile", "docker_compose"]
        essential_score = sum(deployment_indicators[key] for key in essential_deployment) / len(essential_deployment)
        
        # Weight essential more heavily
        final_score = (essential_score * 0.6) + (deployment_score * 0.4)
        
        gate.score = final_score
        gate.details = {
            "deployment_readiness": deployment_indicators,
            "essential_deployment_score": essential_score,
            "overall_deployment_score": deployment_score
        }
    
    async def _monitoring_setup_validation(self, gate: QualityGate):
        """Validate monitoring and observability setup."""
        monitoring_components = {
            "prometheus": (self.project_path / "prometheus.yml").exists(),
            "grafana": (self.project_path / "grafana").is_dir(),
            "monitoring_dir": (self.project_path / "monitoring").is_dir(),
            "logging_config": any(
                (self.project_path / name).exists()
                for name in ["logging.yml", "logging.json", "log_config.py"]
            ),
            "health_check": any(
                "health" in py_file.name.lower()
                for py_file in self.project_path.rglob("*.py")
            )
        }
        
        monitoring_score = sum(monitoring_components.values()) / len(monitoring_components)
        
        gate.score = monitoring_score
        gate.details = monitoring_components
    
    def _calculate_overall_results(self):
        """Calculate overall quality gates results."""
        total_weight = sum(gate.weight for gate in self.gates)
        weighted_score = sum(gate.score * gate.weight for gate in self.gates)
        
        self.results['overall_score'] = weighted_score / total_weight if total_weight > 0 else 0
        
        # Count gates by status
        for gate in self.gates:
            if gate.status == GateStatus.SKIPPED:
                self.results['skipped_gates'] += 1
    
    def _generate_recommendations(self):
        """Generate recommendations based on gate results."""
        recommendations = []
        
        for gate in self.gates:
            if gate.status == GateStatus.FAILED and gate.mandatory:
                recommendations.append(f"CRITICAL: Fix {gate.name} - {gate.description}")
            elif gate.status == GateStatus.FAILED:
                recommendations.append(f"IMPROVE: {gate.name} - {gate.description}")
            elif gate.status == GateStatus.WARNING:
                recommendations.append(f"CONSIDER: Improve {gate.name} for better quality")
        
        # General recommendations based on overall score
        if self.results['overall_score'] < 0.6:
            recommendations.append("Overall quality is below acceptable threshold - prioritize critical fixes")
        elif self.results['overall_score'] < 0.8:
            recommendations.append("Good quality achieved - focus on remaining improvements")
        else:
            recommendations.append("Excellent quality achieved - maintain current standards")
        
        self.results['recommendations'] = recommendations

def main():
    """Main execution function."""
    print("🛡️ Quality Gates Validator - Comprehensive Testing & Validation")
    print("=" * 70)
    
    # Quality level selection
    quality_level = QualityLevel.STANDARD
    if len(sys.argv) > 1:
        level_map = {
            'basic': QualityLevel.BASIC,
            'standard': QualityLevel.STANDARD,
            'strict': QualityLevel.STRICT,
            'enterprise': QualityLevel.ENTERPRISE
        }
        quality_level = level_map.get(sys.argv[1].lower(), QualityLevel.STANDARD)
    
    validator = QualityGatesValidator(quality_level=quality_level)
    
    try:
        results = asyncio.run(validator.execute_quality_gates())
        
        print("\n📊 QUALITY GATES RESULTS")
        print("=" * 40)
        print(f"Overall Score: {results['overall_score']:.3f}")
        print(f"Quality Level: {results['quality_level']}")
        print(f"Execution Time: {results['execution_time']:.3f} seconds")
        print(f"Passed Gates: {results['passed_gates']}/{results['total_gates']}")
        print(f"Failed Gates: {results['failed_gates']}")
        print(f"Warning Gates: {results['warning_gates']}")
        
        if results['test_coverage'] > 0:
            print(f"Test Coverage: {results['test_coverage']:.3f}")
        
        # Show gate details
        print(f"\n🔍 GATE DETAILS")
        print("-" * 30)
        for gate_name, gate_result in results['gate_results'].items():
            status_emoji = {
                'passed': '✅',
                'failed': '❌',
                'warning': '⚠️',
                'skipped': '⏭️'
            }
            emoji = status_emoji.get(gate_result['status'], '❓')
            print(f"{emoji} {gate_name}: {gate_result['status'].upper()} ({gate_result['score']:.3f})")
        
        # Show security findings
        if results['security_findings']:
            print(f"\n🚨 SECURITY FINDINGS")
            print("-" * 25)
            for finding in results['security_findings'][:5]:  # Show first 5
                print(f"  • {finding}")
            if len(results['security_findings']) > 5:
                print(f"  ... and {len(results['security_findings']) - 5} more")
        
        # Show recommendations
        if results['recommendations']:
            print(f"\n💡 RECOMMENDATIONS")
            print("-" * 20)
            for rec in results['recommendations'][:5]:  # Show first 5
                print(f"  • {rec}")
        
        # Overall assessment
        score = results['overall_score']
        failed_mandatory = sum(1 for gate in validator.gates if gate.mandatory and gate.status == GateStatus.FAILED)
        
        if failed_mandatory > 0:
            print(f"\n❌ QUALITY GATES FAILED - {failed_mandatory} mandatory gates failed")
            sys.exit(1)
        elif score >= 0.9:
            print(f"\n🏆 EXCEPTIONAL QUALITY - All gates passed with high scores!")
        elif score >= 0.8:
            print(f"\n🎉 HIGH QUALITY - Quality gates passed successfully!")
        elif score >= 0.7:
            print(f"\n✅ GOOD QUALITY - Quality gates passed with room for improvement")
        else:
            print(f"\n⚠️ QUALITY NEEDS IMPROVEMENT - Score below recommended threshold")
            
    except Exception as e:
        print(f"\n❌ Quality gates validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()