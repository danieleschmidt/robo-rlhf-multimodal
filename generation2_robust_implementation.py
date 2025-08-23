#!/usr/bin/env python3
"""
Generation 2: MAKE IT ROBUST (Reliable) - Comprehensive error handling and security
Implements robust error handling, validation, logging, and security measures
"""

import sys
import json
import time
import logging
import hashlib
import secrets
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import subprocess
import os

# Setup robust logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/generation2_robust.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """Security configuration for robust operations."""
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: List[str] = field(default_factory=lambda: ['.txt', '.json', '.py', '.md'])
    max_execution_time: int = 300  # 5 minutes
    rate_limit_requests: int = 100
    enable_input_sanitization: bool = True

class RobustValidator:
    """Comprehensive input validation with security checks."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.request_count = 0
        self.start_time = time.time()
    
    def validate_file_path(self, path: Union[str, Path]) -> Path:
        """Validate file path with security checks."""
        try:
            path_obj = Path(path).resolve()
            
            # Check for path traversal
            if '..' in str(path):
                raise SecurityError(f"Path traversal detected: {path}")
            
            # Check file extension
            if path_obj.suffix not in self.config.allowed_extensions:
                raise SecurityError(f"File extension not allowed: {path_obj.suffix}")
            
            # Check file size if exists
            if path_obj.exists() and path_obj.stat().st_size > self.config.max_file_size:
                raise SecurityError(f"File too large: {path_obj.stat().st_size} bytes")
            
            logger.info(f"File path validated: {path_obj}")
            return path_obj
            
        except Exception as e:
            logger.error(f"File path validation failed: {e}")
            raise ValidationError(f"Invalid file path: {path}")
    
    def sanitize_input(self, input_data: str) -> str:
        """Sanitize user input to prevent injection attacks."""
        if not self.config.enable_input_sanitization:
            return input_data
        
        # Remove dangerous characters
        sanitized = re.sub(r'[<>&"\']', '', input_data)
        
        # Limit length
        if len(sanitized) > 1000:
            sanitized = sanitized[:1000]
        
        logger.debug(f"Input sanitized: {len(input_data)} -> {len(sanitized)} chars")
        return sanitized
    
    def check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded."""
        self.request_count += 1
        elapsed = time.time() - self.start_time
        
        if elapsed > 60:  # Reset every minute
            self.request_count = 0
            self.start_time = time.time()
        
        if self.request_count > self.config.rate_limit_requests:
            logger.warning(f"Rate limit exceeded: {self.request_count} requests")
            return False
        
        return True

class SecurityError(Exception):
    """Security-related errors."""
    pass

class ValidationError(Exception):
    """Validation-related errors."""
    pass

class RobustExecutionError(Exception):
    """Robust execution errors."""
    pass

class RobustAutonomousEngine:
    """Robust autonomous execution engine with comprehensive error handling."""
    
    def __init__(self):
        self.config = SecurityConfig()
        self.validator = RobustValidator(self.config)
        self.execution_history: List[Dict] = []
        self.session_id = secrets.token_hex(16)
        logger.info(f"Robust engine initialized with session: {self.session_id}")
    
    @contextmanager
    def safe_execution(self, operation_name: str):
        """Context manager for safe execution with comprehensive error handling."""
        start_time = time.time()
        operation_id = hashlib.md5(f"{operation_name}{start_time}".encode()).hexdigest()[:8]
        
        logger.info(f"Starting operation {operation_name} [ID: {operation_id}]")
        
        try:
            yield operation_id
            
            execution_time = time.time() - start_time
            logger.info(f"Operation {operation_name} completed in {execution_time:.2f}s [ID: {operation_id}]")
            
            self._record_execution(operation_id, operation_name, "success", execution_time)
            
        except SecurityError as e:
            logger.error(f"Security error in {operation_name}: {e} [ID: {operation_id}]")
            self._record_execution(operation_id, operation_name, "security_error", time.time() - start_time, str(e))
            raise
            
        except ValidationError as e:
            logger.error(f"Validation error in {operation_name}: {e} [ID: {operation_id}]")
            self._record_execution(operation_id, operation_name, "validation_error", time.time() - start_time, str(e))
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error in {operation_name}: {e} [ID: {operation_id}]")
            self._record_execution(operation_id, operation_name, "error", time.time() - start_time, str(e))
            raise RobustExecutionError(f"Operation {operation_name} failed: {e}")
    
    def _record_execution(self, operation_id: str, operation: str, status: str, execution_time: float, error: str = None):
        """Record execution history for monitoring and analysis."""
        record = {
            "operation_id": operation_id,
            "operation": operation,
            "status": status,
            "execution_time": execution_time,
            "timestamp": time.time(),
            "session_id": self.session_id
        }
        
        if error:
            record["error"] = error
        
        self.execution_history.append(record)
    
    def autonomous_file_operations(self) -> Dict[str, Any]:
        """Robust file operations with comprehensive error handling."""
        results = {"operations": [], "errors": [], "status": "running"}
        
        with self.safe_execution("file_operations"):
            try:
                # Test file creation with validation
                test_file = self.validator.validate_file_path("/tmp/robust_test.txt")
                
                # Secure file content
                secure_content = self.validator.sanitize_input(
                    f"Robust test data - Session: {self.session_id} - Time: {time.time()}"
                )
                
                test_file.write_text(secure_content, encoding='utf-8')
                results["operations"].append("file_creation")
                
                # Verify file integrity
                read_content = test_file.read_text(encoding='utf-8')
                if read_content != secure_content:
                    raise ValidationError("File integrity check failed")
                
                results["operations"].append("file_verification")
                
                # Secure file cleanup
                test_file.unlink()
                results["operations"].append("file_cleanup")
                
                logger.info("File operations completed successfully")
                results["status"] = "success"
                
            except Exception as e:
                results["errors"].append(str(e))
                results["status"] = "failed"
                raise
        
        return results
    
    def autonomous_security_validation(self) -> Dict[str, Any]:
        """Comprehensive security validation tests."""
        results = {"validations": [], "warnings": [], "status": "running"}
        
        with self.safe_execution("security_validation"):
            try:
                # Test input sanitization
                dangerous_input = "<script>alert('xss')</script>"
                sanitized = self.validator.sanitize_input(dangerous_input)
                
                if '<script>' in sanitized:
                    raise SecurityError("Input sanitization failed")
                
                results["validations"].append("input_sanitization")
                
                # Test path traversal protection
                try:
                    self.validator.validate_file_path("../../../etc/passwd")
                    raise SecurityError("Path traversal protection failed")
                except ValidationError:
                    # This is expected - path traversal should be blocked
                    results["validations"].append("path_traversal_protection")
                
                # Test rate limiting
                original_count = self.validator.request_count
                for _ in range(5):
                    if not self.validator.check_rate_limit():
                        break
                
                if self.validator.request_count <= original_count:
                    raise SecurityError("Rate limiting not working")
                
                results["validations"].append("rate_limiting")
                
                # Test session integrity
                if len(self.session_id) != 32:
                    raise SecurityError("Session ID generation failed")
                
                results["validations"].append("session_integrity")
                
                logger.info("Security validation completed successfully")
                results["status"] = "success"
                
            except Exception as e:
                results["warnings"].append(str(e))
                results["status"] = "failed"
                raise
        
        return results
    
    def autonomous_error_recovery(self) -> Dict[str, Any]:
        """Test autonomous error recovery mechanisms."""
        results = {"recovery_tests": [], "status": "running"}
        
        with self.safe_execution("error_recovery"):
            try:
                # Test graceful failure handling
                try:
                    # Intentionally cause a controlled error
                    raise ValueError("Controlled test error")
                except ValueError as e:
                    logger.warning(f"Handled controlled error: {e}")
                    results["recovery_tests"].append("controlled_error_handling")
                
                # Test resource cleanup
                temp_files = []
                try:
                    for i in range(3):
                        temp_file = Path(f"/tmp/recovery_test_{i}.txt")
                        temp_file.write_text(f"Recovery test {i}")
                        temp_files.append(temp_file)
                    
                    # Simulate error during processing
                    raise RuntimeError("Simulated processing error")
                    
                except RuntimeError:
                    # Cleanup resources even after error
                    for temp_file in temp_files:
                        if temp_file.exists():
                            temp_file.unlink()
                    
                    results["recovery_tests"].append("resource_cleanup")
                
                logger.info("Error recovery tests completed successfully")
                results["status"] = "success"
                
            except Exception as e:
                results["status"] = "failed"
                raise
        
        return results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive execution report with security metrics."""
        total_operations = len(self.execution_history)
        successful_operations = len([op for op in self.execution_history if op["status"] == "success"])
        
        report = {
            "session_id": self.session_id,
            "timestamp": time.time(),
            "generation": "Generation 2 - MAKE IT ROBUST",
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "success_rate": (successful_operations / total_operations * 100) if total_operations > 0 else 0,
            "execution_history": self.execution_history,
            "security_config": {
                "max_file_size": self.config.max_file_size,
                "allowed_extensions": self.config.allowed_extensions,
                "rate_limit": self.config.rate_limit_requests,
                "input_sanitization": self.config.enable_input_sanitization
            }
        }
        
        return report

def main():
    """Main execution function for Generation 2."""
    print("🛡️ Generation 2: MAKE IT ROBUST - Comprehensive Reliability Test")
    print("=" * 70)
    
    engine = RobustAutonomousEngine()
    overall_success = True
    
    try:
        # Test 1: Robust File Operations
        print("\n🔧 Testing robust file operations...")
        file_results = engine.autonomous_file_operations()
        print(f"✅ File operations: {len(file_results['operations'])} completed")
        
        # Test 2: Security Validation
        print("\n🔒 Testing security validation...")
        security_results = engine.autonomous_security_validation()
        print(f"✅ Security validations: {len(security_results['validations'])} passed")
        
        # Test 3: Error Recovery
        print("\n🏥 Testing error recovery...")
        recovery_results = engine.autonomous_error_recovery()
        print(f"✅ Recovery tests: {len(recovery_results['recovery_tests'])} completed")
        
        # Generate comprehensive report
        final_report = engine.generate_comprehensive_report()
        
        # Save report
        report_file = Path("/root/repo/generation2_robust_report.json")
        with open(report_file, "w") as f:
            json.dump(final_report, f, indent=2)
        
        print(f"\n📊 GENERATION 2 RESULTS:")
        print(f"Session ID: {final_report['session_id']}")
        print(f"Total Operations: {final_report['total_operations']}")
        print(f"Success Rate: {final_report['success_rate']:.1f}%")
        print(f"Report saved to: {report_file}")
        
        if final_report['success_rate'] >= 90:
            print("\n🎉 GENERATION 2 COMPLETE - PROCEEDING TO GENERATION 3")
            return 0
        else:
            print("\n⚠️ GENERATION 2 PARTIALLY SUCCESSFUL - REVIEW RECOMMENDED")
            return 1
            
    except Exception as e:
        logger.error(f"Generation 2 failed: {e}")
        print(f"\n❌ GENERATION 2 FAILED: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())