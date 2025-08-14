"""
Security utilities for robo-rlhf-multimodal.
"""

import os
import re
import hashlib
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

from robo_rlhf.core.exceptions import SecurityError


logger = logging.getLogger(__name__)


# Dangerous patterns to detect
DANGEROUS_PATTERNS = [
    r'<script[^>]*>.*?</script>',  # JavaScript
    r'javascript:',                # JavaScript protocol
    r'vbscript:',                 # VBScript protocol
    r'data:text/html',            # Data URLs with HTML
    r'<iframe[^>]*>',             # Iframes
    r'<object[^>]*>',             # Objects
    r'<embed[^>]*>',              # Embeds
    r'eval\s*\(',                 # eval() calls
    r'exec\s*\(',                 # exec() calls
    r'import\s+os',               # OS imports
    r'import\s+subprocess',        # Subprocess imports
    r'__import__',                # Dynamic imports
    r'\.\./',                     # Path traversal
    r'\\\.\\',                    # Windows path traversal
]

# Safe file extensions
SAFE_EXTENSIONS = {
    '.json', '.yaml', '.yml', '.txt', '.csv', '.log',
    '.npy', '.npz', '.pkl', '.pickle',
    '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff',
    '.mp4', '.avi', '.mov', '.mkv', '.webm',
    '.wav', '.mp3', '.flac', '.ogg'
}

# Dangerous file extensions
DANGEROUS_EXTENSIONS = {
    '.exe', '.bat', '.cmd', '.com', '.scr', '.pif',
    '.sh', '.bash', '.zsh', '.fish',
    '.js', '.vbs', '.ps1', '.jar',
    '.zip', '.rar', '.7z', '.tar', '.gz'  # Archives (can contain dangerous files)
}

# Max file sizes (bytes)
MAX_FILE_SIZES = {
    'default': 100 * 1024 * 1024,  # 100MB
    'image': 50 * 1024 * 1024,     # 50MB
    'video': 1024 * 1024 * 1024,   # 1GB
    'data': 500 * 1024 * 1024,     # 500MB
}


def sanitize_input(
    value: Any,
    max_length: Optional[int] = None,
    allowed_chars: Optional[str] = None,
    strip_html: bool = True
) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        value: Input value to sanitize
        max_length: Maximum allowed length
        allowed_chars: Regex pattern of allowed characters
        strip_html: Whether to strip HTML tags
        
    Returns:
        Sanitized string
        
    Raises:
        SecurityError: If input is dangerous
    """
    if value is None:
        return ""
    
    # Convert to string
    sanitized = str(value)
    
    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, sanitized, re.IGNORECASE | re.DOTALL):
            logger.warning(f"Dangerous pattern detected: {pattern}")
            raise SecurityError(
                "Input contains potentially dangerous content",
                threat_type="injection_attempt",
                details={"pattern": pattern}
            )
    
    # Strip HTML if requested
    if strip_html:
        sanitized = re.sub(r'<[^>]+>', '', sanitized)
    
    # Apply character restrictions
    if allowed_chars:
        if not re.match(f'^[{allowed_chars}]*$', sanitized):
            raise SecurityError(
                f"Input contains disallowed characters",
                threat_type="invalid_characters",
                details={"allowed_pattern": allowed_chars}
            )
    
    # Apply length restrictions
    if max_length and len(sanitized) > max_length:
        logger.warning(f"Input too long: {len(sanitized)} > {max_length}")
        raise SecurityError(
            f"Input too long: {len(sanitized)} characters > {max_length} limit",
            threat_type="input_too_long"
        )
    
    return sanitized


def check_file_safety(
    file_path: Union[str, Path],
    max_size: Optional[int] = None,
    allowed_extensions: Optional[List[str]] = None,
    scan_content: bool = True
) -> Dict[str, Any]:
    """
    Check if a file is safe to process.
    
    Args:
        file_path: Path to file to check
        max_size: Maximum allowed file size in bytes
        allowed_extensions: List of allowed file extensions
        scan_content: Whether to scan file content for threats
        
    Returns:
        Dictionary with safety information
        
    Raises:
        SecurityError: If file is unsafe
    """
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        raise SecurityError(f"File does not exist: {file_path}", threat_type="file_not_found")
    
    if not file_path.is_file():
        raise SecurityError(f"Path is not a file: {file_path}", threat_type="invalid_file_type")
    
    # Get file info
    file_size = file_path.stat().st_size
    file_extension = file_path.suffix.lower()
    mime_type, _ = mimetypes.guess_type(str(file_path))
    
    safety_info = {
        "path": str(file_path),
        "size": file_size,
        "extension": file_extension,
        "mime_type": mime_type,
        "is_safe": True,
        "warnings": [],
        "threats": []
    }
    
    # Check file extension
    if file_extension in DANGEROUS_EXTENSIONS:
        safety_info["is_safe"] = False
        safety_info["threats"].append(f"Dangerous file extension: {file_extension}")
        raise SecurityError(
            f"Dangerous file extension: {file_extension}",
            threat_type="dangerous_extension"
        )
    
    if allowed_extensions and file_extension not in allowed_extensions:
        safety_info["is_safe"] = False
        safety_info["threats"].append(f"Extension not allowed: {file_extension}")
        raise SecurityError(
            f"File extension not allowed: {file_extension}",
            threat_type="extension_not_allowed",
            details={"allowed": allowed_extensions}
        )
    
    # Check file size
    if max_size is None:
        # Use default based on file type
        if mime_type and mime_type.startswith('image/'):
            max_size = MAX_FILE_SIZES['image']
        elif mime_type and mime_type.startswith('video/'):
            max_size = MAX_FILE_SIZES['video']
        else:
            max_size = MAX_FILE_SIZES['default']
    
    if file_size > max_size:
        safety_info["is_safe"] = False
        safety_info["threats"].append(f"File too large: {file_size} > {max_size}")
        raise SecurityError(
            f"File too large: {file_size} bytes > {max_size} limit",
            threat_type="file_too_large"
        )
    
    # Scan file content for threats
    if scan_content:
        try:
            threats = _scan_file_content(file_path)
            if threats:
                safety_info["is_safe"] = False
                safety_info["threats"].extend(threats)
                raise SecurityError(
                    f"File content contains threats: {threats}",
                    threat_type="malicious_content"
                )
        except Exception as e:
            safety_info["warnings"].append(f"Could not scan file content: {e}")
            logger.warning(f"Failed to scan file content: {e}")
    
    return safety_info


def _scan_file_content(file_path: Path) -> List[str]:
    """
    Scan file content for potential threats.
    
    Args:
        file_path: Path to file to scan
        
    Returns:
        List of detected threats
    """
    threats = []
    
    try:
        # Only scan text-based files
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and not mime_type.startswith(('text/', 'application/json', 'application/yaml')):
            return threats
        
        # Read file content (limit to first 1MB to avoid memory issues)
        max_read_size = 1024 * 1024  # 1MB
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(max_read_size)
        
        # Check for dangerous patterns
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                threats.append(f"Dangerous pattern found: {pattern}")
        
        # Check for suspicious keywords
        suspicious_keywords = [
            'eval', 'exec', 'subprocess', '__import__',
            'os.system', 'os.popen', 'shell=True',
            'pickle.loads', 'marshal.loads'
        ]
        
        for keyword in suspicious_keywords:
            if keyword in content:
                threats.append(f"Suspicious keyword found: {keyword}")
    
    except Exception as e:
        logger.warning(f"Error scanning file content: {e}")
    
    return threats


def generate_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """
    Generate hash of file content for integrity checking.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256, sha512)
        
    Returns:
        Hex digest of file hash
        
    Raises:
        SecurityError: If hashing fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise SecurityError(f"File does not exist: {file_path}")
    
    try:
        hasher = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    except Exception as e:
        raise SecurityError(f"Failed to hash file: {e}")


def verify_file_integrity(file_path: Union[str, Path], expected_hash: str, algorithm: str = "sha256") -> bool:
    """
    Verify file integrity using hash comparison.
    
    Args:
        file_path: Path to file to verify
        expected_hash: Expected hash value
        algorithm: Hash algorithm used
        
    Returns:
        True if file integrity is verified
        
    Raises:
        SecurityError: If verification fails
    """
    actual_hash = generate_file_hash(file_path, algorithm)
    
    if actual_hash != expected_hash:
        raise SecurityError(
            f"File integrity check failed: expected {expected_hash}, got {actual_hash}",
            threat_type="integrity_violation",
            details={
                "expected_hash": expected_hash,
                "actual_hash": actual_hash,
                "algorithm": algorithm
            }
        )
    
    return True


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and other attacks.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = os.path.basename(filename)
    
    # Remove dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Ensure not empty
    if not filename:
        filename = "unnamed_file"
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    
    return filename


class RateLimiter:
    """Simple rate limiter for API endpoints."""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed in time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed for client.
        
        Args:
            client_id: Unique client identifier
            
        Returns:
            True if request is allowed
        """
        import time
        
        current_time = time.time()
        
        # Clean old entries
        if client_id in self.requests:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if current_time - req_time < self.time_window
            ]
        else:
            self.requests[client_id] = []
        
        # Check if limit exceeded
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_id].append(current_time)
        return True


class QuantumSecurityValidator:
    """Advanced security validator for quantum RLHF operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        
        # Quantum-specific security thresholds
        self.max_qubits = self.config.get("max_qubits", 50)
        self.max_circuit_depth = self.config.get("max_circuit_depth", 1000)
        self.max_preference_pairs = self.config.get("max_preference_pairs", 10000)
        self.max_multimodal_features = self.config.get("max_multimodal_features", 2048)
        
        # Security patterns for quantum algorithms
        self.quantum_threat_patterns = [
            r'quantum_tunneling.*infinite',
            r'superposition.*exploit',
            r'entanglement.*backdoor',
            r'measurement.*bypass',
            r'collapse.*overflow'
        ]
        
        # Rate limiting for quantum operations
        self.quantum_rate_limiter = RateLimiter(max_requests=100, time_window=300)  # 100 ops per 5 min
        
    def validate_quantum_problem(self, problem: Dict[str, Any], algorithm_type: str) -> Dict[str, Any]:
        """
        Validate quantum problem parameters for security.
        
        Args:
            problem: Problem parameters dictionary
            algorithm_type: Type of quantum algorithm
            
        Returns:
            Validation results
            
        Raises:
            SecurityError: If problem parameters are unsafe
        """
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "sanitized_problem": problem.copy(),
            "security_score": 1.0
        }
        
        # Validate quantum state sizes
        if "num_qubits" in problem:
            num_qubits = problem["num_qubits"]
            if num_qubits > self.max_qubits:
                raise SecurityError(
                    f"Quantum system too large: {num_qubits} qubits > {self.max_qubits} limit",
                    threat_type="resource_exhaustion",
                    details={"requested_qubits": num_qubits, "max_allowed": self.max_qubits}
                )
            
            # Exponential resource scaling warning
            if num_qubits > 20:
                validation_results["warnings"].append(
                    f"Large quantum system ({num_qubits} qubits) - exponential resource usage"
                )
                validation_results["security_score"] *= 0.8
        
        # Validate preference data
        if "preference_pairs" in problem:
            preference_pairs = problem["preference_pairs"]
            if len(preference_pairs) > self.max_preference_pairs:
                raise SecurityError(
                    f"Too many preference pairs: {len(preference_pairs)} > {self.max_preference_pairs}",
                    threat_type="data_overload"
                )
            
            # Sanitize preference pairs
            sanitized_pairs = []
            for i, pair in enumerate(preference_pairs):
                if i >= self.max_preference_pairs:
                    break
                    
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    # Validate preference data types and sizes
                    pref_a, pref_b = pair[0], pair[1]
                    label = pair[2] if len(pair) > 2 else 0
                    
                    # Sanitize preference vectors
                    if isinstance(pref_a, list):
                        pref_a = self._sanitize_feature_vector(pref_a)
                    if isinstance(pref_b, list):
                        pref_b = self._sanitize_feature_vector(pref_b)
                    
                    # Validate label
                    if not isinstance(label, (int, float)) or label not in [0, 1]:
                        label = 0  # Default to neutral
                        validation_results["warnings"].append(f"Invalid label at pair {i}, defaulting to 0")
                    
                    sanitized_pairs.append((pref_a, pref_b, int(label)))
            
            validation_results["sanitized_problem"]["preference_pairs"] = sanitized_pairs
        
        # Validate multimodal features
        for feature_type in ["vision_features", "proprioceptive_features", "audio_features"]:
            if feature_type in problem:
                features = problem[feature_type]
                if isinstance(features, list) and len(features) > self.max_multimodal_features:
                    validation_results["warnings"].append(
                        f"Truncating {feature_type} from {len(features)} to {self.max_multimodal_features}"
                    )
                    validation_results["sanitized_problem"][feature_type] = features[:self.max_multimodal_features]
                    validation_results["security_score"] *= 0.9
        
        # Validate training data
        if "training_data" in problem and "preference_labels" in problem:
            training_data = problem["training_data"]
            labels = problem["preference_labels"]
            
            if len(training_data) != len(labels):
                raise SecurityError(
                    f"Training data/label mismatch: {len(training_data)} data vs {len(labels)} labels",
                    threat_type="data_inconsistency"
                )
            
            # Sanitize training data
            sanitized_data = []
            sanitized_labels = []
            
            for i, (data, label) in enumerate(zip(training_data, labels)):
                if i >= 5000:  # Limit training data size
                    break
                
                # Sanitize data point
                if isinstance(data, list):
                    data = self._sanitize_feature_vector(data)
                elif isinstance(data, dict):
                    data = self._sanitize_data_dict(data)
                
                # Sanitize label
                if isinstance(label, (int, float)):
                    label = max(0, min(1, int(label)))  # Clamp to [0, 1]
                else:
                    label = 0
                
                sanitized_data.append(data)
                sanitized_labels.append(label)
            
            validation_results["sanitized_problem"]["training_data"] = sanitized_data
            validation_results["sanitized_problem"]["preference_labels"] = sanitized_labels
        
        # Check for malicious patterns in problem description
        problem_str = str(problem)
        for pattern in self.quantum_threat_patterns:
            if re.search(pattern, problem_str, re.IGNORECASE):
                raise SecurityError(
                    f"Potentially malicious quantum pattern detected: {pattern}",
                    threat_type="quantum_exploit_attempt"
                )
        
        return validation_results
    
    def _sanitize_feature_vector(self, features: List) -> List[float]:
        """Sanitize feature vector for safe processing."""
        sanitized = []
        for feature in features[:1024]:  # Limit vector size
            if isinstance(feature, (int, float)):
                # Clamp extreme values
                clamped = max(-1000.0, min(1000.0, float(feature)))
                # Check for NaN/inf
                if not (clamped != clamped or clamped == float('inf') or clamped == float('-inf')):
                    sanitized.append(clamped)
                else:
                    sanitized.append(0.0)
            else:
                sanitized.append(0.0)
        
        return sanitized
    
    def _sanitize_data_dict(self, data: Dict) -> Dict:
        """Sanitize data dictionary for safe processing."""
        sanitized = {}
        for key, value in data.items():
            # Sanitize key
            clean_key = sanitize_input(str(key), max_length=100, allowed_chars=r'a-zA-Z0-9_')
            
            # Sanitize value
            if isinstance(value, (list, tuple)):
                sanitized[clean_key] = self._sanitize_feature_vector(list(value))
            elif isinstance(value, (int, float)):
                clamped = max(-1000.0, min(1000.0, float(value)))
                sanitized[clean_key] = clamped if clamped == clamped else 0.0
            elif isinstance(value, str):
                sanitized[clean_key] = sanitize_input(value, max_length=500)
            else:
                sanitized[clean_key] = str(value)[:100]  # Convert to string and limit
        
        return sanitized
    
    def validate_quantum_circuit_safety(self, circuit_params: Dict[str, Any]) -> bool:
        """
        Validate quantum circuit parameters for safety.
        
        Args:
            circuit_params: Circuit parameters to validate
            
        Returns:
            True if circuit is safe
            
        Raises:
            SecurityError: If circuit is unsafe
        """
        # Check circuit depth
        depth = circuit_params.get("depth", 0)
        if depth > self.max_circuit_depth:
            raise SecurityError(
                f"Circuit too deep: {depth} > {self.max_circuit_depth}",
                threat_type="computational_complexity"
            )
        
        # Check gate count
        gates = circuit_params.get("gates", [])
        if len(gates) > self.max_circuit_depth * 10:
            raise SecurityError(
                f"Too many gates: {len(gates)} > {self.max_circuit_depth * 10}",
                threat_type="resource_exhaustion"
            )
        
        # Validate gate parameters
        for gate in gates:
            if isinstance(gate, dict):
                params = gate.get("params", {})
                for param_name, param_value in params.items():
                    if isinstance(param_value, (int, float)):
                        if abs(param_value) > 1000:  # Extreme parameter values
                            raise SecurityError(
                                f"Extreme gate parameter: {param_name}={param_value}",
                                threat_type="parameter_overflow"
                            )
        
        return True
    
    def check_quantum_rate_limit(self, client_id: str) -> bool:
        """
        Check if quantum operation is within rate limits.
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if within rate limits
            
        Raises:
            SecurityError: If rate limit exceeded
        """
        if not self.quantum_rate_limiter.is_allowed(client_id):
            raise SecurityError(
                "Quantum operation rate limit exceeded",
                threat_type="rate_limit_exceeded",
                details={"client_id": client_id}
            )
        
        return True
    
    def validate_human_preference_data(self, preferences: List) -> List:
        """
        Validate and sanitize human preference data.
        
        Args:
            preferences: Raw preference data
            
        Returns:
            Sanitized preference data
        """
        sanitized_preferences = []
        
        for i, pref in enumerate(preferences[:1000]):  # Limit number of preferences
            if isinstance(pref, (int, float)):
                # Normalize preference to [0, 1] range
                normalized = max(0.0, min(1.0, float(pref)))
                if normalized == normalized:  # Check for NaN
                    sanitized_preferences.append(normalized)
                else:
                    sanitized_preferences.append(0.5)  # Neutral preference
            elif isinstance(pref, str):
                # Convert text preferences to numeric
                pref_lower = pref.lower().strip()
                if pref_lower in ["good", "better", "prefer", "yes", "positive"]:
                    sanitized_preferences.append(1.0)
                elif pref_lower in ["bad", "worse", "reject", "no", "negative"]:
                    sanitized_preferences.append(0.0)
                else:
                    sanitized_preferences.append(0.5)  # Neutral
            else:
                sanitized_preferences.append(0.5)  # Default neutral
        
        return sanitized_preferences
    
    def audit_quantum_operation(self, operation_type: str, params: Dict[str, Any], 
                               result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Audit quantum operation for security analysis.
        
        Args:
            operation_type: Type of quantum operation
            params: Operation parameters
            result: Operation result
            
        Returns:
            Audit log entry
        """
        import time
        
        audit_entry = {
            "timestamp": time.time(),
            "operation_type": operation_type,
            "parameters_hash": hashlib.sha256(str(sorted(params.items())).encode()).hexdigest()[:16],
            "result_size": len(str(result)),
            "security_flags": [],
            "risk_score": 0.0
        }
        
        # Analyze operation for security risks
        if "quantum_speedup" in result:
            speedup = result["quantum_speedup"]
            if speedup > 1000:  # Suspicious speedup claims
                audit_entry["security_flags"].append("suspicious_speedup_claim")
                audit_entry["risk_score"] += 0.3
        
        if "accuracy" in result or "consistency_score" in result:
            accuracy = result.get("accuracy", result.get("consistency_score", 0))
            if accuracy > 0.99:  # Suspiciously high accuracy
                audit_entry["security_flags"].append("suspiciously_high_accuracy")
                audit_entry["risk_score"] += 0.2
        
        # Check for resource consumption patterns
        if "execution_time" in result:
            exec_time = result["execution_time"]
            if exec_time < 0.001:  # Suspiciously fast execution
                audit_entry["security_flags"].append("suspiciously_fast_execution")
                audit_entry["risk_score"] += 0.2
        
        return audit_entry


class QuantumDataEncryption:
    """Quantum-safe encryption for sensitive RLHF data."""
    
    def __init__(self, key_size: int = 256):
        self.key_size = key_size
        self.logger = logger
        
    def encrypt_preference_data(self, preference_data: List, key: Optional[str] = None) -> Dict[str, Any]:
        """
        Encrypt preference data using quantum-safe methods.
        
        Args:
            preference_data: Preference data to encrypt
            key: Encryption key (auto-generated if None)
            
        Returns:
            Encrypted data package
        """
        import json
        import secrets
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        import base64
        
        try:
            # Generate key if not provided
            if key is None:
                key = secrets.token_urlsafe(32)
            
            # Create encryption key from password
            password = key.encode()
            salt = secrets.token_bytes(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            encryption_key = base64.urlsafe_b64encode(kdf.derive(password))
            
            # Encrypt data
            fernet = Fernet(encryption_key)
            data_json = json.dumps(preference_data)
            encrypted_data = fernet.encrypt(data_json.encode())
            
            return {
                "encrypted_data": base64.b64encode(encrypted_data).decode(),
                "salt": base64.b64encode(salt).decode(),
                "encryption_method": "quantum_safe_pbkdf2_aes256",
                "data_hash": hashlib.sha256(data_json.encode()).hexdigest()
            }
        
        except Exception as e:
            raise SecurityError(f"Failed to encrypt preference data: {e}")
    
    def decrypt_preference_data(self, encrypted_package: Dict[str, Any], key: str) -> List:
        """
        Decrypt preference data.
        
        Args:
            encrypted_package: Encrypted data package
            key: Decryption key
            
        Returns:
            Decrypted preference data
        """
        import json
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        import base64
        
        try:
            # Reconstruct encryption key
            password = key.encode()
            salt = base64.b64decode(encrypted_package["salt"])
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            encryption_key = base64.urlsafe_b64encode(kdf.derive(password))
            
            # Decrypt data
            fernet = Fernet(encryption_key)
            encrypted_data = base64.b64decode(encrypted_package["encrypted_data"])
            decrypted_data = fernet.decrypt(encrypted_data)
            
            # Verify integrity
            data_json = decrypted_data.decode()
            actual_hash = hashlib.sha256(data_json.encode()).hexdigest()
            expected_hash = encrypted_package["data_hash"]
            
            if actual_hash != expected_hash:
                raise SecurityError("Data integrity check failed during decryption")
            
            return json.loads(data_json)
        
        except Exception as e:
            raise SecurityError(f"Failed to decrypt preference data: {e}")