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