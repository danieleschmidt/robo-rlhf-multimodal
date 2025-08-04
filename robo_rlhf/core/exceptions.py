"""
Custom exceptions for robo-rlhf-multimodal.
"""


class RoboRLHFError(Exception):
    """Base exception for all robo-rlhf errors."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        """
        Initialize exception with structured error information.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error context
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
    
    def to_dict(self) -> dict:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class ValidationError(RoboRLHFError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: str = None, value=None, **kwargs):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.field = field
        self.value = value
        if field:
            self.details["field"] = field
        if value is not None:
            self.details["value"] = str(value)


class DataCollectionError(RoboRLHFError):
    """Raised when data collection fails."""
    
    def __init__(self, message: str, operation: str = None, **kwargs):
        super().__init__(message, error_code="DATA_COLLECTION_ERROR", **kwargs)
        self.operation = operation
        if operation:
            self.details["operation"] = operation


class PreferenceError(RoboRLHFError):
    """Raised when preference handling fails."""
    
    def __init__(self, message: str, pair_id: str = None, **kwargs):
        super().__init__(message, error_code="PREFERENCE_ERROR", **kwargs)
        self.pair_id = pair_id
        if pair_id:
            self.details["pair_id"] = pair_id


class ModelError(RoboRLHFError):
    """Raised when model operations fail."""
    
    def __init__(self, message: str, model_type: str = None, **kwargs):
        super().__init__(message, error_code="MODEL_ERROR", **kwargs)
        self.model_type = model_type
        if model_type:
            self.details["model_type"] = model_type


class ConfigurationError(RoboRLHFError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        super().__init__(message, error_code="CONFIGURATION_ERROR", **kwargs)
        self.config_key = config_key
        if config_key:
            self.details["config_key"] = config_key


class SecurityError(RoboRLHFError):
    """Raised when security validation fails."""
    
    def __init__(self, message: str, threat_type: str = None, **kwargs):
        super().__init__(message, error_code="SECURITY_ERROR", **kwargs)
        self.threat_type = threat_type
        if threat_type:
            self.details["threat_type"] = threat_type


class ResourceError(RoboRLHFError):
    """Raised when resource operations fail."""
    
    def __init__(self, message: str, resource_type: str = None, **kwargs):
        super().__init__(message, error_code="RESOURCE_ERROR", **kwargs)
        self.resource_type = resource_type
        if resource_type:
            self.details["resource_type"] = resource_type


class NetworkError(RoboRLHFError):
    """Raised when network operations fail."""
    
    def __init__(self, message: str, endpoint: str = None, **kwargs):
        super().__init__(message, error_code="NETWORK_ERROR", **kwargs)
        self.endpoint = endpoint
        if endpoint:
            self.details["endpoint"] = endpoint


class TimeoutError(RoboRLHFError):
    """Raised when operations timeout."""
    
    def __init__(self, message: str, operation: str = None, timeout_seconds: float = None, **kwargs):
        super().__init__(message, error_code="TIMEOUT_ERROR", **kwargs)
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        if operation:
            self.details["operation"] = operation
        if timeout_seconds:
            self.details["timeout_seconds"] = timeout_seconds