"""
Pipeline Security: Comprehensive security layer for self-healing pipeline operations.

Implements authentication, authorization, encryption, and security monitoring.
"""

import asyncio
import hashlib
import hmac
import logging
import secrets
import time
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class Permission(Enum):
    """Available permissions for pipeline operations."""
    READ_METRICS = "read_metrics"
    WRITE_METRICS = "write_metrics"
    MONITOR_HEALTH = "monitor_health"
    TRIGGER_HEALING = "trigger_healing"
    CONFIGURE_RULES = "configure_rules"
    MANAGE_COMPONENTS = "manage_components"
    ADMIN_ACCESS = "admin_access"


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: str
    permissions: Set[Permission]
    security_level: SecurityLevel
    session_token: str
    expires_at: float
    ip_address: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityEvent:
    """Security-related event for auditing."""
    event_type: str
    user_id: str
    component: str
    action: str
    success: bool
    timestamp: float
    ip_address: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class SecurityManager:
    """
    Comprehensive security management for pipeline operations.
    
    Features:
    - JWT-based authentication
    - Role-based access control (RBAC)
    - Data encryption at rest and in transit
    - Security event logging and monitoring
    - Rate limiting and abuse prevention
    - Secure communication channels
    """
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        encryption_key: Optional[bytes] = None,
        token_expiry: int = 3600,  # 1 hour
        enable_rate_limiting: bool = True
    ):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.token_expiry = token_expiry
        self.enable_rate_limiting = enable_rate_limiting
        
        # Security state
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.user_permissions: Dict[str, Set[Permission]] = {}
        self.security_events: List[SecurityEvent] = []
        self.blocked_ips: Set[str] = set()
        self.rate_limits: Dict[str, List[float]] = {}  # user_id -> timestamps
        
        # Default admin user
        self._setup_default_admin()
        
        logger.info("Security manager initialized with encryption and RBAC")
    
    def _setup_default_admin(self) -> None:
        """Setup default admin user."""
        admin_permissions = {
            Permission.READ_METRICS,
            Permission.WRITE_METRICS,
            Permission.MONITOR_HEALTH,
            Permission.TRIGGER_HEALING,
            Permission.CONFIGURE_RULES,
            Permission.MANAGE_COMPONENTS,
            Permission.ADMIN_ACCESS
        }
        self.user_permissions["admin"] = admin_permissions
        logger.info("Default admin user configured")
    
    def create_user(
        self,
        user_id: str,
        permissions: Set[Permission],
        security_level: SecurityLevel = SecurityLevel.INTERNAL
    ) -> None:
        """Create a new user with specified permissions."""
        self.user_permissions[user_id] = permissions
        logger.info(f"Created user {user_id} with {len(permissions)} permissions")
    
    def authenticate(
        self,
        user_id: str,
        credentials: Dict[str, Any],
        ip_address: Optional[str] = None
    ) -> Optional[str]:
        """
        Authenticate user and return session token.
        
        Args:
            user_id: User identifier
            credentials: Authentication credentials (password, API key, etc.)
            ip_address: Client IP address for security logging
            
        Returns:
            Session token if authentication successful, None otherwise
        """
        # Check if IP is blocked
        if ip_address and ip_address in self.blocked_ips:
            self._log_security_event(
                "authentication_blocked",
                user_id,
                "system",
                "authenticate",
                False,
                ip_address,
                {"reason": "blocked_ip"}
            )
            return None
        
        # Rate limiting check
        if self.enable_rate_limiting and self._is_rate_limited(user_id):
            self._log_security_event(
                "authentication_rate_limited",
                user_id,
                "system", 
                "authenticate",
                False,
                ip_address,
                {"reason": "rate_limit_exceeded"}
            )
            return None
        
        # Validate credentials (simplified - in real implementation use proper auth)
        if not self._validate_credentials(user_id, credentials):
            self._log_security_event(
                "authentication_failed",
                user_id,
                "system",
                "authenticate", 
                False,
                ip_address,
                {"reason": "invalid_credentials"}
            )
            return None
        
        # Generate session token
        session_token = self._generate_session_token(user_id, ip_address)
        
        # Create security context
        permissions = self.user_permissions.get(user_id, set())
        security_level = self._determine_security_level(permissions)
        
        context = SecurityContext(
            user_id=user_id,
            permissions=permissions,
            security_level=security_level,
            session_token=session_token,
            expires_at=time.time() + self.token_expiry,
            ip_address=ip_address
        )
        
        self.active_sessions[session_token] = context
        
        self._log_security_event(
            "authentication_success",
            user_id,
            "system",
            "authenticate",
            True,
            ip_address
        )
        
        return session_token
    
    def _validate_credentials(self, user_id: str, credentials: Dict[str, Any]) -> bool:
        """Validate user credentials."""
        # Simplified validation - in production use proper password hashing
        if user_id == "admin" and credentials.get("password") == "admin":
            return True
        
        # Check API key authentication
        if "api_key" in credentials:
            expected_key = self._generate_api_key(user_id)
            return hmac.compare_digest(credentials["api_key"], expected_key)
        
        return False
    
    def _generate_api_key(self, user_id: str) -> str:
        """Generate API key for user."""
        data = f"{user_id}:{self.secret_key}".encode()
        return hashlib.sha256(data).hexdigest()
    
    def _generate_session_token(self, user_id: str, ip_address: Optional[str]) -> str:
        """Generate JWT session token."""
        payload = {
            "user_id": user_id,
            "ip_address": ip_address,
            "iat": time.time(),
            "exp": time.time() + self.token_expiry
        }
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def _determine_security_level(self, permissions: Set[Permission]) -> SecurityLevel:
        """Determine security level based on permissions."""
        if Permission.ADMIN_ACCESS in permissions:
            return SecurityLevel.RESTRICTED
        elif Permission.MANAGE_COMPONENTS in permissions:
            return SecurityLevel.CONFIDENTIAL
        elif Permission.TRIGGER_HEALING in permissions:
            return SecurityLevel.INTERNAL
        else:
            return SecurityLevel.PUBLIC
    
    def _is_rate_limited(self, user_id: str) -> bool:
        """Check if user is rate limited."""
        current_time = time.time()
        window = 300  # 5 minutes
        max_attempts = 10
        
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = []
        
        # Clean old timestamps
        self.rate_limits[user_id] = [
            ts for ts in self.rate_limits[user_id]
            if current_time - ts < window
        ]
        
        # Check limit
        if len(self.rate_limits[user_id]) >= max_attempts:
            return True
        
        # Add current attempt
        self.rate_limits[user_id].append(current_time)
        return False
    
    def validate_session(self, token: str, ip_address: Optional[str] = None) -> Optional[SecurityContext]:
        """Validate session token and return security context."""
        try:
            # Decode JWT token
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            
            # Check if session exists
            if token not in self.active_sessions:
                return None
            
            context = self.active_sessions[token]
            
            # Check expiration
            if time.time() > context.expires_at:
                del self.active_sessions[token]
                return None
            
            # Verify IP address (optional security measure)
            if ip_address and context.ip_address and ip_address != context.ip_address:
                logger.warning(f"IP address mismatch for session {token[:8]}...")
                # Could be made stricter in high-security environments
            
            return context
            
        except jwt.InvalidTokenError:
            return None
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return None
    
    def authorize(
        self,
        context: SecurityContext,
        required_permission: Permission,
        component: str = "system"
    ) -> bool:
        """Check if user has required permission."""
        has_permission = required_permission in context.permissions
        
        self._log_security_event(
            "authorization_check",
            context.user_id,
            component,
            required_permission.value,
            has_permission,
            context.ip_address
        )
        
        return has_permission
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        try:
            encrypted = self.cipher_suite.encrypt(data.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        try:
            encrypted = base64.b64decode(encrypted_data.encode())
            decrypted = self.cipher_suite.decrypt(encrypted)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def _log_security_event(
        self,
        event_type: str,
        user_id: str,
        component: str,
        action: str,
        success: bool,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log security event for auditing."""
        event = SecurityEvent(
            event_type=event_type,
            user_id=user_id,
            component=component,
            action=action,
            success=success,
            timestamp=time.time(),
            ip_address=ip_address,
            details=details or {}
        )
        
        self.security_events.append(event)
        
        # Keep only recent events (last 10000)
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-10000:]
        
        # Log high-severity events
        if not success or event_type in ["authentication_failed", "authorization_failed"]:
            logger.warning(f"Security event: {event_type} - {user_id} - {component} - {success}")
    
    def logout(self, token: str) -> bool:
        """Logout user and invalidate session."""
        if token in self.active_sessions:
            context = self.active_sessions[token]
            del self.active_sessions[token]
            
            self._log_security_event(
                "logout",
                context.user_id,
                "system",
                "logout",
                True,
                context.ip_address
            )
            return True
        
        return False
    
    def block_ip(self, ip_address: str, reason: str = "security_violation") -> None:
        """Block IP address for security reasons."""
        self.blocked_ips.add(ip_address)
        logger.warning(f"Blocked IP address {ip_address}: {reason}")
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        current_time = time.time()
        recent_events = [
            event for event in self.security_events
            if current_time - event.timestamp < 3600  # Last hour
        ]
        
        failed_auth = len([
            event for event in recent_events
            if event.event_type == "authentication_failed"
        ])
        
        successful_auth = len([
            event for event in recent_events
            if event.event_type == "authentication_success"
        ])
        
        return {
            "active_sessions": len(self.active_sessions),
            "total_users": len(self.user_permissions),
            "blocked_ips": len(self.blocked_ips),
            "security_events_last_hour": len(recent_events),
            "failed_authentications_last_hour": failed_auth,
            "successful_authentications_last_hour": successful_auth,
            "rate_limited_users": len(self.rate_limits)
        }


def secure_pipeline_operation(
    security_manager: SecurityManager,
    required_permission: Permission,
    component: str = "pipeline"
):
    """
    Decorator for securing pipeline operations.
    
    Usage:
        @secure_pipeline_operation(security_manager, Permission.TRIGGER_HEALING)
        async def heal_component(context, component_name):
            # This operation requires TRIGGER_HEALING permission
            pass
    """
    def decorator(func: Callable):
        async def wrapper(context: SecurityContext, *args, **kwargs):
            # Validate session
            if not security_manager.validate_session(context.session_token):
                raise PermissionError("Invalid or expired session")
            
            # Check authorization
            if not security_manager.authorize(context, required_permission, component):
                raise PermissionError(f"Insufficient permissions: {required_permission.value}")
            
            # Execute operation
            try:
                result = await func(context, *args, **kwargs)
                
                # Log successful operation
                security_manager._log_security_event(
                    "operation_success",
                    context.user_id,
                    component,
                    func.__name__,
                    True,
                    context.ip_address
                )
                
                return result
                
            except Exception as e:
                # Log failed operation
                security_manager._log_security_event(
                    "operation_failed",
                    context.user_id,
                    component,
                    func.__name__,
                    False,
                    context.ip_address,
                    {"error": str(e)}
                )
                raise
        
        return wrapper
    return decorator


class SecureCommunication:
    """
    Secure communication layer for pipeline components.
    
    Provides encrypted channels for inter-component communication.
    """
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        self.encryption_keys: Dict[str, bytes] = {}
        
    def establish_secure_channel(self, component_a: str, component_b: str) -> str:
        """Establish secure communication channel between components."""
        channel_id = f"{component_a}:{component_b}"
        
        # Generate shared encryption key
        key = Fernet.generate_key()
        self.encryption_keys[channel_id] = key
        
        logger.info(f"Established secure channel: {channel_id}")
        return channel_id
    
    def encrypt_message(self, channel_id: str, message: Dict[str, Any]) -> str:
        """Encrypt message for secure transmission."""
        if channel_id not in self.encryption_keys:
            raise ValueError(f"No encryption key for channel: {channel_id}")
        
        cipher = Fernet(self.encryption_keys[channel_id])
        message_json = json.dumps(message).encode()
        encrypted = cipher.encrypt(message_json)
        
        return base64.b64encode(encrypted).decode()
    
    def decrypt_message(self, channel_id: str, encrypted_message: str) -> Dict[str, Any]:
        """Decrypt received message."""
        if channel_id not in self.encryption_keys:
            raise ValueError(f"No encryption key for channel: {channel_id}")
        
        cipher = Fernet(self.encryption_keys[channel_id])
        encrypted_data = base64.b64decode(encrypted_message.encode())
        decrypted = cipher.decrypt(encrypted_data)
        
        return json.loads(decrypted.decode())
    
    def revoke_channel(self, channel_id: str) -> None:
        """Revoke secure channel and delete encryption keys."""
        if channel_id in self.encryption_keys:
            del self.encryption_keys[channel_id]
            logger.info(f"Revoked secure channel: {channel_id}")


# Additional imports needed
import json