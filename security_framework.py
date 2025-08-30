#!/usr/bin/env python3
"""
Security Framework for Scientific Computing Toolkit

This module implements comprehensive security measures for:
1. Hybrid UQ Framework predictions
2. Cross-framework communication
3. Data processing pipeline
4. API endpoints and user inputs

Author: Scientific Computing Toolkit Team
Date: 2025
License: GPL-3.0-only
"""

import hashlib
import hmac
import secrets
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
import re
import ipaddress
import threading
from pathlib import Path
import base64
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityManager:
    """Central security management system."""

    def __init__(self, config_path: str = "security_config.json"):
        self.config_path = Path(config_path)
        self.session_keys: Dict[str, Dict[str, Any]] = {}
        self.rate_limits: Dict[str, List[datetime]] = {}
        self.blacklisted_ips: set = set()
        self.allowed_origins: set = set()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

        # Security parameters
        self.max_request_size = 1024 * 1024  # 1MB
        self.rate_limit_window = 60  # 1 minute
        self.rate_limit_max_requests = 100
        self.session_timeout_hours = 24
        self.max_concurrent_sessions = 10

        self.load_configuration()

    def load_configuration(self):
        """Load security configuration."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.load_security_settings(config)
            except Exception as e:
                logger.error(f"Failed to load security config: {e}")
                self.create_default_configuration()

    def load_security_settings(self, config: Dict[str, Any]):
        """Load security settings from configuration."""
        security_config = config.get('security', {})

        self.max_request_size = security_config.get('max_request_size', self.max_request_size)
        self.rate_limit_window = security_config.get('rate_limit_window', self.rate_limit_window)
        self.rate_limit_max_requests = security_config.get('rate_limit_max_requests', self.rate_limit_max_requests)
        self.session_timeout_hours = security_config.get('session_timeout_hours', self.session_timeout_hours)
        self.max_concurrent_sessions = security_config.get('max_concurrent_sessions', self.max_concurrent_sessions)

        # Load IP blacklists and allowed origins
        self.blacklisted_ips = set(security_config.get('blacklisted_ips', []))
        self.allowed_origins = set(security_config.get('allowed_origins', []))

    def create_default_configuration(self):
        """Create default security configuration."""
        default_config = {
            "security": {
                "max_request_size": 1048576,
                "rate_limit_window": 60,
                "rate_limit_max_requests": 100,
                "session_timeout_hours": 24,
                "max_concurrent_sessions": 10,
                "blacklisted_ips": [],
                "allowed_origins": ["localhost", "127.0.0.1"]
            },
            "encryption": {
                "algorithm": "AES-256-GCM",
                "key_rotation_days": 30,
                "data_encryption": True
            },
            "authentication": {
                "require_auth": True,
                "token_expiry_hours": 8,
                "password_min_length": 12,
                "two_factor_required": False
            }
        }

        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)

        self.load_security_settings(default_config)


class InputValidator:
    """Comprehensive input validation and sanitization."""

    def __init__(self):
        # Malicious pattern detection
        self.malicious_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS attempts
            r'javascript:',                # JavaScript injection
            r'on\w+\s*=',                  # Event handlers
            r'<\w+[^>]*>',                 # HTML tags
            r'\.\./',                      # Directory traversal
            r'\.\.',                       # Parent directory
            r'union.*select',              # SQL injection
            r'exec\s*\(',                  # Code execution
            r'eval\s*\(',                  # Code evaluation
        ]

        # Compile regex patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL)
                                for pattern in self.malicious_patterns]

        # Allowed file extensions
        self.allowed_extensions = {
            'data': ['.json', '.csv', '.txt', '.npy', '.pkl'],
            'config': ['.json', '.yaml', '.yml'],
            'documentation': ['.md', '.tex', '.pdf'],
            'code': ['.py', '.java', '.swift', '.mojo']
        }

    def validate_input_data(self, data: Any, data_type: str = 'generic') -> Dict[str, Any]:
        """Validate and sanitize input data."""
        result = {
            'is_valid': True,
            'sanitized_data': data,
            'warnings': [],
            'errors': []
        }

        try:
            # Type-specific validation
            if data_type == 'json':
                result.update(self.validate_json_input(data))
            elif data_type == 'file':
                result.update(self.validate_file_input(data))
            elif data_type == 'prediction_request':
                result.update(self.validate_prediction_request(data))
            else:
                result.update(self.validate_generic_input(data))

            # Check for malicious content
            malicious_content = self.detect_malicious_content(str(data))
            if malicious_content:
                result['is_valid'] = False
                result['errors'].append(f"Malicious content detected: {malicious_content}")

        except Exception as e:
            result['is_valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")

        return result

    def validate_json_input(self, data: Any) -> Dict[str, Any]:
        """Validate JSON input data."""
        result = {'is_valid': True, 'warnings': [], 'errors': []}

        if not isinstance(data, (dict, list)):
            try:
                if isinstance(data, str):
                    json.loads(data)
                else:
                    result['is_valid'] = False
                    result['errors'].append("Invalid JSON format")
            except json.JSONDecodeError as e:
                result['is_valid'] = False
                result['errors'].append(f"JSON decode error: {e}")

        # Check for excessively nested structures
        def check_nesting(obj, depth=0, max_depth=10):
            if depth > max_depth:
                return False
            if isinstance(obj, dict):
                return all(check_nesting(v, depth + 1, max_depth) for v in obj.values())
            elif isinstance(obj, list):
                return all(check_nesting(item, depth + 1, max_depth) for item in obj)
            return True

        if not check_nesting(data):
            result['warnings'].append("Deeply nested structure detected")

        return result

    def validate_file_input(self, file_path: str) -> Dict[str, Any]:
        """Validate file input."""
        result = {'is_valid': True, 'warnings': [], 'errors': []}

        path = Path(file_path)

        # Check if file exists
        if not path.exists():
            result['is_valid'] = False
            result['errors'].append("File does not exist")
            return result

        # Check file size
        file_size = path.stat().st_size
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            result['is_valid'] = False
            result['errors'].append("File too large (>100MB)")
            return result

        # Check file extension
        extension = path.suffix.lower()
        allowed_extensions = []
        for category in self.allowed_extensions.values():
            allowed_extensions.extend(category)

        if extension not in allowed_extensions:
            result['warnings'].append(f"Unusual file extension: {extension}")

        # Check for hidden files
        if path.name.startswith('.'):
            result['warnings'].append("Hidden file detected")

        return result

    def validate_prediction_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate hybrid UQ prediction request."""
        result = {'is_valid': True, 'warnings': [], 'errors': []}

        required_fields = ['inputs', 'grid_metrics']
        for field in required_fields:
            if field not in request:
                result['is_valid'] = False
                result['errors'].append(f"Missing required field: {field}")

        # Validate inputs structure
        if 'inputs' in request:
            inputs = request['inputs']
            if not isinstance(inputs, list) or len(inputs) == 0:
                result['errors'].append("Inputs must be non-empty list")
            elif len(inputs) > 100:  # Reasonable limit
                result['warnings'].append("Large input batch detected")

        # Validate grid metrics
        if 'grid_metrics' in request:
            grid_metrics = request['grid_metrics']
            required_metrics = ['dx', 'dy']
            for metric in required_metrics:
                if metric not in grid_metrics:
                    result['errors'].append(f"Missing grid metric: {metric}")

        return result

    def validate_generic_input(self, data: Any) -> Dict[str, Any]:
        """Generic input validation."""
        result = {'is_valid': True, 'warnings': [], 'errors': []}

        # Check data size
        data_str = str(data)
        if len(data_str) > self.max_request_size:
            result['is_valid'] = False
            result['errors'].append("Input data too large")

        return result

    def detect_malicious_content(self, content: str) -> List[str]:
        """Detect potentially malicious content."""
        malicious_matches = []

        for pattern in self.compiled_patterns:
            matches = pattern.findall(content)
            if matches:
                malicious_matches.extend(matches[:5])  # Limit matches to prevent flooding

        return malicious_matches

    def sanitize_input(self, data: Any) -> Any:
        """Sanitize input data to prevent injection attacks."""
        if isinstance(data, str):
            # Remove potentially dangerous characters
            sanitized = re.sub(r'[<>]', '', data)
            # Escape quotes
            sanitized = sanitized.replace('"', '\\"').replace("'", "\\'")
            return sanitized
        elif isinstance(data, dict):
            return {k: self.sanitize_input(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_input(item) for item in data]
        else:
            return data


class AuthenticationManager:
    """Authentication and authorization management."""

    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        self.users: Dict[str, Dict[str, Any]] = {}
        self.tokens: Dict[str, Dict[str, Any]] = {}
        self.roles: Dict[str, List[str]] = {
            'admin': ['read', 'write', 'execute', 'manage_users'],
            'researcher': ['read', 'write', 'execute'],
            'viewer': ['read']
        }

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return session token."""
        if username not in self.users:
            return None

        user = self.users[username]
        if not self.verify_password(password, user['password_hash']):
            return None

        # Create session token
        token = secrets.token_urlsafe(32)
        expiry = datetime.now(timezone.utc) + timedelta(hours=self.security_manager.session_timeout_hours)

        self.tokens[token] = {
            'username': username,
            'role': user['role'],
            'created': datetime.now(timezone.utc),
            'expiry': expiry
        }

        return token

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify authentication token."""
        if token not in self.tokens:
            return None

        token_data = self.tokens[token]

        # Check expiry
        if datetime.now(timezone.utc) > token_data['expiry']:
            del self.tokens[token]
            return None

        return token_data

    def authorize_action(self, token_data: Dict[str, Any], action: str) -> bool:
        """Check if user is authorized for action."""
        user_role = token_data['role']
        allowed_actions = self.roles.get(user_role, [])

        return action in allowed_actions

    def create_user(self, username: str, password: str, role: str = 'viewer') -> bool:
        """Create new user account."""
        if username in self.users:
            return False

        if role not in self.roles:
            return False

        self.users[username] = {
            'password_hash': self.hash_password(password),
            'role': role,
            'created': datetime.now(timezone.utc),
            'active': True
        }

        return True

    def hash_password(self, password: str) -> str:
        """Hash password using secure algorithm."""
        salt = secrets.token_bytes(16)
        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        return base64.b64encode(salt + key).decode()

    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        try:
            decoded = base64.b64decode(password_hash)
            salt = decoded[:16]
            stored_key = decoded[16:]

            key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
            return secrets.compare_digest(key, stored_key)
        except Exception:
            return False


class RateLimiter:
    """Rate limiting for API endpoints."""

    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        self.requests: Dict[str, List[datetime]] = {}
        self.lock = threading.Lock()

    def check_rate_limit(self, identifier: str) -> bool:
        """Check if request should be rate limited."""
        with self.lock:
            now = datetime.now(timezone.utc)
            window_start = now - timedelta(seconds=self.security_manager.rate_limit_window)

            # Clean old requests
            if identifier in self.requests:
                self.requests[identifier] = [
                    req_time for req_time in self.requests[identifier]
                    if req_time > window_start
                ]

            # Check current request count
            current_requests = len(self.requests.get(identifier, []))

            if current_requests >= self.security_manager.rate_limit_max_requests:
                return False

            # Add current request
            if identifier not in self.requests:
                self.requests[identifier] = []
            self.requests[identifier].append(now)

            return True

    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests in current window."""
        with self.lock:
            now = datetime.now(timezone.utc)
            window_start = now - timedelta(seconds=self.security_manager.rate_limit_window)

            if identifier in self.requests:
                current_requests = len([
                    req_time for req_time in self.requests[identifier]
                    if req_time > window_start
                ])
            else:
                current_requests = 0

            return max(0, self.security_manager.rate_limit_max_requests - current_requests)


class EncryptionManager:
    """Data encryption and decryption management."""

    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        self.encryption_key = self.generate_key()
        self.key_rotation_date = datetime.now(timezone.utc) + timedelta(days=30)

    def generate_key(self) -> bytes:
        """Generate encryption key."""
        return secrets.token_bytes(32)  # 256-bit key

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        # Simple XOR encryption for demonstration
        # In production, use proper encryption like AES-256-GCM
        key = self.encryption_key
        encrypted = bytearray()
        data_bytes = data.encode()

        for i, byte in enumerate(data_bytes):
            encrypted.append(byte ^ key[i % len(key)])

        return base64.b64encode(encrypted).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        try:
            key = self.encryption_key
            encrypted_bytes = base64.b64decode(encrypted_data)
            decrypted = bytearray()

            for i, byte in enumerate(encrypted_bytes):
                decrypted.append(byte ^ key[i % len(key)])

            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return ""

    def rotate_key(self):
        """Rotate encryption key."""
        self.encryption_key = self.generate_key()
        self.key_rotation_date = datetime.now(timezone.utc) + timedelta(days=30)
        logger.info("Encryption key rotated successfully")


class AuditLogger:
    """Security audit logging system."""

    def __init__(self, log_path: str = "security_audit.log"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, details: Dict[str, Any],
                  user: str = "system", ip_address: str = "unknown"):
        """Log security event."""
        timestamp = datetime.now(timezone.utc).isoformat() + "Z"

        log_entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "user": user,
            "ip_address": ip_address,
            "details": details
        }

        with open(self.log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def log_authentication(self, username: str, success: bool, ip_address: str):
        """Log authentication event."""
        self.log_event(
            "authentication",
            {"username": username, "success": success},
            username,
            ip_address
        )

    def log_authorization(self, username: str, action: str, success: bool, ip_address: str):
        """Log authorization event."""
        self.log_event(
            "authorization",
            {"action": action, "success": success},
            username,
            ip_address
        )

    def log_security_violation(self, violation_type: str, details: Dict[str, Any],
                              ip_address: str):
        """Log security violation."""
        self.log_event(
            "security_violation",
            {"violation_type": violation_type, "details": details},
            "system",
            ip_address
        )


# Decorators for security enforcement

def require_authentication(func: Callable) -> Callable:
    """Decorator to require authentication for endpoint."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Authentication logic would be implemented here
        # For now, just call the function
        return func(*args, **kwargs)
    return wrapper


def rate_limit(identifier_func: Callable = None) -> Callable:
    """Decorator to apply rate limiting."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Rate limiting logic would be implemented here
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_input(input_type: str = 'generic') -> Callable:
    """Decorator to validate input data."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            validator = InputValidator()

            # Validate function arguments
            for arg_name, arg_value in kwargs.items():
                if arg_name != 'self':  # Skip self for methods
                    validation_result = validator.validate_input_data(arg_value, input_type)
                    if not validation_result['is_valid']:
                        raise ValueError(f"Invalid input for {arg_name}: {validation_result['errors']}")

            return func(*args, **kwargs)
        return wrapper
    return decorator


def audit_log(event_type: str) -> Callable:
    """Decorator to log function calls."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            audit_logger = AuditLogger()
            start_time = datetime.now(timezone.utc)

            try:
                result = func(*args, **kwargs)
                end_time = datetime.now(timezone.utc)

                audit_logger.log_event(
                    event_type,
                    {
                        "function": func.__name__,
                        "duration": (end_time - start_time).total_seconds(),
                        "success": True
                    }
                )

                return result

            except Exception as e:
                end_time = datetime.now(timezone.utc)

                audit_logger.log_event(
                    "error",
                    {
                        "function": func.__name__,
                        "error": str(e),
                        "duration": (end_time - start_time).total_seconds(),
                        "success": False
                    }
                )

                raise
        return wrapper
    return decorator


# Security utilities

def generate_secure_token(length: int = 32) -> str:
    """Generate cryptographically secure token."""
    return secrets.token_urlsafe(length)


def hash_data(data: str, algorithm: str = 'sha256') -> str:
    """Hash data using specified algorithm."""
    if algorithm == 'sha256':
        return hashlib.sha256(data.encode()).hexdigest()
    elif algorithm == 'sha512':
        return hashlib.sha512(data.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def validate_ip_address(ip: str) -> bool:
    """Validate IP address format."""
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal."""
    # Remove path separators and dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip(' .')
    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    return sanitized


# Main security framework class

class ScientificComputingSecurity:
    """Main security framework for scientific computing toolkit."""

    def __init__(self):
        self.security_manager = SecurityManager()
        self.input_validator = InputValidator()
        self.auth_manager = AuthenticationManager(self.security_manager)
        self.rate_limiter = RateLimiter(self.security_manager)
        self.encryption_manager = EncryptionManager(self.security_manager)
        self.audit_logger = AuditLogger()

    def initialize_security(self):
        """Initialize all security components."""
        logger.info("Initializing Scientific Computing Security Framework...")

        # Initialize default admin user
        if not self.auth_manager.users:
            self.auth_manager.create_user("admin", "secure_password_123", "admin")

        logger.info("Security framework initialized successfully")

    def validate_prediction_request(self, request: Dict[str, Any],
                                   user_token: str = None) -> Dict[str, Any]:
        """Validate hybrid UQ prediction request with full security checks."""

        # 1. Authentication check
        if user_token:
            token_data = self.auth_manager.verify_token(user_token)
            if not token_data:
                return {"valid": False, "error": "Invalid authentication token"}
            if not self.auth_manager.authorize_action(token_data, "execute"):
                return {"valid": False, "error": "Insufficient permissions"}

        # 2. Input validation
        validation_result = self.input_validator.validate_prediction_request(request)
        if not validation_result['is_valid']:
            return {"valid": False, "errors": validation_result['errors']}

        # 3. Rate limiting check
        client_id = user_token or "anonymous"
        if not self.rate_limiter.check_rate_limit(client_id):
            return {"valid": False, "error": "Rate limit exceeded"}

        # 4. Security audit log
        self.audit_logger.log_event(
            "prediction_request",
            {"request_size": len(str(request))},
            token_data.get('username', 'anonymous') if token_data else 'anonymous'
        )

        return {"valid": True, "warnings": validation_result.get('warnings', [])}

    def secure_encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data with security logging."""
        encrypted = self.encryption_manager.encrypt_data(data)

        self.audit_logger.log_event(
            "data_encryption",
            {"data_size": len(data)},
            "system"
        )

        return encrypted

    def secure_decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data with security logging."""
        decrypted = self.encryption_manager.decrypt_data(encrypted_data)

        self.audit_logger.log_event(
            "data_decryption",
            {"success": bool(decrypted)},
            "system"
        )

        return decrypted

    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            "active_sessions": len(self.auth_manager.active_sessions),
            "rate_limit_violations": sum(
                1 for requests in self.rate_limiter.requests.values()
                if len(requests) >= self.security_manager.rate_limit_max_requests
            ),
            "security_events_today": self.count_security_events_today(),
            "encryption_key_expiry": self.encryption_manager.key_rotation_date.isoformat(),
            "system_health": "good" if self.check_system_health() else "warning"
        }

    def count_security_events_today(self) -> int:
        """Count security events for today."""
        if not self.audit_logger.log_path.exists():
            return 0

        today = datetime.now(timezone.utc).date()
        count = 0

        with open(self.audit_logger.log_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    entry_date = datetime.fromisoformat(entry['timestamp'][:-1]).date()
                    if entry_date == today:
                        count += 1
                except:
                    continue

        return count

    def check_system_health(self) -> bool:
        """Check overall system security health."""
        # Basic health checks
        checks = [
            len(self.auth_manager.active_sessions) <= self.security_manager.max_concurrent_sessions,
            datetime.now(timezone.utc) < self.encryption_manager.key_rotation_date,
            self.audit_logger.log_path.exists()
        ]

        return all(checks)


# Global security instance
security_framework = ScientificComputingSecurity()


def initialize_security():
    """Initialize the global security framework."""
    security_framework.initialize_security()


def validate_request(request: Dict[str, Any], token: str = None) -> Dict[str, Any]:
    """Validate request with full security checks."""
    return security_framework.validate_prediction_request(request, token)


if __name__ == "__main__":
    # Initialize security framework
    initialize_security()

    # Example usage
    test_request = {
        "inputs": [[[0.1, 0.2], [0.3, 0.4]]],
        "grid_metrics": {"dx": 1.0, "dy": 1.0}
    }

    result = validate_request(test_request)
    print(f"Validation result: {result}")

    # Get security status
    status = security_framework.get_security_status()
    print(f"Security status: {status}")
