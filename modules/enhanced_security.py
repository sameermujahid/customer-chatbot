import hashlib
import hmac
import jwt
import bcrypt
import secrets
import time
import logging
import re
import ipaddress
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
from collections import defaultdict, deque
import json
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import requests
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ThreatType(Enum):
    RATE_LIMIT_VIOLATION = "rate_limit_violation"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    MALICIOUS_CONTENT = "malicious_content"
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_FAILURE = "authorization_failure"
    DDoS_ATTACK = "ddos_attack"
    SQL_INJECTION = "sql_injection"
    XSS_ATTACK = "xss_attack"

@dataclass
class SecurityEvent:
    timestamp: float
    event_type: str
    severity: SecurityLevel
    source_ip: str
    user_id: Optional[str]
    session_id: Optional[str]
    details: Dict[str, Any]
    threat_score: float

@dataclass
class UserSession:
    user_id: str
    session_id: str
    ip_address: str
    user_agent: str
    created_at: float
    last_activity: float
    permissions: List[str]
    security_level: SecurityLevel
    failed_attempts: int
    is_locked: bool
    lock_until: Optional[float]

class EnhancedSecurityManager:
    """
    Advanced security manager with authentication, authorization,
    threat detection, and encryption capabilities
    """
    
    def __init__(self, 
                 secret_key: str = None,
                 encryption_key: str = None,
                 max_failed_attempts: int = 5,
                 lockout_duration: int = 900,  # 15 minutes
                 threat_threshold: float = 0.7):
        
        # Security keys
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        
        # User management
        self.users = {}  # In production, use database
        self.sessions = {}
        self.failed_attempts = defaultdict(int)
        self.locked_users = {}
        
        # Threat detection
        self.threat_threshold = threat_threshold
        self.security_events = deque(maxlen=10000)
        self.threat_patterns = self._load_threat_patterns()
        self.ip_blacklist = set()
        self.ip_whitelist = set()
        
        # Rate limiting
        self.rate_limits = defaultdict(lambda: {
            'requests': deque(maxlen=1000),
            'last_reset': time.time()
        })
        
        # Security monitoring
        self.monitoring_enabled = True
        self.alert_callbacks = []
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("EnhancedSecurityManager initialized")
    
    def _load_threat_patterns(self) -> Dict[str, List[str]]:
        """Load threat detection patterns"""
        return {
            'sql_injection': [
                r"(\b(union|select|insert|update|delete|drop|create|alter)\b)",
                r"(\b(or|and)\b\s+\d+\s*=\s*\d+)",
                r"(\b(union|select)\b.*\bfrom\b)",
                r"(--|#|/\*|\*/)",
                r"(\bxp_|sp_|exec\b)",
            ],
            'xss_attack': [
                r"(<script[^>]*>.*?</script>)",
                r"(javascript:.*)",
                r"(on\w+\s*=\s*['\"])",
                r"(<iframe[^>]*>)",
                r"(<object[^>]*>)",
            ],
            'path_traversal': [
                r"(\.\./|\.\.\\)",
                r"(/%2e%2e%2f|%2e%2e%5c)",
                r"(\.\.%2f|\.\.%5c)",
            ],
            'command_injection': [
                r"(\b(cat|ls|pwd|whoami|id|uname)\b)",
                r"(\b(rm|del|mkdir|touch)\b)",
                r"(\b(wget|curl|nc|telnet)\b)",
                r"(\||&|;|\$\(|`)",
            ]
        }
    
    def register_user(self, user_id: str, password: str, permissions: List[str] = None) -> bool:
        """Register a new user with encrypted password"""
        try:
            with self.lock:
                if user_id in self.users:
                    return False
                
                # Hash password with bcrypt
                salt = bcrypt.gensalt()
                hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
                
                self.users[user_id] = {
                    'password_hash': hashed_password,
                    'permissions': permissions or ['basic'],
                    'created_at': time.time(),
                    'last_login': None,
                    'security_level': SecurityLevel.MEDIUM
                }
                
                logger.info(f"User registered: {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error registering user {user_id}: {e}")
            return False
    
    def authenticate_user(self, user_id: str, password: str, ip_address: str) -> Tuple[bool, str, Dict]:
        """Authenticate user with rate limiting and threat detection"""
        try:
            with self.lock:
                # Check if user is locked
                if user_id in self.locked_users:
                    lock_until = self.locked_users[user_id]
                    if time.time() < lock_until:
                        remaining = int(lock_until - time.time())
                        self._log_security_event(
                            ThreatType.AUTHENTICATION_FAILURE,
                            SecurityLevel.HIGH,
                            ip_address,
                            user_id,
                            f"Account locked, {remaining}s remaining"
                        )
                        return False, "Account temporarily locked", {}
                    else:
                        del self.locked_users[user_id]
                        self.failed_attempts[user_id] = 0
                
                # Check if user exists
                if user_id not in self.users:
                    self._log_security_event(
                        ThreatType.AUTHENTICATION_FAILURE,
                        SecurityLevel.MEDIUM,
                        ip_address,
                        user_id,
                        "User not found"
                    )
                    return False, "Invalid credentials", {}
                
                # Verify password
                user_data = self.users[user_id]
                if bcrypt.checkpw(password.encode('utf-8'), user_data['password_hash']):
                    # Successful authentication
                    self.failed_attempts[user_id] = 0
                    user_data['last_login'] = time.time()
                    
                    # Generate session token
                    session_data = {
                        'user_id': user_id,
                        'ip_address': ip_address,
                        'created_at': time.time(),
                        'permissions': user_data['permissions']
                    }
                    
                    session_token = jwt.encode(session_data, self.secret_key, algorithm='HS256')
                    
                    logger.info(f"User authenticated: {user_id}")
                    return True, session_token, user_data
                else:
                    # Failed authentication
                    self.failed_attempts[user_id] += 1
                    
                    # Check for lockout
                    if self.failed_attempts[user_id] >= 5:
                        lockout_time = time.time() + 900  # 15 minutes
                        self.locked_users[user_id] = lockout_time
                        
                        self._log_security_event(
                            ThreatType.AUTHENTICATION_FAILURE,
                            SecurityLevel.HIGH,
                            ip_address,
                            user_id,
                            f"Account locked after {self.failed_attempts[user_id]} failed attempts"
                        )
                        return False, "Account locked due to multiple failed attempts", {}
                    
                    self._log_security_event(
                        ThreatType.AUTHENTICATION_FAILURE,
                        SecurityLevel.MEDIUM,
                        ip_address,
                        user_id,
                        f"Failed attempt {self.failed_attempts[user_id]}"
                    )
                    return False, "Invalid credentials", {}
                    
        except Exception as e:
            logger.error(f"Authentication error for {user_id}: {e}")
            return False, "Authentication error", {}
    
    def verify_session(self, session_token: str, ip_address: str) -> Tuple[bool, Dict]:
        """Verify session token and return user data"""
        try:
            # Decode and verify token
            payload = jwt.decode(session_token, self.secret_key, algorithms=['HS256'])
            
            # Check if token is expired (24 hours)
            if time.time() - payload['created_at'] > 86400:
                return False, {}
            
            # Check IP address (optional security measure)
            if payload.get('ip_address') != ip_address:
                self._log_security_event(
                    ThreatType.AUTHENTICATION_FAILURE,
                    SecurityLevel.MEDIUM,
                    ip_address,
                    payload.get('user_id'),
                    "IP address mismatch"
                )
                return False, {}
            
            return True, payload
            
        except jwt.ExpiredSignatureError:
            return False, {}
        except jwt.InvalidTokenError:
            return False, {}
        except Exception as e:
            logger.error(f"Session verification error: {e}")
            return False, {}
    
    def check_permissions(self, user_data: Dict, required_permissions: List[str]) -> bool:
        """Check if user has required permissions"""
        user_permissions = user_data.get('permissions', [])
        return all(perm in user_permissions for perm in required_permissions)
    
    def detect_threats(self, request_data: Dict, ip_address: str) -> Tuple[bool, float, List[str]]:
        """Detect potential security threats in request data"""
        threat_score = 0.0
        detected_threats = []
        
        try:
            # Check for malicious patterns in query
            query = request_data.get('query', '')
            if query:
                for threat_type, patterns in self.threat_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, query, re.IGNORECASE):
                            threat_score += 0.3
                            detected_threats.append(threat_type)
            
            # Check for suspicious user agent
            user_agent = request_data.get('user_agent', '')
            suspicious_agents = ['curl', 'wget', 'python', 'bot', 'crawler']
            if any(agent in user_agent.lower() for agent in suspicious_agents):
                threat_score += 0.1
                detected_threats.append('suspicious_user_agent')
            
            # Check for rapid requests (DDoS detection)
            if self._is_rapid_request(ip_address):
                threat_score += 0.4
                detected_threats.append('rapid_requests')
            
            # Check IP blacklist
            if ip_address in self.ip_blacklist:
                threat_score += 1.0
                detected_threats.append('blacklisted_ip')
            
            return threat_score > self.threat_threshold, threat_score, detected_threats
            
        except Exception as e:
            logger.error(f"Threat detection error: {e}")
            return False, 0.0, []
    
    def _is_rapid_request(self, ip_address: str) -> bool:
        """Check if IP is making rapid requests"""
        current_time = time.time()
        rate_data = self.rate_limits[ip_address]
        
        # Remove old requests
        while rate_data['requests'] and current_time - rate_data['requests'][0] > 60:
            rate_data['requests'].popleft()
        
        # Add current request
        rate_data['requests'].append(current_time)
        
        # Check if too many requests in last minute
        return len(rate_data['requests']) > 100
    
    def _log_security_event(self, threat_type: ThreatType, severity: SecurityLevel, 
                           ip_address: str, user_id: Optional[str], details: str):
        """Log security event for monitoring"""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=threat_type.value,
            severity=severity,
            source_ip=ip_address,
            user_id=user_id,
            session_id=None,
            details={'message': details},
            threat_score=severity.value * 0.25
        )
        
        self.security_events.append(event)
        
        # Trigger alerts for high severity events
        if severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            self._trigger_alert(event)
        
        logger.warning(f"Security event: {threat_type.value} from {ip_address} - {details}")
    
    def _trigger_alert(self, event: SecurityEvent):
        """Trigger security alerts"""
        for callback in self.alert_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            return self.fernet.encrypt(data.encode()).decode()
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            return data
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            return self.fernet.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            return encrypted_data
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        with self.lock:
            return {
                'total_users': len(self.users),
                'active_sessions': len(self.sessions),
                'locked_users': len(self.locked_users),
                'security_events': len(self.security_events),
                'blacklisted_ips': len(self.ip_blacklist),
                'whitelisted_ips': len(self.ip_whitelist),
                'recent_threats': len([e for e in self.security_events 
                                     if time.time() - e.timestamp < 3600])
            }

# Security decorators
def require_authentication(required_permissions: List[str] = None):
    """Decorator to require authentication and optional permissions"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import request, jsonify
            
            # Get session token
            session_token = request.headers.get('Authorization', '').replace('Bearer ', '')
            if not session_token:
                return jsonify({'error': 'Authentication required'}), 401
            
            # Verify session
            security_manager = get_enhanced_security_manager()
            is_valid, user_data = security_manager.verify_session(session_token, request.remote_addr)
            
            if not is_valid:
                return jsonify({'error': 'Invalid or expired session'}), 401
            
            # Check permissions if required
            if required_permissions:
                if not security_manager.check_permissions(user_data, required_permissions):
                    return jsonify({'error': 'Insufficient permissions'}), 403
            
            # Add user data to request
            request.user_data = user_data
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def rate_limit_by_user(max_requests: int = 100, window: int = 3600):
    """Decorator for user-based rate limiting"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import request, jsonify
            
            # Get user ID from session
            session_token = request.headers.get('Authorization', '').replace('Bearer ', '')
            if session_token:
                security_manager = get_enhanced_security_manager()
                is_valid, user_data = security_manager.verify_session(session_token, request.remote_addr)
                if is_valid:
                    user_id = user_data.get('user_id')
                    if security_manager._is_rapid_request(user_id):
                        return jsonify({'error': 'Rate limit exceeded'}), 429
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Global security manager instance
_enhanced_security_manager = None

def get_enhanced_security_manager() -> EnhancedSecurityManager:
    """Get global enhanced security manager instance"""
    global _enhanced_security_manager
    if _enhanced_security_manager is None:
        _enhanced_security_manager = EnhancedSecurityManager()
    return _enhanced_security_manager 