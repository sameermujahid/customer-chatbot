import time
import logging
import traceback
import sys
import threading
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
from collections import defaultdict, deque
import asyncio
from functools import wraps
import signal
import os
import gc
import psutil
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ErrorCategory(Enum):
    NETWORK = "network"
    DATABASE = "database"
    MEMORY = "memory"
    CPU = "cpu"
    GPU = "gpu"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"

class RecoveryStrategy(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    DEGRADED_MODE = "degraded_mode"
    RESTART = "restart"
    IGNORE = "ignore"
    ESCALATE = "escalate"

@dataclass
class ErrorInfo:
    error_id: str
    timestamp: float
    error_type: str
    error_message: str
    stack_trace: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_data: Optional[Dict] = None
    recovery_attempts: int = 0
    resolved: bool = False
    resolution_time: Optional[float] = None

@dataclass
class RecoveryAction:
    action_id: str
    error_id: str
    strategy: RecoveryStrategy
    timestamp: float
    success: bool
    duration: float
    details: Dict[str, Any]

class CircuitBreaker:
    """Circuit breaker pattern for handling repeated failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.RLock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _on_success(self):
        """Handle successful execution"""
        with self.lock:
            self.failure_count = 0
            self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed execution"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"

class ErrorRecoverySystem:
    """
    Advanced error recovery system with automatic detection,
    classification, and intelligent recovery strategies
    """
    
    def __init__(self, 
                 max_error_history: int = 10000,
                 enable_auto_recovery: bool = True,
                 enable_circuit_breakers: bool = True):
        
        # Error tracking
        self.errors = deque(maxlen=max_error_history)
        self.error_patterns = defaultdict(int)
        self.recovery_actions = deque(maxlen=max_error_history)
        
        # Circuit breakers
        self.circuit_breakers = {}
        self.enable_circuit_breakers = enable_circuit_breakers
        
        # Recovery strategies
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.enable_auto_recovery = enable_auto_recovery
        
        # Error classification patterns
        self.error_patterns_db = self._load_error_patterns()
        
        # Performance monitoring
        self.error_stats = {
            'total_errors': 0,
            'resolved_errors': 0,
            'auto_recovered': 0,
            'manual_intervention': 0,
            'avg_recovery_time': 0.0
        }
        
        # Background processing
        self.running = True
        self.cleanup_thread = None
        self._start_background_processing()
        
        logger.info("ErrorRecoverySystem initialized")
    
    def _initialize_recovery_strategies(self) -> Dict[ErrorCategory, List[RecoveryStrategy]]:
        """Initialize recovery strategies for different error categories"""
        return {
            ErrorCategory.NETWORK: [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.CIRCUIT_BREAKER,
                RecoveryStrategy.FALLBACK
            ],
            ErrorCategory.DATABASE: [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.CIRCUIT_BREAKER,
                RecoveryStrategy.DEGRADED_MODE
            ],
            ErrorCategory.MEMORY: [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.DEGRADED_MODE,
                RecoveryStrategy.RESTART
            ],
            ErrorCategory.CPU: [
                RecoveryStrategy.DEGRADED_MODE,
                RecoveryStrategy.RETRY
            ],
            ErrorCategory.GPU: [
                RecoveryStrategy.FALLBACK,
                RecoveryStrategy.DEGRADED_MODE
            ],
            ErrorCategory.AUTHENTICATION: [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.ESCALATE
            ],
            ErrorCategory.AUTHORIZATION: [
                RecoveryStrategy.ESCALATE
            ],
            ErrorCategory.VALIDATION: [
                RecoveryStrategy.IGNORE,
                RecoveryStrategy.FALLBACK
            ],
            ErrorCategory.TIMEOUT: [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.CIRCUIT_BREAKER
            ],
            ErrorCategory.RESOURCE_EXHAUSTION: [
                RecoveryStrategy.DEGRADED_MODE,
                RecoveryStrategy.RESTART
            ],
            ErrorCategory.CONFIGURATION: [
                RecoveryStrategy.ESCALATE
            ],
            ErrorCategory.DEPENDENCY: [
                RecoveryStrategy.FALLBACK,
                RecoveryStrategy.DEGRADED_MODE
            ],
            ErrorCategory.UNKNOWN: [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.ESCALATE
            ]
        }
    
    def _load_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load error classification patterns"""
        return {
            # Network errors
            "ConnectionError": {
                "category": ErrorCategory.NETWORK,
                "severity": ErrorSeverity.MEDIUM,
                "retryable": True
            },
            "TimeoutError": {
                "category": ErrorCategory.TIMEOUT,
                "severity": ErrorSeverity.MEDIUM,
                "retryable": True
            },
            "ConnectionRefusedError": {
                "category": ErrorCategory.NETWORK,
                "severity": ErrorSeverity.HIGH,
                "retryable": True
            },
            
            # Database errors
            "DatabaseError": {
                "category": ErrorCategory.DATABASE,
                "severity": ErrorSeverity.HIGH,
                "retryable": True
            },
            "IntegrityError": {
                "category": ErrorCategory.DATABASE,
                "severity": ErrorSeverity.HIGH,
                "retryable": False
            },
            
            # Memory errors
            "MemoryError": {
                "category": ErrorCategory.MEMORY,
                "severity": ErrorSeverity.CRITICAL,
                "retryable": True
            },
            "OutOfMemoryError": {
                "category": ErrorCategory.MEMORY,
                "severity": ErrorSeverity.CRITICAL,
                "retryable": True
            },
            
            # Authentication errors
            "AuthenticationError": {
                "category": ErrorCategory.AUTHENTICATION,
                "severity": ErrorSeverity.HIGH,
                "retryable": False
            },
            "PermissionError": {
                "category": ErrorCategory.AUTHORIZATION,
                "severity": ErrorSeverity.HIGH,
                "retryable": False
            },
            
            # Validation errors
            "ValueError": {
                "category": ErrorCategory.VALIDATION,
                "severity": ErrorSeverity.LOW,
                "retryable": False
            },
            "TypeError": {
                "category": ErrorCategory.VALIDATION,
                "severity": ErrorSeverity.LOW,
                "retryable": False
            },
            
            # Resource errors
            "ResourceExhaustedError": {
                "category": ErrorCategory.RESOURCE_EXHAUSTION,
                "severity": ErrorSeverity.CRITICAL,
                "retryable": True
            }
        }
    
    def _start_background_processing(self):
        """Start background error processing threads"""
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_errors, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_old_errors(self):
        """Clean up old error records"""
        while self.running:
            try:
                current_time = time.time()
                cutoff_time = current_time - 86400  # 24 hours
                
                # Remove old errors
                old_errors = [e for e in self.errors if e.timestamp < cutoff_time]
                for error in old_errors:
                    self.errors.remove(error)
                
                # Remove old recovery actions
                old_actions = [a for a in self.recovery_actions if a.timestamp < cutoff_time]
                for action in old_actions:
                    self.recovery_actions.remove(action)
                
                if old_errors or old_actions:
                    logger.info(f"Cleaned up {len(old_errors)} old errors and {len(old_actions)} old actions")
                
                time.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                logger.error(f"Error in cleanup thread: {e}")
                time.sleep(3600)
    
    def classify_error(self, error: Exception, context: Dict[str, Any] = None) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify error based on type and context"""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Check pattern database
        if error_type in self.error_patterns_db:
            pattern = self.error_patterns_db[error_type]
            return pattern["category"], pattern["severity"]
        
        # Check error message for patterns
        error_lower = error_message.lower()
        
        if any(word in error_lower for word in ["connection", "network", "timeout", "refused"]):
            return ErrorCategory.NETWORK, ErrorSeverity.MEDIUM
        elif any(word in error_lower for word in ["database", "sql", "query"]):
            return ErrorCategory.DATABASE, ErrorSeverity.HIGH
        elif any(word in error_lower for word in ["memory", "out of memory"]):
            return ErrorCategory.MEMORY, ErrorSeverity.CRITICAL
        elif any(word in error_lower for word in ["authentication", "login", "password"]):
            return ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH
        elif any(word in error_lower for word in ["permission", "access", "authorization"]):
            return ErrorCategory.AUTHORIZATION, ErrorSeverity.HIGH
        elif any(word in error_lower for word in ["validation", "invalid", "type"]):
            return ErrorCategory.VALIDATION, ErrorSeverity.LOW
        elif any(word in error_lower for word in ["resource", "exhausted", "limit"]):
            return ErrorCategory.RESOURCE_EXHAUSTION, ErrorSeverity.CRITICAL
        
        # Default classification
        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM
    
    def record_error(self, error: Exception, context: Dict[str, Any] = None, 
                    user_id: Optional[str] = None, session_id: Optional[str] = None,
                    request_data: Optional[Dict] = None) -> str:
        """Record an error for analysis and recovery"""
        
        error_id = str(hashlib.md5(f"{time.time()}{str(error)}".encode()).hexdigest())
        
        # Classify error
        category, severity = self.classify_error(error, context)
        
        # Create error info
        error_info = ErrorInfo(
            error_id=error_id,
            timestamp=time.time(),
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            severity=severity,
            category=category,
            context=context or {},
            user_id=user_id,
            session_id=session_id,
            request_data=request_data
        )
        
        # Store error
        self.errors.append(error_info)
        self.error_patterns[error_type] += 1
        self.error_stats['total_errors'] += 1
        
        # Log error
        logger.error(f"Error recorded: {error_id} - {error_type}: {str(error)}")
        
        # Attempt automatic recovery if enabled
        if self.enable_auto_recovery:
            self._attempt_auto_recovery(error_info)
        
        return error_id
    
    def _attempt_auto_recovery(self, error_info: ErrorInfo):
        """Attempt automatic error recovery"""
        try:
            # Get recovery strategies for this error category
            strategies = self.recovery_strategies.get(error_info.category, [])
            
            for strategy in strategies:
                action_id = self._execute_recovery_strategy(error_info, strategy)
                if action_id:
                    # Check if recovery was successful
                    action = self._get_recovery_action(action_id)
                    if action and action.success:
                        error_info.resolved = True
                        error_info.resolution_time = time.time()
                        self.error_stats['auto_recovered'] += 1
                        self.error_stats['resolved_errors'] += 1
                        logger.info(f"Auto-recovered error {error_info.error_id} using {strategy.value}")
                        break
                    else:
                        error_info.recovery_attempts += 1
            
            # If no automatic recovery worked, escalate
            if not error_info.resolved:
                self._escalate_error(error_info)
                
        except Exception as e:
            logger.error(f"Error in auto-recovery: {e}")
    
    def _execute_recovery_strategy(self, error_info: ErrorInfo, strategy: RecoveryStrategy) -> Optional[str]:
        """Execute a specific recovery strategy"""
        action_id = str(hashlib.md5(f"{error_info.error_id}{strategy.value}{time.time()}".encode()).hexdigest())
        start_time = time.time()
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                success = self._retry_operation(error_info)
            elif strategy == RecoveryStrategy.FALLBACK:
                success = self._fallback_operation(error_info)
            elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                success = self._circuit_breaker_operation(error_info)
            elif strategy == RecoveryStrategy.DEGRADED_MODE:
                success = self._degraded_mode_operation(error_info)
            elif strategy == RecoveryStrategy.RESTART:
                success = self._restart_operation(error_info)
            elif strategy == RecoveryStrategy.IGNORE:
                success = True  # Ignoring is always "successful"
            elif strategy == RecoveryStrategy.ESCALATE:
                success = self._escalate_error(error_info)
            else:
                success = False
            
            duration = time.time() - start_time
            
            # Record recovery action
            action = RecoveryAction(
                action_id=action_id,
                error_id=error_info.error_id,
                strategy=strategy,
                timestamp=time.time(),
                success=success,
                duration=duration,
                details={'error_type': error_info.error_type}
            )
            
            self.recovery_actions.append(action)
            return action_id
            
        except Exception as e:
            logger.error(f"Error executing recovery strategy {strategy.value}: {e}")
            return None
    
    def _retry_operation(self, error_info: ErrorInfo) -> bool:
        """Retry the failed operation"""
        try:
            # Get the original function and arguments from context
            func = error_info.context.get('function')
            args = error_info.context.get('args', ())
            kwargs = error_info.context.get('kwargs', {})
            
            if func and callable(func):
                # Simple retry with exponential backoff
                for attempt in range(3):
                    try:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        func(*args, **kwargs)
                        return True
                    except Exception as e:
                        if attempt == 2:  # Last attempt
                            raise e
                        continue
            
            return False
            
        except Exception as e:
            logger.error(f"Retry operation failed: {e}")
            return False
    
    def _fallback_operation(self, error_info: ErrorInfo) -> bool:
        """Use fallback operation"""
        try:
            # Get fallback function from context
            fallback_func = error_info.context.get('fallback_function')
            
            if fallback_func and callable(fallback_func):
                fallback_func()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Fallback operation failed: {e}")
            return False
    
    def _circuit_breaker_operation(self, error_info: ErrorInfo) -> bool:
        """Use circuit breaker pattern"""
        try:
            operation_key = error_info.context.get('operation_key', 'default')
            
            if operation_key not in self.circuit_breakers:
                self.circuit_breakers[operation_key] = CircuitBreaker()
            
            circuit_breaker = self.circuit_breakers[operation_key]
            func = error_info.context.get('function')
            
            if func and callable(func):
                circuit_breaker.call(func, *error_info.context.get('args', ()), 
                                   **error_info.context.get('kwargs', {}))
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Circuit breaker operation failed: {e}")
            return False
    
    def _degraded_mode_operation(self, error_info: ErrorInfo) -> bool:
        """Switch to degraded mode"""
        try:
            # Implement degraded mode logic
            # This could involve using simpler algorithms, cached data, etc.
            logger.info(f"Switching to degraded mode for error {error_info.error_id}")
            return True
            
        except Exception as e:
            logger.error(f"Degraded mode operation failed: {e}")
            return False
    
    def _restart_operation(self, error_info: ErrorInfo) -> bool:
        """Restart the operation or component"""
        try:
            # This is a placeholder for restart logic
            # In a real implementation, this might restart a service, clear caches, etc.
            logger.info(f"Restart operation triggered for error {error_info.error_id}")
            return True
            
        except Exception as e:
            logger.error(f"Restart operation failed: {e}")
            return False
    
    def _escalate_error(self, error_info: ErrorInfo) -> bool:
        """Escalate error for manual intervention"""
        try:
            # Log escalation
            logger.warning(f"Error escalated for manual intervention: {error_info.error_id}")
            
            # In a real implementation, this might:
            # - Send notifications
            # - Create tickets
            # - Trigger alerts
            # - Call emergency procedures
            
            self.error_stats['manual_intervention'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error escalation failed: {e}")
            return False
    
    def _get_recovery_action(self, action_id: str) -> Optional[RecoveryAction]:
        """Get recovery action by ID"""
        for action in self.recovery_actions:
            if action.action_id == action_id:
                return action
        return None
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        current_time = time.time()
        
        # Calculate recent errors (last hour)
        recent_errors = [e for e in self.errors if current_time - e.timestamp < 3600]
        
        # Calculate recovery success rate
        total_actions = len(self.recovery_actions)
        successful_actions = len([a for a in self.recovery_actions if a.success])
        recovery_success_rate = successful_actions / total_actions if total_actions > 0 else 0.0
        
        # Calculate average recovery time
        recovery_times = [a.duration for a in self.recovery_actions if a.success]
        avg_recovery_time = sum(recovery_times) / len(recovery_times) if recovery_times else 0.0
        
        return {
            **self.error_stats,
            'recent_errors': len(recent_errors),
            'recovery_success_rate': recovery_success_rate,
            'avg_recovery_time': avg_recovery_time,
            'error_categories': {cat.value: len([e for e in self.errors if e.category == cat]) 
                               for cat in ErrorCategory},
            'top_error_types': dict(sorted(self.error_patterns.items(), 
                                         key=lambda x: x[1], reverse=True)[:10]),
            'active_circuit_breakers': len([cb for cb in self.circuit_breakers.values() 
                                          if cb.state == "OPEN"])
        }
    
    def get_error_by_id(self, error_id: str) -> Optional[ErrorInfo]:
        """Get error information by ID"""
        for error in self.errors:
            if error.error_id == error_id:
                return error
        return None
    
    def resolve_error(self, error_id: str, resolution_notes: str = None):
        """Manually resolve an error"""
        error = self.get_error_by_id(error_id)
        if error:
            error.resolved = True
            error.resolution_time = time.time()
            self.error_stats['resolved_errors'] += 1
            logger.info(f"Error {error_id} manually resolved: {resolution_notes}")

# Global error recovery system instance
_error_recovery_system = None

def get_error_recovery_system() -> ErrorRecoverySystem:
    """Get global error recovery system instance"""
    global _error_recovery_system
    if _error_recovery_system is None:
        _error_recovery_system = ErrorRecoverySystem()
    return _error_recovery_system

# Decorator for automatic error handling
def handle_errors(recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
                 max_retries: int = 3,
                 fallback_func: Callable = None):
    """Decorator for automatic error handling and recovery"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            error_system = get_error_recovery_system()
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Record error
                    context = {
                        'function': func,
                        'args': args,
                        'kwargs': kwargs,
                        'attempt': attempt,
                        'fallback_function': fallback_func
                    }
                    
                    error_id = error_system.record_error(e, context)
                    
                    if attempt == max_retries:
                        # Final attempt failed, try fallback
                        if fallback_func:
                            try:
                                logger.info(f"Using fallback function for {func.__name__}")
                                return fallback_func(*args, **kwargs)
                            except Exception as fallback_error:
                                logger.error(f"Fallback function also failed: {fallback_error}")
                        
                        # Re-raise the original error
                        raise e
                    
                    # Wait before retry
                    time.sleep(2 ** attempt)
            
        return wrapper
    return decorator

# Context manager for error handling
@contextmanager
def error_context(operation_name: str = "operation", 
                 user_id: Optional[str] = None,
                 session_id: Optional[str] = None):
    """Context manager for error handling"""
    error_system = get_error_recovery_system()
    
    try:
        yield
    except Exception as e:
        context = {
            'operation_name': operation_name,
            'timestamp': time.time()
        }
        
        error_id = error_system.record_error(e, context, user_id, session_id)
        logger.error(f"Error in {operation_name}: {e} (ID: {error_id})")
        raise 