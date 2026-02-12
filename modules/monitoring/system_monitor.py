import logging
import time
import psutil
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import os
from collections import deque
import traceback

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available: int
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    request_count: int
    error_count: int
    avg_response_time: float

@dataclass
class ApplicationMetrics:
    timestamp: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    active_users: int
    rag_queries: int
    chat_messages: int
    cache_hits: int
    cache_misses: int

class SystemMonitor:
    """
    Comprehensive system monitoring and logging
    """
    
    def __init__(self, log_dir: str = "logs", metrics_retention_hours: int = 24):
        self.log_dir = log_dir
        self.metrics_retention_hours = metrics_retention_hours
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Metrics storage
        self.system_metrics: deque = deque(maxlen=1000)
        self.application_metrics: deque = deque(maxlen=1000)
        self.error_log: deque = deque(maxlen=1000)
        self.performance_log: deque = deque(maxlen=1000)
        
        # Current metrics
        self.current_metrics = {
            'request_count': 0,
            'error_count': 0,
            'total_response_time': 0.0,
            'active_users': 0,
            'rag_queries': 0,
            'chat_messages': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Monitoring thread
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # Setup logging
        self._setup_logging()
        
        logger.info("System Monitor initialized")
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start system monitoring"""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"System monitoring started with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics.append(system_metrics)
                
                # Collect application metrics
                app_metrics = self._collect_application_metrics()
                self.application_metrics.append(app_metrics)
                
                # Check for alerts
                self._check_alerts(system_metrics, app_metrics)
                
                # Clean up old metrics
                self._cleanup_old_metrics()
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # Network usage
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Application-specific metrics
            active_connections = self.current_metrics['active_users']
            request_count = self.current_metrics['request_count']
            error_count = self.current_metrics['error_count']
            
            # Calculate average response time
            avg_response_time = 0.0
            if request_count > 0:
                avg_response_time = self.current_metrics['total_response_time'] / request_count
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available=memory_available,
                disk_usage_percent=disk_usage_percent,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                active_connections=active_connections,
                request_count=request_count,
                error_count=error_count,
                avg_response_time=avg_response_time
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            # Return default metrics
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available=0,
                disk_usage_percent=0.0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                active_connections=0,
                request_count=0,
                error_count=0,
                avg_response_time=0.0
            )
    
    def _collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-level metrics"""
        try:
            metrics = ApplicationMetrics(
                timestamp=datetime.now(),
                total_requests=self.current_metrics['request_count'],
                successful_requests=self.current_metrics['request_count'] - self.current_metrics['error_count'],
                failed_requests=self.current_metrics['error_count'],
                avg_response_time=self.current_metrics['total_response_time'] / max(1, self.current_metrics['request_count']),
                active_users=self.current_metrics['active_users'],
                rag_queries=self.current_metrics['rag_queries'],
                chat_messages=self.current_metrics['chat_messages'],
                cache_hits=self.current_metrics['cache_hits'],
                cache_misses=self.current_metrics['cache_misses']
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
            return ApplicationMetrics(
                timestamp=datetime.now(),
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_response_time=0.0,
                active_users=0,
                rag_queries=0,
                chat_messages=0,
                cache_hits=0,
                cache_misses=0
            )
    
    def _check_alerts(self, system_metrics: SystemMetrics, app_metrics: ApplicationMetrics):
        """Check for system alerts"""
        try:
            alerts = []
            
            # CPU usage alert
            if system_metrics.cpu_percent > 80:
                alerts.append(f"High CPU usage: {system_metrics.cpu_percent:.1f}%")
            
            # Memory usage alert
            if system_metrics.memory_percent > 85:
                alerts.append(f"High memory usage: {system_metrics.memory_percent:.1f}%")
            
            # Disk usage alert
            if system_metrics.disk_usage_percent > 90:
                alerts.append(f"High disk usage: {system_metrics.disk_usage_percent:.1f}%")
            
            # Error rate alert
            if app_metrics.total_requests > 0:
                error_rate = (app_metrics.failed_requests / app_metrics.total_requests) * 100
                if error_rate > 10:
                    alerts.append(f"High error rate: {error_rate:.1f}%")
            
            # Response time alert
            if app_metrics.avg_response_time > 5.0:
                alerts.append(f"Slow response time: {app_metrics.avg_response_time:.2f}s")
            
            # Log alerts
            for alert in alerts:
                self.log_alert(alert)
                
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def log_request(self, endpoint: str, method: str, response_time: float, status_code: int, user_id: Optional[str] = None):
        """Log a request"""
        try:
            # Update current metrics
            self.current_metrics['request_count'] += 1
            self.current_metrics['total_response_time'] += response_time
            
            if status_code >= 400:
                self.current_metrics['error_count'] += 1
            
            # Log performance
            performance_entry = {
                'timestamp': datetime.now().isoformat(),
                'endpoint': endpoint,
                'method': method,
                'response_time': response_time,
                'status_code': status_code,
                'user_id': user_id
            }
            
            self.performance_log.append(performance_entry)
            
            # Log to file
            logger.info(f"Request: {method} {endpoint} - {status_code} - {response_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error logging request: {e}")
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log an error"""
        try:
            error_entry = {
                'timestamp': datetime.now().isoformat(),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'traceback': traceback.format_exc(),
                'context': context or {}
            }
            
            self.error_log.append(error_entry)
            
            # Log to file
            logger.error(f"Error: {type(error).__name__}: {str(error)}")
            if context:
                logger.error(f"Context: {context}")
            
        except Exception as e:
            logger.error(f"Error logging error: {e}")
    
    def log_alert(self, alert_message: str, severity: str = "warning"):
        """Log an alert"""
        try:
            alert_entry = {
                'timestamp': datetime.now().isoformat(),
                'severity': severity,
                'message': alert_message
            }
            
            # Log to file
            logger.warning(f"ALERT [{severity.upper()}]: {alert_message}")
            
        except Exception as e:
            logger.error(f"Error logging alert: {e}")
    
    def update_metrics(self, metrics_updates: Dict[str, Any]):
        """Update current metrics"""
        try:
            for key, value in metrics_updates.items():
                if key in self.current_metrics:
                    if isinstance(value, (int, float)):
                        self.current_metrics[key] = value
                    elif isinstance(value, str) and value.isdigit():
                        self.current_metrics[key] = int(value)
                        
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def get_system_metrics(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get system metrics for the last N hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            metrics = []
            
            for metric in self.system_metrics:
                if metric.timestamp >= cutoff_time:
                    metrics.append(asdict(metric))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return []
    
    def get_application_metrics(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get application metrics for the last N hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            metrics = []
            
            for metric in self.application_metrics:
                if metric.timestamp >= cutoff_time:
                    metrics.append(asdict(metric))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting application metrics: {e}")
            return []
    
    def get_error_log(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get error log for the last N hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            errors = []
            
            for error in self.error_log:
                if datetime.fromisoformat(error['timestamp']) >= cutoff_time:
                    errors.append(error)
            
            return errors
            
        except Exception as e:
            logger.error(f"Error getting error log: {e}")
            return []
    
    def get_performance_log(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get performance log for the last N hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            performance = []
            
            for entry in self.performance_log:
                if datetime.fromisoformat(entry['timestamp']) >= cutoff_time:
                    performance.append(entry)
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting performance log: {e}")
            return []
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            # Get latest metrics
            latest_system = list(self.system_metrics)[-1] if self.system_metrics else None
            latest_app = list(self.application_metrics)[-1] if self.application_metrics else None
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'monitoring_active': self.is_monitoring,
                'current_metrics': self.current_metrics.copy()
            }
            
            if latest_system:
                status['system'] = {
                    'cpu_percent': latest_system.cpu_percent,
                    'memory_percent': latest_system.memory_percent,
                    'disk_usage_percent': latest_system.disk_usage_percent,
                    'active_connections': latest_system.active_connections
                }
            
            if latest_app:
                status['application'] = {
                    'total_requests': latest_app.total_requests,
                    'error_rate': (latest_app.failed_requests / max(1, latest_app.total_requests)) * 100,
                    'avg_response_time': latest_app.avg_response_time,
                    'active_users': latest_app.active_users
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting current status: {e}")
            return {'error': str(e)}
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
            
            # Clean system metrics
            while self.system_metrics and self.system_metrics[0].timestamp < cutoff_time:
                self.system_metrics.popleft()
            
            # Clean application metrics
            while self.application_metrics and self.application_metrics[0].timestamp < cutoff_time:
                self.application_metrics.popleft()
            
            # Clean error log
            while self.error_log and datetime.fromisoformat(self.error_log[0]['timestamp']) < cutoff_time:
                self.error_log.popleft()
            
            # Clean performance log
            while self.performance_log and datetime.fromisoformat(self.performance_log[0]['timestamp']) < cutoff_time:
                self.performance_log.popleft()
                
        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {e}")
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        try:
            # Create formatters
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # File handlers
            log_file = os.path.join(self.log_dir, 'application.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            
            error_file = os.path.join(self.log_dir, 'errors.log')
            error_handler = logging.FileHandler(error_file)
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            
            performance_file = os.path.join(self.log_dir, 'performance.log')
            performance_handler = logging.FileHandler(performance_file)
            performance_handler.setFormatter(formatter)
            
            # Add handlers to root logger
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
            root_logger.addHandler(error_handler)
            root_logger.addHandler(performance_handler)
            
            logger.info("Logging setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up logging: {e}")
    
    def export_metrics(self, filepath: str, hours: int = 24):
        """Export metrics to JSON file"""
        try:
            data = {
                'export_timestamp': datetime.now().isoformat(),
                'system_metrics': self.get_system_metrics(hours),
                'application_metrics': self.get_application_metrics(hours),
                'error_log': self.get_error_log(hours),
                'performance_log': self.get_performance_log(hours)
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")

# Global system monitor instance
_system_monitor = None

def get_system_monitor() -> SystemMonitor:
    """Get global system monitor instance"""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
    return _system_monitor
