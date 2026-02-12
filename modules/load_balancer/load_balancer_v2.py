import threading
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

@dataclass
class ServerInstance:
    instance_id: str
    host: str
    port: int
    health_check_url: str
    max_connections: int
    current_connections: int = 0
    is_healthy: bool = True
    last_health_check: datetime = None
    response_time: float = 0.0
    error_count: int = 0
    success_count: int = 0
    created_at: datetime = None

@dataclass
class LoadBalancerConfig:
    health_check_interval: int = 30  # seconds
    health_check_timeout: int = 5    # seconds
    max_retries: int = 3
    retry_delay: float = 1.0         # seconds
    load_balancing_algorithm: str = "round_robin"  # round_robin, least_connections, weighted
    auto_scaling: bool = True
    min_instances: int = 1
    max_instances: int = 10
    scale_up_threshold: float = 0.8  # 80% CPU/memory
    scale_down_threshold: float = 0.3  # 30% CPU/memory

class LoadBalancerV2:
    """
    Advanced load balancer with auto-scaling capabilities
    """
    
    def __init__(self, config: LoadBalancerConfig = None):
        self.config = config or LoadBalancerConfig()
        
        # Server instances
        self.instances: Dict[str, ServerInstance] = {}
        self.healthy_instances: List[str] = []
        
        # Load balancing state
        self.current_index = 0
        self.request_queue = deque()
        self.processing_requests = 0
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'requests_per_second': 0.0
        }
        
        # Threading
        self.lock = threading.RLock()
        self.health_check_thread = None
        self.auto_scaling_thread = None
        self.is_running = False
        
        # Request tracking
        self.request_times = deque(maxlen=1000)
        
        logger.info("Load Balancer V2 initialized")
    
    def add_instance(self, instance_id: str, host: str, port: int, 
                    health_check_url: str = None, max_connections: int = 100) -> bool:
        """Add a new server instance"""
        try:
            with self.lock:
                if instance_id in self.instances:
                    logger.warning(f"Instance {instance_id} already exists")
                    return False
                
                instance = ServerInstance(
                    instance_id=instance_id,
                    host=host,
                    port=port,
                    health_check_url=health_check_url or f"http://{host}:{port}/health",
                    max_connections=max_connections,
                    created_at=datetime.now()
                )
                
                self.instances[instance_id] = instance
                
                # Add to healthy instances if health check passes
                if self._health_check_instance(instance):
                    self.healthy_instances.append(instance_id)
                
                logger.info(f"Added instance: {instance_id} ({host}:{port})")
                return True
                
        except Exception as e:
            logger.error(f"Error adding instance {instance_id}: {e}")
            return False
    
    def remove_instance(self, instance_id: str) -> bool:
        """Remove a server instance"""
        try:
            with self.lock:
                if instance_id not in self.instances:
                    return False
                
                # Remove from instances
                del self.instances[instance_id]
                
                # Remove from healthy instances
                if instance_id in self.healthy_instances:
                    self.healthy_instances.remove(instance_id)
                
                logger.info(f"Removed instance: {instance_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error removing instance {instance_id}: {e}")
            return False
    
    def get_next_instance(self) -> Optional[ServerInstance]:
        """Get next available instance based on load balancing algorithm"""
        try:
            with self.lock:
                if not self.healthy_instances:
                    return None
                
                if self.config.load_balancing_algorithm == "round_robin":
                    return self._round_robin_selection()
                elif self.config.load_balancing_algorithm == "least_connections":
                    return self._least_connections_selection()
                elif self.config.load_balancing_algorithm == "weighted":
                    return self._weighted_selection()
                else:
                    return self._round_robin_selection()
                    
        except Exception as e:
            logger.error(f"Error getting next instance: {e}")
            return None
    
    def _round_robin_selection(self) -> Optional[ServerInstance]:
        """Round-robin instance selection"""
        if not self.healthy_instances:
            return None
        
        instance_id = self.healthy_instances[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.healthy_instances)
        
        return self.instances.get(instance_id)
    
    def _least_connections_selection(self) -> Optional[ServerInstance]:
        """Least connections instance selection"""
        if not self.healthy_instances:
            return None
        
        min_connections = float('inf')
        selected_instance = None
        
        for instance_id in self.healthy_instances:
            instance = self.instances[instance_id]
            if instance.current_connections < min_connections:
                min_connections = instance.current_connections
                selected_instance = instance
        
        return selected_instance
    
    def _weighted_selection(self) -> Optional[ServerInstance]:
        """Weighted instance selection based on performance"""
        if not self.healthy_instances:
            return None
        
        # Calculate weights based on response time and error rate
        total_weight = 0
        weighted_instances = []
        
        for instance_id in self.healthy_instances:
            instance = self.instances[instance_id]
            
            # Weight based on response time (lower is better)
            response_weight = max(0.1, 1.0 / (instance.response_time + 0.1))
            
            # Weight based on error rate (lower is better)
            total_requests = instance.success_count + instance.error_count
            error_rate = instance.error_count / max(1, total_requests)
            error_weight = max(0.1, 1.0 - error_rate)
            
            # Combined weight
            weight = response_weight * error_weight
            total_weight += weight
            weighted_instances.append((instance, weight))
        
        if total_weight == 0:
            return self._round_robin_selection()
        
        # Select based on weights
        random_value = random.uniform(0, total_weight)
        current_weight = 0
        
        for instance, weight in weighted_instances:
            current_weight += weight
            if random_value <= current_weight:
                return instance
        
        return weighted_instances[0][0] if weighted_instances else None
    
    def route_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Route a request to an available instance"""
        start_time = time.time()
        
        try:
            # Get next available instance
            instance = self.get_next_instance()
            if not instance:
                return {
                    'success': False,
                    'error': 'No available instances',
                    'response_time': time.time() - start_time
                }
            
            # Check if instance can handle more connections
            if instance.current_connections >= instance.max_connections:
                return {
                    'success': False,
                    'error': 'Instance at capacity',
                    'response_time': time.time() - start_time
                }
            
            # Increment connection count
            with self.lock:
                instance.current_connections += 1
            
            try:
                # Route request to instance
                result = self._send_request_to_instance(instance, request_data)
                
                # Update instance stats
                with self.lock:
                    instance.current_connections -= 1
                    instance.success_count += 1
                    instance.response_time = (instance.response_time + result['response_time']) / 2
                
                # Update load balancer stats
                self._update_stats(result['success'], result['response_time'])
                
                return result
                
            except Exception as e:
                # Handle instance failure
                with self.lock:
                    instance.current_connections -= 1
                    instance.error_count += 1
                
                logger.error(f"Error routing request to {instance.instance_id}: {e}")
                
                # Retry with different instance
                return self._retry_request(request_data, start_time)
                
        except Exception as e:
            logger.error(f"Error in route_request: {e}")
            return {
                'success': False,
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    def _send_request_to_instance(self, instance: ServerInstance, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to specific instance"""
        # This is a placeholder - in real implementation, you'd make HTTP request
        # For now, simulate request processing
        time.sleep(random.uniform(0.1, 0.5))  # Simulate processing time
        
        # Simulate success/failure
        success = random.random() > 0.1  # 90% success rate
        
        return {
            'success': success,
            'instance_id': instance.instance_id,
            'response_time': random.uniform(0.1, 2.0),
            'data': f"Response from {instance.instance_id}" if success else None,
            'error': "Simulated error" if not success else None
        }
    
    def _retry_request(self, request_data: Dict[str, Any], start_time: float, retry_count: int = 0) -> Dict[str, Any]:
        """Retry request with different instance"""
        if retry_count >= self.config.max_retries:
            return {
                'success': False,
                'error': 'Max retries exceeded',
                'response_time': time.time() - start_time
            }
        
        # Wait before retry
        time.sleep(self.config.retry_delay)
        
        # Try again
        return self.route_request(request_data)
    
    def start_health_checks(self):
        """Start health check monitoring"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start health check thread
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self.health_check_thread.start()
        
        # Start auto-scaling thread if enabled
        if self.config.auto_scaling:
            self.auto_scaling_thread = threading.Thread(
                target=self._auto_scaling_loop,
                daemon=True
            )
            self.auto_scaling_thread.start()
        
        logger.info("Load balancer health checks started")
    
    def stop_health_checks(self):
        """Stop health check monitoring"""
        self.is_running = False
        
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)
        
        if self.auto_scaling_thread:
            self.auto_scaling_thread.join(timeout=5)
        
        logger.info("Load balancer health checks stopped")
    
    def _health_check_loop(self):
        """Health check monitoring loop"""
        while self.is_running:
            try:
                with self.lock:
                    for instance_id, instance in self.instances.items():
                        is_healthy = self._health_check_instance(instance)
                        
                        if is_healthy and instance_id not in self.healthy_instances:
                            self.healthy_instances.append(instance_id)
                            logger.info(f"Instance {instance_id} is now healthy")
                        elif not is_healthy and instance_id in self.healthy_instances:
                            self.healthy_instances.remove(instance_id)
                            logger.warning(f"Instance {instance_id} is now unhealthy")
                
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(self.config.health_check_interval)
    
    def _health_check_instance(self, instance: ServerInstance) -> bool:
        """Perform health check on instance"""
        try:
            # This is a placeholder - in real implementation, you'd make HTTP request
            # For now, simulate health check
            time.sleep(0.1)  # Simulate network delay
            
            # Simulate health check result
            is_healthy = random.random() > 0.05  # 95% healthy rate
            
            instance.is_healthy = is_healthy
            instance.last_health_check = datetime.now()
            
            return is_healthy
            
        except Exception as e:
            logger.error(f"Error checking health of {instance.instance_id}: {e}")
            instance.is_healthy = False
            instance.last_health_check = datetime.now()
            return False
    
    def _auto_scaling_loop(self):
        """Auto-scaling monitoring loop"""
        while self.is_running:
            try:
                self._check_scaling_needs()
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in auto-scaling loop: {e}")
                time.sleep(60)
    
    def _check_scaling_needs(self):
        """Check if scaling is needed"""
        try:
            with self.lock:
                # Calculate current load
                total_connections = sum(instance.current_connections for instance in self.instances.values())
                total_capacity = sum(instance.max_connections for instance in self.instances.values())
                
                if total_capacity == 0:
                    return
                
                current_load = total_connections / total_capacity
                
                # Check if we need to scale up
                if (current_load > self.config.scale_up_threshold and 
                    len(self.instances) < self.config.max_instances):
                    self._scale_up()
                
                # Check if we need to scale down
                elif (current_load < self.config.scale_down_threshold and 
                      len(self.instances) > self.config.min_instances):
                    self._scale_down()
                    
        except Exception as e:
            logger.error(f"Error checking scaling needs: {e}")
    
    def _scale_up(self):
        """Scale up by adding new instance"""
        try:
            # Generate new instance ID
            new_instance_id = f"instance_{len(self.instances) + 1}"
            
            # Add new instance (placeholder implementation)
            success = self.add_instance(
                instance_id=new_instance_id,
                host=f"server{len(self.instances) + 1}",
                port=8000 + len(self.instances),
                max_connections=100
            )
            
            if success:
                logger.info(f"Scaled up: Added instance {new_instance_id}")
            else:
                logger.error(f"Failed to scale up: Could not add instance {new_instance_id}")
                
        except Exception as e:
            logger.error(f"Error scaling up: {e}")
    
    def _scale_down(self):
        """Scale down by removing instance"""
        try:
            # Find instance with least load
            if not self.instances:
                return
            
            least_loaded_instance = min(
                self.instances.values(),
                key=lambda x: x.current_connections
            )
            
            # Remove instance
            success = self.remove_instance(least_loaded_instance.instance_id)
            
            if success:
                logger.info(f"Scaled down: Removed instance {least_loaded_instance.instance_id}")
            else:
                logger.error(f"Failed to scale down: Could not remove instance {least_loaded_instance.instance_id}")
                
        except Exception as e:
            logger.error(f"Error scaling down: {e}")
    
    def _update_stats(self, success: bool, response_time: float):
        """Update load balancer statistics"""
        try:
            self.stats['total_requests'] += 1
            
            if success:
                self.stats['successful_requests'] += 1
            else:
                self.stats['failed_requests'] += 1
            
            # Update average response time
            self.request_times.append(response_time)
            self.stats['avg_response_time'] = sum(self.request_times) / len(self.request_times)
            
            # Calculate requests per second (simplified)
            if len(self.request_times) > 1:
                time_span = self.request_times[-1] - self.request_times[0]
                if time_span > 0:
                    self.stats['requests_per_second'] = len(self.request_times) / time_span
                    
        except Exception as e:
            logger.error(f"Error updating stats: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        try:
            with self.lock:
                instance_stats = []
                for instance in self.instances.values():
                    instance_stats.append({
                        'instance_id': instance.instance_id,
                        'host': instance.host,
                        'port': instance.port,
                        'is_healthy': instance.is_healthy,
                        'current_connections': instance.current_connections,
                        'max_connections': instance.max_connections,
                        'response_time': instance.response_time,
                        'success_count': instance.success_count,
                        'error_count': instance.error_count,
                        'last_health_check': instance.last_health_check.isoformat() if instance.last_health_check else None
                    })
                
                return {
                    **self.stats,
                    'total_instances': len(self.instances),
                    'healthy_instances': len(self.healthy_instances),
                    'instances': instance_stats,
                    'config': {
                        'load_balancing_algorithm': self.config.load_balancing_algorithm,
                        'auto_scaling': self.config.auto_scaling,
                        'min_instances': self.config.min_instances,
                        'max_instances': self.config.max_instances
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}

# Global load balancer instance
_load_balancer = None

def get_load_balancer() -> LoadBalancerV2:
    """Get global load balancer instance"""
    global _load_balancer
    if _load_balancer is None:
        config = LoadBalancerConfig()
        _load_balancer = LoadBalancerV2(config)
    return _load_balancer
