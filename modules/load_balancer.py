import threading
import queue
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import random
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESPONSE_TIME = "response_time"
    RANDOM = "random"

@dataclass
class WorkerNode:
    """Represents a worker node in the load balancer"""
    id: str
    name: str
    weight: int = 1
    max_connections: int = 10
    current_connections: int = 0
    response_time: float = 0.0
    is_healthy: bool = True
    last_health_check: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0

class LoadBalancer:
    """Load balancer for distributing requests across multiple workers"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.workers: List[WorkerNode] = []
        self.current_worker_index = 0
        self.lock = threading.RLock()
        
        # Health check thread
        self.health_check_thread = threading.Thread(target=self._health_check_worker, daemon=True)
        self.health_check_thread.start()
    
    def add_worker(self, worker_id: str, name: str, weight: int = 1, max_connections: int = 10) -> bool:
        """Add a worker node to the load balancer"""
        try:
            with self.lock:
                # Check if worker already exists
                for worker in self.workers:
                    if worker.id == worker_id:
                        logger.warning(f"Worker {worker_id} already exists")
                        return False
                
                worker = WorkerNode(
                    id=worker_id,
                    name=name,
                    weight=weight,
                    max_connections=max_connections
                )
                
                self.workers.append(worker)
                logger.info(f"Added worker {worker_id} ({name}) to load balancer")
                return True
                
        except Exception as e:
            logger.error(f"Error adding worker {worker_id}: {e}")
            return False
    
    def remove_worker(self, worker_id: str) -> bool:
        """Remove a worker node from the load balancer"""
        try:
            with self.lock:
                for i, worker in enumerate(self.workers):
                    if worker.id == worker_id:
                        del self.workers[i]
                        logger.info(f"Removed worker {worker_id} from load balancer")
                        return True
                
                logger.warning(f"Worker {worker_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error removing worker {worker_id}: {e}")
            return False
    
    def get_worker(self) -> Optional[WorkerNode]:
        """Get the next available worker based on the load balancing strategy"""
        try:
            with self.lock:
                if not self.workers:
                    return None
                
                # Filter healthy workers with available connections
                available_workers = [
                    w for w in self.workers 
                    if w.is_healthy and w.current_connections < w.max_connections
                ]
                
                if not available_workers:
                    logger.warning("No available workers")
                    return None
                
                if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                    return self._round_robin_select(available_workers)
                elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                    return self._least_connections_select(available_workers)
                elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                    return self._weighted_round_robin_select(available_workers)
                elif self.strategy == LoadBalancingStrategy.RESPONSE_TIME:
                    return self._response_time_select(available_workers)
                elif self.strategy == LoadBalancingStrategy.RANDOM:
                    return self._random_select(available_workers)
                else:
                    return self._round_robin_select(available_workers)
                    
        except Exception as e:
            logger.error(f"Error getting worker: {e}")
            return None
    
    def _round_robin_select(self, available_workers: List[WorkerNode]) -> WorkerNode:
        """Round-robin selection"""
        worker = available_workers[self.current_worker_index % len(available_workers)]
        self.current_worker_index += 1
        return worker
    
    def _least_connections_select(self, available_workers: List[WorkerNode]) -> WorkerNode:
        """Least connections selection"""
        return min(available_workers, key=lambda w: w.current_connections)
    
    def _weighted_round_robin_select(self, available_workers: List[WorkerNode]) -> WorkerNode:
        """Weighted round-robin selection"""
        total_weight = sum(w.weight for w in available_workers)
        if total_weight == 0:
            return available_workers[0]
        
        # Simple weighted selection
        weights = [w.weight for w in available_workers]
        selected = random.choices(available_workers, weights=weights, k=1)[0]
        return selected
    
    def _response_time_select(self, available_workers: List[WorkerNode]) -> WorkerNode:
        """Response time-based selection"""
        return min(available_workers, key=lambda w: w.response_time)
    
    def _random_select(self, available_workers: List[WorkerNode]) -> WorkerNode:
        """Random selection"""
        return random.choice(available_workers)
    
    def acquire_worker(self) -> Optional[WorkerNode]:
        """Acquire a worker and increment its connection count"""
        worker = self.get_worker()
        if worker:
            with self.lock:
                worker.current_connections += 1
                worker.total_requests += 1
        return worker
    
    def release_worker(self, worker_id: str, response_time: float = 0.0, success: bool = True):
        """Release a worker and update its statistics"""
        try:
            with self.lock:
                for worker in self.workers:
                    if worker.id == worker_id:
                        worker.current_connections = max(0, worker.current_connections - 1)
                        
                        # Update response time (exponential moving average)
                        if response_time > 0:
                            alpha = 0.1  # Smoothing factor
                            worker.response_time = (alpha * response_time + 
                                                  (1 - alpha) * worker.response_time)
                        
                        if not success:
                            worker.failed_requests += 1
                        
                        break
                        
        except Exception as e:
            logger.error(f"Error releasing worker {worker_id}: {e}")
    
    def _health_check_worker(self):
        """Background worker for health checks"""
        while True:
            try:
                current_time = time.time()
                with self.lock:
                    for worker in self.workers:
                        # Simple health check based on failure rate
                        if worker.total_requests > 0:
                            failure_rate = worker.failed_requests / worker.total_requests
                            worker.is_healthy = failure_rate < 0.5  # Mark unhealthy if >50% failures
                        
                        worker.last_health_check = current_time
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health check worker: {e}")
                time.sleep(30)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        with self.lock:
            total_workers = len(self.workers)
            healthy_workers = sum(1 for w in self.workers if w.is_healthy)
            total_connections = sum(w.current_connections for w in self.workers)
            total_requests = sum(w.total_requests for w in self.workers)
            total_failures = sum(w.failed_requests for w in self.workers)
            
            return {
                'strategy': self.strategy.value,
                'total_workers': total_workers,
                'healthy_workers': healthy_workers,
                'total_connections': total_connections,
                'total_requests': total_requests,
                'total_failures': total_failures,
                'failure_rate': (total_failures / total_requests * 100) if total_requests > 0 else 0,
                'workers': [
                    {
                        'id': w.id,
                        'name': w.name,
                        'weight': w.weight,
                        'current_connections': w.current_connections,
                        'max_connections': w.max_connections,
                        'response_time': w.response_time,
                        'is_healthy': w.is_healthy,
                        'total_requests': w.total_requests,
                        'failed_requests': w.failed_requests
                    }
                    for w in self.workers
                ]
            }

class RequestQueue:
    """Priority queue for managing requests"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.high_priority_queue = queue.Queue(maxsize=max_size // 4)
        self.normal_priority_queue = queue.Queue(maxsize=max_size // 2)
        self.low_priority_queue = queue.Queue(maxsize=max_size // 4)
        
        self.stats = {
            'enqueued': 0,
            'dequeued': 0,
            'dropped': 0
        }
    
    def enqueue(self, request: Any, priority: str = 'normal') -> bool:
        """Enqueue a request with priority"""
        try:
            if priority == 'high':
                queue_obj = self.high_priority_queue
            elif priority == 'low':
                queue_obj = self.low_priority_queue
            else:
                queue_obj = self.normal_priority_queue
            
            # Try to put in queue with timeout
            queue_obj.put(request, timeout=1.0)
            self.stats['enqueued'] += 1
            return True
            
        except queue.Full:
            self.stats['dropped'] += 1
            logger.warning(f"Request queue full, dropping {priority} priority request")
            return False
    
    def dequeue(self) -> Optional[Any]:
        """Dequeue the highest priority request"""
        try:
            # Try high priority first
            try:
                return self.high_priority_queue.get_nowait()
            except queue.Empty:
                pass
            
            # Try normal priority
            try:
                return self.normal_priority_queue.get_nowait()
            except queue.Empty:
                pass
            
            # Try low priority
            try:
                return self.low_priority_queue.get_nowait()
            except queue.Empty:
                pass
            
            return None
            
        except Exception as e:
            logger.error(f"Error dequeuing request: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            'high_priority_size': self.high_priority_queue.qsize(),
            'normal_priority_size': self.normal_priority_queue.qsize(),
            'low_priority_size': self.low_priority_queue.qsize(),
            'total_size': (self.high_priority_queue.qsize() + 
                          self.normal_priority_queue.qsize() + 
                          self.low_priority_queue.qsize()),
            'max_size': self.max_size,
            **self.stats
        }

class ProcessingManager:
    """Manages request processing with load balancing and queuing"""
    
    def __init__(self, num_workers: int = 4, max_queue_size: int = 1000):
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        
        # Load balancer
        self.load_balancer = LoadBalancer(LoadBalancingStrategy.LEAST_CONNECTIONS)
        
        # Request queues
        self.text_generation_queue = RequestQueue(max_queue_size)
        self.rag_processing_queue = RequestQueue(max_queue_size)
        self.location_processing_queue = RequestQueue(max_queue_size)
        
        # Worker pools
        self.text_generation_pool = ThreadPoolExecutor(max_workers=num_workers, 
                                                      thread_name_prefix="text_gen")
        self.rag_processing_pool = ThreadPoolExecutor(max_workers=num_workers, 
                                                     thread_name_prefix="rag_proc")
        self.location_processing_pool = ThreadPoolExecutor(max_workers=num_workers, 
                                                          thread_name_prefix="loc_proc")
        
        # Initialize workers
        self._initialize_workers()
        
        # Processing threads
        self.processing_threads = []
        self._start_processing_threads()
    
    def _initialize_workers(self):
        """Initialize worker nodes"""
        for i in range(self.num_workers):
            # Text generation workers
            self.load_balancer.add_worker(
                f"text_gen_{i}",
                f"Text Generation Worker {i}",
                weight=1,
                max_connections=5
            )
            
            # RAG processing workers
            self.load_balancer.add_worker(
                f"rag_proc_{i}",
                f"RAG Processing Worker {i}",
                weight=1,
                max_connections=5
            )
            
            # Location processing workers
            self.load_balancer.add_worker(
                f"loc_proc_{i}",
                f"Location Processing Worker {i}",
                weight=1,
                max_connections=5
            )
    
    def _start_processing_threads(self):
        """Start background processing threads"""
        thread_configs = [
            (self.text_generation_queue, self.text_generation_pool, "text_generation"),
            (self.rag_processing_queue, self.rag_processing_pool, "rag_processing"),
            (self.location_processing_queue, self.location_processing_pool, "location_processing")
        ]
        
        for queue_obj, pool, name in thread_configs:
            thread = threading.Thread(
                target=self._processing_worker,
                args=(queue_obj, pool, name),
                daemon=True
            )
            thread.start()
            self.processing_threads.append(thread)
    
    def _processing_worker(self, request_queue: RequestQueue, thread_pool: ThreadPoolExecutor, 
                          worker_type: str):
        """Background worker for processing requests"""
        while True:
            try:
                # Get request from queue
                request = request_queue.dequeue()
                if request is None:
                    time.sleep(0.1)  # Small delay if queue is empty
                    continue
                
                # Get worker from load balancer
                worker = self.load_balancer.acquire_worker()
                if worker is None:
                    # No available workers, requeue request
                    request_queue.enqueue(request, 'normal')
                    time.sleep(0.1)
                    continue
                
                # Process request
                start_time = time.time()
                success = False
                
                try:
                    # Submit to thread pool
                    future = thread_pool.submit(self._process_request, request, worker_type)
                    result = future.result(timeout=30)  # 30 second timeout
                    success = True
                    
                except Exception as e:
                    logger.error(f"Error processing {worker_type} request: {e}")
                    success = False
                
                finally:
                    # Release worker
                    response_time = time.time() - start_time
                    self.load_balancer.release_worker(worker.id, response_time, success)
                    request_queue.stats['dequeued'] += 1
                
            except Exception as e:
                logger.error(f"Error in {worker_type} processing worker: {e}")
                time.sleep(1)
    
    def _process_request(self, request: Any, worker_type: str) -> Any:
        """Process a request based on its type"""
        try:
            if worker_type == "text_generation":
                return self._process_text_generation(request)
            elif worker_type == "rag_processing":
                return self._process_rag_request(request)
            elif worker_type == "location_processing":
                return self._process_location_request(request)
            else:
                raise ValueError(f"Unknown worker type: {worker_type}")
                
        except Exception as e:
            logger.error(f"Error in request processing: {e}")
            raise
    
    def _process_text_generation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process text generation request"""
        # Import here to avoid circular imports
        from modules.response import generate_response
        from modules.models import load_tokenizer_and_model
        
        query = request.get('query', '')
        session_id = request.get('session_id', '')
        
        # Get tokenizer and model
        tokenizer, model_llm = load_tokenizer_and_model()
        
        # Generate response
        response, duration = generate_response(query, tokenizer, model_llm)
        
        return {
            'request_id': request.get('request_id'),
            'session_id': session_id,
            'response': response,
            'duration': duration,
            'status': 'success'
        }
    
    def _process_rag_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process RAG request"""
        from modules.models import get_global_retriever
        
        query = request.get('query', '')
        session_id = request.get('session_id', '')
        
        # Get retriever
        retriever = get_global_retriever()
        if retriever is None:
            raise Exception("RAG retriever not available")
        
        # Perform retrieval
        results = retriever.retrieve(query, top_k=5)
        
        return {
            'request_id': request.get('request_id'),
            'session_id': session_id,
            'results': results,
            'status': 'success'
        }
    
    def _process_location_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process location request"""
        from modules.location_processor import LocationProcessor
        
        session_id = request.get('session_id', '')
        latitude = request.get('latitude')
        longitude = request.get('longitude')
        
        if not latitude or not longitude:
            raise Exception("Location coordinates not provided")
        
        # Process location
        location_processor = LocationProcessor()
        result = location_processor.set_location(latitude, longitude, session_id)
        
        return {
            'request_id': request.get('request_id'),
            'session_id': session_id,
            'location_data': result,
            'status': 'success'
        }
    
    def submit_text_generation(self, query: str, session_id: str, priority: str = 'normal') -> str:
        """Submit a text generation request"""
        request_id = f"text_gen_{int(time.time() * 1000)}"
        
        request = {
            'request_id': request_id,
            'query': query,
            'session_id': session_id,
            'type': 'text_generation'
        }
        
        success = self.text_generation_queue.enqueue(request, priority)
        if not success:
            raise Exception("Text generation queue is full")
        
        return request_id
    
    def submit_rag_request(self, query: str, session_id: str, priority: str = 'normal') -> str:
        """Submit a RAG request"""
        request_id = f"rag_{int(time.time() * 1000)}"
        
        request = {
            'request_id': request_id,
            'query': query,
            'session_id': session_id,
            'type': 'rag_processing'
        }
        
        success = self.rag_processing_queue.enqueue(request, priority)
        if not success:
            raise Exception("RAG processing queue is full")
        
        return request_id
    
    def submit_location_request(self, latitude: float, longitude: float, 
                              session_id: str, priority: str = 'normal') -> str:
        """Submit a location request"""
        request_id = f"loc_{int(time.time() * 1000)}"
        
        request = {
            'request_id': request_id,
            'latitude': latitude,
            'longitude': longitude,
            'session_id': session_id,
            'type': 'location_processing'
        }
        
        success = self.location_processing_queue.enqueue(request, priority)
        if not success:
            raise Exception("Location processing queue is full")
        
        return request_id
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing manager statistics"""
        return {
            'load_balancer': self.load_balancer.get_stats(),
            'text_generation_queue': self.text_generation_queue.get_stats(),
            'rag_processing_queue': self.rag_processing_queue.get_stats(),
            'location_processing_queue': self.location_processing_queue.get_stats()
        }
    
    def shutdown(self):
        """Shutdown the processing manager"""
        self.text_generation_pool.shutdown(wait=True)
        self.rag_processing_pool.shutdown(wait=True)
        self.location_processing_pool.shutdown(wait=True)

# Global processing manager
processing_manager = ProcessingManager() 