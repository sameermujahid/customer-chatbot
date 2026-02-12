import asyncio
import threading
import multiprocessing
import queue
import time
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import uuid
import weakref
from collections import defaultdict, deque
import torch
import numpy as np
from functools import partial, lru_cache
import psutil
import gc
from pathlib import Path
import pickle
import json
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class PerformanceTask:
    task_id: str
    task_type: str
    priority: TaskPriority
    func: Callable
    args: tuple
    kwargs: dict
    created_at: float
    timeout: Optional[float] = None
    callback: Optional[Callable] = None
    metadata: Optional[Dict] = None

class PerformanceOptimizer:
    """
    High-performance optimizer with intelligent task distribution,
    caching, and resource management
    """
    
    def __init__(self, 
                 max_cpu_workers: int = None,
                 max_io_workers: int = None,
                 max_gpu_workers: int = 2,
                 cache_size: int = 10000,
                 enable_gpu_optimization: bool = True):
        
        # Determine optimal worker counts
        cpu_count = multiprocessing.cpu_count()
        self.max_cpu_workers = max_cpu_workers or min(cpu_count, 8)
        self.max_io_workers = max_io_workers or min(cpu_count * 2, 16)
        self.max_gpu_workers = max_gpu_workers if torch.cuda.is_available() and enable_gpu_optimization else 0
        
        # Thread and process pools
        self.cpu_pool = ProcessPoolExecutor(max_workers=self.max_cpu_workers)
        self.io_pool = ThreadPoolExecutor(max_workers=self.max_io_workers)
        self.gpu_pool = ThreadPoolExecutor(max_workers=self.max_gpu_workers) if self.max_gpu_workers > 0 else None
        
        # Priority queues for different task types
        self.cpu_queue = queue.PriorityQueue(maxsize=1000)
        self.io_queue = queue.PriorityQueue(maxsize=1000)
        self.gpu_queue = queue.PriorityQueue(maxsize=500) if self.max_gpu_workers > 0 else None
        
        # Task tracking
        self.active_tasks = {}
        self.completed_tasks = {}
        self.task_stats = defaultdict(int)
        
        # Intelligent caching
        self.result_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_size = cache_size
        
        # Resource monitoring
        self.system_stats = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'gpu_usage': 0.0 if torch.cuda.is_available() else None,
            'active_threads': 0,
            'queue_sizes': {}
        }
        
        # Performance metrics
        self.performance_metrics = {
            'avg_response_time': 0.0,
            'throughput': 0.0,
            'error_rate': 0.0,
            'cache_hit_rate': 0.0
        }
        
        # Background workers
        self.running = True
        self.workers = []
        self.monitor_thread = None
        
        # Start background processing
        self._start_background_workers()
        self._start_monitoring()
        
        logger.info(f"PerformanceOptimizer initialized with {self.max_cpu_workers} CPU, "
                   f"{self.max_io_workers} I/O, {self.max_gpu_workers} GPU workers")
    
    def _start_background_workers(self):
        """Start background worker threads"""
        worker_threads = [
            threading.Thread(target=self._cpu_worker, daemon=True, name="CPU-Worker"),
            threading.Thread(target=self._io_worker, daemon=True, name="IO-Worker"),
        ]
        
        if self.gpu_pool:
            worker_threads.append(
                threading.Thread(target=self._gpu_worker, daemon=True, name="GPU-Worker")
            )
        
        for worker in worker_threads:
                worker.start()
                self.workers.append(worker)
        
    def _start_monitoring(self):
        """Start system monitoring thread"""
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_system(self):
        """Monitor system resources and adjust performance"""
        while self.running:
            try:
                # Update system stats
                self.system_stats['cpu_usage'] = psutil.cpu_percent(interval=1)
                self.system_stats['memory_usage'] = psutil.virtual_memory().percent
                self.system_stats['active_threads'] = threading.active_count()
                
                # GPU monitoring
                if torch.cuda.is_available():
                    try:
                        gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                        self.system_stats['gpu_usage'] = gpu_memory * 100
                    except:
                        pass
                
                # Queue sizes
                self.system_stats['queue_sizes'] = {
                    'cpu': self.cpu_queue.qsize(),
                    'io': self.io_queue.qsize(),
                    'gpu': self.gpu_queue.qsize() if self.gpu_queue else 0
                }
                
                # Adaptive performance tuning
                self._adaptive_tuning()
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(10)
    
    def _adaptive_tuning(self):
        """Adaptively tune performance based on system load"""
        cpu_usage = self.system_stats['cpu_usage']
        memory_usage = self.system_stats['memory_usage']
        
        # Memory pressure - trigger garbage collection
        if memory_usage > 80:
            gc.collect()
            logger.info("Triggered garbage collection due to high memory usage")
        
        # CPU pressure - adjust worker priorities
        if cpu_usage > 90:
            # Reduce background task priority
            logger.info("High CPU usage detected - adjusting task priorities")
    
    def _cpu_worker(self):
        """Background worker for CPU-intensive tasks"""
        while self.running:
            try:
                priority, task = self.cpu_queue.get(timeout=1.0)
                
                # Submit to process pool
                future = self.cpu_pool.submit(task.func, *task.args, **task.kwargs)
                
                self.active_tasks[task.task_id] = {
                    'task': task,
                    'future': future,
                    'start_time': time.time()
                }
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in CPU worker: {e}")
    
    def _io_worker(self):
        """Background worker for I/O-bound tasks"""
        while self.running:
            try:
                priority, task = self.io_queue.get(timeout=1.0)
                
                # Submit to thread pool
                future = self.io_pool.submit(task.func, *task.args, **task.kwargs)
                
                self.active_tasks[task.task_id] = {
                    'task': task,
                    'future': future,
                    'start_time': time.time()
                }
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in I/O worker: {e}")
    
    def _gpu_worker(self):
        """Background worker for GPU-intensive tasks"""
        while self.running:
            try:
                priority, task = self.gpu_queue.get(timeout=1.0)
                
                # Submit to GPU thread pool
                future = self.gpu_pool.submit(task.func, *task.args, **task.kwargs)
                
                self.active_tasks[task.task_id] = {
                    'task': task,
                    'future': future,
                    'start_time': time.time()
                }
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in GPU worker: {e}")
    
    def submit_task(self, 
                   func: Callable,
                   task_type: str = "io",
                   priority: TaskPriority = TaskPriority.NORMAL,
                   timeout: Optional[float] = None,
                   callback: Optional[Callable] = None,
                   *args, **kwargs) -> str:
        """Submit a task for execution"""
        
        task_id = str(uuid.uuid4())
        
        # Check cache first
        cache_key = self._generate_cache_key(func, args, kwargs)
        if cache_key in self.result_cache:
            self.cache_hits += 1
            result = self.result_cache[cache_key]
            if callback:
                callback(result)
            return task_id
        
        self.cache_misses += 1
        
        task = PerformanceTask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            func=func,
            args=args,
            kwargs=kwargs,
            created_at=time.time(),
            timeout=timeout,
            callback=callback
        )
        
        # Add to appropriate queue
        priority_value = priority.value
        if task_type == "cpu":
            self.cpu_queue.put((priority_value, task))
        elif task_type == "gpu" and self.gpu_queue:
            self.gpu_queue.put((priority_value, task))
        else:
            self.io_queue.put((priority_value, task))
        
        self.task_stats[f'{task_type}_submitted'] += 1
        return task_id
    
    def _generate_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate a cache key for function call"""
        try:
            # Create a hash of the function and arguments
            func_name = func.__name__ if hasattr(func, '__name__') else str(func)
            args_str = json.dumps(args, sort_keys=True, default=str)
            kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
            
            key_data = f"{func_name}:{args_str}:{kwargs_str}"
            return hashlib.md5(key_data.encode()).hexdigest()
        except:
            return str(uuid.uuid4())
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[Any]:
        """Get the result of a task"""
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        
        if task_id in self.active_tasks:
            task_data = self.active_tasks[task_id]
            future = task_data['future']
            
            try:
                result = future.result(timeout=timeout)
                
                # Cache the result
                task = task_data['task']
                cache_key = self._generate_cache_key(task.func, task.args, task.kwargs)
                self._cache_result(cache_key, result)
                
                # Move to completed
                self.completed_tasks[task_id] = result
                del self.active_tasks[task_id]
                
                # Update metrics
                duration = time.time() - task_data['start_time']
                self._update_performance_metrics(duration, True)
                
                return result
                
            except Exception as e:
                self._update_performance_metrics(0, False)
                logger.error(f"Task {task_id} failed: {e}")
                return None
        
        return None
    
    def _cache_result(self, key: str, result: Any):
        """Cache a result with LRU eviction"""
        if len(self.result_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.result_cache))
            del self.result_cache[oldest_key]
        
        self.result_cache[key] = result
    
    def _update_performance_metrics(self, duration: float, success: bool):
        """Update performance metrics"""
        # Update average response time
        total_requests = self.cache_hits + self.cache_misses
        if total_requests > 0:
            self.performance_metrics['avg_response_time'] = (
                (self.performance_metrics['avg_response_time'] * (total_requests - 1) + duration) / total_requests
            )
        
        # Update error rate
        if not success:
            self.performance_metrics['error_rate'] += 1
        
        # Update cache hit rate
        if total_requests > 0:
            self.performance_metrics['cache_hit_rate'] = self.cache_hits / total_requests
    
    def batch_submit(self, tasks: List[Dict]) -> List[str]:
        """Submit multiple tasks in batch"""
        task_ids = []
        
        for task_spec in tasks:
            task_id = self.submit_task(
                func=task_spec['func'],
                task_type=task_spec.get('type', 'io'),
                priority=TaskPriority(task_spec.get('priority', 3)),
                timeout=task_spec.get('timeout'),
                callback=task_spec.get('callback'),
                *task_spec.get('args', ()),
                **task_spec.get('kwargs', {})
            )
            task_ids.append(task_id)
        
        return task_ids
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            'system_stats': self.system_stats,
            'performance_metrics': self.performance_metrics,
            'task_stats': dict(self.task_stats),
            'cache_stats': {
                'size': len(self.result_cache),
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_rate': self.performance_metrics['cache_hit_rate']
            },
            'queue_stats': {
                'cpu_queue_size': self.cpu_queue.qsize(),
                'io_queue_size': self.io_queue.qsize(),
                'gpu_queue_size': self.gpu_queue.qsize() if self.gpu_queue else 0,
            'active_tasks': len(self.active_tasks),
                'completed_tasks': len(self.completed_tasks)
            }
        }
    
    def clear_cache(self):
        """Clear the result cache"""
        self.result_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def shutdown(self):
        """Shutdown the optimizer and clean up resources"""
        self.running = False
        
        # Shutdown pools
        self.cpu_pool.shutdown(wait=True)
        self.io_pool.shutdown(wait=True)
        if self.gpu_pool:
            self.gpu_pool.shutdown(wait=True)
        
        # Wait for workers
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("PerformanceOptimizer shutdown complete")

# Global performance optimizer instance
performance_optimizer = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer instance"""
    global performance_optimizer
    if performance_optimizer is None:
        performance_optimizer = PerformanceOptimizer()
    return performance_optimizer

# Utility functions for common optimizations
@lru_cache(maxsize=1000)
def cached_embedding(text: str, model_name: str = "jinaai/jina-embeddings-v3") -> np.ndarray:
    """Cached embedding generation"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, trust_remote_code=True)
    return model.encode(text)

def parallel_embed_batch(texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
    """Generate embeddings for a batch of texts in parallel"""
    optimizer = get_performance_optimizer()
    
    def embed_batch(batch_texts):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True)
        return model.encode(batch_texts)
    
    # Split into batches
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    # Submit batch tasks
    task_ids = []
    for batch in batches:
        task_id = optimizer.submit_task(
            embed_batch,
            task_type="gpu" if torch.cuda.is_available() else "cpu",
            priority=TaskPriority.NORMAL,
            timeout=None,
            callback=None,
            *batch
        )
        task_ids.append(task_id)
    
    # Collect results
    results = []
    for task_id in task_ids:
        result = optimizer.get_task_result(task_id)
        if result is not None:
            results.extend(result)
    
    return results

def parallel_property_processing(properties: List[Dict], 
                               processor_func: Callable,
                               batch_size: int = 50) -> List[Dict]:
    """Process properties in parallel batches"""
    optimizer = get_performance_optimizer()
    
    # Split into batches
    batches = [properties[i:i + batch_size] for i in range(0, len(properties), batch_size)]
    
    # Submit batch tasks
    task_ids = []
    for batch in batches:
        task_id = optimizer.submit_task(
            processor_func,
            task_type="io",
            priority=TaskPriority.NORMAL,
            timeout=None,
            callback=None,
            *batch
        )
        task_ids.append(task_id)
    
    # Collect results
    results = []
    for task_id in task_ids:
        result = optimizer.get_task_result(task_id)
        if result is not None:
            results.extend(result)
    
    return results 