import asyncio
import threading
import multiprocessing
import queue
import time
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
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
import signal
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class TaskType(Enum):
    CPU_INTENSIVE = "cpu_intensive"
    IO_BOUND = "io_bound"
    GPU_INTENSIVE = "gpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    NETWORK_BOUND = "network_bound"
    MIXED = "mixed"

class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

class ProcessingStrategy(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    BATCH = "batch"
    STREAMING = "streaming"

@dataclass
class ProcessingTask:
    task_id: str
    task_type: TaskType
    priority: TaskPriority
    strategy: ProcessingStrategy
    func: Callable
    args: tuple
    kwargs: dict
    created_at: float
    timeout: Optional[float] = None
    callback: Optional[Callable] = None
    metadata: Optional[Dict] = None
    dependencies: List[str] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class ProcessingResult:
    task_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = None

class AdvancedProcessor:
    """
    Advanced processor with intelligent task distribution,
    adaptive threading, and real-time performance optimization
    """
    
    def __init__(self, 
                 max_cpu_workers: int = None,
                 max_io_workers: int = None,
                 max_gpu_workers: int = 2,
                 max_memory_workers: int = 4,
                 enable_adaptive_scaling: bool = True,
                 enable_gpu_optimization: bool = True):
        
        # System information
        self.cpu_count = multiprocessing.cpu_count()
        self.total_memory = psutil.virtual_memory().total
        self.gpu_available = torch.cuda.is_available()
        
        # Worker configuration
        self.max_cpu_workers = max_cpu_workers or min(self.cpu_count, 8)
        self.max_io_workers = max_io_workers or min(self.cpu_count * 2, 16)
        self.max_gpu_workers = max_gpu_workers if self.gpu_available and enable_gpu_optimization else 0
        self.max_memory_workers = max_memory_workers
        
        # Adaptive scaling
        self.enable_adaptive_scaling = enable_adaptive_scaling
        self.adaptive_metrics = {
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'gpu_usage': deque(maxlen=100),
            'queue_sizes': deque(maxlen=100),
            'response_times': deque(maxlen=100)
        }
        
        # Thread and process pools
        self.cpu_pool = ProcessPoolExecutor(max_workers=self.max_cpu_workers)
        self.io_pool = ThreadPoolExecutor(max_workers=self.max_io_workers)
        self.gpu_pool = ThreadPoolExecutor(max_workers=self.max_gpu_workers) if self.max_gpu_workers > 0 else None
        self.memory_pool = ThreadPoolExecutor(max_workers=self.max_memory_workers)
        
        # Priority queues for different task types
        self.cpu_queue = queue.PriorityQueue(maxsize=1000)
        self.io_queue = queue.PriorityQueue(maxsize=1000)
        self.gpu_queue = queue.PriorityQueue(maxsize=500) if self.max_gpu_workers > 0 else None
        self.memory_queue = queue.PriorityQueue(maxsize=500)
        
        # Task tracking
        self.active_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.task_dependencies = {}
        self.task_stats = defaultdict(int)
        
        # Performance monitoring
        self.performance_metrics = {
            'total_tasks_processed': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'avg_response_time': 0.0,
            'throughput': 0.0,
            'error_rate': 0.0,
            'queue_utilization': 0.0
        }
        
        # Resource monitoring
        self.system_stats = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'gpu_usage': 0.0 if self.gpu_available else None,
            'active_threads': 0,
            'queue_sizes': {}
        }
        
        # Background workers and monitoring
        self.running = True
        self.workers = []
        self.monitor_thread = None
        self.adaptive_thread = None
        
        # Start background processing
        self._start_background_workers()
        self._start_monitoring()
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"AdvancedProcessor initialized with {self.max_cpu_workers} CPU, "
                   f"{self.max_io_workers} I/O, {self.max_gpu_workers} GPU, "
                   f"{self.max_memory_workers} memory workers")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown()
    
    def _start_background_workers(self):
        """Start background worker threads"""
        worker_configs = [
            (self._cpu_worker, "CPU-Worker"),
            (self._io_worker, "IO-Worker"),
            (self._memory_worker, "Memory-Worker"),
        ]
        
        if self.gpu_pool:
            worker_configs.append((self._gpu_worker, "GPU-Worker"))
        
        for worker_func, name in worker_configs:
            worker = threading.Thread(target=worker_func, daemon=True, name=name)
            worker.start()
            self.workers.append(worker)
    
    def _start_monitoring(self):
        """Start system monitoring and adaptive scaling threads"""
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
        
        if self.enable_adaptive_scaling:
            self.adaptive_thread = threading.Thread(target=self._adaptive_scaling, daemon=True)
            self.adaptive_thread.start()
    
    def _monitor_system(self):
        """Monitor system resources and performance"""
        while self.running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.system_stats['cpu_usage'] = cpu_percent
                self.adaptive_metrics['cpu_usage'].append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.system_stats['memory_usage'] = memory.percent
                self.adaptive_metrics['memory_usage'].append(memory.percent)
                
                # GPU usage (if available)
                if self.gpu_available:
                    try:
                        gpu_percent = torch.cuda.utilization()
                        self.system_stats['gpu_usage'] = gpu_percent
                        self.adaptive_metrics['gpu_usage'].append(gpu_percent)
                    except:
                        pass
                
                # Queue sizes
                queue_sizes = {
                    'cpu': self.cpu_queue.qsize(),
                    'io': self.io_queue.qsize(),
                    'memory': self.memory_queue.qsize()
                }
                if self.gpu_queue:
                    queue_sizes['gpu'] = self.gpu_queue.qsize()
                
                self.system_stats['queue_sizes'] = queue_sizes
                self.adaptive_metrics['queue_sizes'].append(queue_sizes)
                
                # Active threads
                self.system_stats['active_threads'] = threading.active_count()
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(10)
    
    def _adaptive_scaling(self):
        """Adaptive scaling based on system metrics"""
        while self.running:
            try:
                # Calculate average metrics
                avg_cpu = np.mean(self.adaptive_metrics['cpu_usage']) if self.adaptive_metrics['cpu_usage'] else 0
                avg_memory = np.mean(self.adaptive_metrics['memory_usage']) if self.adaptive_metrics['memory_usage'] else 0
                avg_queue_size = np.mean([sum(q.values()) for q in self.adaptive_metrics['queue_sizes']]) if self.adaptive_metrics['queue_sizes'] else 0
                
                # Adaptive scaling logic
                if avg_cpu > 80 and avg_queue_size > 50:
                    # High CPU usage and queue backlog - increase workers
                    self._scale_up_workers()
                elif avg_cpu < 30 and avg_queue_size < 10:
                    # Low CPU usage and small queue - scale down workers
                    self._scale_down_workers()
                
                # Memory pressure handling
                if avg_memory > 85:
                    self._handle_memory_pressure()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in adaptive scaling: {e}")
                time.sleep(60)
    
    def _scale_up_workers(self):
        """Scale up worker pools when under load"""
        try:
            # Increase I/O workers (most flexible)
            current_io_workers = self.io_pool._max_workers
            if current_io_workers < 32:  # Max limit
                new_io_workers = min(current_io_workers + 2, 32)
                self._recreate_pool('io', new_io_workers)
                logger.info(f"Scaled up I/O workers from {current_io_workers} to {new_io_workers}")
            
            # Increase memory workers if needed
            current_memory_workers = self.memory_pool._max_workers
            if current_memory_workers < 8:  # Max limit
                new_memory_workers = min(current_memory_workers + 1, 8)
                self._recreate_pool('memory', new_memory_workers)
                logger.info(f"Scaled up memory workers from {current_memory_workers} to {new_memory_workers}")
                
        except Exception as e:
            logger.error(f"Error scaling up workers: {e}")
    
    def _scale_down_workers(self):
        """Scale down worker pools when underutilized"""
        try:
            # Decrease I/O workers
            current_io_workers = self.io_pool._max_workers
            if current_io_workers > 4:  # Min limit
                new_io_workers = max(current_io_workers - 1, 4)
                self._recreate_pool('io', new_io_workers)
                logger.info(f"Scaled down I/O workers from {current_io_workers} to {new_io_workers}")
                
        except Exception as e:
            logger.error(f"Error scaling down workers: {e}")
    
    def _handle_memory_pressure(self):
        """Handle high memory pressure"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear some caches if available
            if hasattr(self, 'result_cache'):
                # Keep only recent results
                cache_size = len(self.result_cache)
                if cache_size > 1000:
                    # Remove oldest 20% of cache
                    items_to_remove = int(cache_size * 0.2)
                    for _ in range(items_to_remove):
                        if self.result_cache:
                            self.result_cache.popitem()
            
            logger.info("Handled memory pressure with garbage collection and cache cleanup")
            
        except Exception as e:
            logger.error(f"Error handling memory pressure: {e}")
    
    def _recreate_pool(self, pool_type: str, new_workers: int):
        """Recreate a thread/process pool with new worker count"""
        try:
            if pool_type == 'io':
                old_pool = self.io_pool
                self.io_pool = ThreadPoolExecutor(max_workers=new_workers)
                old_pool.shutdown(wait=False)
            elif pool_type == 'memory':
                old_pool = self.memory_pool
                self.memory_pool = ThreadPoolExecutor(max_workers=new_workers)
                old_pool.shutdown(wait=False)
            elif pool_type == 'gpu' and self.gpu_pool:
                old_pool = self.gpu_pool
                self.gpu_pool = ThreadPoolExecutor(max_workers=new_workers)
                old_pool.shutdown(wait=False)
                
        except Exception as e:
            logger.error(f"Error recreating {pool_type} pool: {e}")
    
    def _cpu_worker(self):
        """Background worker for CPU-intensive tasks"""
        while self.running:
            try:
                priority, task = self.cpu_queue.get(timeout=1.0)
                
                # Submit to process pool
                future = self.cpu_pool.submit(self._execute_task, task)
                
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
                future = self.io_pool.submit(self._execute_task, task)
                
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
                future = self.gpu_pool.submit(self._execute_task, task)
                
                self.active_tasks[task.task_id] = {
                    'task': task,
                    'future': future,
                    'start_time': time.time()
                }
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in GPU worker: {e}")
    
    def _memory_worker(self):
        """Background worker for memory-intensive tasks"""
        while self.running:
            try:
                priority, task = self.memory_queue.get(timeout=1.0)
                
                # Submit to memory thread pool
                future = self.memory_pool.submit(self._execute_task, task)
                
                self.active_tasks[task.task_id] = {
                    'task': task,
                    'future': future,
                    'start_time': time.time()
                }
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in memory worker: {e}")
    
    def _execute_task(self, task: ProcessingTask) -> ProcessingResult:
        """Execute a processing task with error handling and retries"""
        start_time = time.time()
        
        try:
            # Check dependencies
            if task.dependencies:
                for dep_id in task.dependencies:
                    if dep_id not in self.completed_tasks:
                        raise Exception(f"Dependency {dep_id} not completed")
            
            # Execute task
            result = task.func(*task.args, **task.kwargs)
            
            duration = time.time() - start_time
            
            # Update performance metrics
            self.adaptive_metrics['response_times'].append(duration)
            
            return ProcessingResult(
                task_id=task.task_id,
                success=True,
                result=result,
                duration=duration,
                metadata={'task_type': task.task_type.value}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries}): {e}")
                
                # Exponential backoff
                time.sleep(2 ** task.retry_count)
                
                # Re-submit task
                self._submit_task_to_queue(task)
                return None
            else:
                logger.error(f"Task {task.task_id} failed after {task.max_retries} retries: {e}")
                
                return ProcessingResult(
                    task_id=task.task_id,
                    success=False,
                    error=str(e),
                    duration=duration,
                    metadata={'task_type': task.task_type.value, 'retry_count': task.retry_count}
                )
    
    def _submit_task_to_queue(self, task: ProcessingTask):
        """Submit task to appropriate queue based on type"""
        try:
            if task.task_type == TaskType.CPU_INTENSIVE:
                self.cpu_queue.put((task.priority.value, task))
            elif task.task_type == TaskType.IO_BOUND:
                self.io_queue.put((task.priority.value, task))
            elif task.task_type == TaskType.GPU_INTENSIVE and self.gpu_queue:
                self.gpu_queue.put((task.priority.value, task))
            elif task.task_type == TaskType.MEMORY_INTENSIVE:
                self.memory_queue.put((task.priority.value, task))
            else:
                # Default to I/O queue
                self.io_queue.put((task.priority.value, task))
                
        except Exception as e:
            logger.error(f"Error submitting task to queue: {e}")
    
    def submit_task(self, 
                   func: Callable,
                   task_type: TaskType = TaskType.IO_BOUND,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   strategy: ProcessingStrategy = ProcessingStrategy.PARALLEL,
                   timeout: Optional[float] = None,
                   callback: Optional[Callable] = None,
                   dependencies: List[str] = None,
                   *args, **kwargs) -> str:
        """Submit a task for processing"""
        
        task_id = str(uuid.uuid4())
        
        task = ProcessingTask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            strategy=strategy,
            func=func,
            args=args,
            kwargs=kwargs,
            created_at=time.time(),
            timeout=timeout,
            callback=callback,
            dependencies=dependencies or []
        )
        
        # Store task dependencies
        if dependencies:
            self.task_dependencies[task_id] = dependencies
        
        # Submit to appropriate queue
        self._submit_task_to_queue(task)
        
        # Update statistics
        self.task_stats[task_type.value] += 1
        self.performance_metrics['total_tasks_processed'] += 1
        
        logger.debug(f"Submitted task {task_id} ({task_type.value}, {priority.value})")
        return task_id
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[ProcessingResult]:
        """Get the result of a completed task"""
        try:
            # Check if task is completed
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            
            # Check if task is active
            if task_id in self.active_tasks:
                task_info = self.active_tasks[task_id]
                future = task_info['future']
                
                # Wait for result
                result = future.result(timeout=timeout)
                
                if result:
                    # Move to completed tasks
                    self.completed_tasks[task_id] = result
                    del self.active_tasks[task_id]
                    
                    # Update statistics
                    if result.success:
                        self.performance_metrics['successful_tasks'] += 1
                    else:
                        self.performance_metrics['failed_tasks'] += 1
                        self.failed_tasks[task_id] = result
                    
                    # Execute callback if provided
                    if task_info['task'].callback:
                        try:
                            task_info['task'].callback(result)
                        except Exception as e:
                            logger.error(f"Error in task callback: {e}")
                    
                    return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting task result: {e}")
            return None
    
    def batch_submit(self, tasks: List[Dict]) -> List[str]:
        """Submit multiple tasks in batch"""
        task_ids = []
        
        for task_config in tasks:
            task_id = self.submit_task(
                func=task_config['func'],
                task_type=task_config.get('task_type', TaskType.IO_BOUND),
                priority=task_config.get('priority', TaskPriority.NORMAL),
                strategy=task_config.get('strategy', ProcessingStrategy.PARALLEL),
                timeout=task_config.get('timeout'),
                callback=task_config.get('callback'),
                dependencies=task_config.get('dependencies'),
                *task_config.get('args', ()),
                **task_config.get('kwargs', {})
            )
            task_ids.append(task_id)
        
        return task_ids
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        # Calculate derived metrics
        total_tasks = self.performance_metrics['total_tasks_processed']
        if total_tasks > 0:
            success_rate = self.performance_metrics['successful_tasks'] / total_tasks
            error_rate = self.performance_metrics['failed_tasks'] / total_tasks
        else:
            success_rate = 0.0
            error_rate = 0.0
        
        # Calculate average response time
        response_times = self.adaptive_metrics['response_times']
        avg_response_time = np.mean(response_times) if response_times else 0.0
        
        # Calculate throughput (tasks per second)
        if response_times:
            throughput = len(response_times) / max(np.sum(response_times), 1)
        else:
            throughput = 0.0
        
        return {
            **self.performance_metrics,
            'success_rate': success_rate,
            'error_rate': error_rate,
            'avg_response_time': avg_response_time,
            'throughput': throughput,
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'queue_sizes': self.system_stats['queue_sizes'],
            'system_stats': self.system_stats,
            'task_type_distribution': dict(self.task_stats)
        }
    
    def clear_completed_tasks(self, max_age: int = 3600):
        """Clear old completed tasks to free memory"""
        current_time = time.time()
        tasks_to_remove = []
        
        for task_id, result in self.completed_tasks.items():
            if current_time - result.metadata.get('completed_at', current_time) > max_age:
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.completed_tasks[task_id]
        
        logger.info(f"Cleared {len(tasks_to_remove)} old completed tasks")
    
    def shutdown(self):
        """Gracefully shutdown the processor"""
        logger.info("Shutting down AdvancedProcessor...")
        
        self.running = False
        
        # Shutdown pools
        self.cpu_pool.shutdown(wait=True)
        self.io_pool.shutdown(wait=True)
        self.memory_pool.shutdown(wait=True)
        if self.gpu_pool:
            self.gpu_pool.shutdown(wait=True)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
        
        logger.info("AdvancedProcessor shutdown complete")

# Global processor instance
_advanced_processor = None

def get_advanced_processor() -> AdvancedProcessor:
    """Get global advanced processor instance"""
    global _advanced_processor
    if _advanced_processor is None:
        _advanced_processor = AdvancedProcessor()
    return _advanced_processor

# Utility functions for common processing patterns
def parallel_map(func: Callable, items: List[Any], 
                task_type: TaskType = TaskType.IO_BOUND,
                max_workers: int = None) -> List[Any]:
    """Process items in parallel using the advanced processor"""
    processor = get_advanced_processor()
    
    # Submit all tasks
    task_ids = []
    for item in items:
        task_id = processor.submit_task(
            func=func,
            task_type=task_type,
            priority=TaskPriority.NORMAL,
            args=(item,)
        )
        task_ids.append(task_id)
    
    # Collect results
    results = []
    for task_id in task_ids:
        result = processor.get_task_result(task_id)
        if result and result.success:
            results.append(result.result)
        else:
            results.append(None)
    
    return results

@contextmanager
def processing_context():
    """Context manager for processing operations"""
    processor = get_advanced_processor()
    try:
        yield processor
    finally:
        # Clean up old tasks
        processor.clear_completed_tasks() 