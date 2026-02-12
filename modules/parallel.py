import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import torch
import logging
from functools import partial
import queue
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global thread pool for I/O bound tasks
thread_pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2)

# Global process pool for CPU bound tasks
process_pool = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())

# Queue for managing async tasks
task_queue = queue.Queue()

def get_device():
    """Get the appropriate device for computation"""
    return "cuda" if torch.cuda.is_available() else "cpu"

def parallel_map(func, items, use_processes=False):
    """Execute a function in parallel on a list of items"""
    executor = process_pool if use_processes else thread_pool
    return list(executor.map(func, items))

def batch_process(items, batch_size=32, func=None):
    """Process items in batches"""
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        if func:
            batch_results = parallel_map(func, batch)
        else:
            batch_results = batch
        results.extend(batch_results)
    return results

class AsyncTaskManager:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.tasks = {}
        self.results = {}
        self.lock = threading.Lock()

    def submit_task(self, task_id, func, *args, **kwargs):
        """Submit a task to be executed asynchronously"""
        future = self.thread_pool.submit(func, *args, **kwargs)
        with self.lock:
            self.tasks[task_id] = future
        return task_id

    def get_result(self, task_id, timeout=None):
        """Get the result of a task"""
        if task_id not in self.tasks:
            return None
        
        if task_id in self.results:
            return self.results[task_id]
        
        try:
            result = self.tasks[task_id].result(timeout=timeout)
            with self.lock:
                self.results[task_id] = result
            return result
        except Exception as e:
            logger.error(f"Error getting result for task {task_id}: {str(e)}")
            return None

    def cancel_task(self, task_id):
        """Cancel a running task"""
        if task_id in self.tasks:
            self.tasks[task_id].cancel()
            with self.lock:
                del self.tasks[task_id]

    def cleanup(self):
        """Clean up completed tasks"""
        with self.lock:
            completed_tasks = [task_id for task_id, future in self.tasks.items() 
                             if future.done()]
            for task_id in completed_tasks:
                if task_id not in self.results:
                    try:
                        self.results[task_id] = self.tasks[task_id].result()
                    except Exception:
                        pass
                del self.tasks[task_id]

class ModelParallelizer:
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
        self.device = get_device()
        self.model = self.model.to(self.device)
        self.model.eval()

    def parallel_predict(self, inputs):
        """Run predictions in parallel using batching"""
        results = []
        with torch.no_grad():
            for i in range(0, len(inputs), self.batch_size):
                batch = inputs[i:i + self.batch_size]
                batch = torch.stack(batch).to(self.device)
                batch_results = self.model(batch)
                results.extend(batch_results.cpu().numpy())
        return results

    def parallel_encode(self, texts):
        """Encode texts in parallel using batching"""
        return batch_process(texts, self.batch_size, self.model.encode)

# Global task manager instance
task_manager = AsyncTaskManager()

def cleanup_resources():
    """Clean up all parallel processing resources"""
    thread_pool.shutdown(wait=True)
    process_pool.shutdown(wait=True)
    task_manager.cleanup() 