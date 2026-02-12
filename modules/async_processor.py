import asyncio
import threading
import queue
import time
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import uuid
import weakref
from collections import defaultdict
import torch
import numpy as np
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RequestPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class ProcessingRequest:
    request_id: str
    session_id: str
    query: str
    priority: RequestPriority
    timestamp: float
    callback: Optional[Callable] = None
    metadata: Optional[Dict] = None

class AsyncRequestProcessor:
    """Handles async processing of chatbot requests with priority queuing"""
    
    def __init__(self, max_workers: int = 8, max_queue_size: int = 1000):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        
        # Priority queues for different request types
        self.text_generation_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.rag_retrieval_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.location_processing_queue = queue.PriorityQueue(maxsize=max_queue_size)
        
        # Thread pools for different types of work
        self.text_generation_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="text_gen")
        self.rag_processing_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag_proc")
        self.io_processing_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="io_proc")
        
        # Process pool for CPU-intensive tasks
        self.cpu_pool = ProcessPoolExecutor(max_workers=2)
        
        # Request tracking
        self.active_requests = {}
        self.completed_requests = {}
        self.request_stats = defaultdict(int)
        
        # Model batching
        self.batch_size = 4
        self.batch_timeout = 0.1  # seconds
        self.current_batch = []
        self.batch_lock = threading.Lock()
        
        # Start background workers
        self.running = True
        self.workers = []
        self._start_workers()
        
    def _start_workers(self):
        """Start background worker threads"""
        worker_threads = [
            threading.Thread(target=self._text_generation_worker, daemon=True),
            threading.Thread(target=self._rag_processing_worker, daemon=True),
            threading.Thread(target=self._location_processing_worker, daemon=True),
            threading.Thread(target=self._batch_processor, daemon=True)
        ]
        
        for worker in worker_threads:
            worker.start()
            self.workers.append(worker)
    
    def submit_text_generation(self, session_id: str, query: str, priority: RequestPriority = RequestPriority.NORMAL, 
                              callback: Optional[Callable] = None, metadata: Optional[Dict] = None) -> str:
        """Submit a text generation request"""
        request_id = str(uuid.uuid4())
        request = ProcessingRequest(
            request_id=request_id,
            session_id=session_id,
            query=query,
            priority=priority,
            timestamp=time.time(),
            callback=callback,
            metadata=metadata
        )
        
        # Add to priority queue (lower number = higher priority)
        priority_value = -priority.value  # Negative for proper ordering
        self.text_generation_queue.put((priority_value, request))
        
        # Track active request
        self.active_requests[request_id] = request
        self.request_stats['text_generation_submitted'] += 1
        
        logger.info(f"Submitted text generation request {request_id} with priority {priority.name}")
        return request_id
    
    def submit_rag_retrieval(self, session_id: str, query: str, priority: RequestPriority = RequestPriority.NORMAL,
                           callback: Optional[Callable] = None, metadata: Optional[Dict] = None) -> str:
        """Submit a RAG retrieval request"""
        request_id = str(uuid.uuid4())
        request = ProcessingRequest(
            request_id=request_id,
            session_id=session_id,
            query=query,
            priority=priority,
            timestamp=time.time(),
            callback=callback,
            metadata=metadata
        )
        
        priority_value = -priority.value
        self.rag_retrieval_queue.put((priority_value, request))
        
        self.active_requests[request_id] = request
        self.request_stats['rag_retrieval_submitted'] += 1
        
        logger.info(f"Submitted RAG retrieval request {request_id} with priority {priority.name}")
        return request_id
    
    def _text_generation_worker(self):
        """Background worker for text generation requests"""
        while self.running:
            try:
                # Get request from queue with timeout
                priority_value, request = self.text_generation_queue.get(timeout=1.0)
                
                # Process the request
                future = self.text_generation_pool.submit(
                    self._process_text_generation, request
                )
                
                # Store future for result retrieval
                self.active_requests[request.request_id] = {
                    'request': request,
                    'future': future
                }
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in text generation worker: {e}")
    
    def _rag_processing_worker(self):
        """Background worker for RAG processing requests"""
        while self.running:
            try:
                priority_value, request = self.rag_retrieval_queue.get(timeout=1.0)
                
                future = self.rag_processing_pool.submit(
                    self._process_rag_retrieval, request
                )
                
                self.active_requests[request.request_id] = {
                    'request': request,
                    'future': future
                }
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in RAG processing worker: {e}")
    
    def _location_processing_worker(self):
        """Background worker for location processing requests"""
        while self.running:
            try:
                priority_value, request = self.location_processing_queue.get(timeout=1.0)
                
                future = self.io_processing_pool.submit(
                    self._process_location_request, request
                )
                
                self.active_requests[request.request_id] = {
                    'request': request,
                    'future': future
                }
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in location processing worker: {e}")
    
    def _batch_processor(self):
        """Background worker for batch processing"""
        while self.running:
            try:
                with self.batch_lock:
                    if len(self.current_batch) >= self.batch_size:
                        batch = self.current_batch.copy()
                        self.current_batch.clear()
                    else:
                        batch = None
                
                if batch:
                    self._process_batch(batch)
                
                time.sleep(self.batch_timeout)
                
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
    
    def _process_text_generation(self, request: ProcessingRequest) -> Dict[str, Any]:
        """Process a text generation request"""
        try:
            start_time = time.time()
            
            # Import here to avoid circular imports
            from modules.response import generate_response
            from modules.models import load_tokenizer_and_model
            
            # Get tokenizer and model (these should be cached globally)
            tokenizer, model_llm = load_tokenizer_and_model()
            
            # Generate response
            response, duration = generate_response(
                request.query, 
                tokenizer, 
                model_llm,
                max_new_tokens=256,
                temperature=0.7
            )
            
            result = {
                'request_id': request.request_id,
                'session_id': request.session_id,
                'response': response,
                'duration': duration,
                'processing_time': time.time() - start_time,
                'status': 'success'
            }
            
            # Call callback if provided
            if request.callback:
                try:
                    request.callback(result)
                except Exception as e:
                    logger.error(f"Error in callback for request {request.request_id}: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing text generation request {request.request_id}: {e}")
            return {
                'request_id': request.request_id,
                'session_id': request.session_id,
                'error': str(e),
                'status': 'error'
            }
    
    def _process_rag_retrieval(self, request: ProcessingRequest) -> Dict[str, Any]:
        """Process a RAG retrieval request"""
        try:
            start_time = time.time()
            
            # Import here to avoid circular imports
            from modules.models import get_global_retriever
            
            # Get global retriever
            retriever = get_global_retriever()
            
            if retriever is None:
                raise Exception("RAG retriever not available")
            
            # Perform retrieval
            results = retriever.retrieve(request.query, top_k=5)
            
            result = {
                'request_id': request.request_id,
                'session_id': request.session_id,
                'results': results,
                'processing_time': time.time() - start_time,
                'status': 'success'
            }
            
            if request.callback:
                try:
                    request.callback(result)
                except Exception as e:
                    logger.error(f"Error in callback for request {request.request_id}: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing RAG retrieval request {request.request_id}: {e}")
            return {
                'request_id': request.request_id,
                'session_id': request.session_id,
                'error': str(e),
                'status': 'error'
            }
    
    def _process_location_request(self, request: ProcessingRequest) -> Dict[str, Any]:
        """Process a location-based request"""
        try:
            start_time = time.time()
            
            # Import here to avoid circular imports
            from modules.location_processor import LocationProcessor
            
            location_processor = LocationProcessor()
            
            # Extract location data from metadata
            metadata = request.metadata or {}
            latitude = metadata.get('latitude')
            longitude = metadata.get('longitude')
            
            if not latitude or not longitude:
                raise Exception("Location coordinates not provided")
            
            # Process location
            result_data = location_processor.set_location(latitude, longitude, request.session_id)
            
            result = {
                'request_id': request.request_id,
                'session_id': request.session_id,
                'location_data': result_data,
                'processing_time': time.time() - start_time,
                'status': 'success'
            }
            
            if request.callback:
                try:
                    request.callback(result)
                except Exception as e:
                    logger.error(f"Error in callback for request {request.request_id}: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing location request {request.request_id}: {e}")
            return {
                'request_id': request.request_id,
                'session_id': request.session_id,
                'error': str(e),
                'status': 'error'
            }
    
    def _process_batch(self, batch: List[ProcessingRequest]):
        """Process a batch of requests"""
        try:
            # Group requests by type
            text_requests = [req for req in batch if req.query]
            
            if text_requests:
                # Process text generation in batch
                futures = []
                for request in text_requests:
                    future = self.text_generation_pool.submit(
                        self._process_text_generation, request
                    )
                    futures.append((request, future))
                
                # Collect results
                for request, future in futures:
                    try:
                        result = future.result(timeout=30)
                        if request.callback:
                            request.callback(result)
                    except Exception as e:
                        logger.error(f"Error in batch processing for request {request.request_id}: {e}")
        
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a request"""
        if request_id in self.completed_requests:
            return self.completed_requests[request_id]
        
        if request_id in self.active_requests:
            request_data = self.active_requests[request_id]
            if isinstance(request_data, dict) and 'future' in request_data:
                future = request_data['future']
                if future.done():
                    try:
                        result = future.result(timeout=0)
                        self.completed_requests[request_id] = result
                        del self.active_requests[request_id]
                        return result
                    except Exception as e:
                        return {'status': 'error', 'error': str(e)}
                else:
                    return {'status': 'processing'}
            else:
                return {'status': 'queued'}
        
        return None
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the processing queues"""
        return {
            'text_generation_queue_size': self.text_generation_queue.qsize(),
            'rag_retrieval_queue_size': self.rag_retrieval_queue.qsize(),
            'location_processing_queue_size': self.location_processing_queue.qsize(),
            'active_requests': len(self.active_requests),
            'completed_requests': len(self.completed_requests),
            'request_stats': dict(self.request_stats)
        }
    
    def shutdown(self):
        """Shutdown the processor and clean up resources"""
        self.running = False
        
        # Shutdown thread pools
        self.text_generation_pool.shutdown(wait=True)
        self.rag_processing_pool.shutdown(wait=True)
        self.io_processing_pool.shutdown(wait=True)
        self.cpu_pool.shutdown(wait=True)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        logger.info("AsyncRequestProcessor shutdown complete")

# Global instance with error handling
try:
    async_processor = AsyncRequestProcessor()
except Exception as e:
    logger.error(f"Failed to initialize AsyncRequestProcessor: {e}")
    async_processor = None 