import torch
import threading
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import gc
import weakref
from collections import OrderedDict
import pickle
import hashlib
import json

# Import existing modules
from .global_models import get_jina_model, get_llm_tokenizer_and_model
from .performance_optimizer import get_performance_optimizer, TaskPriority

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelCache:
    """Intelligent model caching with memory management"""
    
    def __init__(self, max_models: int = 10, max_memory_gb: float = 8.0):
        self.max_models = max_models
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.models = OrderedDict()
        self.model_sizes = {}
        self.access_times = {}
        self.lock = threading.RLock()
    
    def _estimate_model_size(self, model) -> int:
        """Estimate model size in bytes"""
        try:
            if hasattr(model, 'state_dict'):
                total_params = sum(p.numel() * p.element_size() for p in model.parameters())
                return total_params
            else:
                return 100 * 1024 * 1024  # Default 100MB estimate
        except:
            return 100 * 1024 * 1024
    
    def add_model(self, key: str, model: Any) -> bool:
        """Add a model to cache"""
        with self.lock:
            model_size = self._estimate_model_size(model)
            
            # Check if we need to evict models
            while (len(self.models) >= self.max_models or 
                   sum(self.model_sizes.values()) + model_size > self.max_memory_bytes):
                self._evict_oldest()
            
            self.models[key] = model
            self.model_sizes[key] = model_size
            self.access_times[key] = time.time()
            return True
    
    def get_model(self, key: str) -> Optional[Any]:
        """Get a model from cache"""
        with self.lock:
            if key in self.models:
                self.access_times[key] = time.time()
                # Move to end (most recently used)
                self.models.move_to_end(key)
                return self.models[key]
            return None
    
    def _evict_oldest(self):
        """Evict the least recently used model"""
        if not self.models:
            return
        
        oldest_key = next(iter(self.models))
        del self.models[oldest_key]
        del self.model_sizes[oldest_key]
        del self.access_times[oldest_key]
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class EnhancedModelManager:
    """
    Enhanced model manager with parallel processing, intelligent caching,
    and optimized resource management
    """
    
    def __init__(self):
        self.model_cache = ModelCache()
        self.performance_optimizer = get_performance_optimizer()
        
        # Model loading locks
        self.model_locks = {}
        self.lock = threading.Lock()
        
        # Preloaded models
        self.preloaded_models = {}
        
        # Background preloading
        self.preload_thread = None
        self.running = True
        
        # Start background preloading
        self._start_preloading()
        
        logger.info("EnhancedModelManager initialized")
    
    def _start_preloading(self):
        """Start background model preloading"""
        self.preload_thread = threading.Thread(target=self._preload_models, daemon=True)
        self.preload_thread.start()
    
    def _preload_models(self):
        """Preload commonly used models in background"""
        try:
            # Preload Jina embeddings
            logger.info("Preloading Jina embeddings model...")
            jina_model = get_jina_model()
            self.model_cache.add_model("jina_embeddings", jina_model)
            
            # Preload LLM components
            logger.info("Preloading LLM tokenizer and model...")
            tokenizer, llm_model = get_llm_tokenizer_and_model()
            self.model_cache.add_model("llm_tokenizer", tokenizer)
            self.model_cache.add_model("llm_model", llm_model)
            
            logger.info("Model preloading completed")
            
        except Exception as e:
            logger.error(f"Error in model preloading: {e}")
    
    def get_model(self, model_name: str, force_reload: bool = False) -> Any:
        """Get a model with caching and parallel loading"""
        
        # Check cache first
        if not force_reload:
            cached_model = self.model_cache.get_model(model_name)
            if cached_model is not None:
                return cached_model
        
        # Get or create lock for this model
        with self.lock:
            if model_name not in self.model_locks:
                self.model_locks[model_name] = threading.Lock()
        
        model_lock = self.model_locks[model_name]
        
        with model_lock:
            # Double-check cache after acquiring lock
            if not force_reload:
                cached_model = self.model_cache.get_model(model_name)
                if cached_model is not None:
                    return cached_model
            
            # Load model
            model = self._load_model(model_name)
            if model is not None:
                self.model_cache.add_model(model_name, model)
            
            return model
    
    def _load_model(self, model_name: str) -> Any:
        """Load a specific model"""
        try:
            if model_name == "jina_embeddings":
                return get_jina_model()
            elif model_name == "llm_tokenizer":
                tokenizer, _ = get_llm_tokenizer_and_model()
                return tokenizer
            elif model_name == "llm_model":
                _, model = get_llm_tokenizer_and_model()
                return model
            elif model_name == "feature_matcher":
                from .feature_matcher import DynamicFeatureMatcher
                return DynamicFeatureMatcher(load_saved=True)
            else:
                logger.warning(f"Unknown model: {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def parallel_embed_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for multiple texts in parallel"""
        
        def embed_batch(batch_texts: List[str]) -> List[np.ndarray]:
            model = self.get_model("jina_embeddings")
            if model is None:
                return [np.zeros(768) for _ in batch_texts]  # Fallback
            
            try:
                embeddings = model.encode(batch_texts, convert_to_tensor=False)
                return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
            except Exception as e:
                logger.error(f"Error in batch embedding: {e}")
                return [np.zeros(768) for _ in batch_texts]
        
        # Split into batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        # Submit batch tasks
        task_ids = []
        for batch in batches:
            task_id = self.performance_optimizer.submit_task(
                embed_batch,
                task_type="gpu" if torch.cuda.is_available() else "cpu",
                priority=TaskPriority.NORMAL,
                timeout=None,
                callback=None,
                *batch
            )
            task_ids.append(task_id)
        
        # Collect results
        all_embeddings = []
        for task_id in task_ids:
            result = self.performance_optimizer.get_task_result(task_id)
            if result is not None:
                all_embeddings.extend(result)
        
        return all_embeddings
    
    def parallel_generate_responses(self, 
                                  queries: List[str], 
                                  tokenizer: Any = None,
                                  model: Any = None,
                                  batch_size: int = 4) -> List[str]:
        """Generate responses for multiple queries in parallel"""
        
        if tokenizer is None:
            tokenizer = self.get_model("llm_tokenizer")
        if model is None:
            model = self.get_model("llm_model")
        
        if tokenizer is None or model is None:
            return ["Error: Models not available"] * len(queries)
        
        def generate_batch(batch_queries: List[str]) -> List[str]:
            from .response import generate_response
            responses = []
            
            for query in batch_queries:
                try:
                    response, _ = generate_response(
                        query, tokenizer, model,
                        max_new_tokens=256,
                        temperature=0.7
                    )
                    responses.append(response)
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    responses.append("Error generating response")
            
            return responses
        
        # Split into batches
        batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
        
        # Submit batch tasks
        task_ids = []
        for batch in batches:
            task_id = self.performance_optimizer.submit_task(
                generate_batch,
                task_type="gpu" if torch.cuda.is_available() else "cpu",
                priority=TaskPriority.HIGH,
                timeout=None,
                callback=None,
                *batch
            )
            task_ids.append(task_id)
        
        # Collect results
        all_responses = []
        for task_id in task_ids:
            result = self.performance_optimizer.get_task_result(task_id)
            if result is not None:
                all_responses.extend(result)
        
        return all_responses
    
    def parallel_process_properties(self, 
                                  properties: List[Dict],
                                  processor_func: str = "format_property_details",
                                  batch_size: int = 50) -> List[Dict]:
        """Process properties in parallel batches"""
        
        def process_batch(batch_properties: List[Dict]) -> List[Dict]:
            if processor_func == "format_property_details":
                from .property_processor import PropertyProcessor
                processor = PropertyProcessor()
                return [processor.format_property_details(prop) for prop in batch_properties]
            elif processor_func == "extract_features":
                from .feature_matcher import DynamicFeatureMatcher
                matcher = DynamicFeatureMatcher(load_saved=True)
                return [matcher.analyze_query(str(prop)) for prop in batch_properties]
            else:
                return batch_properties
        
        # Split into batches
        batches = [properties[i:i + batch_size] for i in range(0, len(properties), batch_size)]
        
        # Submit batch tasks
        task_ids = []
        for batch in batches:
            task_id = self.performance_optimizer.submit_task(
                process_batch,
                task_type="io",
                priority=TaskPriority.NORMAL,
                timeout=None,
                callback=None,
                *batch
            )
            task_ids.append(task_id)
        
        # Collect results
        all_results = []
        for task_id in task_ids:
            result = self.performance_optimizer.get_task_result(task_id)
            if result is not None:
                all_results.extend(result)
        
        return all_results
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model manager statistics"""
        return {
            'cached_models': len(self.model_cache.models),
            'model_sizes': dict(self.model_cache.model_sizes),
            'total_cached_size_mb': sum(self.model_cache.model_sizes.values()) / (1024 * 1024),
            'performance_stats': self.performance_optimizer.get_performance_stats()
        }
    
    def clear_cache(self):
        """Clear model cache"""
        self.model_cache.models.clear()
        self.model_cache.model_sizes.clear()
        self.model_cache.access_times.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def shutdown(self):
        """Shutdown the model manager"""
        self.running = False
        if self.preload_thread:
            self.preload_thread.join(timeout=5.0)
        self.clear_cache()

# Global enhanced model manager instance
enhanced_model_manager = None

def get_enhanced_model_manager() -> EnhancedModelManager:
    """Get the global enhanced model manager instance"""
    global enhanced_model_manager
    if enhanced_model_manager is None:
        enhanced_model_manager = EnhancedModelManager()
    return enhanced_model_manager

# Utility functions for easy access
def parallel_embed_texts(texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
    """Generate embeddings for texts in parallel"""
    manager = get_enhanced_model_manager()
    return manager.parallel_embed_texts(texts, batch_size)

def parallel_generate_responses(queries: List[str], batch_size: int = 4) -> List[str]:
    """Generate responses for queries in parallel"""
    manager = get_enhanced_model_manager()
    return manager.parallel_generate_responses(queries, batch_size)

def parallel_process_properties(properties: List[Dict], 
                              processor_func: str = "format_property_details",
                              batch_size: int = 50) -> List[Dict]:
    """Process properties in parallel"""
    manager = get_enhanced_model_manager()
    return manager.parallel_process_properties(properties, processor_func, batch_size) 