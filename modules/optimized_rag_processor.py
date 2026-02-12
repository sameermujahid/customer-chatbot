import torch
import numpy as np
import faiss
import logging
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
from functools import lru_cache
import pickle
import hashlib
import json
from pathlib import Path

# Import existing modules
from .performance_optimizer import get_performance_optimizer, TaskPriority
from .enhanced_model_manager import get_enhanced_model_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedRAGProcessor:
    """
    High-performance RAG processor with parallel retrieval,
    intelligent caching, and optimized vector operations
    """
    
    def __init__(self, 
                 model=None, 
                 index=None, 
                 tokenizer=None, 
                 pca=None, 
                 feature_matcher=None):
        
        self.model = model
        self.index = index
        self.tokenizer = tokenizer
        self.pca = pca
        self.feature_matcher = feature_matcher
        
        # Performance components
        self.performance_optimizer = get_performance_optimizer()
        self.model_manager = get_enhanced_model_manager()
        
        # Caching
        self.query_cache = {}
        self.embedding_cache = {}
        self.result_cache = {}
        
        # Thread safety
        self.cache_lock = threading.RLock()
        
        # Performance metrics
        self.metrics = {
            'total_queries': 0,
            'cache_hits': 0,
            'avg_retrieval_time': 0.0,
            'avg_processing_time': 0.0
        }
        
        # Batch processing
        self.batch_size = 32
        self.max_concurrent_queries = 8
        
        logger.info("OptimizedRAGProcessor initialized")
    
    def _generate_cache_key(self, query: str, top_k: int) -> str:
        """Generate cache key for query"""
        cache_data = f"{query}:{top_k}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[Dict]]:
        """Get cached result"""
        with self.cache_lock:
            if cache_key in self.result_cache:
                self.metrics['cache_hits'] += 1
                return self.result_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: List[Dict]):
        """Cache result"""
        with self.cache_lock:
            # Simple LRU cache with max size
            if len(self.result_cache) > 1000:
                # Remove oldest entry
                oldest_key = next(iter(self.result_cache))
                del self.result_cache[oldest_key]
            
            self.result_cache[cache_key] = result
    
    def parallel_retrieve(self, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
        """Retrieve results for multiple queries in parallel"""
        
        def retrieve_single(query: str) -> List[Dict]:
            return self.retrieve(query, top_k)
        
        # Submit parallel tasks
        task_ids = []
        for query in queries:
            task_id = self.performance_optimizer.submit_task(
                retrieve_single,
                "gpu" if torch.cuda.is_available() else "cpu",
                TaskPriority.HIGH,
                None,
                None,
                query
            )
            task_ids.append(task_id)
        
        # Collect results
        results = []
        for task_id in task_ids:
            result = self.performance_optimizer.get_task_result(task_id)
            if result is not None:
                results.append(result)
            else:
                results.append([])
        
        return results
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Enhanced retrieve method with caching and optimization"""
        
        start_time = time.time()
        self.metrics['total_queries'] += 1
        
        # Check cache first
        cache_key = self._generate_cache_key(query, top_k)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            # Generate query embedding
            query_embedding = self._get_query_embedding(query)
            if query_embedding is None:
                return []
            
            # Apply PCA if available
            if self.pca is not None:
                query_embedding = self.pca.transform(query_embedding.reshape(1, -1)).flatten()
            
            # Search index
            if self.index is not None:
                # Convert to float32 for FAISS
                query_embedding = query_embedding.astype(np.float32)
                
                # Search with increased top_k for better filtering
                search_k = min(top_k * 3, 100)
                distances, indices = self.index.search(
                    query_embedding.reshape(1, -1), 
                    search_k
                )
                
                # Get properties from cache
                from .models import get_cached_properties
                properties = get_cached_properties()
                
                if not properties:
                    return []
                
                # Process results
                results = []
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx < len(properties):
                        property_data = properties[idx]
                        
                        # Apply feature matching if available
                        if self.feature_matcher:
                            try:
                                analysis = self.feature_matcher.analyze_query(query)
                                if not self.feature_matcher.check_property_features(
                                    property_data, analysis['feature_requirements']
                                ):
                                    continue
                            except Exception as e:
                                logger.warning(f"Feature matching failed: {e}")
                        
                        # Format property details
                        formatted_property = self._format_property(property_data)
                        
                        results.append({
                            'property': formatted_property,
                            'distance': float(distance),
                            'index': int(idx)
                        })
                        
                        if len(results) >= top_k:
                            break
                
                # Cache result
                self._cache_result(cache_key, results)
                
                # Update metrics
                retrieval_time = time.time() - start_time
                self.metrics['avg_retrieval_time'] = (
                    (self.metrics['avg_retrieval_time'] * (self.metrics['total_queries'] - 1) + retrieval_time) / 
                    self.metrics['total_queries']
                )
                
                return results
            
            else:
                logger.warning("No FAISS index available")
                return []
                
        except Exception as e:
            logger.error(f"Error in retrieve: {e}")
            return []
    
    def _get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Get query embedding with caching"""
        
        # Check embedding cache
        with self.cache_lock:
            if query in self.embedding_cache:
                return self.embedding_cache[query]
        
        try:
            if self.model is None:
                # Get model from manager
                self.model = self.model_manager.get_model("jina_embeddings")
                if self.model is None:
                    return None
            
            # Generate embedding
            embedding = self.model.encode(query, convert_to_tensor=False)
            
            # Cache embedding
            with self.cache_lock:
                self.embedding_cache[query] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return None
    
    def _format_property(self, property_data: Dict) -> Dict:
        """Format property data for response"""
        try:
            from .property_processor import PropertyProcessor
            processor = PropertyProcessor()
            return processor.format_property_details(property_data)
        except Exception as e:
            logger.error(f"Error formatting property: {e}")
            return property_data
    
    def batch_embed_properties(self, properties: List[Dict]) -> np.ndarray:
        """Generate embeddings for properties in parallel batches"""
        
        def extract_property_text(property_data: Dict) -> str:
            """Extract text representation of property"""
            try:
                # Combine key property information
                text_parts = []
                
                # Basic info
                if property_data.get('PropertyName'):
                    text_parts.append(property_data['PropertyName'])
                
                if property_data.get('Address'):
                    text_parts.append(property_data['Address'])
                
                if property_data.get('Description'):
                    text_parts.append(property_data['Description'])
                
                # Property details
                if property_data.get('PropertyType'):
                    text_parts.append(f"Type: {property_data['PropertyType']}")
                
                if property_data.get('NumberOfRooms'):
                    text_parts.append(f"Rooms: {property_data['NumberOfRooms']}")
                
                if property_data.get('MarketValue'):
                    text_parts.append(f"Value: {property_data['MarketValue']}")
                
                # Features
                if property_data.get('KeyFeatures'):
                    text_parts.append(f"Features: {property_data['KeyFeatures']}")
                
                return " | ".join(text_parts)
                
            except Exception as e:
                logger.error(f"Error extracting property text: {e}")
                return str(property_data)
        
        # Extract text representations
        property_texts = []
        for prop in properties:
            text = extract_property_text(prop)
            property_texts.append(text)
        
        # Generate embeddings in parallel
        embeddings = self.model_manager.parallel_embed_texts(
            property_texts, 
            batch_size=self.batch_size
        )
        
        return np.array(embeddings)
    
    def build_optimized_index(self, properties: List[Dict]) -> bool:
        """Build optimized FAISS index with parallel processing"""
        
        try:
            logger.info(f"Building optimized index for {len(properties)} properties...")
            start_time = time.time()
            
            # Generate embeddings in parallel
            embeddings = self.batch_embed_properties(properties)
            
            # Apply PCA if needed
            if self.pca is not None:
                embeddings = self.pca.fit_transform(embeddings)
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            
            # Use GPU if available
            if torch.cuda.is_available():
                try:
                    # GPU index
                    self.index = faiss.IndexFlatIP(dimension)
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    logger.info("Using GPU FAISS index")
                except Exception as e:
                    logger.warning(f"GPU FAISS failed, falling back to CPU: {e}")
                    self.index = faiss.IndexFlatIP(dimension)
            else:
                self.index = faiss.IndexFlatIP(dimension)
            
            # Add vectors to index
            self.index.add(embeddings.astype(np.float32))
            
            build_time = time.time() - start_time
            logger.info(f"Index built successfully in {build_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'metrics': self.metrics,
            'cache_stats': {
                'query_cache_size': len(self.query_cache),
                'embedding_cache_size': len(self.embedding_cache),
                'result_cache_size': len(self.result_cache),
                'cache_hit_rate': self.metrics['cache_hits'] / max(self.metrics['total_queries'], 1)
            },
            'index_stats': {
                'index_size': self.index.ntotal if self.index else 0,
                'dimension': self.index.d if self.index else 0
            } if self.index else {},
            'performance_optimizer_stats': self.performance_optimizer.get_performance_stats()
        }
    
    def clear_caches(self):
        """Clear all caches"""
        with self.cache_lock:
            self.query_cache.clear()
            self.embedding_cache.clear()
            self.result_cache.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Global optimized RAG processor instance
optimized_rag_processor = None

def get_optimized_rag_processor() -> OptimizedRAGProcessor:
    """Get the global optimized RAG processor instance"""
    global optimized_rag_processor
    if optimized_rag_processor is None:
        # Skip ultimate RAG system - return None to use original retriever
        print("⚠️ Skipping optimized RAG processor - using original retriever only")
        return None
    
    return optimized_rag_processor

# Utility functions
def parallel_rag_retrieve(queries: List[str], top_k: int = 5) -> List[List[Dict]]:
    """Retrieve results for multiple queries in parallel"""
    processor = get_optimized_rag_processor()
    return processor.parallel_retrieve(queries, top_k)

def optimized_rag_retrieve(query: str, top_k: int = 5) -> List[Dict]:
    """Optimized single query retrieval"""
    processor = get_optimized_rag_processor()
    return processor.retrieve(query, top_k) 