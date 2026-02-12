import time
import threading
import logging
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict
import hashlib
import json
import pickle
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CachePolicy(Enum):
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"

@dataclass
class CacheEntry:
    """Represents a cache entry"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float]
    size: int

class InMemoryCache:
    """High-performance in-memory cache with TTL and eviction policies"""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 300, 
                 policy: CachePolicy = CachePolicy.LRU):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.policy = policy
        
        # Cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_worker(self):
        """Background worker to clean up expired entries"""
        while True:
            try:
                current_time = time.time()
                with self.lock:
                    expired_keys = []
                    
                    for key, entry in self.cache.items():
                        if entry.ttl and (current_time - entry.created_at) > entry.ttl:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        del self.cache[key]
                        self.evictions += 1
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in cache cleanup worker: {e}")
                time.sleep(60)
    
    def _evict_if_needed(self):
        """Evict entries if cache is full"""
        if len(self.cache) >= self.max_size:
            if self.policy == CachePolicy.LRU:
                # Remove least recently used
                key, entry = self.cache.popitem(last=False)
            elif self.policy == CachePolicy.LFU:
                # Remove least frequently used
                key, entry = min(self.cache.items(), key=lambda x: x[1].access_count)
                del self.cache[key]
            elif self.policy == CachePolicy.FIFO:
                # Remove first in
                key, entry = self.cache.popitem(last=False)
            
            self.evictions += 1
            logger.debug(f"Evicted cache entry: {key}")
    
    def _update_access(self, key: str):
        """Update access information for an entry"""
        if key in self.cache:
            entry = self.cache[key]
            entry.last_accessed = time.time()
            entry.access_count += 1
            
            # Move to end for LRU
            if self.policy == CachePolicy.LRU:
                self.cache.move_to_end(key)
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if expired
                if entry.ttl and (time.time() - entry.created_at) > entry.ttl:
                    del self.cache[key]
                    self.misses += 1
                    return None
                
                self._update_access(key)
                self.hits += 1
                return entry.value
            
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set a value in cache"""
        try:
            with self.lock:
                # Calculate entry size (rough estimate)
                size = len(str(value)) if isinstance(value, str) else 100
                
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    access_count=1,
                    ttl=ttl or self.default_ttl,
                    size=size
                )
                
                # Evict if needed
                self._evict_if_needed()
                
                # Add to cache
                self.cache[key] = entry
                
                # Move to end for LRU
                if self.policy == CachePolicy.LRU:
                    self.cache.move_to_end(key)
                
                return True
                
        except Exception as e:
            logger.error(f"Error setting cache entry {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete a key from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'hit_rate': hit_rate,
                'policy': self.policy.value
            }

class QueryCache:
    """Specialized cache for query results with semantic similarity"""
    
    def __init__(self, max_size: int = 500, similarity_threshold: float = 0.8):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.cache = InMemoryCache(max_size=max_size)
        
        # Query embeddings for similarity search
        self.query_embeddings = {}
        
        # Import embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load embedding model for query cache: {e}")
            self.embedding_model = None
    
    def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Get embedding for a query"""
        if self.embedding_model is None:
            return None
        
        try:
            embedding = self.embedding_model.encode([query])[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error getting query embedding: {e}")
            return None
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            import numpy as np
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def get(self, query: str) -> Optional[Any]:
        """Get cached result for a query, including similar queries"""
        # First try exact match
        result = self.cache.get(query)
        if result is not None:
            return result
        
        # Try semantic similarity if embedding model is available
        if self.embedding_model is not None:
            query_embedding = self._get_query_embedding(query)
            if query_embedding is not None:
                # Find most similar cached query
                best_similarity = 0.0
                best_result = None
                
                for cached_query, cached_embedding in self.query_embeddings.items():
                    similarity = self._calculate_similarity(query_embedding, cached_embedding)
                    if similarity > self.similarity_threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_result = self.cache.get(cached_query)
                
                if best_result is not None:
                    logger.debug(f"Found similar query with similarity {best_similarity:.3f}")
                    return best_result
        
        return None
    
    def set(self, query: str, result: Any, ttl: Optional[float] = None) -> bool:
        """Cache a query result"""
        success = self.cache.set(query, result, ttl)
        
        if success and self.embedding_model is not None:
            # Store query embedding
            embedding = self._get_query_embedding(query)
            if embedding is not None:
                self.query_embeddings[query] = embedding
        
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = self.cache.get_stats()
        stats['query_embeddings'] = len(self.query_embeddings)
        stats['similarity_threshold'] = self.similarity_threshold
        return stats

class ModelCache:
    """Cache for model outputs and intermediate results"""
    
    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.cache = InMemoryCache(max_size=max_size, default_ttl=1800)  # 30 minutes TTL
    
    def _generate_key(self, model_name: str, inputs: Any) -> str:
        """Generate a cache key for model inputs"""
        try:
            # Create a hash of the inputs
            if isinstance(inputs, str):
                input_str = inputs
            else:
                input_str = json.dumps(inputs, sort_keys=True, default=str)
            
            key_data = f"{model_name}:{input_str}"
            return hashlib.md5(key_data.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return f"{model_name}:{hash(str(inputs))}"
    
    def get_model_output(self, model_name: str, inputs: Any) -> Optional[Any]:
        """Get cached model output"""
        key = self._generate_key(model_name, inputs)
        return self.cache.get(key)
    
    def set_model_output(self, model_name: str, inputs: Any, output: Any, 
                        ttl: Optional[float] = None) -> bool:
        """Cache model output"""
        key = self._generate_key(model_name, inputs)
        return self.cache.set(key, output, ttl)
    
    def invalidate_model(self, model_name: str):
        """Invalidate all cached outputs for a model"""
        with self.cache.lock:
            keys_to_remove = []
            for key in self.cache.cache.keys():
                if key.startswith(f"{model_name}:"):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache.cache[key]

class PropertyCache:
    """Specialized cache for property data"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = InMemoryCache(max_size=max_size, default_ttl=3600)  # 1 hour TTL
        
        # Index for property searches
        self.property_index = {}
    
    def cache_properties(self, properties: List[Dict[str, Any]]) -> bool:
        """Cache a list of properties"""
        try:
            for property_data in properties:
                property_id = property_data.get('propertyId', property_data.get('PropertyID', str(hash(str(property_data)))))
                
                # Cache individual property
                self.cache.set(f"property:{property_id}", property_data)
                
                # Index by various criteria
                self._index_property(property_id, property_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error caching properties: {e}")
            return False
    
    def _index_property(self, property_id: str, property_data: Dict[str, Any]):
        """Index property by various criteria for fast lookup"""
        try:
            # Index by location
            latitude = property_data.get('Latitude')
            longitude = property_data.get('Longitude')
            if latitude and longitude:
                location_key = f"location:{latitude:.3f}:{longitude:.3f}"
                if location_key not in self.property_index:
                    self.property_index[location_key] = []
                self.property_index[location_key].append(property_id)
            
            # Index by property type
            property_type = property_data.get('PropertyType', '').lower()
            if property_type:
                type_key = f"type:{property_type}"
                if type_key not in self.property_index:
                    self.property_index[type_key] = []
                self.property_index[type_key].append(property_id)
            
            # Index by price range
            price = property_data.get('MarketValue', 0)
            if price > 0:
                price_range = f"price:{price // 10000 * 10000}-{(price // 10000 + 1) * 10000}"
                if price_range not in self.property_index:
                    self.property_index[price_range] = []
                self.property_index[price_range].append(property_id)
                
        except Exception as e:
            logger.error(f"Error indexing property {property_id}: {e}")
    
    def get_property(self, property_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific property by ID"""
        return self.cache.get(f"property:{property_id}")
    
    def search_properties(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search properties by criteria"""
        try:
            matching_ids = set()
            
            # Search by location
            if 'latitude' in criteria and 'longitude' in criteria:
                lat = criteria['latitude']
                lon = criteria['longitude']
                location_key = f"location:{lat:.3f}:{lon:.3f}"
                if location_key in self.property_index:
                    matching_ids.update(self.property_index[location_key])
            
            # Search by property type
            if 'property_type' in criteria:
                property_type = criteria['property_type'].lower()
                type_key = f"type:{property_type}"
                if type_key in self.property_index:
                    if matching_ids:
                        matching_ids &= set(self.property_index[type_key])
                    else:
                        matching_ids.update(self.property_index[type_key])
            
            # Search by price range
            if 'max_price' in criteria:
                max_price = criteria['max_price']
                price_range = f"price:0-{max_price}"
                if price_range in self.property_index:
                    if matching_ids:
                        matching_ids &= set(self.property_index[price_range])
                    else:
                        matching_ids.update(self.property_index[price_range])
            
            # Get matching properties
            results = []
            for property_id in matching_ids:
                property_data = self.get_property(property_id)
                if property_data:
                    results.append(property_data)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching properties: {e}")
            return []

# Global cache instances
query_cache = QueryCache()
model_cache = ModelCache()
property_cache = PropertyCache()

# Cache manager for coordinating all caches
class CacheManager:
    """Manages all cache instances and provides unified interface"""
    
    def __init__(self):
        self.caches = {
            'query': query_cache,
            'model': model_cache,
            'property': property_cache
        }
    
    def get_cache(self, cache_type: str) -> Optional[Any]:
        """Get a specific cache instance"""
        return self.caches.get(cache_type)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches"""
        stats = {}
        for cache_type, cache in self.caches.items():
            if hasattr(cache, 'get_stats'):
                stats[cache_type] = cache.get_stats()
            else:
                stats[cache_type] = {'size': len(cache.cache.cache) if hasattr(cache, 'cache') else 0}
        return stats
    
    def clear_all(self):
        """Clear all caches"""
        for cache in self.caches.values():
            if hasattr(cache, 'clear'):
                cache.clear()
            elif hasattr(cache, 'cache') and hasattr(cache.cache, 'clear'):
                cache.cache.clear()

# Global cache manager
cache_manager = CacheManager() 