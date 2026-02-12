import faiss
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import logging
from dataclasses import dataclass
from enum import Enum
import pickle
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import OrderedDict
import hashlib
import json

logger = logging.getLogger(__name__)

class SearchType(Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    FILTERED = "filtered"

@dataclass
class SearchResult:
    property_id: str
    property_data: Dict[str, Any]
    similarity_score: float
    search_type: SearchType
    ranking_factors: Dict[str, float]
    metadata: Dict[str, Any]

class AdvancedVectorSearch:
    """
    Advanced vector search with FAISS indexing, hybrid search, and intelligent ranking
    Optimized for high-performance, high-concurrency operation
    """
    
    def __init__(self, 
                 model_name: str = "jinaai/jina-embeddings-v3",
                 embedding_dim: int = 768,
                 index_type: str = "IVF100,Flat",
                 use_gpu: bool = True,
                 enable_caching: bool = True,
                 cache_size: int = 50000,
                 max_workers: int = 8):
        
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.enable_caching = enable_caching
        self.max_workers = max_workers
        
        # Initialize model with optimizations
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        if self.use_gpu:
            self.model = self.model.to('cuda')
        
        # FAISS index
        self.index = None
        self.property_texts = []
        self.property_embeddings = []
        self.property_metadata = []
        
        # Search configuration optimized for speed
        self.search_config = {
            'semantic_weight': 0.7,
            'keyword_weight': 0.3,
            'filter_weight': 0.1,
            'rerank_top_k': 50,  # Increased for better results
            'final_top_k': 5,
            'nprobe': 10,  # FAISS search parameter for speed/accuracy trade-off
            'batch_size': 100  # Batch size for embeddings
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Intelligent caching system
        self.search_cache = OrderedDict()
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'avg_search_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_hit_rate': 0.0
        }
        
        # Pre-computed embeddings cache
        self.embedding_cache = {}
        self.max_embedding_cache_size = 10000
        
        logger.info(f"AdvancedVectorSearch initialized with optimizations (GPU: {self.use_gpu}, Cache: {enable_caching})")
    
    def _generate_cache_key(self, query: str, search_type: SearchType, top_k: int) -> str:
        """Generate cache key for search results"""
        try:
            key_data = {
                'query': query.lower().strip(),
                'search_type': search_type.value,
                'top_k': top_k
            }
            key_string = json.dumps(key_data, sort_keys=True)
            return hashlib.md5(key_string.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return None
    
    def _get_cached_results(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Get cached search results if available"""
        if not self.enable_caching or not cache_key:
            return None
        
        with self.lock:
            if cache_key in self.search_cache:
                # Move to end (LRU)
                results = self.search_cache.pop(cache_key)
                self.search_cache[cache_key] = results
                self.cache_hits += 1
                return results
        
        self.cache_misses += 1
        return None
    
    def _cache_results(self, cache_key: str, results: List[SearchResult]):
        """Cache search results with LRU eviction"""
        if not self.enable_caching or not cache_key:
            return
        
        with self.lock:
            # Remove if already exists
            if cache_key in self.search_cache:
                self.search_cache.pop(cache_key)
            
            # Add to cache
            self.search_cache[cache_key] = results
            
            # Evict oldest if cache is full
            if len(self.search_cache) > self.cache_size:
                self.search_cache.popitem(last=False)
    
    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding if available"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        return None
    
    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding with size limit"""
        if len(self.embedding_cache) >= self.max_embedding_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]
        
        self.embedding_cache[text] = embedding
    
    def build_index(self, properties: List[Dict[str, Any]], 
                   batch_size: int = 100) -> bool:
        """Build FAISS index from properties with optimizations"""
        try:
            logger.info(f"Building optimized FAISS index for {len(properties)} properties")
            start_time = time.time()
            
            # Clear existing data
            self.property_texts = []
            self.property_embeddings = []
            self.property_metadata = []
            
            # Process properties in parallel batches
            all_embeddings = []
            
            def process_batch(batch):
                batch_texts = []
                batch_metadata = []
                
                for prop in batch:
                    # Format property for embedding
                    formatted_text = self._format_property_for_embedding(prop)
                    batch_texts.append(formatted_text)
                    batch_metadata.append({
                        'property_id': prop.get('propertyId', prop.get('id', str(hash(str(prop))))),
                        'property_type': prop.get('typeName', 'unknown'),
                        'price': prop.get('marketValue', 0),
                        'location': f"{prop.get('city', '')}, {prop.get('state', '')}"
                    })
                
                # Generate embeddings
                batch_embeddings = self.model.encode(
                    batch_texts, 
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    batch_size=min(len(batch_texts), self.search_config['batch_size'])
                )
                
                return batch_embeddings, batch_texts, batch_metadata
            
            # Process in parallel batches
            futures = []
            for i in range(0, len(properties), batch_size):
                batch = properties[i:i + batch_size]
                future = self.thread_pool.submit(process_batch, batch)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    batch_embeddings, batch_texts, batch_metadata = future.result()
                    all_embeddings.append(batch_embeddings)
                    self.property_texts.extend(batch_texts)
                    self.property_metadata.extend(batch_metadata)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
            
            # Combine all embeddings
            if all_embeddings:
                self.property_embeddings = np.vstack(all_embeddings).astype('float32')
                
                # Create optimized FAISS index
                if self.index_type == "IVF100,Flat":
                    # Quantizer for IVF
                    quantizer = faiss.IndexFlatIP(self.embedding_dim)
                    self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
                    
                    # Train the index
                    self.index.train(self.property_embeddings)
                    
                    # Set search parameters for speed
                    self.index.nprobe = self.search_config['nprobe']
                else:
                    # Simple flat index
                    self.index = faiss.IndexFlatIP(self.embedding_dim)
                
                # Add vectors to index
                self.index.add(self.property_embeddings)
                
                # Move to GPU if available
                if self.use_gpu:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            
            build_time = time.time() - start_time
            logger.info(f"Optimized FAISS index built in {build_time:.2f}s with {len(properties)} properties")
            
            return True
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            return False
    
    def _format_property_for_embedding(self, property_data: Dict[str, Any]) -> str:
        """Format property data for optimal embedding generation"""
        try:
            # Extract key information
            name = property_data.get('propertyName', property_data.get('PropertyName', 'Unknown Property'))
            property_type = property_data.get('typeName', property_data.get('PropertyType', 'Unknown'))
            city = property_data.get('city', property_data.get('City', ''))
            state = property_data.get('state', property_data.get('State', ''))
            price = property_data.get('marketValue', property_data.get('MarketValue', 0))
            description = property_data.get('description', property_data.get('Description', ''))
            
            # Extract features
            features = []
            
            # PG/Hostel features
            pg_details = property_data.get('pgPropertyDetails', {})
            if pg_details:
                if pg_details.get('wifiAvailable'):
                    features.append('WiFi available')
                if pg_details.get('isACAvailable'):
                    features.append('Air conditioning')
                if pg_details.get('isParkingAvailable'):
                    features.append('Parking available')
                if pg_details.get('powerBackup'):
                    features.append('Power backup')
            
            # Commercial features
            commercial_details = property_data.get('commercialPropertyDetails', {})
            if commercial_details:
                if commercial_details.get('wifiAvailable'):
                    features.append('WiFi available')
                if commercial_details.get('isACAvailable'):
                    features.append('Air conditioning')
                if commercial_details.get('hasParking'):
                    features.append('Parking available')
                if commercial_details.get('powerBackup'):
                    features.append('Power backup')
            
            # Format as structured text
            formatted_text = f"""
            Property: {name}
            Type: {property_type}
            Location: {city}, {state}
            Price: ₹{price:,.0f}
            Features: {', '.join(features) if features else 'Standard amenities'}
            Description: {description}
            """
            
            return formatted_text.strip()
            
        except Exception as e:
            logger.error(f"Error formatting property for embedding: {e}")
            return str(property_data)
    
    def semantic_search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """Perform optimized semantic search using FAISS"""
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = self._generate_cache_key(query, SearchType.SEMANTIC, top_k)
            cached_results = self._get_cached_results(cache_key)
            if cached_results:
                return cached_results
            
            # Get or generate query embedding
            query_embedding = self._get_cached_embedding(query)
            if query_embedding is None:
                query_embedding = self.model.encode([query], convert_to_tensor=False).astype('float32')
                self._cache_embedding(query, query_embedding)
            
            # Search in FAISS index with optimized parameters
            scores, indices = self.index.search(query_embedding, top_k)
            
            # Format results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.property_metadata):
                    result = SearchResult(
                        property_id=self.property_metadata[idx]['property_id'],
                        property_data={},  # Will be filled by caller
                        similarity_score=float(score),
                        search_type=SearchType.SEMANTIC,
                        ranking_factors={'semantic_score': float(score)},
                        metadata=self.property_metadata[idx]
                    )
                    results.append(result)
            
            # Cache results
            self._cache_results(cache_key, results)
            
            search_time = time.time() - start_time
            self._update_search_stats(search_time)
            
            logger.debug(f"Optimized semantic search completed in {search_time:.3f}s, found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def keyword_search(self, query: str, properties: List[Dict[str, Any]], 
                      top_k: int = 20) -> List[SearchResult]:
        """Perform optimized keyword-based search"""
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = self._generate_cache_key(query, SearchType.KEYWORD, top_k)
            cached_results = self._get_cached_results(cache_key)
            if cached_results:
                return cached_results
            
            # Extract keywords from query
            keywords = self._extract_keywords(query.lower())
            
            # Score properties in parallel
            def score_property(prop):
                score = self._calculate_keyword_score(prop, keywords)
                return prop, score
            
            # Process properties in parallel
            futures = []
            for prop in properties:
                future = self.thread_pool.submit(score_property, prop)
                futures.append(future)
            
            # Collect scored properties
            scored_properties = []
            for future in as_completed(futures):
                try:
                    prop, score = future.result()
                    if score > 0:
                        scored_properties.append((prop, score))
                except Exception as e:
                    logger.error(f"Error scoring property: {e}")
            
            # Sort by score and take top_k
            scored_properties.sort(key=lambda x: x[1], reverse=True)
            top_properties = scored_properties[:top_k]
            
            # Format results
            results = []
            for prop, score in top_properties:
                result = SearchResult(
                    property_id=prop.get('propertyId', prop.get('id', str(hash(str(prop))))),
                    property_data=prop,
                    similarity_score=score,
                    search_type=SearchType.KEYWORD,
                    ranking_factors={'keyword_score': score},
                    metadata={}
                )
                results.append(result)
            
            # Cache results
            self._cache_results(cache_key, results)
            
            search_time = time.time() - start_time
            self._update_search_stats(search_time)
            
            logger.debug(f"Optimized keyword search completed in {search_time:.3f}s, found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Extract words
        words = query.split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _calculate_keyword_score(self, property_data: Dict[str, Any], keywords: List[str]) -> float:
        """Calculate keyword match score for a property"""
        score = 0.0
        
        # Check property name
        name = property_data.get('propertyName', property_data.get('PropertyName', '')).lower()
        for keyword in keywords:
            if keyword in name:
                score += 2.0  # Higher weight for name matches
        
        # Check description
        description = property_data.get('description', property_data.get('Description', '')).lower()
        for keyword in keywords:
            if keyword in description:
                score += 1.0
        
        # Check location
        city = property_data.get('city', property_data.get('City', '')).lower()
        state = property_data.get('state', property_data.get('State', '')).lower()
        for keyword in keywords:
            if keyword in city or keyword in state:
                score += 1.5
        
        # Check property type
        property_type = property_data.get('typeName', property_data.get('PropertyType', '')).lower()
        for keyword in keywords:
            if keyword in property_type:
                score += 1.0
        
        return score
    
    def hybrid_search(self, query: str, properties: List[Dict[str, Any]], 
                     top_k: int = 5) -> List[SearchResult]:
        """Perform optimized hybrid search combining semantic and keyword search"""
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = self._generate_cache_key(query, SearchType.HYBRID, top_k)
            cached_results = self._get_cached_results(cache_key)
            if cached_results:
                return cached_results
            
            # Perform both searches in parallel
            semantic_future = self.thread_pool.submit(self.semantic_search, query, self.search_config['rerank_top_k'])
            keyword_future = self.thread_pool.submit(self.keyword_search, query, properties, self.search_config['rerank_top_k'])
            
            # Wait for both results
            semantic_results = semantic_future.result()
            keyword_results = keyword_future.result()
            
            # Combine results
            combined_results = self._combine_search_results(semantic_results, keyword_results)
            
            # Re-rank combined results
            reranked_results = self._rerank_results(query, combined_results, properties)
            
            # Take top_k final results
            final_results = reranked_results[:top_k]
            
            # Cache results
            self._cache_results(cache_key, final_results)
            
            search_time = time.time() - start_time
            self._update_search_stats(search_time)
            
            logger.info(f"Optimized hybrid search completed in {search_time:.3f}s, returning {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def _combine_search_results(self, semantic_results: List[SearchResult], 
                              keyword_results: List[SearchResult]) -> List[SearchResult]:
        """Combine semantic and keyword search results"""
        combined_dict = {}
        
        # Add semantic results
        for result in semantic_results:
            combined_dict[result.property_id] = {
                'semantic_score': result.similarity_score,
                'keyword_score': 0.0,
                'property_data': result.property_data,
                'metadata': result.metadata
            }
        
        # Add keyword results
        for result in keyword_results:
            if result.property_id in combined_dict:
                combined_dict[result.property_id]['keyword_score'] = result.similarity_score
            else:
                combined_dict[result.property_id] = {
                    'semantic_score': 0.0,
                    'keyword_score': result.similarity_score,
                    'property_data': result.property_data,
                    'metadata': result.metadata
                }
        
        # Calculate combined scores
        combined_results = []
        for prop_id, scores in combined_dict.items():
            combined_score = (
                self.search_config['semantic_weight'] * scores['semantic_score'] +
                self.search_config['keyword_weight'] * scores['keyword_score']
            )
            
            result = SearchResult(
                property_id=prop_id,
                property_data=scores['property_data'],
                similarity_score=combined_score,
                search_type=SearchType.HYBRID,
                ranking_factors={
                    'semantic_score': scores['semantic_score'],
                    'keyword_score': scores['keyword_score'],
                    'combined_score': combined_score
                },
                metadata=scores['metadata']
            )
            combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return combined_results
    
    def _rerank_results(self, query: str, results: List[SearchResult], 
                       properties: List[Dict[str, Any]]) -> List[SearchResult]:
        """Re-rank results based on multiple factors"""
        try:
            # Create property lookup
            property_lookup = {prop.get('propertyId', prop.get('id', str(hash(str(prop))))): prop 
                             for prop in properties}
            
            reranked_results = []
            
            for result in results:
                prop_data = property_lookup.get(result.property_id, {})
                
                # Calculate additional ranking factors
                price_factor = self._calculate_price_factor(prop_data, query)
                location_factor = self._calculate_location_factor(prop_data, query)
                feature_factor = self._calculate_feature_factor(prop_data, query)
                
                # Update ranking factors
                result.ranking_factors.update({
                    'price_factor': price_factor,
                    'location_factor': location_factor,
                    'feature_factor': feature_factor
                })
                
                # Calculate final score
                final_score = (
                    result.similarity_score * 0.6 +
                    price_factor * 0.15 +
                    location_factor * 0.15 +
                    feature_factor * 0.1
                )
                
                result.similarity_score = final_score
                result.property_data = prop_data
                reranked_results.append(result)
            
            # Sort by final score
            reranked_results.sort(key=lambda x: x.similarity_score, reverse=True)
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error in re-ranking: {e}")
            return results
    
    def _calculate_price_factor(self, property_data: Dict[str, Any], query: str) -> float:
        """Calculate price relevance factor"""
        try:
            price = float(property_data.get('marketValue', property_data.get('MarketValue', 0)))
            if price <= 0:
                return 0.5  # Neutral score for unknown prices
            
            # Extract price information from query
            query_lower = query.lower()
            
            # Check for price ranges
            if 'under' in query_lower or 'below' in query_lower:
                # Prefer lower prices
                return max(0.1, 1.0 - (price / 1000000))  # Normalize to 1M range
            elif 'over' in query_lower or 'above' in query_lower:
                # Prefer higher prices
                return min(1.0, price / 1000000)
            else:
                return 0.5  # Neutral
            
        except Exception as e:
            logger.error(f"Error calculating price factor: {e}")
            return 0.5
    
    def _calculate_location_factor(self, property_data: Dict[str, Any], query: str) -> float:
        """Calculate location relevance factor"""
        try:
            city = property_data.get('city', property_data.get('City', '')).lower()
            state = property_data.get('state', property_data.get('State', '')).lower()
            query_lower = query.lower()
            
            # Check for location matches
            if city in query_lower or state in query_lower:
                return 1.0
            elif any(word in query_lower for word in ['near', 'close', 'nearby']):
                return 0.8
            else:
                return 0.5  # Neutral
            
        except Exception as e:
            logger.error(f"Error calculating location factor: {e}")
            return 0.5
    
    def _calculate_feature_factor(self, property_data: Dict[str, Any], query: str) -> float:
        """Calculate feature relevance factor"""
        try:
            query_lower = query.lower()
            feature_score = 0.5  # Base score
            
            # Check for specific features
            features_to_check = {
                'wifi': ['wifi', 'internet', 'wireless'],
                'ac': ['ac', 'air conditioning', 'cooling'],
                'parking': ['parking', 'garage', 'car space'],
                'power': ['power backup', 'generator', 'ups']
            }
            
            for feature, keywords in features_to_check.items():
                if any(keyword in query_lower for keyword in keywords):
                    # Check if property has this feature
                    if self._property_has_feature(property_data, feature):
                        feature_score += 0.2
                    else:
                        feature_score -= 0.1
            
            return max(0.0, min(1.0, feature_score))
            
        except Exception as e:
            logger.error(f"Error calculating feature factor: {e}")
            return 0.5
    
    def _property_has_feature(self, property_data: Dict[str, Any], feature: str) -> bool:
        """Check if property has a specific feature"""
        try:
            if feature == 'wifi':
                pg_details = property_data.get('pgPropertyDetails', {})
                commercial_details = property_data.get('commercialPropertyDetails', {})
                return (pg_details.get('wifiAvailable', False) or 
                       commercial_details.get('wifiAvailable', False))
            
            elif feature == 'ac':
                pg_details = property_data.get('pgPropertyDetails', {})
                commercial_details = property_data.get('commercialPropertyDetails', {})
                return (pg_details.get('isACAvailable', False) or 
                       commercial_details.get('isACAvailable', False))
            
            elif feature == 'parking':
                pg_details = property_data.get('pgPropertyDetails', {})
                commercial_details = property_data.get('commercialPropertyDetails', {})
                return (pg_details.get('isParkingAvailable', False) or 
                       commercial_details.get('hasParking', False))
            
            elif feature == 'power':
                pg_details = property_data.get('pgPropertyDetails', {})
                commercial_details = property_data.get('commercialPropertyDetails', {})
                return (pg_details.get('powerBackup', False) or 
                       commercial_details.get('powerBackup', False))
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking property feature: {e}")
            return False
    
    def _update_search_stats(self, search_time: float):
        """Update search statistics with cache metrics"""
        self.search_stats['total_searches'] += 1
        self.search_stats['avg_search_time'] = (
            (self.search_stats['avg_search_time'] * (self.search_stats['total_searches'] - 1) + search_time) /
            self.search_stats['total_searches']
        )
        
        # Update cache hit rate
        total_requests = self.cache_hits + self.cache_misses
        if total_requests > 0:
            self.search_stats['cache_hit_rate'] = self.cache_hits / total_requests
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get comprehensive search performance statistics"""
        stats = self.search_stats.copy()
        stats.update({
            'index_size': len(self.property_embeddings) if self.property_embeddings else 0,
            'index_type': self.index_type,
            'use_gpu': self.use_gpu,
            'cache_size': len(self.search_cache),
            'max_cache_size': self.cache_size,
            'embedding_cache_size': len(self.embedding_cache),
            'max_embedding_cache_size': self.max_embedding_cache_size
        })
        return stats
    
    def clear_cache(self):
        """Clear all caches"""
        with self.lock:
            self.search_cache.clear()
            self.embedding_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
        logger.info("Vector search caches cleared")
    
    def shutdown(self):
        """Shutdown the vector search system"""
        try:
            self.thread_pool.shutdown(wait=True)
            logger.info("AdvancedVectorSearch shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down vector search: {e}")
