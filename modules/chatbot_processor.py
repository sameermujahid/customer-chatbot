import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import OrderedDict
import hashlib
import json

from .nlp_processor import NLPProcessor
from .location_processor import LocationProcessor
from .property_processor import PropertyProcessor
from .models import fetch_and_cache_properties, OptimizedRAGRetriever, get_global_retriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotProcessor:
    """
    Advanced chatbot processor with optimized performance for high-concurrency
    """
    
    def __init__(self, 
                 model_name: str = "jinaai/jina-embeddings-v3",
                 use_gpu: bool = True,
                 max_workers: int = 8,
                 enable_caching: bool = True,
                 cache_size: int = 10000):
        
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        
        # Initialize embedding model with optimizations
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        if self.use_gpu:
            self.model = self.model.to('cuda')
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Intelligent caching system
        self.query_cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Performance tracking
        self.processing_stats = {
            'total_queries': 0,
            'avg_processing_time': 0.0,
            'cache_hit_rate': 0.0
        }
        
        logger.info("ChatbotProcessor initialized with optimizations")
    
    def __init__(self):
        self.nlp_processor = NLPProcessor()
        self.location_processor = LocationProcessor()
        self.property_processor = PropertyProcessor()
        logger.info("Initialized ChatbotProcessor")
        
    def _generate_cache_key(self, message: str, session_id: str) -> str:
        """Generate cache key for query results"""
        try:
            key_data = {
                'message': message.lower().strip(),
                'session_id': session_id
            }
            key_string = json.dumps(key_data, sort_keys=True)
            return hashlib.md5(key_string.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return None
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get cached result if available"""
        if not self.enable_caching or not cache_key:
            return None
        
        with self.lock:
            if cache_key in self.query_cache:
                # Move to end (LRU)
                result = self.query_cache.pop(cache_key)
                self.query_cache[cache_key] = result
                self.cache_hits += 1
                return result
        
        self.cache_misses += 1
        return None
    
    def _cache_result(self, cache_key: str, result: Dict):
        """Cache result with LRU eviction"""
        if not self.enable_caching or not cache_key:
            return
        
        with self.lock:
            # Remove if already exists
            if cache_key in self.query_cache:
                self.query_cache.pop(cache_key)
            
            # Add to cache
            self.query_cache[cache_key] = result
            
            # Evict oldest if cache is full
            if len(self.query_cache) > self.cache_size:
                self.query_cache.popitem(last=False)
        
    def process_query(self, 
                     query: str, 
                     user_location: Optional[Tuple[float, float]] = None,
                     retriever: Optional[Any] = None) -> List[Dict]:
        """Process query with optimizations for high performance"""
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, "default")
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                cached_result['processing_time'] = time.time() - start_time
                cached_result['cached'] = True
                return cached_result['properties']
            
            # Use optimized retriever if available
            if retriever:
                # Use the optimized retriever directly
                results = retriever.retrieve(query, top_k=5)
                
                # Format results for response
                properties = []
                for result in results:
                    property_data = result.property_data.copy()
                    property_data['similarity_score'] = result.similarity_score
                    property_data['search_type'] = result.search_type.value
                    properties.append(property_data)
                
                # Cache the result
                result_data = {
                    'properties': properties,
                    'processing_time': time.time() - start_time,
                    'cached': False
                }
                self._cache_result(cache_key, result_data)
                
                # Update statistics
                self._update_processing_stats(time.time() - start_time)
                
                return properties
            
            # Fallback processing
            logger.warning("No retriever available, using fallback processing")
            return []
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return []
    
    def _update_processing_stats(self, processing_time: float):
        """Update processing statistics"""
        self.processing_stats['total_queries'] += 1
        
        # Update average processing time
        total_queries = self.processing_stats['total_queries']
        current_avg = self.processing_stats['avg_processing_time']
        self.processing_stats['avg_processing_time'] = (
            (current_avg * (total_queries - 1) + processing_time) / total_queries
        )
        
        # Update cache hit rate
        total_requests = self.cache_hits + self.cache_misses
        if total_requests > 0:
            self.processing_stats['cache_hit_rate'] = self.cache_hits / total_requests
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.processing_stats.copy()
        stats.update({
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_size': len(self.query_cache),
            'max_cache_size': self.cache_size
        })
        return stats
    
    def clear_cache(self):
        """Clear query cache"""
        with self.lock:
            self.query_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
        logger.info("Chatbot processor cache cleared")
    
    def shutdown(self):
        """Shutdown the chatbot processor"""
        try:
            self.thread_pool.shutdown(wait=True)
            logger.info("ChatbotProcessor shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down chatbot processor: {e}")
            
    def get_similar_properties(self,
                             reference_property: Dict,
                             properties: List[Dict],
                             top_k: int = 5) -> List[Dict]:
        """Find properties similar to reference property"""
        try:
            logger.info(f"Finding similar properties to: {reference_property.get('PropertyName', 'Unknown')}")
            
            # Process properties
            processed_properties = [
                self.property_processor.process_property_data(p)
                for p in properties
            ]
            
            # Find similar properties
            similar_properties = self.property_processor.find_similar_properties(
                reference_property, processed_properties, top_k
            )
            
            # Format results
            formatted_results = []
            for property_data, similarity in similar_properties:
                formatted_property = {
                    'details': self.property_processor.format_property_details(property_data),
                    'data': property_data,
                    'similarity': similarity
                }
                formatted_results.append(formatted_property)
                
            logger.info(f"Found {len(formatted_results)} similar properties")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error finding similar properties: {str(e)}")
            return []
            
    def get_nearby_landmarks(self,
                           property_data: Dict,
                           radius_miles: float = 5.0) -> List[Dict]:
        """Get landmarks near a property"""
        try:
            logger.info(f"Finding landmarks near property: {property_data.get('PropertyName', 'Unknown')}")
            
            # Get property coordinates
            latitude = float(property_data.get('Latitude', 0))
            longitude = float(property_data.get('Longitude', 0))
            
            if latitude and longitude:
                landmarks = self.location_processor.find_nearby_landmarks(
                    latitude, longitude, radius_miles
                )
                logger.info(f"Found {len(landmarks)} nearby landmarks")
                return landmarks
            return []
            
        except Exception as e:
            logger.error(f"Error finding nearby landmarks: {str(e)}")
            return []
            
    def get_location_details(self, property_data: Dict) -> Dict:
        """Get detailed location information for a property"""
        try:
            logger.info(f"Getting location details for property: {property_data.get('PropertyName', 'Unknown')}")
            
            # Get property coordinates
            latitude = float(property_data.get('Latitude', 0))
            longitude = float(property_data.get('Longitude', 0))
            
            if latitude and longitude:
                details = self.location_processor.get_location_details(
                    latitude, longitude
                )
                logger.info(f"Location details: {details}")
                return details
            return {}
            
        except Exception as e:
            logger.error(f"Error getting location details: {str(e)}")
            return {}

    def find_similar_properties(self, query, top_k=5):
        """Find similar properties using the retriever"""
        try:
            # Get global retriever if not provided
            retriever, vector_db = get_global_retriever()
            
            if retriever is None:
                logger.error("No retriever available")
                return []
            
            # Get properties from retriever
            results = retriever.retrieve(query, top_k=top_k)
            
            # Ensure we have exactly 5 properties
            while len(results) < 5:
                # Add remaining properties with high distance scores
                remaining_idx = len(results)
                properties = fetch_and_cache_properties()
                if remaining_idx < len(properties):
                    property_data = properties[remaining_idx]
                    formatted_property = self.property_processor.format_property_details(property_data)
                    if formatted_property:
                        results.append({
                            "property": formatted_property,
                            "distance": 1.0  # High distance score for additional properties
                        })
            
            return results[:5]  # Return exactly 5 properties
        except Exception as e:
            logger.error(f"Error finding similar properties: {str(e)}")
            return [] 

def process_chat_message(message: str, session_id: str = None, conversation_history: List[Dict] = None) -> Dict:
    """
    Process a chat message using the ultimate RAG system with AI conversation memory
    OPTIMIZED: High-performance processing with caching and parallel execution
    """
    start_time = time.time()
    
    try:
        # Get the optimized RAG system
        from .rag.ultimate_rag_system import get_ultimate_rag_system
        rag_system = get_ultimate_rag_system()
        
        # Get AI conversation memory for context
        from .ai_conversation_memory import get_ai_conversation_memory
        ai_memory = get_ai_conversation_memory()
        
        # Get conversation history from AI memory
        if session_id and ai_memory:
            conversation_history = ai_memory.get_conversation_history(session_id, max_turns=3)
        
        # Process the query through the complete RAG system with AI context
        result = rag_system.process_query(
            query=message,
            session_id=session_id,
            conversation_history=conversation_history,
            top_k=5,
            generate_response=True
        )
        
        # Format response for API
        response_data = {
            'response': result.generated_response.response_text if result.generated_response else "I'm here to help with your real estate search.",
            'properties': [r.property_data for r in result.retrieval_results],
            'confidence': result.confidence,
            'processing_time': result.processing_time,
            'system_mode': result.system_mode.value,
            'follow_up_suggestions': result.generated_response.follow_up_suggestions if result.generated_response else [],
            'session_id': session_id,
            'cached': result.metadata.get('cached', False),
            'is_follow_up': result.metadata.get('is_follow_up', False),
            'context_relevance': result.metadata.get('context_relevance', 0.0)
        }
        
        # Add conversation turn to AI memory
        if session_id and ai_memory and result.generated_response:
            ai_memory.add_conversation_turn(
                session_id=session_id,
                user_query=message,
                assistant_response=result.generated_response.response_text,
                retrieved_properties=[r.property_data for r in result.retrieval_results]
            )
        
        total_time = time.time() - start_time
        logger.info(f"Chat message processed in {total_time:.3f}s (cached: {response_data['cached']})")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        return {
            'response': "I apologize, but I'm having trouble processing your request. Please try again.",
            'properties': [],
            'confidence': 0.0,
            'processing_time': time.time() - start_time,
            'system_mode': 'error',
            'follow_up_suggestions': ["Please try rephrasing your question"],
            'session_id': session_id,
            'cached': False,
            'error': str(e)
        } 