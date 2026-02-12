import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from collections import OrderedDict
import hashlib

from .advanced_rag_retriever import AdvancedRAGRetriever, RetrievalResult, RetrievalStrategy
from .advanced_response_generator import AdvancedResponseGenerator, GeneratedResponse, ResponseStyle, ResponseType

logger = logging.getLogger(__name__)

class SystemMode(Enum):
    SEARCH_ONLY = "search_only"
    CHAT_ONLY = "chat_only"
    HYBRID = "hybrid"
    AUTO = "auto"

@dataclass
class RAGSystemResult:
    query: str
    retrieval_results: List[RetrievalResult]
    generated_response: Optional[GeneratedResponse]
    system_mode: SystemMode
    processing_time: float
    confidence: float
    metadata: Dict[str, Any]

class UltimateRAGSystem:
    """
    Ultimate RAG system combining advanced retrieval and response generation
    Optimized for high-performance, high-concurrency operation
    """
    
    def __init__(self, 
                 model_name: str = "jinaai/jina-embeddings-v3",
                 llm_model_path: str = None,
                 use_gpu: bool = True,
                 max_workers: int = 8,
                 system_mode: SystemMode = SystemMode.AUTO,
                 enable_caching: bool = True,
                 cache_size: int = 10000):
        
        self.system_mode = system_mode
        self.use_gpu = use_gpu
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        
        # Initialize components with optimizations
        self.retriever = AdvancedRAGRetriever(
            model_name, 
            use_gpu, 
            max_workers,
            enable_caching=enable_caching,
            cache_size=cache_size
        )
        self.response_generator = AdvancedResponseGenerator(
            llm_model_path, 
            use_gpu,
            enable_caching=enable_caching,
            cache_size=cache_size,
            max_workers=max_workers
        )
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Intelligent caching system
        self.system_cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Performance tracking
        self.system_stats = {
            'total_queries': 0,
            'avg_processing_time': 0.0,
            'mode_distribution': {},
            'success_rate': 0.0,
            'avg_confidence': 0.0,
            'cache_hit_rate': 0.0
        }
        
        logger.info(f"Ultimate RAG System initialized with optimizations (GPU: {use_gpu}, Cache: {enable_caching})")
    
    def _generate_cache_key(self, query: str, session_id: Optional[str], 
                           top_k: int, generate_response: bool) -> str:
        """Generate cache key for system results"""
        try:
            key_data = {
                'query': query.lower().strip(),
                'session_id': session_id or 'anonymous',
                'top_k': top_k,
                'generate_response': generate_response,
                'system_mode': self.system_mode.value
            }
            key_string = json.dumps(key_data, sort_keys=True)
            return hashlib.md5(key_string.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return None
    
    def _get_cached_result(self, cache_key: str) -> Optional[RAGSystemResult]:
        """Get cached system result if available"""
        if not self.enable_caching or not cache_key:
            return None
        
        with self.lock:
            if cache_key in self.system_cache:
                # Move to end (LRU)
                result = self.system_cache.pop(cache_key)
                self.system_cache[cache_key] = result
                self.cache_hits += 1
                return result
        
        self.cache_misses += 1
        return None
    
    def _cache_result(self, cache_key: str, result: RAGSystemResult):
        """Cache system result with LRU eviction"""
        if not self.enable_caching or not cache_key:
            return
        
        with self.lock:
            # Remove if already exists
            if cache_key in self.system_cache:
                self.system_cache.pop(cache_key)
            
            # Add to cache
            self.system_cache[cache_key] = result
            
            # Evict oldest if cache is full
            if len(self.system_cache) > self.cache_size:
                self.system_cache.popitem(last=False)
    
    def process_query(self,
                     query: str,
                     session_id: Optional[str] = None,
                     conversation_history: Optional[List[Dict]] = None,
                     user_context: Optional[Dict] = None,
                     top_k: int = 5,
                     generate_response: bool = True) -> RAGSystemResult:
        """Process a query through the complete RAG system with AI conversation memory"""
        
        start_time = time.time()
        
        try:
            # Initialize AI conversation memory
            from ..ai_conversation_memory import get_ai_conversation_memory
            ai_memory = get_ai_conversation_memory()
            
            # Check if this is a follow-up query using AI understanding
            is_follow_up, context_relevance = ai_memory.is_follow_up_query(session_id, query)
            
            # Enhance query with conversation context if it's a follow-up
            enhanced_query = query
            if is_follow_up and context_relevance > 0.5:
                enhanced_query = ai_memory.get_contextual_query(session_id, query)
                logger.info(f"Enhanced query with AI context: {enhanced_query}")
            
            # Check cache with enhanced query
            cache_key = self._generate_cache_key(enhanced_query, session_id, top_k, generate_response)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                # Update metadata with cache info
                cached_result.metadata['cached'] = True
                cached_result.metadata['is_follow_up'] = is_follow_up
                cached_result.metadata['context_relevance'] = context_relevance
                cached_result.processing_time = time.time() - start_time
                return cached_result
            
            # Determine system mode
            mode = self._determine_system_mode(enhanced_query, generate_response)
            
            # Get conversation history for context
            conversation_history = ai_memory.get_conversation_history(session_id, max_turns=3)
            
            # Step 1: Retrieve relevant properties with AI context
            retrieval_results = self.retriever.retrieve(
                query=enhanced_query,
                top_k=top_k,
                session_id=session_id,
                conversation_history=conversation_history,
                user_context=user_context,
                strategy=RetrievalStrategy.CONTEXTUAL
            )
            
            # Step 2: Generate response if requested (parallel with retrieval)
            generated_response = None
            if generate_response and mode != SystemMode.SEARCH_ONLY:
                # Generate response with conversation context
                generated_response = self.response_generator.generate_response(
                    query=enhanced_query,
                    retrieval_results=retrieval_results,
                    user_context=user_context,
                    conversation_history=conversation_history
                )
            
            # Step 3: Calculate overall confidence
            confidence = self._calculate_overall_confidence(retrieval_results, generated_response)
            
            # Step 4: Create result
            result = RAGSystemResult(
                query=query,
                retrieval_results=retrieval_results,
                generated_response=generated_response,
                system_mode=mode,
                processing_time=time.time() - start_time,
                confidence=confidence,
                metadata={
                    'session_id': session_id,
                    'top_k': top_k,
                    'is_follow_up': is_follow_up,
                    'context_relevance': context_relevance,
                    'enhanced_query': enhanced_query,
                    'generate_response': generate_response,
                    'cached': False
                }
            )
            
            # Cache the result
            self._cache_result(cache_key, result)
            
            # Step 5: Update statistics
            self._update_system_stats(result.processing_time, mode, confidence, len(retrieval_results) > 0)
            
            logger.info(f"Processed query in {result.processing_time:.3f}s using {mode.value} mode (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return self._generate_error_result(query, e, start_time)
    
    def _determine_system_mode(self, query: str, generate_response: bool) -> SystemMode:
        """Determine the appropriate system mode"""
        try:
            if self.system_mode != SystemMode.AUTO:
                return self.system_mode
            
            query_lower = query.lower()
            
            # Check for search-only indicators
            if any(word in query_lower for word in ['list', 'show', 'find', 'search', 'properties']):
                return SystemMode.SEARCH_ONLY
            
            # Check for chat-only indicators
            if any(word in query_lower for word in ['hello', 'hi', 'help', 'what can you do', 'explain']):
                return SystemMode.CHAT_ONLY
            
            # Default to hybrid
            return SystemMode.HYBRID
            
        except Exception as e:
            logger.error(f"Error determining system mode: {e}")
            return SystemMode.HYBRID
    
    def _calculate_overall_confidence(self, 
                                    retrieval_results: List[RetrievalResult],
                                    generated_response: Optional[GeneratedResponse]) -> float:
        """Calculate overall system confidence"""
        try:
            confidence = 0.5  # Base confidence
            
            # Retrieval confidence
            if retrieval_results:
                avg_retrieval_score = sum(r.similarity_score for r in retrieval_results) / len(retrieval_results)
                confidence += avg_retrieval_score * 0.3
            
            # Generation confidence
            if generated_response:
                confidence += generated_response.confidence * 0.4
            
            # Results quality factor
            if len(retrieval_results) > 0:
                confidence += 0.1
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating overall confidence: {e}")
            return 0.5
    
    def _generate_error_result(self, query: str, error: Exception, start_time: float) -> RAGSystemResult:
        """Generate error result when processing fails"""
        processing_time = time.time() - start_time
        
        return RAGSystemResult(
            query=query,
            retrieval_results=[],
            generated_response=None,
            system_mode=SystemMode.HYBRID,
            processing_time=processing_time,
            confidence=0.0,
            metadata={'error': str(error)}
        )
    
    def batch_process(self,
                     queries: List[str],
                     session_ids: Optional[List[str]] = None,
                     user_contexts: Optional[List[Dict]] = None,
                     top_k: int = 5,
                     generate_responses: bool = True) -> List[RAGSystemResult]:
        """Process multiple queries in parallel for high throughput"""
        try:
            if session_ids is None:
                session_ids = [None] * len(queries)
            if user_contexts is None:
                user_contexts = [None] * len(queries)
            
            # Submit all processing tasks
            futures = []
            for i, query in enumerate(queries):
                future = self.thread_pool.submit(
                    self.process_query,
                    query,
                    session_ids[i],
                    None,  # No conversation history
                    user_contexts[i],
                    top_k,
                    generate_responses
                )
                futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=60)  # 60 second timeout
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in batch processing: {e}")
                    # Create error result
                    error_result = RAGSystemResult(
                        query="",
                        retrieval_results=[],
                        generated_response=None,
                        system_mode=SystemMode.HYBRID,
                        processing_time=0.0,
                        confidence=0.0,
                        metadata={'error': str(e)}
                    )
                    results.append(error_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return []
    
    def search_only(self,
                   query: str,
                   session_id: Optional[str] = None,
                   conversation_history: Optional[List[Dict]] = None,
                   user_context: Optional[Dict] = None,
                   top_k: int = 5) -> List[RetrievalResult]:
        """Perform search-only operation"""
        try:
            result = self.process_query(
                query=query,
                session_id=session_id,
                conversation_history=conversation_history,
                user_context=user_context,
                top_k=top_k,
                generate_response=False
            )
            
            return result.retrieval_results
            
        except Exception as e:
            logger.error(f"Error in search-only operation: {e}")
            return []
    
    def chat_only(self,
                 query: str,
                 session_id: Optional[str] = None,
                 conversation_history: Optional[List[Dict]] = None,
                 user_context: Optional[Dict] = None) -> GeneratedResponse:
        """Perform chat-only operation"""
        try:
            # For chat-only, we might not need retrieval results
            # Create empty retrieval results for the response generator
            empty_results = []
            
            response = self.response_generator.generate_response(
                query=query,
                retrieval_results=empty_results,
                user_context=user_context,
                conversation_history=conversation_history
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat-only operation: {e}")
            return self.response_generator._generate_fallback_response(query, e)
    
    def hybrid_search(self,
                     query: str,
                     session_id: Optional[str] = None,
                     conversation_history: Optional[List[Dict]] = None,
                     user_context: Optional[Dict] = None,
                     top_k: int = 5) -> Tuple[List[RetrievalResult], GeneratedResponse]:
        """Perform hybrid search with both retrieval and response generation"""
        try:
            result = self.process_query(
                query=query,
                session_id=session_id,
                conversation_history=conversation_history,
                user_context=user_context,
                top_k=top_k,
                generate_response=True
            )
            
            return result.retrieval_results, result.generated_response
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return [], self.response_generator._generate_fallback_response(query, e)
    
    def get_similar_properties(self,
                             property_id: str,
                             top_k: int = 5,
                             session_id: Optional[str] = None) -> List[RetrievalResult]:
        """Find properties similar to a given property"""
        try:
            # Get property details
            property_data = self.retriever.properties_lookup.get(property_id)
            if not property_data:
                logger.warning(f"Property {property_id} not found")
                return []
            
            # Create a query based on property features
            query = self._create_similarity_query(property_data)
            
            # Search for similar properties
            results = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                session_id=session_id
            )
            
            # Filter out the original property
            filtered_results = [r for r in results if r.property_id != property_id]
            
            return filtered_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar properties: {e}")
            return []
    
    def _create_similarity_query(self, property_data: Dict[str, Any]) -> str:
        """Create a query for finding similar properties"""
        try:
            query_parts = []
            
            # Add property type
            property_type = property_data.get('typeName', property_data.get('PropertyType', ''))
            if property_type:
                query_parts.append(property_type)
            
            # Add location
            city = property_data.get('city', property_data.get('City', ''))
            if city:
                query_parts.append(city)
            
            # Add features
            features = []
            
            # PG/Hostel features
            pg_details = property_data.get('pgPropertyDetails', {})
            if pg_details:
                if pg_details.get('wifiAvailable'):
                    features.append('wifi')
                if pg_details.get('isACAvailable'):
                    features.append('air conditioning')
                if pg_details.get('isParkingAvailable'):
                    features.append('parking')
                if pg_details.get('powerBackup'):
                    features.append('power backup')
            
            # Commercial features
            commercial_details = property_data.get('commercialPropertyDetails', {})
            if commercial_details:
                if commercial_details.get('wifiAvailable'):
                    features.append('wifi')
                if commercial_details.get('isACAvailable'):
                    features.append('air conditioning')
                if commercial_details.get('hasParking'):
                    features.append('parking')
                if commercial_details.get('powerBackup'):
                    features.append('power backup')
            
            if features:
                query_parts.extend(features)
            
            return " ".join(query_parts)
            
        except Exception as e:
            logger.error(f"Error creating similarity query: {e}")
            return "similar property"
    
    def get_recommendations(self,
                          user_context: Dict,
                          top_k: int = 5,
                          session_id: Optional[str] = None) -> List[RetrievalResult]:
        """Get personalized property recommendations"""
        try:
            # Create recommendation query based on user context
            query = self._create_recommendation_query(user_context)
            
            # Search for recommendations
            results = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                session_id=session_id,
                user_context=user_context
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def _create_recommendation_query(self, user_context: Dict) -> str:
        """Create a query for recommendations based on user context"""
        try:
            query_parts = []
            
            if user_context.get('preferred_location'):
                query_parts.append(user_context['preferred_location'])
            
            if user_context.get('preferred_property_type'):
                query_parts.append(user_context['preferred_property_type'])
            
            if user_context.get('preferred_features'):
                query_parts.extend(user_context['preferred_features'])
            
            if user_context.get('budget_range'):
                query_parts.append(user_context['budget_range'])
            
            if not query_parts:
                query_parts.append('recommended properties')
            
            return " ".join(query_parts)
            
        except Exception as e:
            logger.error(f"Error creating recommendation query: {e}")
            return "recommended properties"
    
    def _update_system_stats(self, processing_time: float, mode: SystemMode, confidence: float, success: bool):
        """Update system statistics with cache metrics"""
        self.system_stats['total_queries'] += 1
        self.system_stats['mode_distribution'][mode.value] = \
            self.system_stats['mode_distribution'].get(mode.value, 0) + 1
        
        # Update average processing time
        total_queries = self.system_stats['total_queries']
        current_avg = self.system_stats['avg_processing_time']
        self.system_stats['avg_processing_time'] = (
            (current_avg * (total_queries - 1) + processing_time) / total_queries
        )
        
        # Update success rate
        current_success_rate = self.system_stats['success_rate']
        self.system_stats['success_rate'] = (
            (current_success_rate * (total_queries - 1) + (1 if success else 0)) / total_queries
        )
        
        # Update average confidence
        current_avg_conf = self.system_stats['avg_confidence']
        self.system_stats['avg_confidence'] = (
            (current_avg_conf * (total_queries - 1) + confidence) / total_queries
        )
        
        # Update cache hit rate
        total_requests = self.cache_hits + self.cache_misses
        if total_requests > 0:
            self.system_stats['cache_hit_rate'] = self.cache_hits / total_requests
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = self.system_stats.copy()
        stats.update({
            'retriever_stats': self.retriever.get_retrieval_stats(),
            'generator_stats': self.response_generator.get_generation_stats(),
            'properties_count': len(self.retriever.properties_cache),
            'cache_size': len(self.system_cache),
            'max_cache_size': self.cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses
        })
        return stats
    
    def build_index(self, properties: List[Dict[str, Any]]) -> bool:
        """Build the complete RAG index"""
        try:
            logger.info(f"Building ultimate RAG system index for {len(properties)} properties")
            start_time = time.time()
            
            # Build retriever index
            success = self.retriever.build_index(properties)
            
            if success:
                build_time = time.time() - start_time
                logger.info(f"Ultimate RAG system index built successfully in {build_time:.2f}s")
                return True
            else:
                logger.error("Failed to build retriever index")
                return False
                
        except Exception as e:
            logger.error(f"Error building ultimate RAG system index: {e}")
            return False
    
    def save_system(self, filepath: str):
        """Save the complete RAG system"""
        try:
            # Save retriever
            retriever_path = filepath.replace('.system', '_retriever.index')
            self.retriever.save_index(retriever_path)
            
            # Save system configuration
            config_path = filepath.replace('.system', '_config.json')
            config = {
                'system_mode': self.system_mode.value,
                'use_gpu': self.use_gpu,
                'system_stats': self.system_stats
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Ultimate RAG system saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving ultimate RAG system: {e}")
    
    def load_system(self, filepath: str):
        """Load the complete RAG system"""
        try:
            # Load retriever
            retriever_path = filepath.replace('.system', '_retriever.index')
            self.retriever.load_index(retriever_path)
            
            # Load system configuration
            config_path = filepath.replace('.system', '_config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update system configuration
            self.system_mode = SystemMode(config.get('system_mode', 'auto'))
            self.system_stats.update(config.get('system_stats', {}))
            
            logger.info(f"Ultimate RAG system loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading ultimate RAG system: {e}")
    
    def clear_cache(self):
        """Clear all system caches"""
        try:
            with self.lock:
                self.system_cache.clear()
                self.cache_hits = 0
                self.cache_misses = 0
            
            # Clear component caches
            self.retriever.clear_cache()
            self.response_generator.clear_cache()
            
            logger.info("Ultimate RAG system caches cleared")
            
        except Exception as e:
            logger.error(f"Error clearing system caches: {e}")
    
    def shutdown(self):
        """Shutdown the complete RAG system"""
        try:
            self.retriever.shutdown()
            self.response_generator.shutdown()
            self.thread_pool.shutdown(wait=True)
            logger.info("Ultimate RAG system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down RAG system: {e}")

    def get_retriever(self):
        """Get the underlying retriever component for backward compatibility"""
        return self.retriever
    
    def get_response_generator(self):
        """Get the underlying response generator component"""
        return self.response_generator
    
    def get_system_mode(self) -> SystemMode:
        """Get current system mode"""
        return self.system_mode
    
    def set_system_mode(self, mode: SystemMode):
        """Set system mode"""
        self.system_mode = mode
        logger.info(f"System mode changed to {mode.value}")

# Global ultimate RAG system instance
_ultimate_rag_system = None

def get_ultimate_rag_system() -> UltimateRAGSystem:
    """Get the global ultimate RAG system instance"""
    global _ultimate_rag_system
    if _ultimate_rag_system is None:
        _ultimate_rag_system = UltimateRAGSystem()
    return _ultimate_rag_system
