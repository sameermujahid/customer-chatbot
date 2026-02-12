import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from .advanced_vector_search import AdvancedVectorSearch, SearchResult, SearchType
from .query_processor import AdvancedQueryProcessor, ProcessedQuery, QueryIntent
from .feature_matcher import DynamicFeatureMatcher

logger = logging.getLogger(__name__)

class RetrievalStrategy(Enum):
    SEMANTIC_ONLY = "semantic_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"
    FILTERED_HYBRID = "filtered_hybrid"
    CONTEXTUAL = "contextual"

@dataclass
class RetrievalResult:
    property_id: str
    property_data: Dict[str, Any]
    similarity_score: float
    retrieval_strategy: RetrievalStrategy
    ranking_factors: Dict[str, float]
    query_analysis: ProcessedQuery
    metadata: Dict[str, Any]

class AdvancedRAGRetriever:
    """
    Advanced RAG retriever combining vector search, query processing, and intelligent ranking
    """
    
    def __init__(self, 
                 model_name: str = "jinaai/jina-embeddings-v3",
                 use_gpu: bool = True,
                 max_workers: int = 4,
                 enable_caching: bool = True,
                 cache_size: int = 50000):
        
        # Initialize components
        self.vector_search = AdvancedVectorSearch(
            model_name,
            use_gpu=use_gpu,
            enable_caching=enable_caching,
            cache_size=cache_size,
            max_workers=max_workers
        )
        self.query_processor = AdvancedQueryProcessor(model_name, use_gpu=use_gpu)
        self.feature_matcher = DynamicFeatureMatcher(load_saved=True)
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Properties cache
        self.properties_cache = []
        self.properties_lookup = {}
        
        # Performance and cache settings
        self.retrieval_stats = {
            'total_retrievals': 0,
            'avg_retrieval_time': 0.0,
            'strategy_distribution': {},
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("AdvancedRAGRetriever initialized")
    
    def build_index(self, properties: List[Dict[str, Any]]) -> bool:
        """Build search index from properties"""
        try:
            logger.info(f"Building advanced RAG index for {len(properties)} properties")
            start_time = time.time()
            
            # Cache properties
            self.properties_cache = properties
            self.properties_lookup = {
                prop.get('propertyId', prop.get('id', str(hash(str(prop))))): prop 
                for prop in properties
            }
            
            # Build vector search index
            success = self.vector_search.build_index(properties)
            
            if success:
                build_time = time.time() - start_time
                logger.info(f"Advanced RAG index built successfully in {build_time:.2f}s")
                return True
            else:
                logger.error("Failed to build vector search index")
                return False
                
        except Exception as e:
            logger.error(f"Error building advanced RAG index: {e}")
            return False
    
    def retrieve(self, 
                query: str, 
                top_k: int = 5,
                session_id: Optional[str] = None,
                conversation_history: Optional[List[Dict]] = None,
                user_context: Optional[Dict] = None,
                strategy: RetrievalStrategy = RetrievalStrategy.HYBRID) -> List[RetrievalResult]:
        """Retrieve properties using advanced RAG techniques"""
        
        start_time = time.time()
        
        try:
            # Step 1: Process query
            processed_query = self.query_processor.process_query(
                query, session_id, conversation_history, user_context
            )
            
            # Step 2: Choose retrieval strategy based on query intent and complexity
            if strategy == RetrievalStrategy.CONTEXTUAL:
                strategy = self._choose_strategy(processed_query)
            
            # Step 3: Perform retrieval based on strategy
            if strategy == RetrievalStrategy.SEMANTIC_ONLY:
                results = self._semantic_retrieval(processed_query, top_k)
            elif strategy == RetrievalStrategy.KEYWORD_ONLY:
                results = self._keyword_retrieval(processed_query, top_k)
            elif strategy == RetrievalStrategy.HYBRID:
                results = self._hybrid_retrieval(processed_query, top_k)
            elif strategy == RetrievalStrategy.FILTERED_HYBRID:
                results = self._filtered_hybrid_retrieval(processed_query, top_k)
            else:
                results = self._hybrid_retrieval(processed_query, top_k)
            
            # Step 4: Apply feature matching if needed
            if processed_query.entities.get('features'):
                results = self._apply_feature_matching(results, processed_query)
            
            # Step 5: Final ranking and filtering
            final_results = self._final_ranking(results, processed_query, top_k)
            
            # Step 6: Format results
            retrieval_results = []
            for i, result in enumerate(final_results):
                retrieval_result = RetrievalResult(
                    property_id=result.property_id,
                    property_data=result.property_data,
                    similarity_score=result.similarity_score,
                    retrieval_strategy=strategy,
                    ranking_factors=result.ranking_factors,
                    query_analysis=processed_query,
                    metadata={
                        'rank': i + 1,
                        'retrieval_time': time.time() - start_time,
                        'strategy_used': strategy.value
                    }
                )
                retrieval_results.append(retrieval_result)
            
            # Update statistics
            retrieval_time = time.time() - start_time
            self._update_retrieval_stats(retrieval_time, strategy)
            
            logger.info(f"Retrieved {len(retrieval_results)} properties using {strategy.value} strategy in {retrieval_time:.3f}s")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error in advanced RAG retrieval: {e}")
            return []
    
    def _choose_strategy(self, processed_query: ProcessedQuery) -> RetrievalStrategy:
        """Choose optimal retrieval strategy based on query analysis"""
        try:
            # Simple queries with clear intent -> semantic only
            if (processed_query.complexity.value == 'simple' and 
                processed_query.intent in [QueryIntent.SEARCH_PROPERTY, QueryIntent.GET_DETAILS]):
                return RetrievalStrategy.SEMANTIC_ONLY
            
            # Complex queries with multiple entities -> filtered hybrid
            if (processed_query.complexity.value == 'complex' or 
                len(processed_query.entities.get('locations', [])) > 0 or
                len(processed_query.entities.get('price_ranges', [])) > 0):
                return RetrievalStrategy.FILTERED_HYBRID
            
            # Queries with specific features -> hybrid with feature matching
            if processed_query.entities.get('features'):
                return RetrievalStrategy.HYBRID
            
            # Default to hybrid
            return RetrievalStrategy.HYBRID
            
        except Exception as e:
            logger.error(f"Error choosing strategy: {e}")
            return RetrievalStrategy.HYBRID
    
    def _semantic_retrieval(self, processed_query: ProcessedQuery, top_k: int) -> List[SearchResult]:
        """Perform semantic-only retrieval"""
        try:
            # Use processed query for semantic search
            results = self.vector_search.semantic_search(processed_query.processed_query, top_k * 2)
            
            # Add property data to results
            for result in results:
                result.property_data = self.properties_lookup.get(result.property_id, {})
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic retrieval: {e}")
            return []
    
    def _keyword_retrieval(self, processed_query: ProcessedQuery, top_k: int) -> List[SearchResult]:
        """Perform keyword-only retrieval"""
        try:
            # Use original query for keyword search
            results = self.vector_search.keyword_search(
                processed_query.original_query, 
                self.properties_cache, 
                top_k * 2
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in keyword retrieval: {e}")
            return []
    
    def _hybrid_retrieval(self, processed_query: ProcessedQuery, top_k: int) -> List[SearchResult]:
        """Perform hybrid retrieval"""
        try:
            # Use processed query for hybrid search
            results = self.vector_search.hybrid_search(
                processed_query.processed_query,
                self.properties_cache,
                top_k * 2
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            return []
    
    def _filtered_hybrid_retrieval(self, processed_query: ProcessedQuery, top_k: int) -> List[SearchResult]:
        """Perform filtered hybrid retrieval with pre-filtering"""
        try:
            # Step 1: Apply filters first
            filtered_properties = self._apply_filters(processed_query.filters)
            
            if not filtered_properties:
                logger.warning("No properties match the filters, falling back to hybrid search")
                return self._hybrid_retrieval(processed_query, top_k)
            
            # Step 2: Perform hybrid search on filtered properties
            results = self.vector_search.hybrid_search(
                processed_query.processed_query,
                filtered_properties,
                top_k * 2
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in filtered hybrid retrieval: {e}")
            return []
    
    def _apply_filters(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to properties"""
        try:
            filtered_properties = []
            
            for prop in self.properties_cache:
                if self._property_matches_filters(prop, filters):
                    filtered_properties.append(prop)
            
            logger.debug(f"Filtered {len(self.properties_cache)} properties to {len(filtered_properties)}")
            return filtered_properties
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return self.properties_cache
    
    def _property_matches_filters(self, property_data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if property matches all filters"""
        try:
            # Location filter
            if filters.get('location'):
                location = filters['location'].lower()
                prop_city = property_data.get('city', '').lower()
                prop_state = property_data.get('state', '').lower()
                
                if location not in prop_city and location not in prop_state:
                    return False
            
            # Price filters
            if filters.get('price_min') is not None:
                prop_price = float(property_data.get('marketValue', 0))
                if prop_price < filters['price_min']:
                    return False
            
            if filters.get('price_max') is not None:
                prop_price = float(property_data.get('marketValue', 0))
                if prop_price > filters['price_max']:
                    return False
            
            # Property type filter
            if filters.get('property_type'):
                prop_type = property_data.get('typeName', '').lower()
                filter_type = filters['property_type'].lower()
                
                if filter_type not in prop_type:
                    return False
            
            # BHK filter
            if filters.get('bhk'):
                prop_bhk = property_data.get('numberOfRooms', 0)
                if prop_bhk != filters['bhk']:
                    return False
            
            # Furnished filter
            if filters.get('furnished'):
                # This would need to be implemented based on your data structure
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking property filters: {e}")
            return True  # Default to include if error
    
    def _apply_feature_matching(self, results: List[SearchResult], 
                               processed_query: ProcessedQuery) -> List[SearchResult]:
        """Apply feature matching to results"""
        try:
            if not processed_query.entities.get('features'):
                return results
            
            # Use feature matcher to filter results
            feature_requirements = {}
            for feature in processed_query.entities['features']:
                feature_requirements[feature] = True
            
            filtered_results = []
            for result in results:
                if self.feature_matcher.check_property_features(result.property_data, feature_requirements):
                    filtered_results.append(result)
            
            logger.debug(f"Feature matching filtered {len(results)} results to {len(filtered_results)}")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in feature matching: {e}")
            return results
    
    def _final_ranking(self, results: List[SearchResult], 
                      processed_query: ProcessedQuery, 
                      top_k: int) -> List[SearchResult]:
        """Apply final ranking and selection"""
        try:
            if not results:
                return []
            
            # Apply additional ranking factors
            for result in results:
                # Query intent factor
                intent_factor = self._calculate_intent_factor(result, processed_query.intent)
                result.ranking_factors['intent_factor'] = intent_factor
                
                # Complexity factor
                complexity_factor = self._calculate_complexity_factor(result, processed_query.complexity)
                result.ranking_factors['complexity_factor'] = complexity_factor
                
                # Context factor
                context_factor = self._calculate_context_factor(result, processed_query.context)
                result.ranking_factors['context_factor'] = context_factor
                
                # Calculate final score
                final_score = (
                    result.similarity_score * 0.5 +
                    intent_factor * 0.2 +
                    complexity_factor * 0.15 +
                    context_factor * 0.15
                )
                
                result.similarity_score = final_score
            
            # Sort by final score
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Return top_k results
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in final ranking: {e}")
            return results[:top_k]
    
    def _calculate_intent_factor(self, result: SearchResult, intent: QueryIntent) -> float:
        """Calculate ranking factor based on query intent"""
        try:
            if intent == QueryIntent.SEARCH_PROPERTY:
                return 1.0  # Neutral
            elif intent == QueryIntent.GET_DETAILS:
                return 1.0  # Neutral
            elif intent == QueryIntent.COMPARE_PROPERTIES:
                return 0.8  # Slightly lower for comparison
            elif intent == QueryIntent.GET_RECOMMENDATIONS:
                return 1.2  # Higher for recommendations
            elif intent == QueryIntent.GET_NEARBY:
                return 1.1  # Higher for location-based queries
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error calculating intent factor: {e}")
            return 1.0
    
    def _calculate_complexity_factor(self, result: SearchResult, complexity) -> float:
        """Calculate ranking factor based on query complexity"""
        try:
            if complexity.value == 'simple':
                return 1.0  # Neutral for simple queries
            elif complexity.value == 'moderate':
                return 0.9  # Slightly lower for moderate complexity
            elif complexity.value == 'complex':
                return 0.8  # Lower for complex queries
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error calculating complexity factor: {e}")
            return 1.0
    
    def _calculate_context_factor(self, result: SearchResult, context: Dict[str, Any]) -> float:
        """Calculate ranking factor based on conversation context - DISABLED to prevent session sharing"""
        try:
            # DISABLED to prevent session sharing between different session IDs
            factor = 1.0  # Base factor - no context influence
            
            # REMOVED conversation history processing to prevent session sharing
            # if context.get('conversation_history'):
            #     recent_locations = []
            #     recent_types = []
            #     
            #     for msg in context['conversation_history'][-3:]:  # Last 3 messages
            #         if msg.get('type') == 'user':
            #             # Extract entities from recent messages
            #             # This is a simplified version
            #             pass
            #     
            #     # Boost score if property matches recent context
            #     if recent_locations or recent_types:
            #         factor += 0.1
            
            return factor
            
        except Exception as e:
            logger.error(f"Error calculating context factor: {e}")
            return 1.0
    
    def batch_retrieve(self, 
                      queries: List[str], 
                      top_k: int = 5,
                      session_ids: Optional[List[str]] = None,
                      conversation_histories: Optional[List[List[Dict]]] = None,
                      user_contexts: Optional[List[Dict]] = None) -> List[List[RetrievalResult]]:
        """Perform batch retrieval for multiple queries"""
        try:
            if session_ids is None:
                session_ids = [None] * len(queries)
            if conversation_histories is None:
                conversation_histories = [None] * len(queries)
            if user_contexts is None:
                user_contexts = [None] * len(queries)
            
            # Submit batch tasks
            futures = []
            for i, query in enumerate(queries):
                future = self.thread_pool.submit(
                    self.retrieve,
                    query,
                    top_k,
                    session_ids[i],
                    conversation_histories[i],
                    user_contexts[i]
                )
                futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in batch retrieval: {e}")
                    results.append([])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch retrieval: {e}")
            return [[] for _ in queries]
    
    def _update_retrieval_stats(self, retrieval_time: float, strategy: RetrievalStrategy):
        """Update retrieval statistics"""
        self.retrieval_stats['total_retrievals'] += 1
        self.retrieval_stats['strategy_distribution'][strategy.value] = \
            self.retrieval_stats['strategy_distribution'].get(strategy.value, 0) + 1
        
        # Update average retrieval time
        total_retrievals = self.retrieval_stats['total_retrievals']
        current_avg = self.retrieval_stats['avg_retrieval_time']
        self.retrieval_stats['avg_retrieval_time'] = (
            (current_avg * (total_retrievals - 1) + retrieval_time) / total_retrievals
        )
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        return {
            **self.retrieval_stats,
            'vector_search_stats': self.vector_search.get_search_stats(),
            'query_processing_stats': self.query_processor.get_processing_stats(),
            'properties_count': len(self.properties_cache)
        }
    
    def save_index(self, filepath: str):
        """Save the complete RAG index"""
        try:
            # Save vector search index
            vector_index_path = filepath.replace('.index', '_vector.index')
            self.vector_search.save_index(vector_index_path)
            
            # Save query processor context
            query_processor_path = filepath.replace('.index', '_query_processor.pkl')
            import pickle
            with open(query_processor_path, 'wb') as f:
                pickle.dump({
                    'conversation_contexts': dict(self.query_processor.conversation_contexts),
                    'processing_stats': self.query_processor.processing_stats
                }, f)
            
            logger.info(f"Advanced RAG index saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving advanced RAG index: {e}")
    
    def load_index(self, filepath: str):
        """Load the complete RAG index"""
        try:
            # Load vector search index
            vector_index_path = filepath.replace('.index', '_vector.index')
            self.vector_search.load_index(vector_index_path)
            
            # Load query processor context
            query_processor_path = filepath.replace('.index', '_query_processor.pkl')
            import pickle
            with open(query_processor_path, 'rb') as f:
                data = pickle.load(f)
                self.query_processor.conversation_contexts.update(data['conversation_contexts'])
                self.query_processor.processing_stats.update(data['processing_stats'])
            
            logger.info(f"Advanced RAG index loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading advanced RAG index: {e}")
    
    def clear_cache(self):
        """Clear all caches"""
        try:
            self.properties_cache.clear()
            self.properties_lookup.clear()
            
            # Clear query processor contexts
            self.query_processor.conversation_contexts.clear()
            
            logger.info("Advanced RAG caches cleared")
            
        except Exception as e:
            logger.error(f"Error clearing caches: {e}")
    
    def shutdown(self):
        """Shutdown the retriever"""
        try:
            self.thread_pool.shutdown(wait=True)
            logger.info("Advanced RAG retriever shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down retriever: {e}")
