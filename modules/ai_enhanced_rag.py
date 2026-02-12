"""
AI-Enhanced RAG System with Post-Processing

This module integrates the AI post-processor with the existing RAG system
to provide precise, intent-matching property search results.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ai_postprocessor import AIPropertyPostProcessor
from rag.advanced_rag_retriever import AdvancedRAGRetriever, RetrievalStrategy
from rag.query_processor import AdvancedQueryProcessor, ProcessedQuery

logger = logging.getLogger(__name__)

@dataclass
class EnhancedRAGResult:
    """Result from AI-enhanced RAG system"""
    query: str
    original_results: List[Dict[str, Any]]
    filtered_results: List[Dict[str, Any]]
    processing_time: float
    filter_stats: Dict[str, Any]
    confidence: float
    metadata: Dict[str, Any]

class AIEnhancedRAGSystem:
    """
    AI-enhanced RAG system that combines advanced retrieval with semantic post-processing.
    
    This system ensures that retrieved properties exactly match user query intent
    through semantic filtering and precision-focused matching.
    """
    
    def __init__(self, 
                 retriever: AdvancedRAGRetriever,
                 postprocessor: AIPropertyPostProcessor,
                 query_processor: AdvancedQueryProcessor):
        """
        Initialize the AI-enhanced RAG system.
        
        Args:
            retriever: Advanced RAG retriever for initial property retrieval
            postprocessor: AI post-processor for semantic filtering
            query_processor: Query processor for intent analysis
        """
        self.retriever = retriever
        self.postprocessor = postprocessor
        self.query_processor = query_processor
        
        logger.info("AI-Enhanced RAG System initialized")
    
    def search_properties(self, 
                         query: str,
                         top_k: int = 5,
                         session_id: Optional[str] = None,
                         conversation_history: Optional[List[Dict]] = None,
                         user_context: Optional[Dict] = None,
                         retrieval_strategy: RetrievalStrategy = RetrievalStrategy.CONTEXTUAL) -> EnhancedRAGResult:
        """
        Search for properties with AI-enhanced precision.
        
        Args:
            query: User search query
            top_k: Maximum number of results to return
            session_id: Session identifier for context
            conversation_history: Previous conversation context
            user_context: User-specific context
            retrieval_strategy: Strategy for initial retrieval
            
        Returns:
            EnhancedRAGResult with filtered and validated results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: '{query}' with AI-enhanced RAG")
            
            # Step 1: Process query for intent analysis
            processed_query = self.query_processor.process_query(
                query, session_id, conversation_history, user_context
            )
            
            # Step 2: Retrieve initial candidates (more than needed for filtering)
            retrieval_top_k = min(top_k * 3, 20)  # Get 3x more for filtering
            retrieval_results = self.retriever.retrieve(
                query=processed_query.processed_query,
                top_k=retrieval_top_k,
                session_id=session_id,
                conversation_history=conversation_history,
                user_context=user_context,
                strategy=retrieval_strategy
            )
            
            # Extract property data from retrieval results
            original_properties = []
            for result in retrieval_results:
                if hasattr(result, 'property_data'):
                    original_properties.append(result.property_data)
                elif isinstance(result, dict) and 'property' in result:
                    original_properties.append(result['property'])
                else:
                    original_properties.append(result)
            
            logger.info(f"Retrieved {len(original_properties)} candidate properties")
            
            # Step 3: Apply AI post-processing for semantic filtering
            filtered_properties = self.postprocessor.process_results(
                query, original_properties, top_k
            )
            
            # Step 4: Calculate overall confidence
            confidence = self._calculate_overall_confidence(
                processed_query, filtered_properties, original_properties
            )
            
            # Step 5: Get filter statistics
            filter_stats = self.postprocessor.get_stats()
            
            processing_time = time.time() - start_time
            
            result = EnhancedRAGResult(
                query=query,
                original_results=original_properties,
                filtered_results=filtered_properties,
                processing_time=processing_time,
                filter_stats=filter_stats,
                confidence=confidence,
                metadata={
                    'session_id': session_id,
                    'retrieval_strategy': retrieval_strategy.value,
                    'processed_query': processed_query.processed_query,
                    'query_intent': {
                        'intent': processed_query.intent.value,
                        'complexity': processed_query.complexity.value,
                        'entities': processed_query.entities,
                        'filters': processed_query.filters
                    },
                    'retrieval_count': len(original_properties),
                    'filtered_count': len(filtered_properties),
                    'filter_rate': len(filtered_properties) / max(1, len(original_properties))
                }
            )
            
            logger.info(f"AI-enhanced search complete: {len(filtered_properties)}/{len(original_properties)} "
                       f"properties match query intent (Confidence: {confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in AI-enhanced search: {e}")
            # Return empty result on error
            return EnhancedRAGResult(
                query=query,
                original_results=[],
                filtered_results=[],
                processing_time=time.time() - start_time,
                filter_stats={},
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def _calculate_overall_confidence(self, 
                                    processed_query: ProcessedQuery,
                                    filtered_properties: List[Dict[str, Any]],
                                    original_properties: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in the search results."""
        try:
            # Base confidence from query processing
            base_confidence = processed_query.confidence
            
            # Filter quality factor
            if not original_properties:
                return 0.0
            
            filter_quality = len(filtered_properties) / len(original_properties)
            
            # Adjust confidence based on filter quality
            if filter_quality > 0.5:  # Good filter rate
                confidence = base_confidence * 1.1
            elif filter_quality > 0.2:  # Moderate filter rate
                confidence = base_confidence * 0.9
            else:  # Low filter rate (might be too restrictive)
                confidence = base_confidence * 0.7
            
            # Boost confidence if we have good matches
            if filtered_properties:
                confidence = min(1.0, confidence + 0.1)
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            retriever_stats = getattr(self.retriever, 'get_stats', lambda: {})()
            postprocessor_stats = self.postprocessor.get_stats()
            query_processor_stats = getattr(self.query_processor, 'get_processing_stats', lambda: {})()
            
            return {
                'retriever': retriever_stats,
                'postprocessor': postprocessor_stats,
                'query_processor': query_processor_stats,
                'system_info': {
                    'components': ['retriever', 'postprocessor', 'query_processor'],
                    'enhancement': 'AI-powered semantic filtering'
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {'error': str(e)}
    
    def reset_stats(self):
        """Reset all system statistics."""
        try:
            if hasattr(self.retriever, 'reset_stats'):
                self.retriever.reset_stats()
            
            self.postprocessor.reset_stats()
            
            if hasattr(self.query_processor, 'reset_stats'):
                self.query_processor.reset_stats()
            
            logger.info("System statistics reset")
            
        except Exception as e:
            logger.error(f"Error resetting stats: {e}")

# Factory function for easy initialization
def create_ai_enhanced_rag_system(use_gpu: bool = True) -> AIEnhancedRAGSystem:
    """
    Factory function to create an AI-enhanced RAG system.
    
    Args:
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Initialized AIEnhancedRAGSystem
    """
    try:
        # Initialize components
        from rag.advanced_rag_retriever import AdvancedRAGRetriever
        from rag.query_processor import AdvancedQueryProcessor
        
        # Create retriever (assuming it's already initialized with models)
        retriever = AdvancedRAGRetriever()
        
        # Create query processor
        query_processor = AdvancedQueryProcessor(use_gpu=use_gpu)
        
        # Create post-processor
        postprocessor = AIPropertyPostProcessor(use_gpu=use_gpu)
        
        # Create enhanced RAG system
        enhanced_rag = AIEnhancedRAGSystem(
            retriever=retriever,
            postprocessor=postprocessor,
            query_processor=query_processor
        )
        
        logger.info("AI-Enhanced RAG System created successfully")
        return enhanced_rag
        
    except Exception as e:
        logger.error(f"Error creating AI-enhanced RAG system: {e}")
        raise
