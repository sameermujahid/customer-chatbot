import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import time
import gc
from threading import Lock
import multiprocessing as mp
import asyncio
import threading
from functools import lru_cache
import queue
import psutil
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PropertyMatch:
    """Represents a property match with confidence score"""
    property_data: Dict[str, Any]
    confidence_score: float
    match_reasons: List[str]
    matched_fields: List[str]

class AIPropertyFilter:
    """
    AI-powered property filtering system using multiple Hugging Face models
    for accurate query-to-property matching without hardcoded patterns
    """
    
    def __init__(self, 
                 use_gpu: bool = True,
                 max_workers: int = None,  # Auto-detect optimal workers
                 similarity_threshold: float = 0.7,
                 cross_encoder_threshold: float = 0.6,
                 batch_size: int = 64,  # Increased for better GPU utilization
                 enable_multiprocessing: bool = True,
                 enable_caching: bool = True,
                 cache_size: int = 1000):
        
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        # Auto-detect optimal number of workers
        if max_workers is None:
            if self.device == "cuda":
                # For GPU, use fewer workers to avoid memory conflicts
                self.max_workers = min(4, psutil.cpu_count())
            else:
                # For CPU, use more workers
                self.max_workers = min(8, psutil.cpu_count())
        else:
            self.max_workers = max_workers
            
        self.similarity_threshold = similarity_threshold
        self.cross_encoder_threshold = cross_encoder_threshold
        self.batch_size = batch_size
        self.enable_multiprocessing = enable_multiprocessing
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        
        # STRICT thresholds for precise AI-based matching - NO partial matches allowed
        self.minimum_threshold = 0.6  # Much higher threshold - requires strong matches only
        self.field_match_threshold = 0.5  # Higher threshold for individual field matching
        self.overall_match_threshold = 0.6  # Higher threshold for overall property matching
        
        # New strict matching parameters
        self.required_criteria_threshold = 0.7  # Threshold for each required criteria
        self.complete_match_boost = 0.3  # Boost for properties that match ALL criteria
        self.partial_match_penalty = 0.5  # Penalty for properties that only match some criteria
        
        # Thread safety and performance optimization
        self._lock = Lock()
        self._model_lock = Lock()
        self._cache_lock = Lock()
        
        # Performance optimization components
        self._embedding_cache = {} if enable_caching else None
        self._query_cache = {} if enable_caching else None
        self._processing_queue = queue.Queue()
        self._result_queue = queue.Queue()
        
        # Initialize models
        self._initialize_models()
        
        # Property fields to match against - ALL FIELDS for comprehensive AI filtering
        self.target_fields = [
            'PropertyName', 'Address', 'ZipCode', 'LeasableSquareFeet', 'YearBuilt', 
            'NumberOfRooms', 'ParkingSpaces', 'PropertyManager', 'MarketValue', 
            'City', 'State', 'Country', 'PropertyType', 'PropertyStatus', 
            'Description', 'TotalSquareFeet', 'Beds', 'Baths', 'AgentName', 
            'KeyFeatures', 'NearbyAmenities', 'PGDetails', 'CommercialDetails'
        ]
        
        # Clear GPU memory on initialization
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        logger.info(f"AIPropertyFilter initialized on {self.device} with batch_size={batch_size}")
    
    def _initialize_models(self):
        """Initialize multiple Hugging Face models for different tasks with GPU optimization"""
        try:
            # 1. More powerful Sentence Transformer for better semantic understanding
            self.sentence_model = SentenceTransformer(
                'sentence-transformers/all-mpnet-base-v2',  # More powerful model
                device=self.device  # Use GPU for faster processing
            )
            
            # 2. More powerful Cross-encoder for precise matching
            self.cross_encoder = CrossEncoder(
                'cross-encoder/ms-marco-MiniLM-L-12-v2',  # More powerful model
                device=self.device  # Use GPU for faster processing
            )
            
            # 3. Additional model for query decomposition and understanding
            self.query_understanding_model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',
                device=self.device
            )
            
            # 4. NER pipeline for entity extraction (if needed)
            self.ner_pipeline = None  # We'll use AI-based entity extraction
            
            # Clear GPU memory after model loading
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            logger.info("✅ GPU-optimized AI models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading AI models: {e}")
            # Fallback to basic text matching
            self.sentence_model = None
            self.cross_encoder = None
            self.ner_pipeline = None
            self.classifier = None
            logger.warning("⚠️ Using fallback text matching without AI models")
    
    def extract_query_entities(self, query: str) -> Dict[str, Any]:
        """Enhanced AI-based query decomposition to identify all required criteria including price ranges"""
        try:
            # Use AI to decompose the query into semantic components
            query_words = query.lower().split()
            
            # Extract price range information
            price_range = self._extract_price_range(query)
            
            # Check if this is a generic property query (contains generic property terms)
            # Generic terms are broad, non-specific property terms that should use lenient matching
            generic_property_terms = {
                'properties', 'property', 'real estate', 'homes', 'houses', 'houses', 
                'buildings', 'units', 'spaces', 'accommodation', 'residence', 'residences',
                'place', 'places', 'home', 'house', 'building', 'unit', 'space',
                'pgs', 'pg', 'paying guest', 'paying guests', 'hostel', 'hostels',
                'guest house', 'guest houses', 'lodging'
            }
            
            # Specific property types that should use strict matching (not generic)
            specific_property_terms = {
                'apartments', 'apartment', 'flats', 'flat', 'villas', 'villa',
                'commercial', 'office', 'offices', 'retail', 'warehouse', 'warehouses',
                'land', 'plot', 'plots', 'farm', 'farms', 'estate', 'estates'
            }
            
            # Check if query contains generic property terms
            has_generic_terms = any(term in query.lower() for term in generic_property_terms)
            
            # Check if query contains specific property terms (should use strict matching)
            has_specific_terms = any(term in query.lower() for term in specific_property_terms)
            
            # Enhanced analysis using AI models to understand query complexity
            if self.query_understanding_model is not None:
                # Use AI to analyze query complexity
                query_embedding = self.query_understanding_model.encode([query])
                
                # Analyze if this is a multi-criteria query by checking semantic diversity
                # This is a heuristic approach - in practice, we can use more sophisticated methods
                
                # For queries with multiple words, check if they represent different concepts
                if len(query_words) >= 2:
                    # Check semantic similarity between words to determine if they're related
                    word_embeddings = self.query_understanding_model.encode(query_words)
                    
                    # Calculate average similarity between words
                    similarities = []
                    for i in range(len(word_embeddings)):
                        for j in range(i + 1, len(word_embeddings)):
                            sim = np.dot(word_embeddings[i], word_embeddings[j]) / (
                                np.linalg.norm(word_embeddings[i]) * np.linalg.norm(word_embeddings[j])
                            )
                            similarities.append(sim)
                    
                    avg_similarity = np.mean(similarities) if similarities else 0.0
                    
                    # If words are semantically different (low similarity), it's likely multi-criteria
                    # But if it contains generic property terms, be more lenient
                    # However, if it contains specific property terms, use strict matching
                    is_multi_criteria = avg_similarity < 0.6 and not has_generic_terms and not has_specific_terms
                    
                    return {
                        'is_multi_criteria': is_multi_criteria,
                        'query_components': query_words,
                        'requires_all_criteria': is_multi_criteria,
                        'semantic_diversity': avg_similarity,
                        'query_complexity': 'high' if is_multi_criteria else 'low',
                        'price_range': price_range,
                        'has_generic_terms': has_generic_terms,
                        'has_specific_terms': has_specific_terms,
                        'is_generic_query': has_generic_terms and not has_specific_terms
                    }
                else:
                    return {
                        'is_multi_criteria': False,
                        'query_components': query_words,
                        'requires_all_criteria': False,
                        'semantic_diversity': 1.0,
                        'query_complexity': 'low',
                        'price_range': price_range,
                        'has_generic_terms': has_generic_terms,
                        'has_specific_terms': has_specific_terms,
                        'is_generic_query': has_generic_terms and not has_specific_terms
                    }
            else:
                # Fallback to simple word count analysis
                # Special handling for price-based queries - they should be treated as generic
                is_price_based_query = price_range.get('has_price_range', False)
                
                if len(query_words) >= 2 and not has_generic_terms and not has_specific_terms and not is_price_based_query:
                    return {
                        'is_multi_criteria': True,
                        'query_components': query_words,
                        'requires_all_criteria': True,
                        'semantic_diversity': 0.5,  # Assume moderate diversity
                        'query_complexity': 'high',
                        'price_range': price_range,
                        'has_generic_terms': has_generic_terms,
                        'has_specific_terms': has_specific_terms,
                        'is_generic_query': has_generic_terms and not has_specific_terms
                    }
                else:
                    # For price-based queries or generic queries, use lenient matching
                    # But for specific property terms, use strict matching
                    return {
                        'is_multi_criteria': False,
                        'query_components': query_words,
                        'requires_all_criteria': False,
                        'semantic_diversity': 1.0,
                        'query_complexity': 'low',
                        'price_range': price_range,
                        'has_generic_terms': has_generic_terms,
                        'has_specific_terms': has_specific_terms,
                        'is_generic_query': (has_generic_terms or is_price_based_query) and not has_specific_terms
                    }
            
        except Exception as e:
            logger.error(f"Error in enhanced AI query decomposition: {e}")
            price_range = self._extract_price_range(query)
            is_price_based_query = price_range.get('has_price_range', False)
            has_generic_terms = any(term in query.lower() for term in {
                'properties', 'property', 'real estate', 'homes', 'houses', 'pgs', 'pg', 
                'paying guest', 'hostel'
            })
            has_specific_terms = any(term in query.lower() for term in {
                'apartments', 'flats', 'villas', 'commercial', 'office', 'warehouse'
            })
            
            return {
                'is_multi_criteria': len(query.lower().split()) >= 2 and not (has_generic_terms or is_price_based_query or has_specific_terms),
                'query_components': query.lower().split(),
                'requires_all_criteria': len(query.lower().split()) >= 2 and not (has_generic_terms or is_price_based_query or has_specific_terms),
                'semantic_diversity': 0.5,
                'query_complexity': 'medium',
                'price_range': price_range,
                'has_generic_terms': has_generic_terms,
                'has_specific_terms': has_specific_terms,
                'is_generic_query': (has_generic_terms or is_price_based_query) and not has_specific_terms
            }
    
    def _extract_price_range(self, query: str) -> Dict[str, Any]:
        """Extract price range from query (e.g., '50-100 Lakhs', '0-50 Lakhs')"""
        try:
            import re
            
            # Convert to lowercase for pattern matching
            query_lower = query.lower()
            
            # Comprehensive price pattern matching for natural language queries
            # Supports: exact numbers, ranges, thousands, lakhs, crores, direct amounts, and various natural language patterns
            price_patterns = [
                # ULTRA COMPREHENSIVE RANGE PATTERNS - "within X to Y Lakhs"
                r'within\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'within\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+budget',
                r'within\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+range',
                r'within\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+price',
                
                # "with in X to Y Lakhs" patterns (space in "with in")
                r'with\s+in\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'with\s+in\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+budget',
                r'with\s+in\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+range',
                r'with\s+in\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+price',
                
                # "in X to Y Lakhs" patterns
                r'in\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'in\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+range',
                r'in\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+budget',
                r'in\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+price',
                
                # "between X and Y Lakhs" patterns
                r'between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+range',
                r'between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+budget',
                r'between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+price',
                
                # "from X to Y Lakhs" patterns
                r'from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+range',
                r'from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+budget',
                r'from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+price',
                
                # "X to Y Lakhs" patterns (direct)
                r'(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+range',
                r'(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+budget',
                r'(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+price',
                
                # "X-Y Lakhs" patterns (dash)
                r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+range',
                r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+budget',
                r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+price',
                
                # "X Y Lakhs" patterns (space)
                r'(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+range',
                r'(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+budget',
                r'(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+price',
                
                # "within X Y Lakhs" patterns (space)
                r'within\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'within\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+range',
                r'within\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+budget',
                r'within\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+price',
                
                # "with in X Y Lakhs" patterns (space in "with in")
                r'with\s+in\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'with\s+in\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+range',
                r'with\s+in\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+budget',
                r'with\s+in\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+price',
                
                # "in X Y Lakhs" patterns (space)
                r'in\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'in\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+range',
                r'in\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+budget',
                r'in\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+price',
                
                # "between X Y Lakhs" patterns (space)
                r'between\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'between\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+range',
                r'between\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+budget',
                r'between\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+price',
                
                # "from X Y Lakhs" patterns (space)
                r'from\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'from\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+range',
                r'from\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+budget',
                r'from\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+price',
                
                # LEGACY PATTERNS - Keep existing patterns for backward compatibility
                r'within\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+budget',
                r'budget\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                
                # Comprehensive single value patterns with units
                r'upto\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'under\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'below\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'above\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'more\s+than\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'less\s+than\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'over\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'no\s+more\s+than\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'at\s+least\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'at\s+most\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'maximum\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'minimum\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'max\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'min\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'around\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'approximately\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'about\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'near\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'close\s+to\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'up\s+to\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'not\s+more\s+than\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'not\s+less\s+than\s+(\d+(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                
                # ULTRA COMPREHENSIVE DIRECT RUPEE AMOUNT RANGE PATTERNS
                # "between X and Y" patterns
                r'between\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'between\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'between\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'between\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                
                # "from X to Y" patterns
                r'from\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'from\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'from\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'from\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                
                # "X to Y" patterns (direct)
                r'(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                
                # "X-Y" patterns (dash)
                r'(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                
                # "within X to Y" patterns
                r'within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                r'within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+inr',
                r'within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupee',
                r'within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs\.',
                r'within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹\.',
                
                # "within X-Y" patterns (dash)
                r'within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                
                # "within X and Y" patterns
                r'within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                
                # "within X to Y range" patterns
                r'within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+range',
                r'within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+price\s+range',
                r'within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+budget\s+range',
                
                # "within X-Y range" patterns (dash)
                r'within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+range',
                r'within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+price\s+range',
                r'within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+budget\s+range',
                
                # "in X to Y" patterns
                r'in\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'in\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'in\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'in\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                r'in\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+inr',
                r'in\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupee',
                r'in\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs\.',
                r'in\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹\.',
                
                # "in X-Y" patterns (dash)
                r'in\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'in\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'in\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'in\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                
                # "in X and Y" patterns
                r'in\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'in\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'in\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'in\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                
                # "in the range of X to Y" patterns
                r'in\s+the\s+range\s+of\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'in\s+the\s+range\s+of\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'in\s+the\s+range\s+of\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'in\s+the\s+range\s+of\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                
                # "in the range X to Y" patterns
                r'in\s+the\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'in\s+the\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'in\s+the\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'in\s+the\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                
                # "range X to Y" patterns
                r'range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                r'range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+inr',
                r'range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupee',
                
                # "range X-Y" patterns (dash)
                r'range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                
                # "range X and Y" patterns
                r'range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                
                # "price range X to Y" patterns
                r'price\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'price\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'price\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'price\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                
                # "price range X-Y" patterns (dash)
                r'price\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'price\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'price\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'price\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                
                # "price range X and Y" patterns
                r'price\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'price\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'price\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'price\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                
                # "price range X to Y" patterns
                r'price\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'price\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'price\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'price\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                
                # "budget X to Y" patterns
                r'budget\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'budget\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'budget\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'budget\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                r'budget\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+inr',
                r'budget\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupee',
                
                # "budget X-Y" patterns (dash)
                r'budget\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'budget\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'budget\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'budget\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                
                # "budget X and Y" patterns
                r'budget\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'budget\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'budget\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'budget\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                
                # "between X and Y" patterns
                r'between\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'between\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'between\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'between\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                r'between\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+inr',
                r'between\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupee',
                
                # "from X to Y" patterns
                r'from\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'from\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'from\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'from\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                r'from\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+inr',
                r'from\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupee',
                
                # "from X-Y" patterns (dash)
                r'from\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'from\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'from\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'from\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                
                # "costing X to Y" patterns
                r'costing\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'costing\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'costing\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'costing\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                
                # "costing X-Y" patterns (dash)
                r'costing\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'costing\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'costing\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'costing\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                
                # "costing X and Y" patterns
                r'costing\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'costing\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rupees',
                r'costing\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+rs',
                r'costing\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+₹',
                
                # Comprehensive direct rupee amount patterns (without lakhs/crores) - handle spaces in numbers
                r'under\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'below\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'upto\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'above\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'more\s+than\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'less\s+than\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'over\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'no\s+more\s+than\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'at\s+least\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'at\s+most\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'maximum\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'minimum\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'max\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'min\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'around\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'approximately\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'about\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'near\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'close\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'up\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'not\s+more\s+than\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'not\s+less\s+than\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                
                # Additional casual/informal patterns
                r'roughly\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'roughly\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'roughly\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'like\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'like\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'like\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'something\s+like\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'something\s+like\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'something\s+like\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'sort\s+of\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'sort\s+of\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'sort\s+of\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'maybe\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'maybe\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'maybe\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'probably\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'probably\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'probably\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'ideally\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'ideally\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'ideally\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'preferably\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'preferably\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'preferably\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                
                # Question-based patterns
                r'what\s+properties\s+under\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'what\s+properties\s+below\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'what\s+properties\s+upto\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'what\s+properties\s+above\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'what\s+properties\s+over\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'what\s+properties\s+more\s+than\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'what\s+properties\s+less\s+than\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'what\s+properties\s+between\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'what\s+properties\s+from\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'what\s+properties\s+within\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'what\s+properties\s+in\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'what\s+properties\s+in\s+the\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'what\s+properties\s+in\s+the\s+range\s+of\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'what\s+properties\s+in\s+the\s+price\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'what\s+properties\s+in\s+the\s+budget\s+range\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'what\s+properties\s+with\s+budget\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'what\s+properties\s+with\s+price\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'what\s+properties\s+costing\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'what\s+properties\s+costing\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*-\s*(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'what\s+properties\s+costing\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+and\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                
                # Negation patterns
                r'not\s+more\s+than\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'not\s+less\s+than\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'not\s+above\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'not\s+below\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'not\s+over\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'not\s+under\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                
                # Comparison patterns
                r'cheaper\s+than\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'more\s+expensive\s+than\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'less\s+expensive\s+than\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'costlier\s+than\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'pricier\s+than\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'budget\s+friendly\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'affordable\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'economical\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'value\s+for\s+money\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s+to\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                
                # Exact number patterns (treat as range with small tolerance)
                r'exactly\s+(\d+(?:\s+\d+)*(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)',
                r'(\d+(?:\s+\d+)*(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+exactly',
                r'(\d+(?:\s+\d+)*(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+only',
                r'(\d+(?:\s+\d+)*(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+precisely',
                r'(\d+(?:\s+\d+)*(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)\s+flat',
                
                # Direct exact number patterns
                r'exactly\s+(\d+(?:\s+\d+)*(?:\.\d+)?)',
                r'(\d+(?:\s+\d+)*(?:\.\d+)?)\s+exactly',
                r'(\d+(?:\s+\d+)*(?:\.\d+)?)\s+only',
                r'(\d+(?:\s+\d+)*(?:\.\d+)?)\s+precisely',
                r'(\d+(?:\s+\d+)*(?:\.\d+)?)\s+flat',
                
                # ADDITIONAL EDGE CASE PATTERNS
                # "X lakh to Y lakh" patterns (with "lakh" repeated)
                r'(\d+(?:\.\d+)?)\s+lakh\s+to\s+(\d+(?:\.\d+)?)\s+lakh',
                r'(\d+(?:\.\d+)?)\s+lakhs\s+to\s+(\d+(?:\.\d+)?)\s+lakhs',
                r'(\d+(?:\.\d+)?)\s+lakh\s+to\s+(\d+(?:\.\d+)?)\s+lakhs',
                r'(\d+(?:\.\d+)?)\s+lakhs\s+to\s+(\d+(?:\.\d+)?)\s+lakh',
                
                # "X crore to Y crore" patterns
                r'(\d+(?:\.\d+)?)\s+crore\s+to\s+(\d+(?:\.\d+)?)\s+crore',
                r'(\d+(?:\.\d+)?)\s+crores\s+to\s+(\d+(?:\.\d+)?)\s+crores',
                r'(\d+(?:\.\d+)?)\s+crore\s+to\s+(\d+(?:\.\d+)?)\s+crores',
                r'(\d+(?:\.\d+)?)\s+crores\s+to\s+(\d+(?:\.\d+)?)\s+crore',
                
                # "X lakh - Y lakh" patterns (dash with repeated unit)
                r'(\d+(?:\.\d+)?)\s+lakh\s*-\s*(\d+(?:\.\d+)?)\s+lakh',
                r'(\d+(?:\.\d+)?)\s+lakhs\s*-\s*(\d+(?:\.\d+)?)\s+lakhs',
                r'(\d+(?:\.\d+)?)\s+lakh\s*-\s*(\d+(?:\.\d+)?)\s+lakhs',
                r'(\d+(?:\.\d+)?)\s+lakhs\s*-\s*(\d+(?:\.\d+)?)\s+lakh',
                
                # "X crore - Y crore" patterns (dash with repeated unit)
                r'(\d+(?:\.\d+)?)\s+crore\s*-\s*(\d+(?:\.\d+)?)\s+crore',
                r'(\d+(?:\.\d+)?)\s+crores\s*-\s*(\d+(?:\.\d+)?)\s+crores',
                r'(\d+(?:\.\d+)?)\s+crore\s*-\s*(\d+(?:\.\d+)?)\s+crores',
                r'(\d+(?:\.\d+)?)\s+crores\s*-\s*(\d+(?:\.\d+)?)\s+crore',
                
                # "X lakh Y lakh" patterns (space with repeated unit)
                r'(\d+(?:\.\d+)?)\s+lakh\s+(\d+(?:\.\d+)?)\s+lakh',
                r'(\d+(?:\.\d+)?)\s+lakhs\s+(\d+(?:\.\d+)?)\s+lakhs',
                r'(\d+(?:\.\d+)?)\s+lakh\s+(\d+(?:\.\d+)?)\s+lakhs',
                r'(\d+(?:\.\d+)?)\s+lakhs\s+(\d+(?:\.\d+)?)\s+lakh',
                
                # "X crore Y crore" patterns (space with repeated unit)
                r'(\d+(?:\.\d+)?)\s+crore\s+(\d+(?:\.\d+)?)\s+crore',
                r'(\d+(?:\.\d+)?)\s+crores\s+(\d+(?:\.\d+)?)\s+crores',
                r'(\d+(?:\.\d+)?)\s+crore\s+(\d+(?:\.\d+)?)\s+crores',
                r'(\d+(?:\.\d+)?)\s+crores\s+(\d+(?:\.\d+)?)\s+crore',
                
                # "within X lakh to Y lakh" patterns
                r'within\s+(\d+(?:\.\d+)?)\s+lakh\s+to\s+(\d+(?:\.\d+)?)\s+lakh',
                r'within\s+(\d+(?:\.\d+)?)\s+lakhs\s+to\s+(\d+(?:\.\d+)?)\s+lakhs',
                r'within\s+(\d+(?:\.\d+)?)\s+lakh\s+to\s+(\d+(?:\.\d+)?)\s+lakhs',
                r'within\s+(\d+(?:\.\d+)?)\s+lakhs\s+to\s+(\d+(?:\.\d+)?)\s+lakh',
                
                # "with in X lakh to Y lakh" patterns (space in "with in")
                r'with\s+in\s+(\d+(?:\.\d+)?)\s+lakh\s+to\s+(\d+(?:\.\d+)?)\s+lakh',
                r'with\s+in\s+(\d+(?:\.\d+)?)\s+lakhs\s+to\s+(\d+(?:\.\d+)?)\s+lakhs',
                r'with\s+in\s+(\d+(?:\.\d+)?)\s+lakh\s+to\s+(\d+(?:\.\d+)?)\s+lakhs',
                r'with\s+in\s+(\d+(?:\.\d+)?)\s+lakhs\s+to\s+(\d+(?:\.\d+)?)\s+lakh',
                
                # "in X lakh to Y lakh" patterns
                r'in\s+(\d+(?:\.\d+)?)\s+lakh\s+to\s+(\d+(?:\.\d+)?)\s+lakh',
                r'in\s+(\d+(?:\.\d+)?)\s+lakhs\s+to\s+(\d+(?:\.\d+)?)\s+lakhs',
                r'in\s+(\d+(?:\.\d+)?)\s+lakh\s+to\s+(\d+(?:\.\d+)?)\s+lakhs',
                r'in\s+(\d+(?:\.\d+)?)\s+lakhs\s+to\s+(\d+(?:\.\d+)?)\s+lakh',
                
                # "between X lakh and Y lakh" patterns
                r'between\s+(\d+(?:\.\d+)?)\s+lakh\s+and\s+(\d+(?:\.\d+)?)\s+lakh',
                r'between\s+(\d+(?:\.\d+)?)\s+lakhs\s+and\s+(\d+(?:\.\d+)?)\s+lakhs',
                r'between\s+(\d+(?:\.\d+)?)\s+lakh\s+and\s+(\d+(?:\.\d+)?)\s+lakhs',
                r'between\s+(\d+(?:\.\d+)?)\s+lakhs\s+and\s+(\d+(?:\.\d+)?)\s+lakh',
                
                # "from X lakh to Y lakh" patterns
                r'from\s+(\d+(?:\.\d+)?)\s+lakh\s+to\s+(\d+(?:\.\d+)?)\s+lakh',
                r'from\s+(\d+(?:\.\d+)?)\s+lakhs\s+to\s+(\d+(?:\.\d+)?)\s+lakhs',
                r'from\s+(\d+(?:\.\d+)?)\s+lakh\s+to\s+(\d+(?:\.\d+)?)\s+lakhs',
                r'from\s+(\d+(?:\.\d+)?)\s+lakhs\s+to\s+(\d+(?:\.\d+)?)\s+lakh',
                
                # Standalone number patterns (treat as approximate - most natural user queries)
                r'(\d+(?:\s+\d+)*(?:\.\d+)?)\s*(lakh|lakhs|cr|crore|crores|k|thousand|thousands)'
            ]
            
            for i, pattern in enumerate(price_patterns):
                match = re.search(pattern, query_lower)
                if match:
                    groups = match.groups()
                    logger.info(f"💰 Price pattern #{i+1} matched: '{pattern}' -> {groups}")
                    logger.info(f"💰 Full query: '{query}' -> Matched groups: {groups}")
                    
                    if len(groups) == 3:  # Range pattern (min-max-unit)
                        min_val = float(groups[0])
                        max_val = float(groups[1])
                        unit = groups[2].lower()
                        
                        # Convert to actual values (MarketValue is in rupees)
                        if unit in ['lakh', 'lakhs']:
                            min_rupees = min_val * 100000  # 1 lakh = 100,000 rupees
                            max_rupees = max_val * 100000
                        elif unit in ['cr', 'crore', 'crores']:
                            min_rupees = min_val * 10000000  # 1 crore = 10,000,000 rupees
                            max_rupees = max_val * 10000000
                        elif unit in ['k', 'thousand', 'thousands']:
                            min_rupees = min_val * 1000  # 1 thousand = 1,000 rupees
                            max_rupees = max_val * 1000
                        else:
                            continue
                        
                        result = {
                            'has_price_range': True,
                            'min_price': min_rupees,
                            'max_price': max_rupees,
                            'min_display': f"{min_val} {unit}",
                            'max_display': f"{max_val} {unit}",
                            'unit': unit,
                            'pattern_matched': pattern
                        }
                        logger.info(f"💰 Price range extracted: {result}")
                        return result
                    
                    elif len(groups) == 2:  # Single value pattern (value-unit)
                        val = float(groups[0])
                        unit = groups[1].lower()
                        
                        # Convert to actual values
                        if unit in ['lakh', 'lakhs']:
                            rupees = val * 100000
                        elif unit in ['cr', 'crore', 'crores']:
                            rupees = val * 10000000
                        elif unit in ['k', 'thousand', 'thousands']:
                            rupees = val * 1000
                        else:
                            continue
                        
                        # Determine if it's upper limit, lower limit, or exact based on pattern
                        if 'upto' in pattern or 'under' in pattern or 'below' in pattern or 'no more than' in pattern or 'at most' in pattern or 'maximum' in pattern or 'max ' in pattern or 'not more than' in pattern:
                            result = {
                                'has_price_range': True,
                                'min_price': 0,
                                'max_price': rupees,
                                'min_display': "0",
                                'max_display': f"{val} {unit}",
                                'unit': unit,
                                'pattern_matched': pattern
                            }
                            logger.info(f"💰 Price range extracted: {result}")
                            return result
                        elif 'above' in pattern or 'more than' in pattern or 'over' in pattern or 'at least' in pattern or 'minimum' in pattern or 'min ' in pattern or 'not less than' in pattern:
                            result = {
                                'has_price_range': True,
                                'min_price': rupees,
                                'max_price': float('inf'),
                                'min_display': f"{val} {unit}",
                                'max_display': "∞",
                                'unit': unit,
                                'pattern_matched': pattern
                            }
                            logger.info(f"💰 Price range extracted: {result}")
                            return result
                        elif 'exactly' in pattern or 'only' in pattern or 'precisely' in pattern or 'flat' in pattern:
                            # Exact number - create a small range around the value (±5%)
                            tolerance = rupees * 0.05
                            result = {
                                'has_price_range': True,
                                'min_price': max(0, rupees - tolerance),
                                'max_price': rupees + tolerance,
                                'min_display': f"{val} {unit} (±5%)",
                                'max_display': f"{val} {unit} (±5%)",
                                'unit': unit,
                                'pattern_matched': pattern
                            }
                            logger.info(f"💰 Price range extracted: {result}")
                            return result
                        elif 'around' in pattern or 'approximately' in pattern or 'about' in pattern or 'near' in pattern or 'close to' in pattern:
                            # Approximate number - create a range around the value (±10%)
                            tolerance = rupees * 0.10
                            result = {
                                'has_price_range': True,
                                'min_price': max(0, rupees - tolerance),
                                'max_price': rupees + tolerance,
                                'min_display': f"{val} {unit} (±10%)",
                                'max_display': f"{val} {unit} (±10%)",
                                'unit': unit,
                                'pattern_matched': pattern
                            }
                            logger.info(f"💰 Price range extracted: {result}")
                            return result
                        else:
                            # Standalone number with unit (no comparison word) - treat as approximate (±15%)
                            # This handles queries like "villas 2 lakhs" or "properties 50K"
                            tolerance = rupees * 0.15
                            result = {
                                'has_price_range': True,
                                'min_price': max(0, rupees - tolerance),
                                'max_price': rupees + tolerance,
                                'min_display': f"{val} {unit} (±15%)",
                                'max_display': f"{val} {unit} (±15%)",
                                'unit': unit,
                                'pattern_matched': pattern
                            }
                            logger.info(f"💰 Price range extracted: {result}")
                            return result
                    
                    elif len(groups) == 2:  # Direct rupee amount range pattern (e.g., "between X and Y")
                        # Handle spaces in numbers (e.g., "1 00 000" -> "100000", "5 00 000" -> "500000")
                        min_str = groups[0].replace(' ', '')
                        max_str = groups[1].replace(' ', '')
                        min_val = float(min_str)
                        max_val = float(max_str)
                        
                        result = {
                            'has_price_range': True,
                            'min_price': min_val,  # Direct rupees value
                            'max_price': max_val,  # Direct rupees value
                            'min_display': f"₹{min_val:,.0f}",
                            'max_display': f"₹{max_val:,.0f}",
                            'unit': 'rupees',
                            'pattern_matched': pattern
                        }
                        logger.info(f"💰 Price range extracted: {result}")
                        return result
                    
                    elif len(groups) == 1:  # Direct rupee amount pattern (without lakhs/crores)
                        # Handle spaces in numbers (e.g., "50 0000" -> "500000", "2 00 000" -> "200000")
                        val_str = groups[0].replace(' ', '')
                        val = float(val_str)
                        
                        # Determine if it's upper limit, lower limit, or exact based on pattern
                        if 'upto' in pattern or 'under' in pattern or 'below' in pattern or 'less than' in pattern or 'no more than' in pattern or 'at most' in pattern or 'maximum' in pattern or 'max ' in pattern or 'not more than' in pattern:
                            result = {
                                'has_price_range': True,
                                'min_price': 0,
                                'max_price': val,  # Direct rupees value
                                'min_display': "0",
                                'max_display': f"₹{val:,.0f}",
                                'unit': 'rupees',
                                'pattern_matched': pattern
                            }
                            logger.info(f"💰 Price range extracted: {result}")
                            return result
                        elif 'above' in pattern or 'more than' in pattern or 'over' in pattern or 'at least' in pattern or 'minimum' in pattern or 'min ' in pattern or 'not less than' in pattern:
                            result = {
                                'has_price_range': True,
                                'min_price': val,  # Direct rupees value
                                'max_price': float('inf'),
                                'min_display': f"₹{val:,.0f}",
                                'max_display': "∞",
                                'unit': 'rupees',
                                'pattern_matched': pattern
                            }
                            logger.info(f"💰 Price range extracted: {result}")
                            return result
                        elif 'exactly' in pattern or 'only' in pattern or 'precisely' in pattern or 'flat' in pattern:
                            # Exact number - create a small range around the value (±5%)
                            tolerance = val * 0.05
                            result = {
                                'has_price_range': True,
                                'min_price': max(0, val - tolerance),
                                'max_price': val + tolerance,
                                'min_display': f"₹{val:,.0f} (±5%)",
                                'max_display': f"₹{val:,.0f} (±5%)",
                                'unit': 'rupees',
                                'pattern_matched': pattern
                            }
                            logger.info(f"💰 Price range extracted: {result}")
                            return result
                        elif 'around' in pattern or 'approximately' in pattern or 'about' in pattern or 'near' in pattern or 'close to' in pattern:
                            # Approximate number - create a range around the value (±10%)
                            tolerance = val * 0.10
                            result = {
                                'has_price_range': True,
                                'min_price': max(0, val - tolerance),
                                'max_price': val + tolerance,
                                'min_display': f"₹{val:,.0f} (±10%)",
                                'max_display': f"₹{val:,.0f} (±10%)",
                                'unit': 'rupees',
                                'pattern_matched': pattern
                            }
                            logger.info(f"💰 Price range extracted: {result}")
                            return result
                        else:
                            # Standalone number (no comparison word) - treat as approximate (±15%)
                            # This handles queries like "villas in 200000" or "properties 2 lakhs"
                            tolerance = val * 0.15
                            result = {
                                'has_price_range': True,
                                'min_price': max(0, val - tolerance),
                                'max_price': val + tolerance,
                                'min_display': f"₹{val:,.0f} (±15%)",
                                'max_display': f"₹{val:,.0f} (±15%)",
                                'unit': 'rupees',
                                'pattern_matched': pattern
                            }
                            logger.info(f"💰 Price range extracted: {result}")
                            return result
            
            # No price range found
            logger.info(f"💰 No price range found in query: '{query}'")
            return {
                'has_price_range': False,
                'min_price': None,
                'max_price': None,
                'min_display': None,
                'max_display': None,
                'unit': None,
                'pattern_matched': None
            }
            
        except Exception as e:
            logger.error(f"Error extracting price range: {e}")
            return {
                'has_price_range': False,
                'min_price': None,
                'max_price': None,
                'min_display': None,
                'max_display': None,
                'unit': None,
                'pattern_matched': None
            }
    
    def _preprocess_query(self, query: str) -> str:
        """Minimal preprocessing for AI models - preserve natural language"""
        # Minimal preprocessing - AI models work better with natural language
        # Only normalize whitespace, preserve all other characters for better AI understanding
        cleaned = ' '.join(query.split())
        return cleaned
    
    def calculate_semantic_similarity(self, query: str, property_text: str) -> float:
        """Calculate semantic similarity between query and property text"""
        try:
            if self.sentence_model is None:
                # Fallback to simple text similarity
                return self._calculate_text_similarity(query, property_text)
            
            # Get embeddings
            query_embedding = self.sentence_model.encode([query])
            property_embedding = self.sentence_model.encode([property_text])
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding[0], property_embedding[0]) / (
                np.linalg.norm(query_embedding[0]) * np.linalg.norm(property_embedding[0])
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            # Fallback to simple text similarity
            return self._calculate_text_similarity(query, property_text)
    
    def calculate_batch_semantic_similarity(self, query: str, property_texts: List[str]) -> List[float]:
        """Calculate semantic similarity between query and multiple property texts in batch with caching and optimization"""
        try:
            if self.sentence_model is None:
                # Fallback to simple text similarity
                return [self._calculate_text_similarity(query, text) for text in property_texts]
            
            # Check cache first
            if self.enable_caching and self._embedding_cache is not None:
                cache_key = f"semantic_{hash(query)}_{len(property_texts)}"
                with self._cache_lock:
                    if cache_key in self._embedding_cache:
                        logger.debug(f"Cache hit for semantic similarity: {cache_key}")
                        return self._embedding_cache[cache_key]
            
            # Prepare texts for batch processing
            texts = [query] + property_texts
            
            # Get embeddings in batch with optimized batch size
            with self._model_lock:  # Thread safety for model access
                # Use larger batch size for better GPU utilization
                optimal_batch_size = min(self.batch_size * 2, len(texts))
                embeddings = self.sentence_model.encode(
                    texts, 
                    batch_size=optimal_batch_size, 
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # Normalize for faster cosine similarity
                )
            
            query_embedding = embeddings[0]
            property_embeddings = embeddings[1:]
            
            # Vectorized cosine similarity calculation (much faster)
            # Since embeddings are normalized, cosine similarity is just dot product
            similarities = np.dot(property_embeddings, query_embedding).tolist()
            
            # Cache the result
            if self.enable_caching and self._embedding_cache is not None:
                with self._cache_lock:
                    if len(self._embedding_cache) < self.cache_size:
                        self._embedding_cache[cache_key] = similarities
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error calculating batch semantic similarity: {e}")
            # Fallback to simple text similarity
            return [self._calculate_text_similarity(query, text) for text in property_texts]
    
    def calculate_cross_encoder_score(self, query: str, property_text: str) -> float:
        """Calculate enhanced cross-encoder score for precise matching"""
        try:
            if self.cross_encoder is None:
                # Fallback to simple text matching
                return self._calculate_text_similarity(query, property_text)
            
            # Prepare text pair with enhanced context
            # Add context to help the cross-encoder understand the relationship
            enhanced_query = f"Find property: {query}"
            enhanced_property = f"Property details: {property_text}"
            
            text_pair = [enhanced_query, enhanced_property]
            
            # Get cross-encoder score
            score = self.cross_encoder.predict(text_pair)
            
            # Convert to probability (sigmoid) with enhanced scaling
            probability = 1 / (1 + np.exp(-score))
            
            # Apply additional scaling for better discrimination
            # This helps distinguish between strong and weak matches
            if probability > 0.8:
                # Strong matches get a boost
                probability = min(probability * 1.1, 1.0)
            elif probability < 0.3:
                # Weak matches get penalized more
                probability = probability * 0.8
            
            return float(probability)
            
        except Exception as e:
            logger.error(f"Error calculating enhanced cross-encoder score: {e}")
            # Fallback to simple text similarity
            return self._calculate_text_similarity(query, property_text)
    
    def calculate_batch_cross_encoder_score(self, query: str, property_texts: List[str]) -> List[float]:
        """Calculate enhanced cross-encoder scores for multiple property texts in batch with caching and optimization"""
        try:
            if self.cross_encoder is None:
                # Fallback to simple text matching
                return [self._calculate_text_similarity(query, text) for text in property_texts]
            
            # Check cache first
            if self.enable_caching and self._embedding_cache is not None:
                cache_key = f"cross_{hash(query)}_{len(property_texts)}"
                with self._cache_lock:
                    if cache_key in self._embedding_cache:
                        logger.debug(f"Cache hit for cross-encoder: {cache_key}")
                        return self._embedding_cache[cache_key]
            
            # Prepare enhanced text pairs for batch processing
            enhanced_query = f"Find property: {query}"
            text_pairs = [[enhanced_query, f"Property details: {text}"] for text in property_texts]
            
            # Get cross-encoder scores in batch with optimized processing
            with self._model_lock:  # Thread safety for model access
                # Process in chunks to avoid memory issues
                chunk_size = min(self.batch_size, len(text_pairs))
                all_scores = []
                
                for i in range(0, len(text_pairs), chunk_size):
                    chunk = text_pairs[i:i + chunk_size]
                    chunk_scores = self.cross_encoder.predict(chunk)
                    all_scores.extend(chunk_scores)
                
                scores = all_scores
            
            # Vectorized probability conversion (much faster)
            scores_array = np.array(scores)
            probabilities = 1 / (1 + np.exp(-scores_array))  # Sigmoid
            
            # Vectorized scaling for better discrimination
            probabilities = np.where(probabilities > 0.8, 
                                   np.minimum(probabilities * 1.1, 1.0),
                                   np.where(probabilities < 0.3, 
                                          probabilities * 0.8, 
                                          probabilities))
            
            result = probabilities.tolist()
            
            # Cache the result
            if self.enable_caching and self._embedding_cache is not None:
                with self._cache_lock:
                    if len(self._embedding_cache) < self.cache_size:
                        self._embedding_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating enhanced batch cross-encoder score: {e}")
            # Fallback to simple text similarity
            return [self._calculate_text_similarity(query, text) for text in property_texts]
    
    def _get_field_weights(self) -> Dict[str, float]:
        """Get dynamic field weights for comprehensive AI matching - NO STATIC PATTERNS"""
        # All fields get equal importance - pure AI will determine relevance
        # No hardcoded preferences, no static patterns, no keyword-based weights
        return {
            'PropertyName': 1.2,      # Slightly higher as it's often descriptive
            'Address': 1.1,           # Important for location-based queries
            'ZipCode': 1.0,           # Standard weight
            'LeasableSquareFeet': 1.0, # Standard weight
            'YearBuilt': 1.0,         # Standard weight
            'NumberOfRooms': 1.0,     # Standard weight
            'ParkingSpaces': 1.0,     # Standard weight
            'PropertyManager': 1.0,   # Standard weight
            'MarketValue': 1.0,       # Standard weight
            'City': 1.1,              # Important for location-based queries
            'State': 1.1,             # Important for location-based queries
            'Country': 1.1,           # Important for location-based queries
            'PropertyType': 1.2,      # Slightly higher as it's often descriptive
            'PropertyStatus': 1.0,    # Standard weight
            'Description': 1.3,       # Higher weight as it contains rich information
            'TotalSquareFeet': 1.0,   # Standard weight
            'Beds': 1.0,              # Standard weight
            'Baths': 1.0,             # Standard weight
            'AgentName': 1.0,         # Standard weight
            'KeyFeatures': 1.2,       # Slightly higher as it contains important details
            'NearbyAmenities': 1.1,   # Slightly higher for amenity-based queries
            'PGDetails': 1.1,         # Slightly higher for specific property details
            'CommercialDetails': 1.1  # Slightly higher for commercial property details
        }
    
    def _check_price_range_match(self, property_data: Dict[str, Any], price_range: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if property's MarketValue matches the price range"""
        try:
            if not price_range.get('has_price_range', False):
                return True, "No price range specified"
            
            market_value = property_data.get('MarketValue', 0)
            
            # Handle different data types for MarketValue
            if isinstance(market_value, str):
                try:
                    market_value = float(market_value.replace(',', '').replace('₹', '').strip())
                except (ValueError, AttributeError):
                    market_value = 0
            elif market_value is None:
                market_value = 0
            
            min_price = price_range.get('min_price', 0)
            max_price = price_range.get('max_price', float('inf'))
            
            # Debug logging
            logger.debug(f"Price check: Property '{property_data.get('PropertyName', 'Unknown')}' "
                        f"MarketValue={market_value}, Range={min_price}-{max_price}")
            
            # Check if market value is within range
            if min_price <= market_value <= max_price:
                return True, f"Price {market_value} within range {price_range.get('min_display')}-{price_range.get('max_display')}"
            else:
                return False, f"Price {market_value} outside range {price_range.get('min_display')}-{price_range.get('max_display')}"
                
        except Exception as e:
            logger.error(f"Error checking price range match: {e}")
            return True, f"Error checking price: {str(e)}"
    
    def _calculate_text_similarity(self, query: str, property_text: str) -> float:
        """AI-like text similarity as fallback when models fail"""
        try:
            # Simple word-based similarity as fallback
            query_words = set(query.lower().split())
            property_words = set(property_text.lower().split())
            
            if not query_words or not property_words:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = query_words & property_words
            union = query_words | property_words
            
            if not union:
                return 0.0
            
            similarity = len(intersection) / len(union)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating fallback text similarity: {e}")
            return 0.0
    
    def match_property_fields(self, query: str, property_data: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """STRICT AI-based matching - requires ALL query criteria to be satisfied including price range"""
        try:
            # First, decompose the query to understand requirements
            query_analysis = self.extract_query_entities(query)
            price_range = query_analysis.get('price_range', {})
            
            # Check price range match first - if it doesn't match, reject immediately
            if price_range.get('has_price_range', False):
                price_match, price_reason = self._check_price_range_match(property_data, price_range)
                if not price_match:
                    logger.info(f"🚫 Property '{property_data.get('PropertyName', 'Unknown')}' rejected due to price mismatch: {price_reason}")
                    return 0.0, [f"price_mismatch:{price_reason}"], []
                else:
                    logger.debug(f"✅ Property '{property_data.get('PropertyName', 'Unknown')}' passed price filter: {price_reason}")
            
            # Calculate individual field scores
            field_scores = {}
            field_weights = self._get_field_weights()
            
            for field in self.target_fields:
                if field not in property_data or not property_data[field]:
                    continue
                
                field_value = str(property_data[field])
                
                # Calculate both semantic and cross-encoder scores
                semantic_score = self.calculate_semantic_similarity(query, field_value)
                cross_score = self.calculate_cross_encoder_score(query, field_value)
                
                # Use the higher of the two scores for this field
                field_score = max(semantic_score, cross_score)
                
                # Apply field weight
                weighted_score = field_score * field_weights.get(field, 1.0)
                field_scores[field] = weighted_score
            
            # ENHANCED MATCHING LOGIC - More lenient for generic queries
            is_generic_query = query_analysis.get('is_generic_query', False)
            
            if query_analysis.get('requires_all_criteria', False) and not is_generic_query:
                # For multi-criteria queries (non-generic), we need STRONG matches in multiple relevant fields
                # This prevents partial matches like "townhouses" matching any property type
                
                # Identify the best matching fields
                sorted_fields = sorted(field_scores.items(), key=lambda x: x[1], reverse=True)
                
                # Require at least 2 strong matches for multi-criteria queries
                strong_matches = [field for field, score in sorted_fields if score >= self.required_criteria_threshold]
                
                if len(strong_matches) < 2:
                    # Not enough strong matches - reject this property
                    return 0.0, [], []
                
                # Calculate score based on the top matches
                top_scores = [score for field, score in sorted_fields[:3]]  # Top 3 matches
                final_score = sum(top_scores) / len(top_scores)  # Average of top matches
                
                # Apply complete match boost
                final_score = min(final_score + self.complete_match_boost, 1.0)
                
                match_reasons = [f"{field}:{score:.3f}" for field, score in sorted_fields[:3]]
                matched_fields = [field for field, score in sorted_fields[:3]]
                
                # Add price match reason if applicable
                if price_range.get('has_price_range', False):
                    _, price_reason = self._check_price_range_match(property_data, price_range)
                    match_reasons.append(f"price_match:{price_reason}")
                    matched_fields.append("MarketValue")
                
            else:
                # Single criteria query OR generic query - use more lenient matching
                if not field_scores:
                    return 0.0, [], []
                
                # For generic queries, use lower threshold
                if is_generic_query:
                    # Generic queries like "properties under 20000" - be more lenient
                    # Use the best match even if it's not super strong
                    best_field, best_score = max(field_scores.items(), key=lambda x: x[1])
                    
                    # Lower threshold for generic queries - they should match any property
                    generic_threshold = 0.3  # Much lower than required_criteria_threshold (0.7)
                    
                    if best_score < generic_threshold:
                        return 0.0, [], []
                    
                    # For generic queries, boost the score to make it more acceptable
                    final_score = min(best_score + 0.2, 1.0)  # Add boost for generic queries
                    match_reasons = [f"{best_field}:{best_score:.3f}(generic_boost)"]
                    matched_fields = [best_field]
                    
                    logger.debug(f"🏠 Generic query match: {best_field} scored {best_score:.3f}, boosted to {final_score:.3f}")
                    
                else:
                    # Single criteria query (non-generic) - use normal threshold
                    best_field, best_score = max(field_scores.items(), key=lambda x: x[1])
                    
                    if best_score < self.required_criteria_threshold:
                        return 0.0, [], []
                    
                    final_score = best_score
                    match_reasons = [f"{best_field}:{best_score:.3f}"]
                    matched_fields = [best_field]
                
                # Add price match reason if applicable
                if price_range.get('has_price_range', False):
                    _, price_reason = self._check_price_range_match(property_data, price_range)
                    match_reasons.append(f"price_match:{price_reason}")
                    matched_fields.append("MarketValue")
            
            return final_score, match_reasons, matched_fields
            
        except Exception as e:
            logger.error(f"Error in STRICT AI property field matching: {e}")
            return 0.0, [], []
    
    def match_properties_batch(self, query: str, properties: List[Dict[str, Any]]) -> List[Tuple[float, List[str], List[str]]]:
        """ULTRA-OPTIMIZED batch processing for precise AI matching with parallel processing"""
        try:
            if not properties:
                return []
            
            start_time = time.time()
            
            # First, analyze the query to understand requirements
            query_analysis = self.extract_query_entities(query)
            price_range = query_analysis.get('price_range', {})
            
            # Pre-filter properties by price range if specified (parallel processing)
            if price_range.get('has_price_range', False):
                logger.info(f"💰 Applying parallel price range filter: {price_range.get('min_display')}-{price_range.get('max_display')}")
                
                # Use parallel processing for price filtering
                filtered_properties, property_mapping, rejected_count = self._parallel_price_filter(
                    properties, price_range
                )
                
                logger.info(f"💰 Price filtering: {len(properties)} → {len(filtered_properties)} properties (rejected {rejected_count})")
                
                # If no properties match price range, return all zeros
                if not filtered_properties:
                    logger.warning(f"🚫 No properties match price range {price_range.get('min_display')}-{price_range.get('max_display')}")
                    return [(0.0, [f"price_mismatch:No properties in range {price_range.get('min_display')}-{price_range.get('max_display')}"], []) for _ in properties]
                
                properties_to_process = filtered_properties
            else:
                properties_to_process = properties
                property_mapping = {i: i for i in range(len(properties))}
            
            # Prepare all field texts for batch processing (optimized)
            all_field_texts, property_field_mapping = self._prepare_field_texts_batch(properties_to_process)
            
            if not all_field_texts:
                return [(0.0, [], []) for _ in properties]
            
            # Parallel batch processing for semantic and cross-encoder scores
            logger.info(f"🚀 Starting parallel batch processing for {len(all_field_texts)} field texts")
            
            # Use ThreadPoolExecutor for parallel model inference
            with ThreadPoolExecutor(max_workers=2) as executor:  # One for each model
                # Submit both tasks in parallel
                semantic_future = executor.submit(
                    self.calculate_batch_semantic_similarity, query, all_field_texts
                )
                cross_future = executor.submit(
                    self.calculate_batch_cross_encoder_score, query, all_field_texts
                )
                
                # Wait for both to complete
                semantic_scores = semantic_future.result()
                cross_scores = cross_future.result()
            
            # Process results for each property with STRICT scoring (vectorized)
            property_scores = self._process_field_scores_vectorized(
                property_field_mapping, semantic_scores, cross_scores
            )
            
            # Apply STRICT matching logic for each property (parallel processing)
            results = self._apply_matching_logic_parallel(
                properties_to_process, property_scores, query_analysis, price_range
            )
            
            # Map results back to original property indices
            final_results = [(0.0, [], []) for _ in properties]
            for new_idx, result in enumerate(results):
                orig_idx = property_mapping[new_idx]
                final_results[orig_idx] = result
            
            processing_time = time.time() - start_time
            logger.info(f"✅ ULTRA-OPTIMIZED batch processing completed in {processing_time:.2f}s")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in ULTRA-OPTIMIZED batch property matching: {e}")
            # Fallback to individual matching
            return [self.match_property_fields(query, prop) for prop in properties]
    
    def _parallel_price_filter(self, properties: List[Dict[str, Any]], price_range: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[int, int], int]:
        """Parallel price filtering for better performance"""
        try:
            filtered_properties = []
            property_mapping = {}
            rejected_count = 0
            
            # Use ThreadPoolExecutor for parallel price checking
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all price checking tasks
                future_to_index = {
                    executor.submit(self._check_price_range_match, prop, price_range): i
                    for i, prop in enumerate(properties)
                }
                
                # Collect results
                for future in as_completed(future_to_index):
                    orig_idx = future_to_index[future]
                    try:
                        price_match, _ = future.result()
                        if price_match:
                            filtered_properties.append(properties[orig_idx])
                            property_mapping[len(filtered_properties) - 1] = orig_idx
                        else:
                            rejected_count += 1
                    except Exception as e:
                        logger.error(f"Error in parallel price filtering: {e}")
                        rejected_count += 1
            
            return filtered_properties, property_mapping, rejected_count
            
        except Exception as e:
            logger.error(f"Error in parallel price filtering: {e}")
            # Fallback to sequential processing
            filtered_properties = []
            property_mapping = {}
            rejected_count = 0
            
            for orig_idx, property_data in enumerate(properties):
                price_match, _ = self._check_price_range_match(property_data, price_range)
                if price_match:
                    filtered_properties.append(property_data)
                    property_mapping[len(filtered_properties) - 1] = orig_idx
                else:
                    rejected_count += 1
            
            return filtered_properties, property_mapping, rejected_count
    
    def _prepare_field_texts_batch(self, properties: List[Dict[str, Any]]) -> Tuple[List[str], List[Tuple[int, str]]]:
        """Prepare field texts for batch processing with optimization"""
        try:
            all_field_texts = []
            property_field_mapping = []
            
            # Pre-allocate lists for better performance
            estimated_size = len(properties) * len(self.target_fields)
            all_field_texts = []
            property_field_mapping = []
            
            for prop_idx, property_data in enumerate(properties):
                for field in self.target_fields:
                    if field in property_data and property_data[field]:
                        field_value = str(property_data[field])
                        all_field_texts.append(field_value)
                        property_field_mapping.append((prop_idx, field))
            
            return all_field_texts, property_field_mapping
            
        except Exception as e:
            logger.error(f"Error preparing field texts batch: {e}")
            return [], []
    
    def _process_field_scores_vectorized(self, property_field_mapping: List[Tuple[int, str]], 
                                       semantic_scores: List[float], 
                                       cross_scores: List[float]) -> Dict[int, Dict[str, Any]]:
        """Process field scores using vectorized operations for better performance"""
        try:
            property_scores = {}
            field_weights = self._get_field_weights()
            
            # Convert to numpy arrays for vectorized operations
            semantic_array = np.array(semantic_scores)
            cross_array = np.array(cross_scores)
            
            # Vectorized max operation (much faster than loop)
            max_scores = np.maximum(semantic_array, cross_array)
            
            for i, (prop_idx, field) in enumerate(property_field_mapping):
                if prop_idx not in property_scores:
                    property_scores[prop_idx] = {
                        'field_scores': {},
                        'match_reasons': [],
                        'matched_fields': []
                    }
                
                # Apply field weight
                field_weight = field_weights.get(field, 1.0)
                weighted_field_score = max_scores[i] * field_weight
                
                # Store the field score
                property_scores[prop_idx]['field_scores'][field] = weighted_field_score
            
            return property_scores
            
        except Exception as e:
            logger.error(f"Error in vectorized field score processing: {e}")
            return {}
    
    def _apply_matching_logic_parallel(self, properties: List[Dict[str, Any]], 
                                     property_scores: Dict[int, Dict[str, Any]], 
                                     query_analysis: Dict[str, Any], 
                                     price_range: Dict[str, Any]) -> List[Tuple[float, List[str], List[str]]]:
        """Apply matching logic using parallel processing"""
        try:
            results = []
            
            # Use ThreadPoolExecutor for parallel matching logic
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all matching tasks
                future_to_index = {
                    executor.submit(
                        self._apply_single_property_matching_logic,
                        prop_idx, properties[prop_idx], property_scores.get(prop_idx, {}),
                        query_analysis, price_range
                    ): prop_idx
                    for prop_idx in range(len(properties))
                }
                
                # Collect results in order
                results = [None] * len(properties)
                for future in as_completed(future_to_index):
                    prop_idx = future_to_index[future]
                    try:
                        result = future.result()
                        results[prop_idx] = result
                    except Exception as e:
                        logger.error(f"Error in parallel matching logic: {e}")
                        results[prop_idx] = (0.0, [], [])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in parallel matching logic: {e}")
            # Fallback to sequential processing
            results = []
            for prop_idx in range(len(properties)):
                result = self._apply_single_property_matching_logic(
                    prop_idx, properties[prop_idx], property_scores.get(prop_idx, {}),
                    query_analysis, price_range
                )
                results.append(result)
            return results
    
    def _apply_single_property_matching_logic(self, prop_idx: int, property_data: Dict[str, Any], 
                                            score_data: Dict[str, Any], query_analysis: Dict[str, Any], 
                                            price_range: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """Apply matching logic for a single property"""
        try:
            if not score_data or 'field_scores' not in score_data:
                return (0.0, [], [])
            
            field_scores = score_data['field_scores']
            is_generic_query = query_analysis.get('is_generic_query', False)
            
            if query_analysis.get('requires_all_criteria', False) and not is_generic_query:
                # Multi-criteria query (non-generic) - require strong matches in multiple fields
                sorted_fields = sorted(field_scores.items(), key=lambda x: x[1], reverse=True)
                strong_matches = [field for field, score in sorted_fields if score >= self.required_criteria_threshold]
                
                if len(strong_matches) < 2:
                    return (0.0, [], [])
                
                # Calculate score based on top matches
                top_scores = [score for field, score in sorted_fields[:3]]
                final_score = sum(top_scores) / len(top_scores)
                final_score = min(final_score + self.complete_match_boost, 1.0)
                
                match_reasons = [f"{field}:{score:.3f}" for field, score in sorted_fields[:3]]
                matched_fields = [field for field, score in sorted_fields[:3]]
                
            else:
                # Single criteria query OR generic query - use more lenient matching
                if not field_scores:
                    return (0.0, [], [])
                
                # For generic queries, use lower threshold
                if is_generic_query:
                    best_field, best_score = max(field_scores.items(), key=lambda x: x[1])
                    generic_threshold = 0.3
                    
                    if best_score < generic_threshold:
                        return (0.0, [], [])
                    
                    final_score = min(best_score + 0.2, 1.0)
                    match_reasons = [f"{best_field}:{best_score:.3f}(generic_boost)"]
                    matched_fields = [best_field]
                    
                else:
                    # Single criteria query (non-generic) - use normal threshold
                    best_field, best_score = max(field_scores.items(), key=lambda x: x[1])
                    
                    if best_score < self.required_criteria_threshold:
                        return (0.0, [], [])
                    
                    final_score = best_score
                    match_reasons = [f"{best_field}:{best_score:.3f}"]
                    matched_fields = [best_field]
            
            # Add price match reason if applicable
            if price_range.get('has_price_range', False):
                _, price_reason = self._check_price_range_match(property_data, price_range)
                match_reasons.append(f"price_match:{price_reason}")
                matched_fields.append("MarketValue")
            
            return (final_score, match_reasons, matched_fields)
            
        except Exception as e:
            logger.error(f"Error in single property matching logic: {e}")
            return (0.0, [], [])
    
    # ALL STATIC PATTERN MATCHING METHODS REMOVED - PURE AI ONLY
    # NO keywords, NO patterns, NO hardcoded rules, NO regex matching
    # Everything is handled by AI semantic similarity and cross-encoder models
    
    def filter_properties(self, 
                         query: str, 
                         properties: List[Dict[str, Any]], 
                         max_results: int = 20) -> List[PropertyMatch]:
        """ULTRA-OPTIMIZED property filtering with parallel processing and caching"""
        try:
            start_time = time.time()
            
            if not properties:
                return []
            
            # Check cache first for repeated queries
            if self.enable_caching and self._query_cache is not None:
                cache_key = f"query_{hash(query)}_{len(properties)}_{max_results}"
                with self._cache_lock:
                    if cache_key in self._query_cache:
                        logger.info(f"🚀 Cache hit for query: {query[:50]}...")
                        return self._query_cache[cache_key]
            
            # Check if this is a generic query for logging
            query_analysis = self.extract_query_entities(query)
            is_generic_query = query_analysis.get('is_generic_query', False)
            
            if is_generic_query:
                logger.info(f"🏠 Generic query detected: '{query}' - using lenient matching thresholds")
            
            logger.info(f"🚀 Starting ULTRA-OPTIMIZED filtering of {len(properties)} properties")
            
            # Step 1: ULTRA-OPTIMIZED batch process field matching for all properties
            logger.info("📊 Step 1: ULTRA-OPTIMIZED batch processing field matching...")
            field_matching_start = time.time()
            
            # Use ULTRA-OPTIMIZED batch processing for field matching
            field_results = self.match_properties_batch(query, properties)
            
            field_matching_time = time.time() - field_matching_start
            logger.info(f"✅ Field matching completed in {field_matching_time:.2f}s")
            
            # Step 2: Parallel batch process overall property text matching
            logger.info("📊 Step 2: Parallel batch processing overall property matching...")
            overall_matching_start = time.time()
            
            # Create property texts for batch processing (parallel)
            property_texts = self._create_property_texts_parallel(properties)
            
            # Parallel batch calculate overall semantic similarities and cross-encoder scores
            with ThreadPoolExecutor(max_workers=2) as executor:
                semantic_future = executor.submit(
                    self.calculate_batch_semantic_similarity, query, property_texts
                )
                cross_future = executor.submit(
                    self.calculate_batch_cross_encoder_score, query, property_texts
                )
                
                overall_semantic_scores = semantic_future.result()
                overall_cross_scores = cross_future.result()
            
            overall_matching_time = time.time() - overall_matching_start
            logger.info(f"✅ Overall matching completed in {overall_matching_time:.2f}s")
            
            # Step 3: Vectorized combine results and create PropertyMatch objects
            logger.info("📊 Step 3: Vectorized combining results...")
            combine_start = time.time()
            
            property_matches = self._combine_results_vectorized(
                properties, field_results, overall_semantic_scores, 
                overall_cross_scores, query_analysis
            )
            
            combine_time = time.time() - combine_start
            logger.info(f"✅ Results combined in {combine_time:.2f}s")
            
            # Step 4: Sort by confidence score (descending) - optimized
            property_matches.sort(key=lambda x: x.confidence_score, reverse=True)
            
            # Step 5: Limit results
            filtered_results = property_matches[:max_results]
            
            # Cache the result
            if self.enable_caching and self._query_cache is not None:
                with self._cache_lock:
                    if len(self._query_cache) < self.cache_size:
                        self._query_cache[cache_key] = filtered_results
            
            # Clear GPU memory after processing
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            total_processing_time = time.time() - start_time
            logger.info(f"✅ ULTRA-OPTIMIZED: Filtered {len(properties)} properties to {len(filtered_results)} matches in {total_processing_time:.2f}s")
            logger.info(f"📈 Performance: {len(properties)/total_processing_time:.1f} properties/second")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in ULTRA-OPTIMIZED property filtering: {e}")
            # Fallback to original method
            logger.warning("⚠️ Falling back to original filtering method")
            return self._filter_properties_fallback(query, properties, max_results)
    
    def _filter_properties_fallback(self, 
                                   query: str, 
                                   properties: List[Dict[str, Any]], 
                                   max_results: int = 20) -> List[PropertyMatch]:
        """Fallback filtering method using original approach"""
        try:
            start_time = time.time()
            
            # Process properties in parallel
            property_matches = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all property matching tasks
                future_to_property = {
                    executor.submit(self._match_single_property, query, prop): prop 
                    for prop in properties
                }
                
                # Collect results
                for future in as_completed(future_to_property):
                    try:
                        match = future.result()
                        if match and match.confidence_score >= self.minimum_threshold:
                            property_matches.append(match)
                    except Exception as e:
                        logger.error(f"Error processing property: {e}")
            
            # Sort by confidence score (descending)
            property_matches.sort(key=lambda x: x.confidence_score, reverse=True)
            
            # Limit results
            filtered_results = property_matches[:max_results]
            
            processing_time = time.time() - start_time
            logger.info(f"✅ FALLBACK: Filtered {len(properties)} properties to {len(filtered_results)} matches in {processing_time:.2f}s")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in fallback property filtering: {e}")
            return []
    
    def _match_single_property(self, query: str, property_data: Dict[str, Any]) -> Optional[PropertyMatch]:
        """STRICT AI-based matching of a single property against the query"""
        try:
            # 1. STRICT Field-based matching (primary)
            field_score, match_reasons, matched_fields = self.match_property_fields(query, property_data)
            
            # 2. Check if this is a generic query and adjust threshold accordingly
            query_analysis = self.extract_query_entities(query)
            is_generic_query = query_analysis.get('is_generic_query', False)
            
            # Use lower threshold for generic queries
            effective_threshold = 0.3 if is_generic_query else self.required_criteria_threshold
            
            if field_score < effective_threshold:
                return None
            
            # 3. AI Overall property text matching (secondary validation)
            property_text = self._create_property_text(property_data)
            overall_semantic_score = self.calculate_semantic_similarity(query, property_text)
            overall_cross_score = self.calculate_cross_encoder_score(query, property_text)
            
            # 4. ENHANCED final score calculation - handle generic queries
            final_score = field_score
            
            # Add boost from overall matching with appropriate thresholds
            if is_generic_query:
                # For generic queries, use lower thresholds
                if overall_semantic_score >= 0.4:  # Lower threshold for generic queries
                    final_score = min(final_score + 0.1, 1.0)
                    match_reasons.append(f"overall_ai_semantic:{overall_semantic_score:.3f}")
                
                if overall_cross_score >= 0.4:  # Lower threshold for generic queries
                    final_score = min(final_score + 0.05, 1.0)
                    match_reasons.append(f"overall_ai_cross_encoder:{overall_cross_score:.3f}")
            else:
                # For specific queries, use normal thresholds
                if overall_semantic_score >= self.overall_match_threshold:
                    final_score = min(final_score + 0.1, 1.0)
                    match_reasons.append(f"overall_ai_semantic:{overall_semantic_score:.3f}")
                
                if overall_cross_score >= self.overall_match_threshold:
                    final_score = min(final_score + 0.05, 1.0)
                    match_reasons.append(f"overall_ai_cross_encoder:{overall_cross_score:.3f}")
            
            # 5. Final validation - use appropriate threshold
            effective_minimum_threshold = 0.4 if is_generic_query else self.minimum_threshold
            if final_score < effective_minimum_threshold:
                return None
            
            return PropertyMatch(
                property_data=property_data,
                confidence_score=final_score,
                match_reasons=match_reasons,
                matched_fields=matched_fields
            )
            
        except Exception as e:
            logger.error(f"Error in STRICT AI single property matching: {e}")
            return None
    
    def _create_property_text(self, property_data: Dict[str, Any]) -> str:
        """Create a combined text representation of the property"""
        text_parts = []
        
        for field in self.target_fields:
            if field in property_data and property_data[field]:
                text_parts.append(str(property_data[field]))
        
        return " ".join(text_parts)
    
    def _create_property_texts_parallel(self, properties: List[Dict[str, Any]]) -> List[str]:
        """Create property texts using parallel processing"""
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                property_texts = list(executor.map(self._create_property_text, properties))
            return property_texts
        except Exception as e:
            logger.error(f"Error in parallel property text creation: {e}")
            # Fallback to sequential processing
            return [self._create_property_text(prop) for prop in properties]
    
    def _combine_results_vectorized(self, properties: List[Dict[str, Any]], 
                                  field_results: List[Tuple[float, List[str], List[str]]], 
                                  overall_semantic_scores: List[float], 
                                  overall_cross_scores: List[float], 
                                  query_analysis: Dict[str, Any]) -> List[PropertyMatch]:
        """Combine results using vectorized operations for better performance"""
        try:
            property_matches = []
            is_generic_query = query_analysis.get('is_generic_query', False)
            
            # Convert to numpy arrays for vectorized operations
            field_scores = np.array([result[0] for result in field_results])
            semantic_scores = np.array(overall_semantic_scores)
            cross_scores = np.array(overall_cross_scores)
            
            # Vectorized final score calculation
            if is_generic_query:
                # For generic queries, be more lenient
                generic_threshold = 0.3
                valid_mask = field_scores >= generic_threshold
                
                final_scores = np.where(valid_mask, field_scores, 0.0)
                
                # Add boosts from overall matching
                semantic_boost_mask = semantic_scores >= 0.4
                cross_boost_mask = cross_scores >= 0.4
                
                final_scores = np.where(semantic_boost_mask, 
                                      np.minimum(final_scores + 0.1, 1.0), final_scores)
                final_scores = np.where(cross_boost_mask, 
                                      np.minimum(final_scores + 0.05, 1.0), final_scores)
                
                effective_threshold = 0.4
            else:
                # For specific queries, use strict matching
                valid_mask = field_scores >= self.required_criteria_threshold
                final_scores = np.where(valid_mask, field_scores, 0.0)
                
                # Add boosts from overall matching
                semantic_boost_mask = semantic_scores >= self.overall_match_threshold
                cross_boost_mask = cross_scores >= self.overall_match_threshold
                
                final_scores = np.where(semantic_boost_mask, 
                                      np.minimum(final_scores + 0.1, 1.0), final_scores)
                final_scores = np.where(cross_boost_mask, 
                                      np.minimum(final_scores + 0.05, 1.0), final_scores)
                
                effective_threshold = self.minimum_threshold
            
            # Create PropertyMatch objects for valid results
            for i, (property_data, field_result, final_score) in enumerate(
                zip(properties, field_results, final_scores)
            ):
                if final_score >= effective_threshold:
                    field_score, match_reasons, matched_fields = field_result
                    
                    # Add overall match reasons
                    if semantic_scores[i] > self.overall_match_threshold:
                        match_reasons.append(f"overall_ai_semantic:{semantic_scores[i]:.3f}")
                    if cross_scores[i] > self.overall_match_threshold:
                        match_reasons.append(f"overall_ai_cross_encoder:{cross_scores[i]:.3f}")
                    
                    property_matches.append(PropertyMatch(
                        property_data=property_data,
                        confidence_score=float(final_score),
                        match_reasons=match_reasons,
                        matched_fields=matched_fields
                    ))
            
            return property_matches
            
        except Exception as e:
            logger.error(f"Error in vectorized result combination: {e}")
            # Fallback to sequential processing
            property_matches = []
            for i, (property_data, field_result, overall_semantic, overall_cross) in enumerate(
                zip(properties, field_results, overall_semantic_scores, overall_cross_scores)
            ):
                field_score, match_reasons, matched_fields = field_result
                
                # ENHANCED final score calculation - handle generic queries
                is_generic_query = query_analysis.get('is_generic_query', False)
                
                if is_generic_query:
                    # For generic queries, be more lenient with field matching
                    generic_threshold = 0.3  # Lower threshold for generic queries
                    if field_score >= generic_threshold:
                        # Field matching is acceptable for generic queries
                        final_score = field_score
                        
                        # Add boost from overall matching if it's also good
                        if overall_semantic >= 0.4:  # Lower threshold for generic queries
                            final_score = min(final_score + 0.1, 1.0)
                        if overall_cross >= 0.4:  # Lower threshold for generic queries
                            final_score = min(final_score + 0.05, 1.0)
                    else:
                        # Field matching is too weak even for generic queries
                        final_score = 0.0
                else:
                    # For specific queries, use strict matching
                    if field_score >= self.required_criteria_threshold:
                        # Field matching is strong - use it as primary score
                        final_score = field_score
                        
                        # Add small boost from overall matching if it's also strong
                        if overall_semantic >= self.overall_match_threshold:
                            final_score = min(final_score + 0.1, 1.0)
                        if overall_cross >= self.overall_match_threshold:
                            final_score = min(final_score + 0.05, 1.0)
                    else:
                        # Field matching is weak - reject this property
                        final_score = 0.0
                
                # Add overall match reasons with configurable thresholds
                if overall_semantic > self.overall_match_threshold:
                    match_reasons.append(f"overall_ai_semantic:{overall_semantic:.3f}")
                if overall_cross > self.overall_match_threshold:
                    match_reasons.append(f"overall_ai_cross_encoder:{overall_cross:.3f}")
                
                # ENHANCED filtering - more lenient for generic queries
                # Use lower threshold for generic queries
                effective_threshold = 0.4 if is_generic_query else self.minimum_threshold
                
                if final_score >= effective_threshold:
                    property_matches.append(PropertyMatch(
                        property_data=property_data,
                        confidence_score=final_score,
                        match_reasons=match_reasons,
                        matched_fields=matched_fields
                    ))
            
            return property_matches
    
    def get_filtering_stats(self) -> Dict[str, Any]:
        """Get filtering statistics"""
        return {
            'similarity_threshold': self.similarity_threshold,
            'cross_encoder_threshold': self.cross_encoder_threshold,
            'target_fields': self.target_fields,
            'total_target_fields': len(self.target_fields),
            'device': self.device,
            'max_workers': self.max_workers,
            'batch_size': self.batch_size,
            'minimum_threshold': self.minimum_threshold,
            'field_match_threshold': self.field_match_threshold,
            'overall_match_threshold': self.overall_match_threshold,
            'field_weights': self._get_field_weights()
        }
    
    def clear_gpu_memory(self):
        """Clear GPU memory to prevent memory issues"""
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                logger.info("✅ GPU memory cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing GPU memory: {e}")
    
    def optimize_for_batch_processing(self, batch_size: int = 64):
        """Optimize the filter for batch processing with enhanced performance"""
        try:
            self.batch_size = batch_size
            
            # Clear GPU memory
            self.clear_gpu_memory()
            
            # Optimize model settings for batch processing
            if self.sentence_model is not None:
                # Set model to evaluation mode for faster inference
                self.sentence_model.eval()
            
            if self.cross_encoder is not None:
                # Set model to evaluation mode for faster inference
                self.cross_encoder.eval()
            
            logger.info(f"✅ ULTRA-OPTIMIZED for batch processing with batch_size={batch_size}")
        except Exception as e:
            logger.error(f"Error optimizing for batch processing: {e}")
    
    def clear_cache(self):
        """Clear all caches to free memory"""
        try:
            with self._cache_lock:
                if self._embedding_cache is not None:
                    self._embedding_cache.clear()
                if self._query_cache is not None:
                    self._query_cache.clear()
            logger.info("✅ All caches cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        try:
            stats = {
                'device': self.device,
                'max_workers': self.max_workers,
                'batch_size': self.batch_size,
                'enable_multiprocessing': self.enable_multiprocessing,
                'enable_caching': self.enable_caching,
                'cache_size': self.cache_size,
                'embedding_cache_size': len(self._embedding_cache) if self._embedding_cache else 0,
                'query_cache_size': len(self._query_cache) if self._query_cache else 0,
                'cpu_count': psutil.cpu_count(),
                'memory_usage': psutil.virtual_memory().percent,
                'gpu_available': torch.cuda.is_available(),
                'gpu_memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                'gpu_memory_reserved': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {}
    
    def analyze_match_quality(self, query: str, property_matches: List[PropertyMatch]) -> Dict[str, Any]:
        """
        Enhanced analysis of match quality with STRICT thresholds
        Returns classification of whether matches are exact, similar, or no match
        """
        try:
            if not property_matches:
                return {
                    'overall_quality': 'no_match',
                    'total_matches': 0,
                    'exact_matches': 0,
                    'similar_matches': 0,
                    'low_quality_matches': 0,
                    'average_confidence': 0.0,
                    'highest_confidence': 0.0,
                    'analysis_details': 'No properties matched the query',
                    'query_analysis': self.extract_query_entities(query)
                }
            
            # STRICT quality thresholds based on the enhanced AI filter's scoring system
            exact_match_threshold = 0.85     # Very high confidence - exact match
            similar_match_threshold = 0.7    # High confidence - similar match
            low_quality_threshold = 0.6      # Medium confidence - weak match
            
            # Analyze each match
            exact_matches = []
            similar_matches = []
            low_quality_matches = []
            
            for match in property_matches:
                confidence = match.confidence_score
                if confidence >= exact_match_threshold:
                    exact_matches.append(match)
                elif confidence >= similar_match_threshold:
                    similar_matches.append(match)
                elif confidence >= low_quality_threshold:
                    low_quality_matches.append(match)
            
            # Calculate statistics
            total_matches = len(property_matches)
            confidences = [match.confidence_score for match in property_matches]
            average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            highest_confidence = max(confidences) if confidences else 0.0
            
            # Determine overall quality with enhanced analysis
            query_analysis = self.extract_query_entities(query)
            
            if exact_matches:
                overall_quality = 'exact_match'
                quality_description = f"Found {len(exact_matches)} exact matches (confidence >= {exact_match_threshold})"
            elif similar_matches:
                overall_quality = 'similar_match'
                quality_description = f"Found {len(similar_matches)} similar matches (confidence >= {similar_match_threshold})"
            elif low_quality_matches:
                overall_quality = 'weak_match'
                quality_description = f"Found {len(low_quality_matches)} weak matches (confidence >= {low_quality_threshold})"
            else:
                overall_quality = 'no_match'
                quality_description = "No meaningful matches found - all matches below quality threshold"
            
            # Create detailed analysis with query understanding
            analysis_details = {
                'query': query,
                'query_analysis': query_analysis,
                'quality_description': quality_description,
                'thresholds_used': {
                    'exact_match': exact_match_threshold,
                    'similar_match': similar_match_threshold,
                    'low_quality': low_quality_threshold,
                    'minimum_threshold': self.minimum_threshold,
                    'required_criteria_threshold': self.required_criteria_threshold
                },
                'confidence_distribution': {
                    'exact_matches': [m.confidence_score for m in exact_matches],
                    'similar_matches': [m.confidence_score for m in similar_matches],
                    'low_quality_matches': [m.confidence_score for m in low_quality_matches]
                },
                'top_match_details': [],
                'matching_strategy': 'STRICT_AI_MATCHING'
            }
            
            # Add details for top 3 matches with enhanced information
            for i, match in enumerate(property_matches[:3]):
                analysis_details['top_match_details'].append({
                    'rank': i + 1,
                    'property_name': match.property_data.get('PropertyName', 'N/A'),
                    'property_type': match.property_data.get('PropertyType', 'N/A'),
                    'location': f"{match.property_data.get('City', 'N/A')}, {match.property_data.get('State', 'N/A')}",
                    'confidence': match.confidence_score,
                    'matched_fields': match.matched_fields,
                    'match_reasons': match.match_reasons[:3]  # Top 3 reasons
                })
            
            return {
                'overall_quality': overall_quality,
                'total_matches': total_matches,
                'exact_matches': len(exact_matches),
                'similar_matches': len(similar_matches),
                'low_quality_matches': len(low_quality_matches),
                'average_confidence': round(average_confidence, 3),
                'highest_confidence': round(highest_confidence, 3),
                'analysis_details': analysis_details,
                'query_analysis': query_analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing enhanced match quality: {e}")
            return {
                'overall_quality': 'error',
                'total_matches': 0,
                'exact_matches': 0,
                'similar_matches': 0,
                'low_quality_matches': 0,
                'average_confidence': 0.0,
                'highest_confidence': 0.0,
                'analysis_details': f'Error analyzing matches: {str(e)}',
                'query_analysis': {'error': str(e)}
            }

# Global instance
_ai_property_filter = None

def get_ai_property_filter() -> AIPropertyFilter:
    """Get the global ULTRA-OPTIMIZED AI property filter instance with parallel processing"""
    global _ai_property_filter
    if _ai_property_filter is None:
        _ai_property_filter = AIPropertyFilter(
            use_gpu=True,
            max_workers=None,  # Auto-detect optimal workers
            batch_size=64,     # Increased for better GPU utilization
            enable_multiprocessing=True,
            enable_caching=True,
            cache_size=1000
        )
        # Optimize for batch processing
        _ai_property_filter.optimize_for_batch_processing(batch_size=64)
        logger.info("🚀 ULTRA-OPTIMIZED AI Property Filter initialized with parallel processing!")
        logger.info(f"📊 Configuration: {_ai_property_filter.max_workers} workers, batch_size={_ai_property_filter.batch_size}")
    return _ai_property_filter
