"""
AI Post-Processor for Property Search Results

This module provides semantic filtering of property search results to ensure
they exactly match the user query intent. It uses embeddings and cross-encoder
scoring to perform context-aware matching without relying on hard-coded keywords.

Key Features:
- Semantic matching using sentence transformers
- Cross-encoder scoring for precise relevance
- Focus on core property fields (PropertyName, Address, City, State, Country, PropertyType)
- Precision-focused filtering (exclude uncertain matches)
- Context-aware query intent analysis
"""

import logging
import re
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import torch
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class QueryIntent:
    """Represents the extracted intent from user query"""
    locations: List[str]
    property_types: List[str]
    size_requirements: List[str]  # BHK, rooms, etc.
    features: List[str]
    price_indicators: List[str]
    urgency: bool
    confidence: float

@dataclass
class PropertyMatch:
    """Represents a property match with semantic scores"""
    property_data: Dict[str, Any]
    overall_score: float
    location_score: float
    type_score: float
    size_score: float
    feature_score: float
    confidence: float
    match_reason: str

class AIPropertyPostProcessor:
    """
    AI-powered post-processor for property search results.
    
    Performs semantic matching between user queries and retrieved properties
    to ensure results exactly match user intent.
    """
    
    def __init__(self, 
                 embedding_model = None,  # Can be string or model instance
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 use_gpu: bool = True,
                 similarity_threshold: float = 0.7,
                 confidence_threshold: float = 0.6):
        """
        Initialize the AI post-processor.
        
        Args:
            embedding_model: Model for generating embeddings
            cross_encoder_model: Model for cross-encoder scoring
            use_gpu: Whether to use GPU acceleration
            similarity_threshold: Minimum similarity score for matches
            confidence_threshold: Minimum confidence for including results
        """
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        
        # Initialize models
        try:
            # Check if embedding_model is already a model instance or a string
            if embedding_model is None:
                # Default model
                self.embedding_model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
                self.embedding_model = self.embedding_model.to(self.device)
                logger.info("AI Post-Processor: Loaded new Jina model instance")
            elif hasattr(embedding_model, 'encode'):
                # It's already a model instance, reuse it
                self.embedding_model = embedding_model
                logger.info("AI Post-Processor: Reusing existing model instance (memory efficient)")
            else:
                # It's a string, load the model
                self.embedding_model = SentenceTransformer(embedding_model, trust_remote_code=True)
                self.embedding_model = self.embedding_model.to(self.device)
                logger.info(f"AI Post-Processor: Loaded model from string: {embedding_model}")
            
            # Only load cross-encoder if we need it (lazy loading)
            self.cross_encoder = None
            self.cross_encoder_model_name = cross_encoder_model
            self.cross_encoder_loaded = False
                
            logger.info(f"AI Post-Processor initialized with device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing AI Post-Processor: {e}")
            raise
        
        # Core property fields to focus on
        self.core_fields = {
            'PropertyName', 'Address', 'City', 'State', 'Country', 'PropertyType'
        }
        
        # Property type mappings for semantic matching
        self.property_type_mappings = {
            'flat': ['flat', 'apartment', 'residential', 'home'],
            'villa': ['villa', 'house', 'bungalow', 'mansion'],
            'pg': ['pg', 'paying guest', 'hostel', 'accommodation'],
            'office': ['office', 'commercial', 'workspace', 'business'],
            'shop': ['shop', 'retail', 'store', 'commercial space']
        }
        
        # Location normalization patterns
        self.location_patterns = {
            'mumbai': ['mumbai', 'bombay', 'navi mumbai', 'thane'],
            'delhi': ['delhi', 'new delhi', 'ncr', 'gurgaon', 'noida'],
            'bangalore': ['bangalore', 'bengaluru', 'electronic city', 'whitefield'],
            'pune': ['pune', 'hinjewadi', 'wakad', 'kharadi'],
            'hyderabad': ['hyderabad', 'secunderabad', 'hitech city', 'gachibowli', 'madhapur']
        }
        
        # Performance tracking
        self.stats = {
            'total_processed': 0,
            'total_filtered': 0,
            'avg_processing_time': 0.0,
            'filter_reasons': defaultdict(int)
        }
    
    def process_results(self, 
                       query: str, 
                       retrieved_properties: List[Dict[str, Any]], 
                       top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Process retrieved properties to ensure they match query intent.
        
        Args:
            query: Original user query
            retrieved_properties: List of properties from retriever
            top_k: Maximum number of results to return
            
        Returns:
            List of filtered properties that match query intent
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing {len(retrieved_properties)} properties for query: '{query}'")
            
            # Step 1: Extract query intent
            query_intent = self._extract_query_intent(query)
            logger.info(f"Query intent: {query_intent}")
            
            # Step 2: Filter properties based on semantic matching
            matched_properties = []
            
            for prop_data in retrieved_properties:
                property_match = self._evaluate_property_match(
                    query, query_intent, prop_data
                )
                
                if property_match and property_match.confidence >= self.confidence_threshold:
                    matched_properties.append(property_match)
                    logger.debug(f"Property matched: {property_match.property_data.get('PropertyName', 'Unknown')} "
                               f"(Score: {property_match.overall_score:.3f}, Confidence: {property_match.confidence:.3f})")
                else:
                    reason = property_match.match_reason if property_match else "Low confidence"
                    self.stats['filter_reasons'][reason] += 1
                    logger.debug(f"Property filtered: {prop_data.get('PropertyName', 'Unknown')} - {reason}")
            
            # Step 3: Sort by overall score and return top_k
            matched_properties.sort(key=lambda x: x.overall_score, reverse=True)
            final_results = [match.property_data for match in matched_properties[:top_k]]
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(len(retrieved_properties), len(final_results), processing_time)
            
            logger.info(f"Post-processing complete: {len(final_results)}/{len(retrieved_properties)} properties match query intent")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
            # Return empty list on error to ensure precision
            return []
    
    def _extract_query_intent(self, query: str) -> QueryIntent:
        """Extract intent from user query using semantic analysis."""
        try:
            query_lower = query.lower()
            
            # Extract locations
            locations = self._extract_locations(query_lower)
            
            # Extract property types
            property_types = self._extract_property_types(query_lower)
            
            # Extract size requirements (BHK, rooms, etc.)
            size_requirements = self._extract_size_requirements(query_lower)
            
            # Extract features
            features = self._extract_features(query_lower)
            
            # Extract price indicators
            price_indicators = self._extract_price_indicators(query_lower)
            
            # Check for urgency
            urgency = any(word in query_lower for word in ['urgent', 'immediate', 'asap', 'quick'])
            
            # Calculate confidence based on extracted entities
            confidence = self._calculate_intent_confidence(
                locations, property_types, size_requirements, features
            )
            
            return QueryIntent(
                locations=locations,
                property_types=property_types,
                size_requirements=size_requirements,
                features=features,
                price_indicators=price_indicators,
                urgency=urgency,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error extracting query intent: {e}")
            return QueryIntent([], [], [], [], [], False, 0.0)
    
    def _extract_locations(self, query: str) -> List[str]:
        """Extract location entities from query."""
        locations = []
        
        # Check for known city patterns
        for city, patterns in self.location_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    locations.append(city)
                    break
        
        # Extract area/neighborhood names
        area_patterns = [
            r'\b(\w+)\s+(area|neighborhood|district|colony|nagar|pur|nagar)\b',
            r'\b(near|close to|around)\s+(\w+)\b',
            r'\b(\w+)\s+(road|street|lane|avenue)\b'
        ]
        
        for pattern in area_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if isinstance(match, tuple):
                    locations.extend([m for m in match if m and len(m) > 2])
                else:
                    if match and len(match) > 2:
                        locations.append(match)
        
        return list(set(locations))
    
    def _extract_property_types(self, query: str) -> List[str]:
        """Extract property type entities from query."""
        property_types = []
        
        for prop_type, patterns in self.property_type_mappings.items():
            for pattern in patterns:
                if pattern in query:
                    property_types.append(prop_type)
                    break
        
        return list(set(property_types))
    
    def _extract_size_requirements(self, query: str) -> List[str]:
        """Extract size requirements (BHK, rooms, etc.) from query."""
        size_requirements = []
        
        # BHK patterns
        bhk_patterns = [
            r'\b(\d+)\s*(bhk|bedroom|room)\b',
            r'\b(\d+)\s*(bed|bath)\b'
        ]
        
        for pattern in bhk_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if isinstance(match, tuple):
                    size_requirements.extend([m for m in match if m.isdigit()])
                else:
                    if match.isdigit():
                        size_requirements.append(match)
        
        return list(set(size_requirements))
    
    def _extract_features(self, query: str) -> List[str]:
        """Extract feature requirements from query."""
        features = []
        
        feature_keywords = [
            'furnished', 'semi-furnished', 'unfurnished',
            'wifi', 'internet', 'ac', 'air conditioning',
            'parking', 'balcony', 'garden', 'gym',
            'swimming pool', 'power backup', 'security'
        ]
        
        for feature in feature_keywords:
            if feature in query:
                features.append(feature)
        
        return features
    
    def _extract_price_indicators(self, query: str) -> List[str]:
        """Extract price-related indicators from query."""
        price_indicators = []
        
        price_patterns = [
            r'\b(under|below|less than|upto)\s*([\d,]+)\b',
            r'\b(over|above|more than)\s*([\d,]+)\b',
            r'\b(between|from)\s*([\d,]+)\s*(to|-)\s*([\d,]+)\b',
            r'\b(budget|affordable|cheap|economical|premium|luxury)\b'
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if isinstance(match, tuple):
                    price_indicators.extend([m for m in match if m])
                else:
                    if match:
                        price_indicators.append(match)
        
        return price_indicators
    
    def _calculate_intent_confidence(self, locations: List[str], property_types: List[str], 
                                   size_requirements: List[str], features: List[str]) -> float:
        """Calculate confidence in extracted intent."""
        try:
            # Base confidence from entity extraction
            total_entities = len(locations) + len(property_types) + len(size_requirements) + len(features)
            
            if total_entities == 0:
                return 0.3  # Low confidence for vague queries
            
            # Normalize confidence based on entity count
            confidence = min(1.0, total_entities / 4.0)  # Max confidence with 4+ entities
            
            # Boost confidence for specific entities
            if locations:
                confidence += 0.2
            if property_types:
                confidence += 0.2
            if size_requirements:
                confidence += 0.1
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating intent confidence: {e}")
            return 0.5
    
    def _evaluate_property_match(self, query: str, query_intent: QueryIntent, 
                               property_data: Dict[str, Any]) -> Optional[PropertyMatch]:
        """Evaluate if a property matches the query intent."""
        try:
            # Extract core property information
            property_text = self._extract_property_text(property_data)
            
            # Calculate semantic similarity scores
            location_score = self._calculate_location_score(query_intent, property_data)
            type_score = self._calculate_type_score(query_intent, property_data)
            size_score = self._calculate_size_score(query_intent, property_data)
            feature_score = self._calculate_feature_score(query_intent, property_data)
            
            # Calculate overall semantic similarity
            overall_score = self._calculate_semantic_similarity(query, property_text)
            
            # Calculate cross-encoder score for precision
            cross_encoder_score = self._calculate_cross_encoder_score(query, property_text)
            
            # Combine scores with weights
            weighted_score = (
                overall_score * 0.3 +
                cross_encoder_score * 0.4 +
                location_score * 0.15 +
                type_score * 0.1 +
                size_score * 0.03 +
                feature_score * 0.02
            )
            
            # Determine confidence and match reason
            confidence, match_reason = self._determine_confidence_and_reason(
                weighted_score, location_score, type_score, query_intent, property_data
            )
            
            return PropertyMatch(
                property_data=property_data,
                overall_score=weighted_score,
                location_score=location_score,
                type_score=type_score,
                size_score=size_score,
                feature_score=feature_score,
                confidence=confidence,
                match_reason=match_reason
            )
            
        except Exception as e:
            logger.error(f"Error evaluating property match: {e}")
            return None
    
    def _extract_property_text(self, property_data: Dict[str, Any]) -> str:
        """Extract and format core property information as text."""
        try:
            # Focus only on core fields
            text_parts = []
            
            for field in self.core_fields:
                value = property_data.get(field, '')
                if value and str(value).strip() and str(value).lower() != 'n/a':
                    text_parts.append(f"{field}: {value}")
            
            # Add additional relevant fields if available
            additional_fields = ['Beds', 'Baths', 'Description']
            for field in additional_fields:
                value = property_data.get(field, '')
                if value and str(value).strip() and str(value).lower() != 'n/a':
                    text_parts.append(f"{field}: {value}")
            
            return " ".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting property text: {e}")
            return ""
    
    def _calculate_location_score(self, query_intent: QueryIntent, property_data: Dict[str, Any]) -> float:
        """Calculate location matching score."""
        try:
            if not query_intent.locations:
                return 0.5  # Neutral score if no location specified
            
            property_location_text = " ".join([
                str(property_data.get(field, '')).lower() 
                for field in ['City', 'State', 'Address', 'Country']
                if property_data.get(field)
            ])
            
            if not property_location_text:
                return 0.0
            
            # Check for exact matches
            for query_location in query_intent.locations:
                if query_location.lower() in property_location_text:
                    return 1.0
            
            # Calculate semantic similarity for location
            query_location_text = " ".join(query_intent.locations)
            location_similarity = self._calculate_semantic_similarity(
                query_location_text, property_location_text
            )
            
            return location_similarity
            
        except Exception as e:
            logger.error(f"Error calculating location score: {e}")
            return 0.0
    
    def _calculate_type_score(self, query_intent: QueryIntent, property_data: Dict[str, Any]) -> float:
        """Calculate property type matching score."""
        try:
            if not query_intent.property_types:
                return 0.5  # Neutral score if no type specified
            
            property_type = str(property_data.get('PropertyType', '')).lower()
            property_name = str(property_data.get('PropertyName', '')).lower()
            
            if not property_type and not property_name:
                return 0.0
            
            # Check for exact matches
            for query_type in query_intent.property_types:
                if query_type in property_type or query_type in property_name:
                    return 1.0
            
            # Calculate semantic similarity for type
            query_type_text = " ".join(query_intent.property_types)
            property_type_text = f"{property_type} {property_name}"
            
            type_similarity = self._calculate_semantic_similarity(
                query_type_text, property_type_text
            )
            
            return type_similarity
            
        except Exception as e:
            logger.error(f"Error calculating type score: {e}")
            return 0.0
    
    def _calculate_size_score(self, query_intent: QueryIntent, property_data: Dict[str, Any]) -> float:
        """Calculate size requirement matching score."""
        try:
            if not query_intent.size_requirements:
                return 0.5  # Neutral score if no size specified
            
            property_beds = str(property_data.get('Beds', '')).lower()
            property_name = str(property_data.get('PropertyName', '')).lower()
            
            # Check for BHK matches in property name
            for size_req in query_intent.size_requirements:
                if size_req in property_name:
                    return 1.0
                if property_beds and size_req == property_beds:
                    return 1.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating size score: {e}")
            return 0.0
    
    def _calculate_feature_score(self, query_intent: QueryIntent, property_data: Dict[str, Any]) -> float:
        """Calculate feature matching score."""
        try:
            if not query_intent.features:
                return 0.5  # Neutral score if no features specified
            
            property_description = str(property_data.get('Description', '')).lower()
            property_features = str(property_data.get('KeyFeatures', '')).lower()
            
            if not property_description and not property_features:
                return 0.0
            
            # Check for feature matches
            matches = 0
            for feature in query_intent.features:
                if feature in property_description or feature in property_features:
                    matches += 1
            
            return matches / len(query_intent.features) if query_intent.features else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating feature score: {e}")
            return 0.0
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using embeddings."""
        try:
            if not text1 or not text2:
                return 0.0
            
            # Generate embeddings
            embeddings = self.embedding_model.encode([text1, text2], convert_to_numpy=True)
            
            # Calculate cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _calculate_cross_encoder_score(self, query: str, property_text: str) -> float:
        """Calculate cross-encoder score for precise relevance."""
        try:
            if not query or not property_text:
                return 0.0
            
            # Lazy load cross-encoder only when needed
            if not self.cross_encoder_loaded:
                self._load_cross_encoder()
            
            if self.cross_encoder is None:
                return 0.0
            
            # Use cross-encoder for precise scoring
            score = self.cross_encoder.predict([(query, property_text)])
            return float(score[0])
            
        except Exception as e:
            logger.error(f"Error calculating cross-encoder score: {e}")
            return 0.0
    
    def _load_cross_encoder(self):
        """Lazy load the cross-encoder model."""
        try:
            if not self.cross_encoder_loaded:
                logger.info("Loading cross-encoder model (lazy loading)...")
                self.cross_encoder = CrossEncoder(self.cross_encoder_model_name)
                if self.device == 'cuda' and torch.cuda.is_available():
                    self.cross_encoder = self.cross_encoder.to(self.device)
                self.cross_encoder_loaded = True
                logger.info("Cross-encoder model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading cross-encoder: {e}")
            self.cross_encoder = None
            self.cross_encoder_loaded = True  # Mark as loaded to avoid retrying
    
    def _determine_confidence_and_reason(self, weighted_score: float, location_score: float, 
                                       type_score: float, query_intent: QueryIntent, 
                                       property_data: Dict[str, Any]) -> Tuple[float, str]:
        """Determine confidence level and reason for match/no-match."""
        try:
            # Base confidence from weighted score
            confidence = weighted_score
            
            # Adjust confidence based on critical factors
            if query_intent.locations and location_score < 0.3:
                confidence *= 0.3  # Heavy penalty for location mismatch
                reason = "Location mismatch"
            elif query_intent.property_types and type_score < 0.3:
                confidence *= 0.5  # Penalty for type mismatch
                reason = "Property type mismatch"
            elif weighted_score < self.similarity_threshold:
                confidence *= 0.7  # Penalty for low overall similarity
                reason = "Low semantic similarity"
            else:
                reason = "Good match"
            
            # Additional checks for precision
            property_city = str(property_data.get('City', '')).lower()
            property_type = str(property_data.get('PropertyType', '')).lower()
            
            # Strict location matching
            if query_intent.locations:
                location_match = any(
                    loc.lower() in property_city or 
                    loc.lower() in str(property_data.get('Address', '')).lower()
                    for loc in query_intent.locations
                )
                if not location_match:
                    confidence *= 0.1
                    reason = "No location match found"
            
            # Strict type matching
            if query_intent.property_types:
                type_match = any(
                    prop_type in property_type or 
                    prop_type in str(property_data.get('PropertyName', '')).lower()
                    for prop_type in query_intent.property_types
                )
                if not type_match:
                    confidence *= 0.2
                    reason = "No property type match found"
            
            return min(1.0, max(0.0, confidence)), reason
            
        except Exception as e:
            logger.error(f"Error determining confidence: {e}")
            return 0.0, "Error in evaluation"
    
    def _update_stats(self, total_processed: int, total_filtered: int, processing_time: float):
        """Update processing statistics."""
        try:
            self.stats['total_processed'] += total_processed
            self.stats['total_filtered'] += total_filtered
            
            # Update average processing time
            current_avg = self.stats['avg_processing_time']
            total_runs = self.stats['total_processed'] // max(1, total_processed)
            self.stats['avg_processing_time'] = (
                (current_avg * (total_runs - 1) + processing_time) / total_runs
            )
            
        except Exception as e:
            logger.error(f"Error updating stats: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self.stats,
            'filter_rate': (
                (self.stats['total_processed'] - self.stats['total_filtered']) / 
                max(1, self.stats['total_processed'])
            ),
            'device': self.device,
            'similarity_threshold': self.similarity_threshold,
            'confidence_threshold': self.confidence_threshold
        }
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            'total_processed': 0,
            'total_filtered': 0,
            'avg_processing_time': 0.0,
            'filter_reasons': defaultdict(int)
        }
    
    def cleanup_memory(self):
        """Clean up memory by clearing caches and unused models."""
        try:
            # Clear cross-encoder if loaded
            if self.cross_encoder is not None:
                del self.cross_encoder
                self.cross_encoder = None
                self.cross_encoder_loaded = False
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("AI Post-Processor memory cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
    
    def __del__(self):
        """Destructor to clean up resources."""
        try:
            self.cleanup_memory()
        except:
            pass  # Ignore errors during cleanup
