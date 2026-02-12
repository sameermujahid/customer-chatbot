import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from transformers import pipeline
import numpy as np
from sentence_transformers import SentenceTransformer
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    SEARCH_PROPERTY = "search_property"
    GET_DETAILS = "get_details"
    COMPARE_PROPERTIES = "compare_properties"
    GET_RECOMMENDATIONS = "get_recommendations"
    SET_LOCATION = "set_location"
    GET_NEARBY = "get_nearby"
    FILTER_PROPERTIES = "filter_properties"
    GENERAL_QUESTION = "general_question"

class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

@dataclass
class ProcessedQuery:
    original_query: str
    processed_query: str
    intent: QueryIntent
    complexity: QueryComplexity
    entities: Dict[str, Any]
    filters: Dict[str, Any]
    context: Dict[str, Any]
    confidence: float
    metadata: Dict[str, Any]

class AdvancedQueryProcessor:
    """
    Advanced query processor with expansion, context injection, and intent classification
    """
    
    def __init__(self, 
                 model_name: str = "jinaai/jina-embeddings-v3",
                 use_gpu: bool = True):
        
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        if use_gpu:
            self.model = self.model.to('cuda')
        
        # Intent classification
        self.intent_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if use_gpu else -1
        )
        
        # Query expansion patterns
        self.expansion_patterns = self._initialize_expansion_patterns()
        
        # Entity extraction patterns
        self.entity_patterns = self._initialize_entity_patterns()
        
        # Context management - REMOVED to prevent session sharing
        # self.conversation_contexts = defaultdict(lambda: deque(maxlen=10))
        
        # Performance tracking
        self.processing_stats = {
            'total_queries': 0,
            'avg_processing_time': 0.0,
            'intent_distribution': defaultdict(int)
        }
        
        logger.info("AdvancedQueryProcessor initialized")
    
    def _initialize_expansion_patterns(self) -> Dict[str, List[str]]:
        """Initialize query expansion patterns"""
        return {
            'location': {
                'patterns': [
                    r'(near|close to|around|in)\s+(\w+)',
                    r'(\w+)\s+(area|neighborhood|district)',
                    r'(\w+)\s+(city|town)'
                ],
                'expansions': {
                    'mumbai': ['mumbai', 'bombay', 'navi mumbai', 'thane'],
                    'delhi': ['delhi', 'new delhi', 'ncr', 'gurgaon', 'noida'],
                    'bangalore': ['bangalore', 'bengaluru', 'electronic city', 'whitefield'],
                    'pune': ['pune', 'hinjewadi', 'wakad', 'kharadi'],
                    'hyderabad': ['hyderabad', 'secunderabad', 'hitech city', 'gachibowli']
                }
            },
            'property_type': {
                'patterns': [
                    r'(pg|paying guest|hostel|apartment|flat|house|villa|office|shop)',
                    r'(\d+)\s*(bhk|bedroom|room)',
                    r'(furnished|semi-furnished|unfurnished)'
                ],
                'expansions': {
                    'pg': ['pg', 'paying guest', 'hostel', 'accommodation'],
                    'apartment': ['apartment', 'flat', 'residential', 'home'],
                    'office': ['office', 'commercial', 'workspace', 'business'],
                    'shop': ['shop', 'retail', 'store', 'commercial']
                }
            },
            'features': {
                'patterns': [
                    r'(wifi|internet|ac|air conditioning|parking|power backup)',
                    r'(furnished|semi-furnished|unfurnished)',
                    r'(balcony|garden|gym|swimming pool)'
                ],
                'expansions': {
                    'wifi': ['wifi', 'internet', 'wireless', 'high-speed internet'],
                    'ac': ['ac', 'air conditioning', 'cooling', 'climate control'],
                    'parking': ['parking', 'garage', 'car space', 'vehicle parking'],
                    'power': ['power backup', 'generator', 'ups', 'inverter']
                }
            },
            'price': {
                'patterns': [
                    r'(under|below|less than|upto)\s*([\d,]+)',
                    r'(over|above|more than)\s*([\d,]+)',
                    r'(between|from)\s*([\d,]+)\s*(to|-)\s*([\d,]+)'
                ],
                'expansions': {
                    'budget': ['budget', 'affordable', 'cheap', 'economical'],
                    'premium': ['premium', 'luxury', 'high-end', 'expensive']
                }
            }
        }
    
    def _initialize_entity_patterns(self) -> Dict[str, List[str]]:
        """Initialize entity extraction patterns"""
        return {
            'location': [
                r'\b(mumbai|delhi|bangalore|pune|hyderabad|chennai|kolkata|ahmedabad)\b',
                r'\b(\w+)\s+(city|town|area|district)\b',
                r'\b(near|close to|around)\s+(\w+)\b'
            ],
            'price_range': [
                r'\b(under|below|less than|upto)\s*([\d,]+)\b',
                r'\b(over|above|more than)\s*([\d,]+)\b',
                r'\b(between|from)\s*([\d,]+)\s*(to|-)\s*([\d,]+)\b'
            ],
            'property_type': [
                r'\b(pg|paying guest|hostel|apartment|flat|house|villa|office|shop)\b',
                r'\b(\d+)\s*(bhk|bedroom|room)\b',
                r'\b(furnished|semi-furnished|unfurnished)\b'
            ],
            'features': [
                r'\b(wifi|internet|ac|air conditioning|parking|power backup)\b',
                r'\b(balcony|garden|gym|swimming pool)\b',
                r'\b(furnished|semi-furnished|unfurnished)\b'
            ],
            'urgency': [
                r'\b(urgent|immediate|asap|quick|fast)\b',
                r'\b(need|want|looking for)\b'
            ]
        }
    
    def process_query(self, 
                     query: str, 
                     session_id: Optional[str] = None,
                     conversation_history: Optional[List[Dict]] = None,
                     user_context: Optional[Dict] = None) -> ProcessedQuery:
        """Process query with advanced techniques"""
        
        start_time = time.time()
        
        try:
            # Store original query
            original_query = query
            
            # Add to conversation context - REMOVED to prevent session sharing
            # if session_id:
            #     self.conversation_contexts[session_id].append({
            #         'query': query,
            #         'timestamp': time.time()
            #     })
            
            # Step 1: Basic preprocessing
            cleaned_query = self._preprocess_query(query)
            
            # Step 2: Intent classification
            intent, intent_confidence = self._classify_intent(cleaned_query)
            
            # Step 3: Entity extraction
            entities = self._extract_entities(cleaned_query)
            
            # Step 4: Query expansion
            expanded_query = self._expand_query(cleaned_query, entities)
            
            # Step 5: Context injection
            # NO CONVERSATION STORING: Removed conversation_history to reduce processing time
            context_query = self._inject_context(expanded_query, session_id, None, user_context)
            
            # Step 6: Filter extraction
            filters = self._extract_filters(context_query, entities)
            
            # Step 7: Complexity assessment
            complexity = self._assess_complexity(context_query, entities)
            
            # Step 8: Final processing
            processed_query = self._finalize_query(context_query, intent, entities)
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(intent_confidence, entities, complexity)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_processing_stats(processing_time, intent)
            
            result = ProcessedQuery(
                original_query=original_query,
                processed_query=processed_query,
                intent=intent,
                complexity=complexity,
                entities=entities,
                filters=filters,
                context={
                    'session_id': session_id,
                    'conversation_history': None, # Disabled to reduce processing time
                    'user_context': user_context
                },
                confidence=confidence,
                metadata={
                    'processing_time': processing_time,
                    'intent_confidence': intent_confidence,
                    'expansion_applied': expanded_query != cleaned_query
                }
            )
            
            logger.info(f"Query processed: {original_query} -> {processed_query} (Intent: {intent.value}, Confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            # Return fallback result
            return ProcessedQuery(
                original_query=query,
                processed_query=query,
                intent=QueryIntent.GENERAL_QUESTION,
                complexity=QueryComplexity.SIMPLE,
                entities={},
                filters={},
                context={},
                confidence=0.5,
                metadata={'error': str(e)}
            )
    
    def _preprocess_query(self, query: str) -> str:
        """Basic query preprocessing"""
        # Convert to lowercase
        query = query.lower()
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        # Remove special characters but keep important ones
        query = re.sub(r'[^\w\s\-.,₹$]', ' ', query)
        
        # Normalize numbers
        query = re.sub(r'(\d+),(\d+)', r'\1\2', query)  # Remove commas from numbers
        
        return query
    
    def _classify_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Classify query intent using zero-shot classification"""
        try:
            intent_candidates = [
                "search for properties",
                "get property details",
                "compare properties",
                "get recommendations",
                "set location",
                "find nearby properties",
                "filter properties",
                "ask general question"
            ]
            
            result = self.intent_classifier(query, intent_candidates)
            
            # Map to QueryIntent enum
            intent_mapping = {
                "search for properties": QueryIntent.SEARCH_PROPERTY,
                "get property details": QueryIntent.GET_DETAILS,
                "compare properties": QueryIntent.COMPARE_PROPERTIES,
                "get recommendations": QueryIntent.GET_RECOMMENDATIONS,
                "set location": QueryIntent.SET_LOCATION,
                "find nearby properties": QueryIntent.GET_NEARBY,
                "filter properties": QueryIntent.FILTER_PROPERTIES,
                "ask general question": QueryIntent.GENERAL_QUESTION
            }
            
            top_intent = result['labels'][0]
            confidence = result['scores'][0]
            
            return intent_mapping.get(top_intent, QueryIntent.GENERAL_QUESTION), confidence
            
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            return QueryIntent.GENERAL_QUESTION, 0.5
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities from query"""
        entities = {
            'locations': [],
            'price_ranges': [],
            'property_types': [],
            'features': [],
            'urgency': False,
            'numbers': []
        }
        
        try:
            # Extract locations
            for pattern in self.entity_patterns['location']:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    entities['locations'].append(match.group())
            
            # Extract price ranges
            for pattern in self.entity_patterns['price_range']:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    entities['price_ranges'].append(match.group())
            
            # Extract property types
            for pattern in self.entity_patterns['property_type']:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    entities['property_types'].append(match.group())
            
            # Extract features
            for pattern in self.entity_patterns['features']:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    entities['features'].append(match.group())
            
            # Check for urgency
            for pattern in self.entity_patterns['urgency']:
                if re.search(pattern, query, re.IGNORECASE):
                    entities['urgency'] = True
                    break
            
            # Extract numbers
            numbers = re.findall(r'\d+', query)
            entities['numbers'] = [int(num) for num in numbers]
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
        
        return entities
    
    def _expand_query(self, query: str, entities: Dict[str, Any]) -> str:
        """Expand query with synonyms and related terms"""
        expanded_terms = []
        
        try:
            # Location expansion
            for location in entities['locations']:
                for city, expansions in self.expansion_patterns['location']['expansions'].items():
                    if city.lower() in location.lower():
                        expanded_terms.extend(expansions)
            
            # Property type expansion
            for prop_type in entities['property_types']:
                for type_key, expansions in self.expansion_patterns['property_type']['expansions'].items():
                    if type_key.lower() in prop_type.lower():
                        expanded_terms.extend(expansions)
            
            # Feature expansion
            for feature in entities['features']:
                for feature_key, expansions in self.expansion_patterns['features']['expansions'].items():
                    if feature_key.lower() in feature.lower():
                        expanded_terms.extend(expansions)
            
            # Price expansion
            for price_term in entities['price_ranges']:
                for price_key, expansions in self.expansion_patterns['price']['expansions'].items():
                    if price_key.lower() in price_term.lower():
                        expanded_terms.extend(expansions)
            
            # Add expanded terms to query
            if expanded_terms:
                unique_terms = list(set(expanded_terms))
                expanded_query = f"{query} {' '.join(unique_terms)}"
                return expanded_query
            
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
        
        return query
    
    def _inject_context(self, 
                       query: str, 
                       session_id: Optional[str],
                       conversation_history: Optional[List[Dict]],
                       user_context: Optional[Dict]) -> str:
        """Inject context from conversation history and user preferences"""
        
        context_terms = []
        
        try:
            # Add conversation history context - REMOVED to prevent session sharing
            # if session_id and self.conversation_contexts[session_id]:
            #     recent_queries = list(self.conversation_contexts[session_id])[-3:]  # Last 3 queries
            #     for conv in recent_queries:
            #         # Extract key terms from recent queries
            #         recent_entities = self._extract_entities(conv['query'])
            #         if recent_entities['locations']:
            #             context_terms.extend(recent_entities['locations'])
            #         if recent_entities['property_types']:
            #             context_terms.extend(recent_entities['property_types'])
            
            # Add user context
            if user_context:
                if user_context.get('preferred_location'):
                    context_terms.append(user_context['preferred_location'])
                if user_context.get('preferred_property_type'):
                    context_terms.append(user_context['preferred_property_type'])
                if user_context.get('budget_range'):
                    context_terms.append(user_context['budget_range'])
            
            # NO CONVERSATION STORING: Removed conversation history context to reduce processing time
            # if conversation_history:
            #     for msg in conversation_history[-5:]:  # Last 5 messages
            #         if msg.get('type') == 'user':
            #             msg_entities = self._extract_entities(msg.get('content', ''))
            #             if msg_entities['locations']:
            #                 context_terms.extend(msg_entities['locations'])
            #             if msg_entities['property_types']:
            #                 context_terms.extend(msg_entities['property_types'])
            
            # Add context terms to query
            if context_terms:
                unique_context = list(set(context_terms))
                context_query = f"{query} {' '.join(unique_context)}"
                return context_query
            
        except Exception as e:
            logger.error(f"Error injecting context: {e}")
        
        return query
    
    def _extract_filters(self, query: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Extract filters from query and entities"""
        filters = {
            'location': None,
            'price_min': None,
            'price_max': None,
            'property_type': None,
            'features': [],
            'bhk': None,
            'furnished': None,
            'urgency': False
        }
        
        try:
            # Location filter
            if entities['locations']:
                filters['location'] = entities['locations'][0]
            
            # Price filters
            for price_range in entities['price_ranges']:
                if 'under' in price_range or 'below' in price_range:
                    numbers = re.findall(r'\d+', price_range)
                    if numbers:
                        filters['price_max'] = int(numbers[0])
                elif 'over' in price_range or 'above' in price_range:
                    numbers = re.findall(r'\d+', price_range)
                    if numbers:
                        filters['price_min'] = int(numbers[0])
                elif 'between' in price_range:
                    numbers = re.findall(r'\d+', price_range)
                    if len(numbers) >= 2:
                        filters['price_min'] = int(numbers[0])
                        filters['price_max'] = int(numbers[1])
            
            # Property type filter
            if entities['property_types']:
                filters['property_type'] = entities['property_types'][0]
            
            # Features filter
            filters['features'] = entities['features']
            
            # BHK filter
            for number in entities['numbers']:
                if number in [1, 2, 3, 4, 5]:  # Common BHK values
                    filters['bhk'] = number
                    break
            
            # Furnished filter
            if 'furnished' in query:
                filters['furnished'] = 'furnished'
            elif 'semi-furnished' in query:
                filters['furnished'] = 'semi-furnished'
            elif 'unfurnished' in query:
                filters['furnished'] = 'unfurnished'
            
            # Urgency filter
            filters['urgency'] = entities['urgency']
            
        except Exception as e:
            logger.error(f"Error extracting filters: {e}")
        
        return filters
    
    def _assess_complexity(self, query: str, entities: Dict[str, Any]) -> QueryComplexity:
        """Assess query complexity"""
        try:
            complexity_score = 0
            
            # Length factor
            if len(query.split()) > 10:
                complexity_score += 2
            elif len(query.split()) > 5:
                complexity_score += 1
            
            # Entity factor
            complexity_score += len(entities['locations'])
            complexity_score += len(entities['price_ranges'])
            complexity_score += len(entities['property_types'])
            complexity_score += len(entities['features'])
            
            # Special patterns
            if re.search(r'\b(and|or|but)\b', query):
                complexity_score += 1
            
            if re.search(r'\b(compare|difference|similar)\b', query):
                complexity_score += 2
            
            # Determine complexity level
            if complexity_score >= 5:
                return QueryComplexity.COMPLEX
            elif complexity_score >= 2:
                return QueryComplexity.MODERATE
            else:
                return QueryComplexity.SIMPLE
                
        except Exception as e:
            logger.error(f"Error assessing complexity: {e}")
            return QueryComplexity.SIMPLE
    
    def _finalize_query(self, query: str, intent: QueryIntent, entities: Dict[str, Any]) -> str:
        """Finalize processed query"""
        try:
            # Add intent-specific terms
            if intent == QueryIntent.SEARCH_PROPERTY:
                if not any(term in query for term in ['property', 'house', 'apartment', 'pg']):
                    query += " property"
            
            elif intent == QueryIntent.GET_NEARBY:
                if 'nearby' not in query and 'near' not in query:
                    query += " nearby"
            
            elif intent == QueryIntent.COMPARE_PROPERTIES:
                if 'compare' not in query:
                    query += " compare"
            
            # Clean up query
            query = re.sub(r'\s+', ' ', query).strip()
            
            return query
            
        except Exception as e:
            logger.error(f"Error finalizing query: {e}")
            return query
    
    def _calculate_confidence(self, intent_confidence: float, entities: Dict[str, Any], 
                            complexity: QueryComplexity) -> float:
        """Calculate overall confidence score"""
        try:
            # Base confidence from intent classification
            confidence = intent_confidence
            
            # Entity factor
            entity_count = sum(len(v) if isinstance(v, list) else (1 if v else 0) 
                             for v in entities.values())
            entity_factor = min(1.0, entity_count / 5.0)  # Normalize to 0-1
            confidence = (confidence + entity_factor) / 2
            
            # Complexity factor (simpler queries are more confident)
            complexity_factor = {
                QueryComplexity.SIMPLE: 1.0,
                QueryComplexity.MODERATE: 0.8,
                QueryComplexity.COMPLEX: 0.6
            }
            confidence *= complexity_factor.get(complexity, 0.8)
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _update_processing_stats(self, processing_time: float, intent: QueryIntent):
        """Update processing statistics"""
        self.processing_stats['total_queries'] += 1
        self.processing_stats['intent_distribution'][intent.value] += 1
        
        # Update average processing time
        total_queries = self.processing_stats['total_queries']
        current_avg = self.processing_stats['avg_processing_time']
        self.processing_stats['avg_processing_time'] = (
            (current_avg * (total_queries - 1) + processing_time) / total_queries
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.processing_stats,
            'active_sessions': 0,  # REMOVED conversation contexts
            'total_context_entries': 0  # REMOVED conversation contexts
        }
    
    def clear_session_context(self, session_id: str):
        """Clear conversation context for a session - REMOVED to prevent session sharing"""
        # REMOVED to prevent session sharing
        logger.info(f"Session context clearing disabled for {session_id}")
    
    def get_session_context(self, session_id: str) -> List[Dict]:
        """Get conversation context for a session - REMOVED to prevent session sharing"""
        # REMOVED to prevent session sharing
        return []
