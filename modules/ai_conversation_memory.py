import logging
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
import threading
import hashlib
from sentence_transformers import SentenceTransformer
from modules.global_models import get_jina_model
import torch

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    turn_id: str
    session_id: str
    user_query: str
    assistant_response: str
    retrieved_properties: List[Dict[str, Any]]
    query_embedding: np.ndarray
    response_embedding: np.ndarray
    timestamp: datetime
    query_intent: str
    context_relevance: float
    metadata: Dict[str, Any]

@dataclass
class ConversationContext:
    """Represents the current conversation context"""
    session_id: str
    turns: List[ConversationTurn]
    current_topic: str
    topic_embedding: np.ndarray
    conversation_summary: str
    user_preferences: Dict[str, Any]
    property_context: Dict[str, Any]
    last_activity: datetime
    context_strength: float

class AIConversationMemory:
    """
    AI-powered conversation memory system that understands context like ChatGPT
    Uses embeddings and semantic similarity to maintain conversation state
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 max_context_length: int = 10,
                 similarity_threshold: float = 0.5,
                 context_decay_factor: float = 0.9):
        
        try:
            self.model_name = model_name
            self.max_context_length = max_context_length
            # Slightly stricter threshold to reduce false positives
            self.similarity_threshold = max(0.55, similarity_threshold)
            self.context_decay_factor = context_decay_factor
            
            # Initialize embedding model (prefer shared Jina model for consistency)
            try:
                self.embedding_model = get_jina_model()
                if self.embedding_model is not None:
                    logger.info("✅ Using global Jina embedding model for AI conversation memory")
                else:
                    raise Exception("Global Jina model unavailable")
            except Exception as e:
                logger.warning(f"Global Jina model unavailable, falling back to {model_name}: {e}")
                try:
                    logger.info(f"Loading fallback embedding model: {model_name}")
                    self.embedding_model = SentenceTransformer(model_name)
                    logger.info(f"✅ Loaded fallback embedding model: {model_name}")
                except Exception as e2:
                    logger.error(f"Failed to load any embedding model: {e2}")
                    self.embedding_model = None
            
            # Conversation storage
            self.conversations: Dict[str, ConversationContext] = {}
            
            # Thread safety
            self.lock = threading.RLock()
            
            # Performance tracking
            self.stats = {
                'total_conversations': 0,
                'active_conversations': 0,
                'context_hits': 0,
                'context_misses': 0,
                'avg_context_relevance': 0.0
            }

            # Prototype embeddings for detecting fresh search intents (cached)
            self._new_search_prototypes = [
                "list properties",
                "find properties",
                "search properties",
                "show 3 bhk apartments",
                "show houses for rent",
                "properties for sale",
                "apartments in hyderabad",
                "find 3 bhk in gachibowli",
            ]
            self._new_search_proto_embeddings = None
            
            logger.info("✅ AI Conversation Memory initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI Conversation Memory: {e}")
            raise
    
    def _generate_turn_id(self, session_id: str, query: str) -> str:
        """Generate unique turn ID"""
        timestamp = str(time.time())
        content = f"{session_id}:{query}:{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        if text is None or not text.strip():
            logger.warning("Text is None or empty, returning zero embedding")
            return np.zeros(384)  # Fallback for all-MiniLM-L6-v2
        
        if not self.embedding_model:
            return np.zeros(384)  # Fallback for all-MiniLM-L6-v2
        
        try:
            return self.embedding_model.encode(text, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return np.zeros(384)
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def _summarize_properties_text(self, properties: List[Dict[str, Any]]) -> str:
        """Create a compact textual summary of properties (names, types, cities) for similarity checks"""
        try:
            if not properties:
                return ""
            parts = []
            for prop in properties[:5]:
                name = prop.get('PropertyName') or prop.get('propertyName') or ''
                typ = prop.get('PropertyType') or prop.get('typeName') or ''
                city = prop.get('City') or prop.get('city') or ''
                state = prop.get('State') or prop.get('state') or ''
                parts.append(f"{name} | {typ} | {city} {state}")
            return " ; ".join([p for p in parts if p.strip()])
        except Exception as e:
            logger.warning(f"Failed to summarize properties text: {e}")
            return ""

    def _is_new_search_intent(self, query: str) -> bool:
        """Detect if a query is likely a fresh search using prototype similarities."""
        try:
            if self.embedding_model is None:
                return False
            # Initialize prototype embeddings once
            if self._new_search_proto_embeddings is None:
                self._new_search_proto_embeddings = [
                    self._get_embedding(p) for p in self._new_search_prototypes
                ]
            query_emb = self._get_embedding(query)
            # Max similarity to any prototype
            max_sim = 0.0
            for proto_emb in self._new_search_proto_embeddings:
                sim = self._calculate_similarity(query_emb, proto_emb)
                if sim > max_sim:
                    max_sim = sim
            # Heuristic threshold for being a new search intent
            return max_sim > 0.60
        except Exception as e:
            logger.warning(f"New search intent check failed: {e}")
            return False
    
    def _analyze_query_intent(self, query: str, conversation_context: Optional[ConversationContext] = None) -> str:
        """Analyze query intent using AI understanding"""
        query_lower = query.lower()
        
        # Follow-up indicators (narrow, to reduce false positives)
        follow_up_indicators = [
            'tell me more', 'what about', 'how about', 'can you show', 'show me',
            'which one', 'which property', 'tell me about',
            'too', 'also', 'as well', 'additionally', 'more'
        ]
        
        # Clarification indicators
        clarification_indicators = [
            'what do you mean', 'i don\'t understand', 'can you explain',
            'clarify', 'elaborate', 'more details', 'specify'
        ]
        
        # Property-specific follow-ups
        property_follow_ups = [
            'price', 'location', 'size', 'rooms', 'bedrooms', 'bathrooms',
            'amenities', 'features', 'photos', 'images', 'contact', 'agent',
            'availability', 'when', 'where', 'how much', 'cost', 'details'
        ]
        
        # Check for follow-up patterns (avoid classifying generic first-turn queries)
        if any(indicator in query_lower for indicator in follow_up_indicators):
            return "follow_up"
        elif any(indicator in query_lower for indicator in clarification_indicators):
            return "clarification"
        elif any(indicator in query_lower for indicator in property_follow_ups):
            return "property_detail"
        elif conversation_context and len(conversation_context.turns) > 0:
            # Check similarity with previous context
            query_embedding = self._get_embedding(query)
            last_turn = conversation_context.turns[-1]
            similarity = self._calculate_similarity(query_embedding, last_turn.query_embedding)
            
            if similarity > self.similarity_threshold:
                return "contextual_follow_up"
            
            # Fallback: Check if this is likely a follow-up based on keywords
            if self.embedding_model is None:
                # If embeddings failed, use keyword-based detection
                last_query_lower = last_turn.user_query.lower()
                common_words = set(query_lower.split()) & set(last_query_lower.split())
                if len(common_words) >= 2:  # At least 2 common words
                    return "contextual_follow_up"
        
        return "new_query"
    
    def _update_conversation_topic(self, conversation_context: ConversationContext, new_query: str) -> str:
        """Update conversation topic using AI understanding"""
        if not conversation_context.turns:
            return new_query
        
        # Get embeddings for topic analysis
        new_query_embedding = self._get_embedding(new_query)
        topic_embedding = conversation_context.topic_embedding
        
        # Calculate topic similarity
        topic_similarity = self._calculate_similarity(new_query_embedding, topic_embedding)
        
        if topic_similarity > self.similarity_threshold:
            # Same topic, update with weighted combination
            alpha = 0.7  # Weight for new query
            updated_topic_embedding = (alpha * new_query_embedding + 
                                     (1 - alpha) * topic_embedding)
            conversation_context.topic_embedding = updated_topic_embedding
            return conversation_context.current_topic
        else:
            # New topic
            conversation_context.topic_embedding = new_query_embedding
            return new_query
    
    def _generate_conversation_summary(self, turns: List[ConversationTurn]) -> str:
        """Generate AI-based conversation summary"""
        if not turns:
            return ""
        
        # Extract key information from recent turns
        recent_turns = turns[-3:]  # Last 3 turns
        
        summary_parts = []
        for turn in recent_turns:
            if turn.query_intent in ["follow_up", "property_detail", "contextual_follow_up"]:
                summary_parts.append(f"User asked about: {turn.user_query}")
            else:
                summary_parts.append(f"New topic: {turn.user_query}")
        
        return " | ".join(summary_parts)
    
    def _calculate_context_relevance(self, query: str, conversation_context: ConversationContext) -> float:
        """Calculate how relevant the current query is to conversation context"""
        if not conversation_context.turns:
            return 0.0
        
        query_embedding = self._get_embedding(query)
        
        # Calculate similarity with recent turns
        recent_turns = conversation_context.turns[-3:]
        similarities = []
        
        for turn in recent_turns:
            similarity = self._calculate_similarity(query_embedding, turn.query_embedding)
            similarities.append(similarity)
        
        # Calculate weighted average (more recent = higher weight)
        weights = [0.5, 0.3, 0.2][:len(similarities)]
        weighted_similarity = sum(s * w for s, w in zip(similarities, weights))
        
        return weighted_similarity
    
    def add_conversation_turn(self, 
                            session_id: str,
                            user_query: str,
                            assistant_response: str,
                            retrieved_properties: List[Dict[str, Any]] = None) -> ConversationTurn:
        """Add a new turn to the conversation"""
        
        with self.lock:
            # Get or create conversation context
            if session_id not in self.conversations:
                conversation_context = ConversationContext(
                    session_id=session_id,
                    turns=[],
                    current_topic=user_query,
                    topic_embedding=self._get_embedding(user_query),
                    conversation_summary="",
                    user_preferences={},
                    property_context={},
                    last_activity=datetime.now(),
                    context_strength=1.0
                )
                self.conversations[session_id] = conversation_context
                self.stats['total_conversations'] += 1
                self.stats['active_conversations'] += 1
            else:
                conversation_context = self.conversations[session_id]
            
            # Analyze query intent
            query_intent = self._analyze_query_intent(user_query, conversation_context)
            
            # Calculate context relevance
            context_relevance = self._calculate_context_relevance(user_query, conversation_context)
            
            # Create conversation turn
            turn = ConversationTurn(
                turn_id=self._generate_turn_id(session_id, user_query),
                session_id=session_id,
                user_query=user_query,
                assistant_response=assistant_response,
                retrieved_properties=retrieved_properties or [],
                query_embedding=self._get_embedding(user_query),
                response_embedding=self._get_embedding(assistant_response),
                timestamp=datetime.now(),
                query_intent=query_intent,
                context_relevance=context_relevance,
                metadata={
                    'context_strength': conversation_context.context_strength,
                    'topic_similarity': context_relevance
                }
            )
            
            # Add turn to conversation
            conversation_context.turns.append(turn)
            
            # Maintain max context length
            if len(conversation_context.turns) > self.max_context_length:
                conversation_context.turns = conversation_context.turns[-self.max_context_length:]
            
            # Update conversation topic
            conversation_context.current_topic = self._update_conversation_topic(
                conversation_context, user_query
            )
            
            # Update conversation summary
            conversation_context.conversation_summary = self._generate_conversation_summary(
                conversation_context.turns
            )
            
            # Update last activity
            conversation_context.last_activity = datetime.now()
            
            # Update context strength (decay over time)
            time_diff = (datetime.now() - conversation_context.turns[0].timestamp).total_seconds()
            conversation_context.context_strength = max(0.1, self.context_decay_factor ** (time_diff / 3600))
            
            # Update stats
            self.stats['avg_context_relevance'] = (
                (self.stats['avg_context_relevance'] * (len(conversation_context.turns) - 1) + context_relevance) 
                / len(conversation_context.turns)
            )
            
            logger.info(f"Added conversation turn for session {session_id}, intent: {query_intent}, relevance: {context_relevance:.3f}")
            
            return turn
    
    def get_conversation_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context for a session"""
        if session_id is None:
            logger.warning("Session ID is None")
            return None
        
        with self.lock:
            return self.conversations.get(session_id)
    
    def is_follow_up_query(self, session_id: str, query: str) -> Tuple[bool, float]:
        """Determine if query is a follow-up using AI understanding"""
        # Handle None or empty inputs
        if session_id is None or query is None:
            logger.warning("Session ID or query is None")
            return False, 0.0
        
        if not query.strip():
            logger.warning("Query is empty")
            return False, 0.0
        
        conversation_context = self.get_conversation_context(session_id)
        
        if not conversation_context or not conversation_context.turns:
            logger.info(f"No conversation context found for session {session_id}")
            return False, 0.0
        
        query_lower = query.lower()
        
        # Special case: "agent details" queries are almost always follow-ups
        if "agent" in query_lower and "details" in query_lower:
            logger.info(f"Detected agent details follow-up query: '{query}'")
            return True, 0.9  # High relevance for agent details
        
        # Special case: "these properties" or "same properties" indicates follow-up
        if any(phrase in query_lower for phrase in ["these properties", "same properties", "those properties"]):
            logger.info(f"Detected property reference follow-up query: '{query}'")
            return True, 0.8
        
        # Special case: Specific property references (first, second, third, etc.)
        property_ordinals = ["first", "second", "third", "fourth", "fifth", "1st", "2nd", "3rd", "4th", "5th"]
        if any(ordinal in query_lower for ordinal in property_ordinals) and "property" in query_lower:
            logger.info(f"Detected specific property reference follow-up query: '{query}'")
            return True, 0.85
        
        # Analyze query intent (rule-based)
        query_intent = self._analyze_query_intent(query, conversation_context)

        # Baseline relevance over recent turns
        context_relevance = self._calculate_context_relevance(query, conversation_context)

        # Additional multi-signal relevance using last turn
        try:
            last_turn = conversation_context.turns[-1]
            query_emb = self._get_embedding(query)
            last_query_sim = self._calculate_similarity(query_emb, last_turn.query_embedding)
            last_resp_sim = self._calculate_similarity(query_emb, last_turn.response_embedding)
            properties_text = self._summarize_properties_text(last_turn.retrieved_properties)
            prop_sim = 0.0
            if properties_text:
                prop_sim = self._calculate_similarity(query_emb, self._get_embedding(properties_text))

            # Weighted aggregation prioritizing immediate context
            # Emphasize last response and query, then properties, then baseline
            aggregated = (
                0.35 * last_resp_sim +
                0.30 * last_query_sim +
                0.20 * prop_sim +
                0.15 * context_relevance
            )
            # Use the better of baseline vs aggregated
            context_relevance = max(context_relevance, aggregated)
        except Exception as e:
            logger.warning(f"Enhanced follow-up relevance calculation failed: {e}")

        # Determine if it's a follow-up (preliminary)
        is_follow_up = (
            query_intent in ["follow_up", "property_detail", "contextual_follow_up"] or
            context_relevance > self.similarity_threshold
        )

        # Override to NEW if this clearly looks like a fresh search and the user didn't use deictic references
        deictic_refs = any(p in query_lower for p in ["these", "those", "same properties", "this property", "that property"]) or any(
            ord_ref in query_lower for ord_ref in ["first", "second", "third", "fourth", "fifth", "1st", "2nd", "3rd", "4th", "5th"]
        )
        if not deictic_refs:
            if self._is_new_search_intent(query):
                # Only treat as follow-up if the aggregated similarity is very high (strong contextual tie)
                if context_relevance < (self.similarity_threshold + 0.20):
                    is_follow_up = False
        
        logger.info(f"Follow-up analysis for '{query}': intent={query_intent}, relevance={context_relevance:.3f}, threshold={self.similarity_threshold}, is_follow_up={is_follow_up}")
        
        if is_follow_up:
            self.stats['context_hits'] += 1
        else:
            self.stats['context_misses'] += 1
        
        return is_follow_up, context_relevance
    
    def get_contextual_query(self, session_id: str, query: str) -> str:
        """Enhance query with conversation context"""
        conversation_context = self.get_conversation_context(session_id)
        
        if not conversation_context or not conversation_context.turns:
            return query
        
        # Check if it's a follow-up
        is_follow_up, relevance = self.is_follow_up_query(session_id, query)
        
        if is_follow_up and relevance > 0.5:
            # Enhance query with context
            last_turn = conversation_context.turns[-1]
            
            # Handle specific property references
            query_lower = query.lower()
            property_ordinals = ["first", "second", "third", "fourth", "fifth", "1st", "2nd", "3rd", "4th", "5th"]
            
            # Check if this is a specific property reference
            ordinal_index = None
            for i, ordinal in enumerate(property_ordinals):
                if ordinal in query_lower:
                    ordinal_index = i % 5  # Map to 0-4 index
                    break
            
            if ordinal_index is not None and last_turn.retrieved_properties:
                # Get the specific property
                if ordinal_index < len(last_turn.retrieved_properties):
                    specific_property = last_turn.retrieved_properties[ordinal_index]
                    property_name = specific_property.get('PropertyName', f'Property {ordinal_index + 1}')
                    contextual_query = f"Context: {last_turn.user_query} | Current query: {query} | Specific property: {property_name}"
                    logger.info(f"Enhanced query for specific property {ordinal_index + 1}: {contextual_query}")
                    return contextual_query
            
            # Create contextual query
            contextual_query = f"Context: {last_turn.user_query} | Current query: {query}"
            
            # Add property context if available
            if last_turn.retrieved_properties:
                property_names = [prop.get('PropertyName', 'property') for prop in last_turn.retrieved_properties[:2]]
                contextual_query += f" | Related properties: {', '.join(property_names)}"
            
            logger.info(f"Enhanced query with context: {contextual_query}")
            return contextual_query
        
        return query
    
    def get_conversation_history(self, session_id: str, max_turns: int = 5) -> List[Dict[str, Any]]:
        """Get conversation history for context injection"""
        conversation_context = self.get_conversation_context(session_id)
        
        if not conversation_context:
            return []
        
        # Get recent turns
        recent_turns = conversation_context.turns[-max_turns:]
        
        history = []
        for turn in recent_turns:
            history.append({
                'user_query': turn.user_query,
                'assistant_response': turn.assistant_response,
                'query_intent': turn.query_intent,
                'context_relevance': turn.context_relevance,
                'timestamp': turn.timestamp.isoformat()
            })
        
        return history
    
    def update_user_preferences(self, session_id: str, preferences: Dict[str, Any]):
        """Update user preferences based on conversation"""
        with self.lock:
            conversation_context = self.get_conversation_context(session_id)
            if conversation_context:
                conversation_context.user_preferences.update(preferences)
                logger.info(f"Updated preferences for session {session_id}: {preferences}")
    
    def get_user_preferences(self, session_id: str) -> Dict[str, Any]:
        """Get user preferences from conversation"""
        conversation_context = self.get_conversation_context(session_id)
        return conversation_context.user_preferences if conversation_context else {}
    
    def get_specific_property(self, session_id: str, property_index: int) -> Optional[Dict[str, Any]]:
        """Get a specific property from the last conversation turn by index"""
        conversation_context = self.get_conversation_context(session_id)
        
        if not conversation_context or not conversation_context.turns:
            return None
        
        last_turn = conversation_context.turns[-1]
        
        if not last_turn.retrieved_properties or property_index >= len(last_turn.retrieved_properties):
            return None
        
        return last_turn.retrieved_properties[property_index]
    
    def clear_conversation(self, session_id: str):
        """Clear conversation for a session"""
        with self.lock:
            if session_id in self.conversations:
                del self.conversations[session_id]
                self.stats['active_conversations'] -= 1
                logger.info(f"Cleared conversation for session {session_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        with self.lock:
            return {
                **self.stats,
                'total_sessions': len(self.conversations),
                'model_name': self.model_name,
                'max_context_length': self.max_context_length,
                'similarity_threshold': self.similarity_threshold
            }
    
    def cleanup_old_conversations(self, max_age_hours: int = 24):
        """Clean up old conversations"""
        with self.lock:
            current_time = datetime.now()
            sessions_to_remove = []
            
            for session_id, context in self.conversations.items():
                age = (current_time - context.last_activity).total_seconds() / 3600
                if age > max_age_hours:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.conversations[session_id]
                self.stats['active_conversations'] -= 1
            
            if sessions_to_remove:
                logger.info(f"Cleaned up {len(sessions_to_remove)} old conversations")

# Global instance
_ai_conversation_memory = None

def get_ai_conversation_memory() -> AIConversationMemory:
    """Get global AI conversation memory instance"""
    global _ai_conversation_memory
    try:
        if _ai_conversation_memory is None:
            logger.info("Creating new AI conversation memory instance...")
            _ai_conversation_memory = AIConversationMemory()
            if _ai_conversation_memory is None:
                logger.error("Failed to create AI conversation memory instance")
                return None
        return _ai_conversation_memory
    except Exception as e:
        logger.error(f"Error getting AI conversation memory: {e}")
        return None
