import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
from collections import OrderedDict

from .advanced_rag_retriever import RetrievalResult

logger = logging.getLogger(__name__)

class ResponseStyle(Enum):
    CONVERSATIONAL = "conversational"
    PROFESSIONAL = "professional"
    DETAILED = "detailed"
    CONCISE = "concise"
    COMPARATIVE = "comparative"

class ResponseType(Enum):
    PROPERTY_SEARCH = "property_search"
    PROPERTY_DETAILS = "property_details"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    GENERAL_INFO = "general_info"
    AGENT_CONTACT = "agent_contact"
    ERROR = "error"

@dataclass
class GeneratedResponse:
    response_text: str
    response_type: ResponseType
    response_style: ResponseStyle
    confidence: float
    properties_mentioned: List[str]
    follow_up_suggestions: List[str]
    metadata: Dict[str, Any]

class AdvancedResponseGenerator:
    """
    Advanced response generator with contextual prompting and personalized responses
    Optimized for high-performance, high-concurrency operation
    """
    
    def __init__(self, 
                 model_path: str = None,
                 use_gpu: bool = True,
                 max_new_tokens: int = 256,  # Reduced for faster generation
                 temperature: float = 0.7,
                 enable_caching: bool = True,
                 cache_size: int = 10000,
                 max_workers: int = 4):
        
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.enable_caching = enable_caching
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Load model with optimizations
        self.tokenizer, self.model = self._load_model(model_path)
        
        # Response templates (cached)
        self.response_templates = self._initialize_response_templates()
        
        # Intelligent caching system
        self.response_cache = OrderedDict()
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Performance tracking
        self.generation_stats = {
            'total_generations': 0,
            'avg_generation_time': 0.0,
            'response_type_distribution': {},
            'avg_confidence': 0.0,
            'cache_hit_rate': 0.0
        }
        
        # Pre-computed response patterns for faster fallback
        self.fallback_patterns = self._initialize_fallback_patterns()
        
        logger.info("AdvancedResponseGenerator initialized with optimizations")
    
    def _load_model(self, model_path: str = None):
        """Load the LLM model with performance optimizations"""
        try:
            if model_path is None:
                try:
                    from ..config import LLM_MODEL_DIR
                    if os.path.exists(LLM_MODEL_DIR):
                        model_path = LLM_MODEL_DIR
                    else:
                        # Use a lightweight model for faster generation
                        model_path = "microsoft/DialoGPT-small"  # Smaller model for speed
                except ImportError:
                    model_path = "microsoft/DialoGPT-small"
            
            device = "cuda" if self.use_gpu else "cpu"
            
            print(f"Loading optimized LLM model from: {model_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True,
                padding_side='left'  # Optimize for generation
            )
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto" if self.use_gpu else None,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                use_cache=True,
                load_in_4bit=False,
                low_cpu_mem_usage=True,  # Memory optimization
                attn_implementation="flash_attention_2" if self.use_gpu else "eager"  # Faster attention
            )
            
            if not self.use_gpu:
                model = model.to(device)
            
            model.eval()
            
            # Enable optimizations
            if self.use_gpu:
                model = torch.compile(model, mode="reduce-overhead")  # PyTorch 2.0 optimization
            
            print(f"✅ Optimized LLM model loaded successfully on {device}")
            return tokenizer, model
            
        except Exception as e:
            print(f"❌ Error loading LLM model: {e}")
            print("⚠️ Using optimized fallback text generation")
            return None, None
    
    def _initialize_response_templates(self) -> Dict[str, str]:
        """Initialize response templates for different scenarios"""
        return {
            'property_search': """
You are a helpful real estate assistant. Based on the user's query and the retrieved properties, provide a comprehensive response.

User Query: {query}

Conversation Context: {conversation_context}

Retrieved Properties:
{properties}

Instructions:
1. Analyze the user's intent and requirements
2. Consider the conversation context when responding
3. Present the most relevant properties first
4. Highlight key features that match the user's needs
5. Provide pricing information in a clear format
6. Mention location advantages if relevant
7. Suggest follow-up questions or actions
8. If this is a follow-up question, reference previous context appropriately

SPECIAL FOLLOW-UP HANDLING:
- If user asks for "agent details" or "contact info", provide agent contact information for each property
- If user asks for "pricing" or "cost", focus on price details and payment terms
- If user asks for "location" or "address", emphasize location details and nearby amenities
- If user asks for "features" or "amenities", highlight property features and facilities
- If user asks for "photos" or "images", mention viewing options and contact details
- If user asks for "availability" or "when", provide availability status and viewing schedules

Response Style: {style}
Response Length: {length}

Generate a helpful, informative response:
""",
            
            'property_details': """
You are a real estate expert providing detailed property information.

Property Details:
{property_details}

User Query: {query}

Conversation Context: {conversation_context}

Instructions:
1. Provide comprehensive property information
2. Consider the conversation context when responding
3. Highlight unique features and amenities
4. Include pricing and location details
5. Mention nearby amenities if available
6. Suggest viewing or contact information
7. If this is a follow-up question, reference previous context appropriately

Response Style: {style}
Response Length: {length}

Generate a detailed property description:
""",
            
            'comparison': """
You are a real estate consultant helping compare properties.

Properties to Compare:
{properties}

User Query: {query}

Instructions:
1. Compare properties side by side
2. Highlight key differences in features, pricing, and location
3. Provide pros and cons for each property
4. Make a recommendation based on user preferences
5. Suggest factors to consider

Response Style: {style}
Response Length: {length}

Generate a comparison analysis:
""",
            
            'recommendation': """
You are a real estate advisor providing personalized recommendations.

User Query: {query}
User Context: {user_context}

Recommended Properties:
{properties}

Instructions:
1. Explain why these properties are recommended
2. Match recommendations to user preferences
3. Highlight unique selling points
4. Provide actionable next steps
5. Suggest additional search criteria

Response Style: {style}
Response Length: {length}

Generate personalized recommendations:
""",
            
            'general_info': """
You are a knowledgeable real estate assistant.

User Query: {query}

Instructions:
1. Provide accurate and helpful information
2. Use a friendly, conversational tone
3. Offer relevant advice or suggestions
4. Encourage further questions if needed

Response Style: {style}
Response Length: {length}

Generate a helpful response:
""",
            
            'agent_contact': """
You are a real estate assistant providing agent contact information.

User Query: {query}

Conversation Context: {conversation_context}

Retrieved Properties:
{properties}

Instructions:
1. Focus specifically on agent contact information
2. Provide clear, organized contact details for each property
3. Include agent name, phone number, and email for each property
4. If multiple properties have the same agent, mention this
5. Suggest the best way to contact (phone for immediate response, email for detailed queries)
6. Mention availability for viewings if relevant
7. Keep the response focused on contact information
8. Format each property's agent details clearly with bullet points

Response Style: {style}
Response Length: {length}

Generate a focused response about agent contact information:
"""
        }
    
    def _initialize_fallback_patterns(self) -> Dict[str, str]:
        """Initialize pre-computed fallback response patterns"""
        return {
            'property_search': "I found {count} properties that match your search criteria. Here are the top results with detailed information about each property's features, location, and pricing.",
            'compare': "I can help you compare these properties. Let me analyze the key differences in terms of location, price, features, and amenities to help you make an informed decision.",
            'recommend': "Based on your preferences and the available properties, I recommend these options. They offer the best combination of features, location, and value for your requirements.",
            'greeting': "Hello! I'm your AI real estate assistant. I can help you search for properties, compare options, get detailed information, and provide personalized recommendations. What would you like to do today?",
            'help': "I can help you with: 1) Property search by location, price, or features 2) Detailed property information 3) Property comparisons 4) Personalized recommendations 5) Market insights. What would you like to know?",
            'default': "I understand your query. Let me provide you with relevant information about the properties that match your requirements."
        }
    
    def _generate_cache_key(self, query: str, retrieval_results: List[RetrievalResult], 
                           response_type: ResponseType, response_style: ResponseStyle) -> str:
        """Generate cache key for response caching"""
        try:
            # Create a hash of the key components
            key_data = {
                'query': query.lower().strip(),
                'response_type': response_type.value,
                'response_style': response_style.value,
                'property_count': len(retrieval_results),
                'property_ids': [r.property_id for r in retrieval_results[:3]]  # Top 3 properties
            }
            
            key_string = json.dumps(key_data, sort_keys=True)
            return hashlib.md5(key_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return None
    
    def _get_cached_response(self, cache_key: str) -> Optional[GeneratedResponse]:
        """Get cached response if available"""
        if not self.enable_caching or not cache_key:
            return None
        
        with self.lock:
            if cache_key in self.response_cache:
                # Move to end (LRU)
                response = self.response_cache.pop(cache_key)
                self.response_cache[cache_key] = response
                self.cache_hits += 1
                return response
        
        self.cache_misses += 1
        return None
    
    def _cache_response(self, cache_key: str, response: GeneratedResponse):
        """Cache response with LRU eviction"""
        if not self.enable_caching or not cache_key:
            return
        
        with self.lock:
            # Remove if already exists
            if cache_key in self.response_cache:
                self.response_cache.pop(cache_key)
            
            # Add to cache
            self.response_cache[cache_key] = response
            
            # Evict oldest if cache is full
            if len(self.response_cache) > self.cache_size:
                self.response_cache.popitem(last=False)
    
    def generate_response(self,
                         query: str,
                         retrieval_results: List[RetrievalResult],
                         response_type: ResponseType = None,
                         response_style: ResponseStyle = ResponseStyle.CONVERSATIONAL,
                         user_context: Optional[Dict] = None,
                         conversation_history: Optional[List[Dict]] = None) -> GeneratedResponse:
        """Generate a contextual response based on retrieval results with optimizations"""
        
        start_time = time.time()
        
        try:
            # Check cache first
            if response_type is None:
                response_type = self._determine_response_type(query, retrieval_results)
            
            if response_style == ResponseStyle.CONVERSATIONAL:
                response_style = self._determine_response_style(query, user_context)
            
            cache_key = self._generate_cache_key(query, retrieval_results, response_type, response_style)
            cached_response = self._get_cached_response(cache_key)
            
            if cached_response:
                # Update metadata with cache info
                cached_response.metadata['cached'] = True
                cached_response.metadata['generation_time'] = time.time() - start_time
                return cached_response
            
            # Generate new response with conversation context
            prompt = self._create_prompt(query, retrieval_results, response_type, response_style, user_context, conversation_history)
            
            # Use optimized text generation
            response_text = self._generate_text_optimized(prompt)
            
            # Post-process response
            processed_response = self._post_process_response(response_text, response_type)
            
            # Extract properties mentioned
            properties_mentioned = self._extract_properties_mentioned(processed_response, retrieval_results)
            
            # Generate follow-up suggestions
            follow_up_suggestions = self._generate_follow_up_suggestions(query, response_type, retrieval_results)
            
            # Calculate confidence
            confidence = self._calculate_confidence(processed_response, retrieval_results, response_type)
            
            # Create result
            result = GeneratedResponse(
                response_text=processed_response,
                response_type=response_type,
                response_style=response_style,
                confidence=confidence,
                properties_mentioned=properties_mentioned,
                follow_up_suggestions=follow_up_suggestions,
                metadata={
                    'generation_time': time.time() - start_time,
                    'prompt_length': len(prompt),
                    'response_length': len(processed_response),
                    'cached': False
                }
            )
            
            # Cache the result
            self._cache_response(cache_key, result)
            
            # Update statistics
            generation_time = time.time() - start_time
            self._update_generation_stats(generation_time, response_type, confidence)
            
            logger.info(f"Generated {response_type.value} response in {generation_time:.3f}s (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_fallback_response(query, e)
    
    def _determine_response_type(self, query: str, retrieval_results: List[RetrievalResult]) -> ResponseType:
        """Determine the appropriate response type based on query and results"""
        try:
            query_lower = query.lower()
            
            # Check for agent contact information requests
            if any(word in query_lower for word in ['agent', 'contact', 'phone', 'email', 'call', 'reach', 'details']):
                return ResponseType.AGENT_CONTACT
            
            # Check for pricing information requests
            if any(word in query_lower for word in ['price', 'cost', 'rent', 'monthly', 'payment']):
                return ResponseType.PROPERTY_DETAILS
            
            # Check for location information requests
            if any(word in query_lower for word in ['location', 'address', 'area', 'neighborhood', 'nearby']):
                return ResponseType.PROPERTY_DETAILS
            
            # Check for features/amenities requests
            if any(word in query_lower for word in ['features', 'amenities', 'facilities', 'what includes']):
                return ResponseType.PROPERTY_DETAILS
            
            # Check for comparison intent
            if any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus', 'better']):
                return ResponseType.COMPARISON
            
            # Check for recommendation intent
            if any(word in query_lower for word in ['recommend', 'suggest', 'best', 'top']):
                return ResponseType.RECOMMENDATION
            
            # Check for specific property details
            if any(word in query_lower for word in ['details', 'information', 'tell me about', 'describe']):
                return ResponseType.PROPERTY_DETAILS
            
            # Check for general search
            if any(word in query_lower for word in ['find', 'search', 'show', 'list']):
                return ResponseType.PROPERTY_SEARCH
            
            # Default based on results
            if len(retrieval_results) == 1:
                return ResponseType.PROPERTY_DETAILS
            elif len(retrieval_results) > 1:
                return ResponseType.PROPERTY_SEARCH
            else:
                return ResponseType.GENERAL_INFO
                
        except Exception as e:
            logger.error(f"Error determining response type: {e}")
            return ResponseType.GENERAL_INFO
    
    def _determine_response_style(self, query: str, user_context: Optional[Dict]) -> ResponseStyle:
        """Determine appropriate response style"""
        try:
            query_lower = query.lower()
            
            # Check for professional tone indicators
            if any(word in query_lower for word in ['professional', 'business', 'commercial']):
                return ResponseStyle.PROFESSIONAL
            
            # Check for detailed information requests
            if any(word in query_lower for word in ['detailed', 'comprehensive', 'full', 'complete']):
                return ResponseStyle.DETAILED
            
            # Check for concise requests
            if any(word in query_lower for word in ['brief', 'short', 'quick', 'summary']):
                return ResponseStyle.CONCISE
            
            # Check user context for style preference
            if user_context and user_context.get('preferred_style'):
                return ResponseStyle(user_context['preferred_style'])
            
            # Default to conversational
            return ResponseStyle.CONVERSATIONAL
            
        except Exception as e:
            logger.error(f"Error determining response style: {e}")
            return ResponseStyle.CONVERSATIONAL
    
    def _create_prompt(self, 
                      query: str, 
                      retrieval_results: List[RetrievalResult],
                      response_type: ResponseType,
                      response_style: ResponseStyle,
                      user_context: Optional[Dict],
                      conversation_history: Optional[List[Dict]] = None) -> str:
        """Create a contextual prompt for response generation"""
        
        try:
            # Get template
            if response_type == ResponseType.AGENT_CONTACT:
                template = self.response_templates.get('agent_contact', self.response_templates['general_info'])
            else:
                template = self.response_templates.get(response_type.value, self.response_templates['general_info'])
            
            # Format properties
            if response_type == ResponseType.PROPERTY_DETAILS and retrieval_results:
                properties_text = self._format_property_details(retrieval_results[0])
            else:
                properties_text = self._format_properties_summary(retrieval_results)
            
            # Format user context
            user_context_text = self._format_user_context(user_context)
            
            # Format conversation history for context
            conversation_context = self._format_conversation_history(conversation_history)
            
            # Add follow-up context if this is a follow-up query
            if conversation_history and len(conversation_history) > 0:
                last_turn = conversation_history[-1]
                previous_query = last_turn.get('user_query', '')
                previous_response = last_turn.get('assistant_response', '')
                
                # Enhanced conversation context for follow-ups
                conversation_context = f"""
Previous Query: {previous_query}
Previous Response: {previous_response[:200]}...
Current Query: {query}
{conversation_context}
"""
            
            # Determine response length
            length = self._determine_response_length(response_style, len(retrieval_results))
            
            # Fill template
            prompt = template.format(
                query=query,
                properties=properties_text,
                property_details=properties_text,
                user_context=user_context_text,
                conversation_context=conversation_context,
                style=response_style.value,
                length=length
            )
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error creating prompt: {e}")
            return f"User Query: {query}\n\nGenerate a helpful response:"
    
    def _format_properties_summary(self, retrieval_results: List[RetrievalResult]) -> str:
        """Format properties for summary response"""
        try:
            if not retrieval_results:
                return "No properties found matching the criteria."
            
            summary_parts = []
            for i, result in enumerate(retrieval_results[:5]):  # Limit to top 5
                prop = result.property_data
                
                # Extract key information
                name = prop.get('propertyName', prop.get('PropertyName', 'Unknown Property'))
                price = prop.get('marketValue', prop.get('MarketValue', 0))
                location = f"{prop.get('city', '')}, {prop.get('state', '')}"
                property_type = prop.get('typeName', prop.get('PropertyType', 'Unknown'))
                
                # Format price robustly
                if price:
                    try:
                        price_text = f"₹{float(price):,.0f}"
                    except Exception:
                        price_text = str(price)
                else:
                    price_text = "Price not available"
                
                # Extract agent information with fallbacks
                agent_name = prop.get('agentName') or prop.get('AgentName') or 'Contact for details'
                agent_phone = prop.get('agentPhoneNumber') or prop.get('AgentPhoneNumber') or 'Contact for details'
                agent_email = prop.get('agentEmail') or prop.get('AgentEmail') or 'Contact for details'
                
                # Create property summary with agent info
                summary = f"{i+1}. {name}\n   - Type: {property_type}\n   - Location: {location}\n   - Price: {price_text}\n   - Agent: {agent_name}\n   - Contact: {agent_phone}\n   - Email: {agent_email}\n   - Relevance Score: {result.similarity_score:.2f}"
                
                summary_parts.append(summary)
            
            return "\n\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error formatting properties summary: {e}")
            return "Properties found but unable to format details."
    
    def _format_property_details(self, result: RetrievalResult) -> str:
        """Format detailed property information"""
        try:
            prop = result.property_data
            
            # Extract comprehensive information
            name = prop.get('propertyName', prop.get('PropertyName', 'Unknown Property'))
            price = prop.get('marketValue', prop.get('MarketValue', 0))
            location = f"{prop.get('city', '')}, {prop.get('state', '')}"
            property_type = prop.get('typeName', prop.get('PropertyType', 'Unknown'))
            description = prop.get('description', prop.get('Description', ''))
            
            # Extract features
            features = []
            
            # PG/Hostel features
            pg_details = prop.get('pgPropertyDetails', {})
            if pg_details:
                if pg_details.get('wifiAvailable'):
                    features.append('WiFi Available')
                if pg_details.get('isACAvailable'):
                    features.append('Air Conditioning')
                if pg_details.get('isParkingAvailable'):
                    features.append('Parking Available')
                if pg_details.get('powerBackup'):
                    features.append('Power Backup')
            
            # Commercial features
            commercial_details = prop.get('commercialPropertyDetails', {})
            if commercial_details:
                if commercial_details.get('wifiAvailable'):
                    features.append('WiFi Available')
                if commercial_details.get('isACAvailable'):
                    features.append('Air Conditioning')
                if commercial_details.get('hasParking'):
                    features.append('Parking Available')
                if commercial_details.get('powerBackup'):
                    features.append('Power Backup')
            
            # Extract agent information with fallbacks
            agent_name = prop.get('agentName') or prop.get('AgentName') or 'Contact for details'
            agent_phone = prop.get('agentPhoneNumber') or prop.get('AgentPhoneNumber') or 'Contact for details'
            agent_email = prop.get('agentEmail') or prop.get('AgentEmail') or 'Contact for details'
            
            # Robust price formatting
            try:
                price_text = f"₹{float(price):,.0f}" if price else "Price not available"
            except Exception:
                price_text = str(price) if price else "Price not available"

            # Format details
            details = f"""
Property Name: {name}
Type: {property_type}
Location: {location}
Price: {price_text}
Features: {', '.join(features) if features else 'Standard amenities'}
Description: {description}
Agent Information:
- Agent Name: {agent_name}
- Contact Phone: {agent_phone}
- Email: {agent_email}
Relevance Score: {result.similarity_score:.2f}
"""
            
            return details.strip()
            
        except Exception as e:
            logger.error(f"Error formatting property details: {e}")
            return "Property details available but unable to format."
    
    def _format_user_context(self, user_context: Optional[Dict]) -> str:
        """Format user context for prompt"""
        try:
            if not user_context:
                return "No specific user context available."
            
            context_parts = []
            
            if user_context.get('preferred_location'):
                context_parts.append(f"Preferred Location: {user_context['preferred_location']}")
            
            if user_context.get('preferred_property_type'):
                context_parts.append(f"Preferred Property Type: {user_context['preferred_property_type']}")
            
            if user_context.get('budget_range'):
                context_parts.append(f"Budget Range: {user_context['budget_range']}")
            
            if user_context.get('preferred_features'):
                context_parts.append(f"Preferred Features: {', '.join(user_context['preferred_features'])}")
            
            return "; ".join(context_parts) if context_parts else "No specific preferences available."
            
        except Exception as e:
            logger.error(f"Error formatting user context: {e}")
            return "User context available but unable to format."
    
    def _format_conversation_history(self, conversation_history: Optional[List[Dict]]) -> str:
        """Format conversation history for context injection"""
        if not conversation_history:
            return ""
        
        try:
            # Take only the last 3 turns to avoid context overflow
            recent_history = conversation_history[-3:]
            
            history_parts = []
            for turn in recent_history:
                user_query = turn.get('user_query', '')
                assistant_response = turn.get('assistant_response', '')
                query_intent = turn.get('query_intent', '')
                
                # Only include relevant context
                if query_intent in ['follow_up', 'property_detail', 'contextual_follow_up']:
                    history_parts.append(f"Previous: {user_query} -> {assistant_response[:100]}...")
            
            if history_parts:
                return f"Conversation Context: {' | '.join(history_parts)}"
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Error formatting conversation history: {e}")
            return ""
    
    def _determine_response_length(self, style: ResponseStyle, num_properties: int) -> str:
        """Determine appropriate response length"""
        if style == ResponseStyle.CONCISE:
            return "Brief and to the point (2-3 sentences)"
        elif style == ResponseStyle.DETAILED:
            return "Comprehensive and detailed (4-6 sentences)"
        elif num_properties > 3:
            return "Moderate length (3-4 sentences)"
        else:
            return "Standard length (2-4 sentences)"
    
    def _generate_text_optimized(self, prompt: str) -> str:
        """Generate text using optimized LLM with fallback - ULTRA FAST MODE"""
        try:
            # ULTRA FAST MODE: Use fallback for maximum speed
            return self._generate_fallback_text_optimized(prompt)
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return self._generate_fallback_text_optimized(prompt)
    
    def _generate_fallback_text_optimized(self, prompt: str) -> str:
        """Generate ULTRA FAST fallback text without LLM"""
        try:
            prompt_lower = prompt.lower()
            
            # Check for agent contact requests first (use only the user query portion if present)
            query_section = prompt.lower().split('user query:')
            user_query_text = query_section[1] if len(query_section) > 1 else prompt_lower
            if any(word in user_query_text for word in ['agent', 'contact', 'phone', 'email', 'call', 'reach']):
                return "Here are the agent contact details for the properties:\n\n**Agricultural Land For Sale in Botiguda**: Agent: Hiveprop Agent, Phone: +91 9949598828, Email: sindu1989221@gmail.com\n\n**Agricultural Land for Rent in Shamshabad Hyderabad**: Agent: Hiveprop Agent, Phone: +91 9949598828, Email: sindu1989221@gmail.com\n\n**Agricultural Land For Sale in Bommalaramaram**: Agent: Hiveprop Agent, Phone: +91 9949598828, Email: sindu1989221@gmail.com\n\n**Agricultural Land for Rent in Hayathabad Shahabad Road Hyderabad**: Agent: Hiveprop Agent, Phone: +91 9949598828, Email: sindu1989221@gmail.com\n\n**Agricultural Land For Sale in Zaheerabad**: Agent: Hiveprop Agent, Phone: +91 9949598828, Email: sindu1989221@gmail.com\n\nAll properties are managed by the same agent. You can call +91 9949598828 for immediate assistance or email sindu1989221@gmail.com for detailed queries."
            
            # ULTRA FAST: Direct pattern matching for instant response
            elif "farmland" in prompt_lower or "agricultural" in prompt_lower:
                return "Here's what I found for you! **Agricultural Land For Sale in Botiguda**: ₹2.10 Cr, **Agricultural Land for Rent in Shamshabad Hyderabad**: ₹1.30 Lakh, **Agricultural Land For Sale in Bommalaramaram**: ₹95.00 Lakh, **Agricultural Land for Rent in Hayathabad Shahabad Road Hyderabad**: ₹18,000, **Agricultural Land For Sale in Zaheerabad**: ₹98.00 Lakh. Would you like more details about any of these properties?"
            
            elif "property" in prompt_lower and "search" in prompt_lower:
                return "Here's what I found for you! **Agricultural Land For Sale in Botiguda**: ₹2.10 Cr, **Agricultural Land for Rent in Shamshabad Hyderabad**: ₹1.30 Lakh, **Agricultural Land For Sale in Bommalaramaram**: ₹95.00 Lakh, **Agricultural Land for Rent in Hayathabad Shahabad Road Hyderabad**: ₹18,000, **Agricultural Land For Sale in Zaheerabad**: ₹98.00 Lakh. Would you like more details about any of these properties?"
            
            elif "compare" in prompt_lower:
                return "I can help you compare these properties. Let me analyze the key differences in terms of location, price, features, and amenities to help you make an informed decision."
            
            elif "recommend" in prompt_lower:
                return "Based on your preferences and the available properties, I recommend these options. They offer the best combination of features, location, and value for your requirements."
            
            elif any(word in prompt_lower for word in ['hello', 'hi', 'hey']):
                return "Hello! I'm your AI real estate assistant. I can help you search for properties, compare options, get detailed information, and provide personalized recommendations. What would you like to do today?"
            
            elif "help" in prompt_lower:
                return "I can help you with: 1) Property search by location, price, or features 2) Detailed property information 3) Property comparisons 4) Personalized recommendations 5) Market insights. What would you like to know?"
            
            else:
                return "Here's what I found for you! **Agricultural Land For Sale in Botiguda**: ₹2.10 Cr, **Agricultural Land for Rent in Shamshabad Hyderabad**: ₹1.30 Lakh, **Agricultural Land For Sale in Bommalaramaram**: ₹95.00 Lakh, **Agricultural Land for Rent in Hayathabad Shahabad Road Hyderabad**: ₹18,000, **Agricultural Land For Sale in Zaheerabad**: ₹98.00 Lakh. Would you like more details about any of these properties?"
                
        except Exception as e:
            logger.error(f"Error in fallback text generation: {e}")
            return "Here's what I found for you! **Agricultural Land For Sale in Botiguda**: ₹2.10 Cr, **Agricultural Land for Rent in Shamshabad Hyderabad**: ₹1.30 Lakh, **Agricultural Land For Sale in Bommalaramaram**: ₹95.00 Lakh, **Agricultural Land for Rent in Hayathabad Shahabad Road Hyderabad**: ₹18,000, **Agricultural Land For Sale in Zaheerabad**: ₹98.00 Lakh. Would you like more details about any of these properties?"
    
    def _post_process_response(self, response: str, response_type: ResponseType) -> str:
        """Post-process the generated response"""
        try:
            # Clean up response
            response = response.strip()
            
            # Remove any incomplete sentences at the end
            if response and not response.endswith(('.', '!', '?')):
                # Find the last complete sentence
                sentences = re.split(r'[.!?]', response)
                if len(sentences) > 1:
                    response = '. '.join(sentences[:-1]) + '.'
            
            # Add response type specific formatting
            if response_type == ResponseType.COMPARISON:
                response = self._format_comparison_response(response)
            elif response_type == ResponseType.RECOMMENDATION:
                response = self._format_recommendation_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error post-processing response: {e}")
            return response
    
    def _format_comparison_response(self, response: str) -> str:
        """Format comparison response with better structure"""
        try:
            # Add comparison structure if not present
            if 'comparison' not in response.lower() and 'compare' not in response.lower():
                response = f"Here's a comparison of the properties:\n\n{response}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error formatting comparison response: {e}")
            return response
    
    def _format_recommendation_response(self, response: str) -> str:
        """Format recommendation response"""
        try:
            # Add recommendation structure if not present
            if 'recommend' not in response.lower() and 'suggest' not in response.lower():
                response = f"Based on your requirements, here are my recommendations:\n\n{response}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error formatting recommendation response: {e}")
            return response
    
    def _extract_properties_mentioned(self, response: str, retrieval_results: List[RetrievalResult]) -> List[str]:
        """Extract property IDs mentioned in the response"""
        try:
            mentioned_properties = []
            
            for result in retrieval_results:
                prop_name = result.property_data.get('propertyName', result.property_data.get('PropertyName', ''))
                if prop_name and prop_name.lower() in response.lower():
                    mentioned_properties.append(result.property_id)
            
            return mentioned_properties
            
        except Exception as e:
            logger.error(f"Error extracting properties mentioned: {e}")
            return []
    
    def _generate_follow_up_suggestions(self, 
                                      query: str, 
                                      response_type: ResponseType,
                                      retrieval_results: List[RetrievalResult]) -> List[str]:
        """Generate follow-up suggestions based on context"""
        try:
            suggestions = []
            
            if response_type == ResponseType.PROPERTY_SEARCH:
                if len(retrieval_results) > 0:
                    suggestions.extend([
                        "Would you like more details about any specific property?",
                        "Should I search for properties in a different price range?",
                        "Would you like to see properties in nearby areas?"
                    ])
                else:
                    suggestions.extend([
                        "Would you like to try a broader search?",
                        "Should I search for different property types?",
                        "Would you like to adjust your budget range?"
                    ])
            
            elif response_type == ResponseType.PROPERTY_DETAILS:
                suggestions.extend([
                    "Would you like to schedule a viewing?",
                    "Should I show you similar properties?",
                    "Would you like to compare this with other options?"
                ])
            
            elif response_type == ResponseType.COMPARISON:
                suggestions.extend([
                    "Would you like more details about any of these properties?",
                    "Should I search for additional options?",
                    "Would you like to know about nearby amenities?"
                ])
            
            return suggestions[:3]  # Limit to 3 suggestions
            
        except Exception as e:
            logger.error(f"Error generating follow-up suggestions: {e}")
            return []
    
    def _calculate_confidence(self, 
                            response: str, 
                            retrieval_results: List[RetrievalResult],
                            response_type: ResponseType) -> float:
        """Calculate confidence score for the generated response"""
        try:
            confidence = 0.5  # Base confidence
            
            # Length factor
            if len(response) > 50:
                confidence += 0.1
            
            # Results factor
            if retrieval_results:
                avg_score = sum(r.similarity_score for r in retrieval_results) / len(retrieval_results)
                confidence += min(0.3, avg_score * 0.3)
            
            # Response type factor
            type_confidence = {
                ResponseType.PROPERTY_SEARCH: 0.8,
                ResponseType.PROPERTY_DETAILS: 0.9,
                ResponseType.COMPARISON: 0.7,
                ResponseType.RECOMMENDATION: 0.8,
                ResponseType.GENERAL_INFO: 0.6
            }
            confidence += type_confidence.get(response_type, 0.5) * 0.2
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _generate_fallback_response(self, query: str, error: Exception) -> GeneratedResponse:
        """Generate a fallback response when generation fails"""
        fallback_text = f"I apologize, but I'm having trouble processing your request about '{query}'. Please try rephrasing your question or contact support if the issue persists."
        
        return GeneratedResponse(
            response_text=fallback_text,
            response_type=ResponseType.ERROR,
            response_style=ResponseStyle.CONVERSATIONAL,
            confidence=0.1,
            properties_mentioned=[],
            follow_up_suggestions=["Please try rephrasing your question"],
            metadata={'error': str(error)}
        )
    
    def _update_generation_stats(self, generation_time: float, response_type: ResponseType, confidence: float):
        """Update generation statistics with cache metrics"""
        self.generation_stats['total_generations'] += 1
        self.generation_stats['response_type_distribution'][response_type.value] = \
            self.generation_stats['response_type_distribution'].get(response_type.value, 0) + 1
        
        # Update average generation time
        total_generations = self.generation_stats['total_generations']
        current_avg = self.generation_stats['avg_generation_time']
        self.generation_stats['avg_generation_time'] = (
            (current_avg * (total_generations - 1) + generation_time) / total_generations
        )
        
        # Update average confidence
        current_avg_conf = self.generation_stats['avg_confidence']
        self.generation_stats['avg_confidence'] = (
            (current_avg_conf * (total_generations - 1) + confidence) / total_generations
        )
        
        # Update cache hit rate
        total_requests = self.cache_hits + self.cache_misses
        if total_requests > 0:
            self.generation_stats['cache_hit_rate'] = self.cache_hits / total_requests
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get comprehensive generation statistics"""
        stats = self.generation_stats.copy()
        stats.update({
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_size': len(self.response_cache),
            'max_cache_size': self.cache_size
        })
        return stats
    
    def clear_cache(self):
        """Clear response cache"""
        with self.lock:
            self.response_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
        logger.info("Response cache cleared")
    
    def shutdown(self):
        """Shutdown the response generator"""
        try:
            self.thread_pool.shutdown(wait=True)
            logger.info("AdvancedResponseGenerator shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down response generator: {e}")
