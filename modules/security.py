import time
import logging
from collections import defaultdict
from better_profanity import profanity
from modules.config import (
    RATE_LIMIT_WINDOW, 
    MAX_REQUESTS_PER_WINDOW, 
    CACHE_TTL,
    MAX_QUERY_LENGTH,
    UserPlan,
    PLAN_FIELDS
)
import torch
import numpy as np
from sentence_transformers import util
import re
import bleach
import threading
from functools import wraps
from transformers import pipeline

# Thread local storage for user plan
_thread_local = threading.local()

def get_current_plan():
    """Get the current user plan from thread local storage"""
    return getattr(_thread_local, 'current_plan', UserPlan.PLUS)

def set_current_plan(plan):
    """Set the current user plan in thread local storage"""
    _thread_local.current_plan = plan

def with_user_plan(f):
    """Decorator to handle user plan from request"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            from flask import request
            plan = UserPlan.BASIC  # Default to BASIC plan
            
            if request.is_json:
                plan_str = request.json.get('user_plan', 'basic').lower()
                try:
                    plan = UserPlan(plan_str)
                except ValueError:
                    logging.warning(f"Invalid plan value: {plan_str}, defaulting to BASIC")
                    plan = UserPlan.BASIC
            
            set_current_plan(plan)
            return f(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in with_user_plan decorator: {str(e)}")
            set_current_plan(UserPlan.BASIC)  # Ensure BASIC plan is set even on error
            return f(*args, **kwargs)
    return decorated_function

class SecurityManager:
    def __init__(self):
        self.request_counts = defaultdict(lambda: {'count': 0, 'window_start': 0})

    def check_rate_limit(self, ip_address):
        current_time = time.time()
        if current_time - self.request_counts[ip_address]['window_start'] >= RATE_LIMIT_WINDOW:
            self.request_counts[ip_address] = {'count': 0, 'window_start': current_time}

        self.request_counts[ip_address]['count'] += 1
        return self.request_counts[ip_address]['count'] <= MAX_REQUESTS_PER_WINDOW

class QueryValidator:
    def __init__(self, model_embedding):
        self.model_embedding = model_embedding
        self.domain_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Real estate related categories
        self.real_estate_categories = [
            "property search",
            "rental property",
            "property for sale",
            "PG accommodation",
            "hostel accommodation",
            "commercial property",
            "agricultural property",
            "farmland",
            "agricultural land",
            "property details",
            "property location",
            "property price",
            "property features"
        ]
        
        # Initialize with examples
        self.initialize_with_examples()

    def initialize_with_examples(self):
        """Initialize with example queries for better classification"""
        self.real_estate_examples = [
            "Show me 2BHK apartments in Hyderabad",
            "Find PG accommodation near Hitech City",
            "What are the properties for sale in Madhapur?",
            "Looking for a 3BHK villa in Gachibowli",
            "Need a girls hostel in Kondapur",
            "Show me commercial properties for rent",
            "Find properties near my location",
            "What's the price of 2BHK in Gachibowli?",
            "Show me properties with swimming pool",
            "Find PG with food facility",
            "Looking for boys hostel in Madhapur",
            "Show me properties near metro station",
            "Find properties with 24/7 security",
            "Need a furnished apartment",
            "Show me properties with parking",
            "Find agricultural land for sale",
            "Show me farmland properties",
            "Looking for agricultural plots",
            "Need farming land",
            "Show me agricultural properties"
        ]
        
        self.non_real_estate_examples = [
            "What's the weather like today?",
            "Tell me a joke",
            "What's the time?",
            "How to make pasta?",
            "What's the capital of France?",
            "Show me the latest news",
            "Play some music",
            "What's the meaning of life?",
            "How to fix my computer?",
            "Tell me about history"
        ]

    def is_real_estate_query(self, query):
        """Check if the query is related to real estate using zero-shot classification"""
        try:
            # Handle simple responses that are part of a conversation
            simple_responses = ["yes", "no", "ok", "sure", "fine", "alright"]
            if query.lower().strip() in simple_responses:
                # If it's a simple response, check if we're in a real estate context
                # This could be enhanced by checking conversation history
                return True

            # First check for common real estate keywords
            real_estate_keywords = [
                "property", "house", "apartment", "flat", "villa", "pg", "hostel",
                "rent", "sale", "buy", "accommodation", "room", "beds", "baths",
                "bhk", "square feet", "location", "price", "amenities", "facilities",
                "farmland", "agriculture", "agricultural", "land", "farm", "farming",
                "plot", "acre", "hectare", "agricultural land", "farm land"
            ]
            
            query_lower = query.lower()
            if any(keyword in query_lower for keyword in real_estate_keywords):
                return True

            # Use zero-shot classification for more complex cases
            result = self.domain_classifier(
                query,
                candidate_labels=["real estate query", "non real estate query"],
                hypothesis_template="This is a {}."
            )
            
            # Get the confidence score for real estate
            real_estate_score = result['scores'][0] if result['labels'][0] == "real estate query" else result['scores'][1]
            
            # Also check against our example categories
            category_result = self.domain_classifier(
                query,
                candidate_labels=self.real_estate_categories,
                hypothesis_template="This query is about {}."
            )
            
            # If any category has high confidence, consider it a real estate query
            max_category_score = max(category_result['scores'])
            
            # Consider it a real estate query if either the general classification
            # or any specific category has high confidence
            return real_estate_score > 0.6 or max_category_score > 0.7

        except Exception as e:
            logging.error(f"Error in is_real_estate_query: {str(e)}")
            # Default to True if there's an error to be safe
            return True

    def clean_input(self, query):
        """Clean and validate the input query"""
        # Remove any special characters and extra spaces
        cleaned = re.sub(r'[^\w\s]', ' ', query)
        cleaned = ' '.join(cleaned.split())
        return cleaned

    def validate_query_length(self, query):
        """Validate query length"""
        return len(query) <= MAX_QUERY_LENGTH

    def check_profanity(self, query):
        """Check for profanity in the query"""
        # Add profanity checking logic here if needed
        return True 