import re
import spacy
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch
from typing import Dict, List, Tuple, Union
import logging
from modules.global_models import get_jina_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPProcessor:
    def __init__(self):
        try:
            # Load lightweight models with proper device handling
            self.nlp = spacy.load("en_core_web_sm")
            self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
            self.zero_shot = pipeline("zero-shot-classification", 
                                    model="facebook/bart-large-mnli",
                                    device=0 if torch.cuda.is_available() else -1)
            
            # Load sentence transformer with proper device handling
            self.sentence_transformer = get_jina_model()
        except Exception as e:
            logger.error(f"Error initializing NLPProcessor: {e}")
            self.nlp = None
            self.ner_pipeline = None
            self.zero_shot = None
            self.sentence_transformer = None
        
        # Initialize numerical extraction patterns
        self.number_patterns = {
            'price': r'\$?\d+(?:,\d{3})*(?:\.\d{2})?',
            'sqft': r'\d+(?:,\d{3})*\s*(?:sq\s*ft|square\s*feet)',
            'year': r'(?:19|20)\d{2}',
            'beds': r'\d+\s*(?:bed|beds|bedroom|bedrooms)',
            'baths': r'\d+\s*(?:bath|baths|bathroom|bathrooms)'
        }
        
        # Property status categories
        self.status_categories = [
            "available", "sold", "pending", "under contract", 
            "off market", "coming soon", "active", "inactive"
        ]
        
        # Currency conversion rates (example)
        self.currency_rates = {
            'lakh': 100000,
            'crore': 10000000,
            'million': 1000000,
            'billion': 1000000000
        }

    def convert_currency(self, value: str) -> float:
        """Convert various currency formats to a standard number"""
        try:
            # Remove currency symbols and commas
            value = re.sub(r'[^\d.]', '', value)
            
            # Check for word-based numbers
            value_lower = value.lower()
            for word, multiplier in self.currency_rates.items():
                if word in value_lower:
                    # Extract the number and multiply by the rate
                    num = float(re.sub(r'[^\d.]', '', value))
                    return num * multiplier
            
            # If no special words found, return the number as is
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def extract_numerical_values(self, text: str) -> Dict[str, Union[float, int]]:
        """Extract numerical values from text using regex and NLP"""
        values = {}
        
        # Extract numbers using patterns
        for key, pattern in self.number_patterns.items():
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                value = match.group()
                # Clean and convert value
                if key == 'price':
                    values[key] = self.convert_currency(value)
                elif key in ['sqft', 'beds', 'baths']:
                    values[key] = int(re.sub(r'[^\d]', '', value))
                elif key == 'year':
                    values[key] = int(value)
        
        return values

    def classify_property_status(self, text: str) -> str:
        """Classify property status using zero-shot classification"""
        result = self.zero_shot(
            text,
            candidate_labels=self.status_categories,
            multi_label=False
        )
        return result['labels'][0]

    def extract_landmarks(self, text: str) -> List[str]:
        """Extract landmarks and points of interest using NER"""
        doc = self.nlp(text)
        landmarks = []
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['FAC', 'ORG', 'LOC']:
                landmarks.append(ent.text)
        
        return landmarks

    def semantic_similarity(self, query: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """Calculate semantic similarity between query and candidates"""
        query_embedding = self.sentence_transformer.encode(query)
        candidate_embeddings = self.sentence_transformer.encode(candidates)
        
        similarities = []
        for candidate, embedding in zip(candidates, candidate_embeddings):
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((candidate, float(similarity)))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)

    def process_query(self, query: str) -> Dict:
        """Process a natural language query and extract structured information"""
        # Extract numerical values
        numerical_values = self.extract_numerical_values(query)
        
        # Extract landmarks
        landmarks = self.extract_landmarks(query)
        
        # Classify property status if mentioned
        status = None
        if any(status_word in query.lower() for status_word in self.status_categories):
            status = self.classify_property_status(query)
        
        return {
            'numerical_values': numerical_values,
            'landmarks': landmarks,
            'status': status,
            'original_query': query
        }

    def format_property_details(self, property_data: Dict) -> str:
        """Format property details in a natural language format"""
        details = []
        
        # Basic information
        if 'PropertyName' in property_data:
            details.append(f"Property: {property_data['PropertyName']}")
        if 'Address' in property_data:
            details.append(f"Location: {property_data['Address']}")
        
        # Numerical details
        if 'Beds' in property_data:
            details.append(f"{property_data['Beds']} bedrooms")
        if 'Baths' in property_data:
            details.append(f"{property_data['Baths']} bathrooms")
        if 'LeasableSquareFeet' in property_data:
            details.append(f"{property_data['LeasableSquareFeet']} square feet")
        
        # Status and price
        if 'PropertyStatus' in property_data:
            details.append(f"Status: {property_data['PropertyStatus']}")
        if 'MarketValue' in property_data:
            details.append(f"Price: ${property_data['MarketValue']:,.2f}")
        
        return "\n".join(details) 