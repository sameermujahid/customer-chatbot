import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from ..constraint_parser import ConstraintParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicFeatureMatcher:
    def __init__(self, load_saved=False):
        self.model_path = Path("models/saved_models/feature_matcher")
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.constraint_parser = ConstraintParser()
        
        if load_saved:
            self._load_models()
        else:
            self._initialize_models()
            
    def _load_models(self):
        """Load all saved models and patterns"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load base model
        self.models = {
            'base': SentenceTransformer(str(self.model_path / "base_model")).to(device),
            'zero_shot': pipeline("zero-shot-classification", 
                                model="facebook/bart-large-mnli",
                                device=0 if torch.cuda.is_available() else -1)
        }
        
        # Load feature patterns
        with open(self.model_path / "feature_patterns.pkl", 'rb') as f:
            self.feature_patterns = pickle.load(f)
    
        # Initialize embedding cache
        self.embedding_cache = {}

    def _initialize_models(self):
        """Initialize new models"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {
            'base': SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True).to(device),
            'zero_shot': pipeline("zero-shot-classification", 
                                model="facebook/bart-large-mnli",
                                device=0 if torch.cuda.is_available() else -1)
        }
        
        # Initialize feature patterns
        self.feature_patterns = self._initialize_feature_patterns()
        
        # Initialize embedding cache
        self.embedding_cache = {}

    def _initialize_feature_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize feature patterns for different property types"""
        return {
            'wifi': {
                'semantic_patterns': [
                    "wifi available",
                    "internet access",
                    "wireless internet",
                    "high-speed internet"
                ],
                'context_patterns': [
                    "wifi",
                    "internet",
                    "wireless",
                    "wi-fi"
                ],
                'pg': 'wifiAvailable',
                'commercial': 'wifiAvailable'
            },
            'ac': {
                'semantic_patterns': [
                    "air conditioning",
                    "central air",
                    "climate control",
                    "cooling system"
                ],
                'context_patterns': [
                    "ac",
                    "air conditioning",
                    "central air",
                    "cooling"
                ],
                'pg': 'isACAvailable',
                'commercial': 'isACAvailable'
            },
            'parking': {
                'semantic_patterns': [
                    "parking available",
                    "car parking",
                    "garage",
                    "parking space"
                ],
                'context_patterns': [
                    "parking",
                    "garage",
                    "car space",
                    "vehicle parking"
                ],
                'pg': 'isParkingAvailable',
                'commercial': 'hasParking'
            },
            'power_backup': {
                'semantic_patterns': [
                    "power backup",
                    "generator",
                    "backup power",
                    "uninterrupted power"
                ],
                'context_patterns': [
                    "power backup",
                    "generator",
                    "ups",
                    "inverter"
                ],
                'pg': 'powerBackup',
                'commercial': 'powerBackup'
            }
        }

    def _get_embedding(self, text: str) -> torch.Tensor:
        """Get embedding for text with caching"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
            
        embedding = self.models['base'].encode(text, convert_to_tensor=True)
        self.embedding_cache[text] = embedding
        return embedding

    def _analyze_negation(self, query: str) -> Tuple[bool, float]:
        """Analyze if query contains negation"""
        negation_patterns = [
            "no", "not", "without", "lack of", "missing",
            "doesn't have", "don't have", "doesn't need",
            "don't need", "isn't", "aren't"
        ]
        
        query_lower = query.lower()
        has_negation = any(pattern in query_lower for pattern in negation_patterns)
        
        # Calculate confidence based on negation word position
        confidence = 0.0
        if has_negation:
            words = query_lower.split()
            for i, word in enumerate(words):
                if word in negation_patterns:
                    # Higher confidence if negation is closer to feature words
                    confidence = max(confidence, 1.0 - (i / len(words)))
        
        return has_negation, confidence

    def _analyze_feature_presence(self, query: str, feature: str, patterns: Dict[str, Any]) -> Tuple[bool, float]:
        """Analyze if a feature is mentioned in the query using multiple methods"""
        query_lower = query.lower()
        
        # 1. Zero-shot classification
        zero_shot_result = self.models['zero_shot'](
            query,
            [f"This property has {feature}", f"This property does not have {feature}"]
        )
        
        # 2. Context pattern matching
        context_match = any(pattern in query_lower for pattern in patterns['context_patterns'])
        
        # Combine results with adjusted weights (60% zero-shot, 40% context)
        presence_score = (
            0.6 * zero_shot_result['scores'][0] +
            0.4 * (1.0 if context_match else 0.0)
        )
        
        return presence_score > 0.5, presence_score

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to understand feature requirements and constraints"""
        # Analyze negation
        has_negation, negation_confidence = self._analyze_negation(query)
        
        # Analyze each feature
        feature_requirements = {}
        for feature, patterns in self.feature_patterns.items():
            is_present, confidence = self._analyze_feature_presence(query, feature, patterns)
            if is_present:
                feature_requirements[feature] = not has_negation
        
        # Parse constraints
        constraints = self.constraint_parser.parse_all_constraints(query)
        
        return {
            'feature_requirements': feature_requirements,
            'constraints': constraints,
            'has_negation': has_negation,
            'negation_confidence': negation_confidence
        }

    def check_property_features(self, property_data: Dict[str, Any], feature_requirements: Dict[str, bool]) -> bool:
        """Check if property meets the feature requirements"""
        
        # Debug: Check for propertyId in the input data
        property_id = property_data.get('propertyId', property_data.get('id', property_data.get('PropertyID', 'NOT_FOUND')))
        if property_id != 'NOT_FOUND':
            print(f"🔍 Debug: propertyId in feature check: {property_id}")
        
        property_type = property_data.get('typeName', '').lower()
        
        for feature, required in feature_requirements.items():
            patterns = self.feature_patterns[feature]
            
            if 'pg' in property_type or 'hostel' in property_type:
                pg_details = property_data.get('pgPropertyDetails', {})
                if pg_details and patterns['pg']:
                    if pg_details.get(patterns['pg'], False) != required:
                        return False
            elif any(t in property_type for t in ['office', 'shop', 'commercial']):
                commercial_details = property_data.get('commercialPropertyDetails', {})
                if commercial_details and patterns['commercial']:
                    if commercial_details.get(patterns['commercial'], False) != required:
                        return False
        
        return True

    def filter_properties_by_query(self, properties: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Filter properties based on both feature requirements and constraints"""
        analysis = self.analyze_query(query)
        feature_requirements = analysis['feature_requirements']
        constraints = analysis['constraints']
        
        filtered_properties = []
        
        for property_data in properties:
            # Check feature requirements
            if feature_requirements and not self.check_property_features(property_data, feature_requirements):
                continue
            
            # Check constraints
            if not self.constraint_parser.property_matches_constraints(property_data, constraints):
                continue
            
            filtered_properties.append(property_data)
        
        logger.info(f"Filtered {len(properties)} properties to {len(filtered_properties)} based on query analysis")
        return filtered_properties

    def get_query_analysis_summary(self, query: str) -> str:
        """Get a formatted summary of query analysis"""
        analysis = self.analyze_query(query)
        
        summary_parts = []
        
        # Feature requirements summary
        feature_requirements = analysis['feature_requirements']
        if feature_requirements:
            features = [f"{feature} ({'required' if required else 'not required'})" 
                       for feature, required in feature_requirements.items()]
            summary_parts.append(f"Features: {', '.join(features)}")
        
        # Constraints summary
        constraints_summary = self.constraint_parser.format_constraints_summary(analysis['constraints'])
        if constraints_summary != "No specific constraints":
            summary_parts.append(f"Constraints: {constraints_summary}")
        
        # Negation summary
        if analysis['has_negation']:
            summary_parts.append(f"Negation detected (confidence: {analysis['negation_confidence']:.2f})")
        
        return " | ".join(summary_parts) if summary_parts else "No specific requirements detected"

    def save_models(self):
        """Save all models and patterns"""
        try:
            # Save base model
            self.models['base'].save(str(self.model_path / "base_model"))
            
            # Save feature patterns
            with open(self.model_path / "feature_patterns.pkl", 'wb') as f:
                pickle.dump(self.feature_patterns, f)
                
            logger.info("Successfully saved all models and patterns")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise 