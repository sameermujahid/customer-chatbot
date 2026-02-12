from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import torch
from sentence_transformers import SentenceTransformer
from .constraint_parser import ConstraintParser
from modules.global_models import get_jina_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PropertyProcessor:
    def __init__(self):
        try:
            self.sentence_transformer = get_jina_model()
        except Exception as e:
            logger.error(f"Error initializing PropertyProcessor: {e}")
            self.sentence_transformer = None
        self.constraint_parser = ConstraintParser()
        
    def format_property_details(self, property_data: Dict) -> str:
        """Format property details into a natural language string"""
        print(f"\n=== Formatting property details for: {property_data.get('propertyName', 'Unknown')} ===")
        
        # Debug: Check for propertyId in the input data
        property_id = property_data.get('propertyId', property_data.get('id', property_data.get('PropertyID', 'NOT_FOUND')))
        print(f"🔍 Debug: propertyId from input: {property_id}")
        print(f"🔍 Debug: Available keys: {list(property_data.keys())}")
        
        details = []
        
        # Basic information
        if property_data.get('Address'):
            details.append(f"Located at {property_data['Address']}")
            
        # Handle PG properties specifically
        if property_data.get('typeName', '').lower() == 'pg':
            print("Processing PG property")
            pg_details = property_data.get('pgPropertyDetails', {})
            if pg_details:
                details.append("PG Accommodation")
                if pg_details.get('totalBeds'):
                    details.append(f"Total Beds: {pg_details['totalBeds']}")
                if pg_details.get('availableFor'):
                    details.append(f"Available for: {pg_details['availableFor']}")
                if pg_details.get('foodIncluded'):
                    details.append(f"Food: {pg_details['foodIncluded']}")
                if pg_details.get('wifiAvailable'):
                    details.append(f"WiFi: {'Available' if pg_details['wifiAvailable'] else 'Not Available'}")
        else:
            # Regular property details
            if property_data.get('BHK'):
                details.append(f"{property_data['BHK']} BHK")
            
            if property_data.get('Bathrooms'):
                details.append(f"with {property_data['Bathrooms']} bathrooms")
            
            if property_data.get('Square_Footage'):
                details.append(f"covering {property_data['Square_Footage']} sq ft")
            
            if property_data.get('Year_Built'):
                details.append(f"built in {property_data['Year_Built']}")
            
            if property_data.get('Market_Value'):
                details.append(f"priced at ₹{property_data['Market_Value']:,.2f}")
        
        # Enhanced status display
        if property_data.get('Status'):
            status = property_data['Status'].lower()
            status_display = {
                'available': 'Available for purchase/rent',
                'sold': 'Sold',
                'pending': 'Sale/Rental pending',
                'under contract': 'Under contract',
                'off market': 'Currently off market',
                'coming soon': 'Coming soon to market',
                'active': 'Active listing',
                'inactive': 'Inactive listing'
            }.get(status, f"Status: {property_data['Status']}")
            details.append(status_display)
            
        if property_data.get('Distance'):
            details.append(f"Distance: {property_data['Distance']} miles")
            
        # Add landmark information if available
        if property_data.get('Nearby_Landmarks'):
            landmarks = property_data['Nearby_Landmarks']
            if isinstance(landmarks, list):
                details.append(f"Nearby landmarks: {', '.join(landmarks)}")
            elif isinstance(landmarks, str):
                details.append(f"Nearby landmarks: {landmarks}")
        
        formatted_details = " | ".join(details)
        print(f"Formatted details: {formatted_details}")
        return formatted_details

    def filter_by_constraints(self, properties: List[Dict], query: str) -> List[Dict]:
        """Filter properties based on constraints parsed from query"""
        constraints = self.constraint_parser.parse_all_constraints(query)
        
        # Check if any constraints were found
        has_constraints = any(
            any(value is not None for value in constraint_dict.values())
            for constraint_dict in constraints.values()
        )
        
        if not has_constraints:
            return properties
        
        filtered_properties = []
        for property_data in properties:
            if self.constraint_parser.property_matches_constraints(property_data, constraints):
                filtered_properties.append(property_data)
        
        logging.info(f"Filtered {len(properties)} properties to {len(filtered_properties)} based on constraints")
        return filtered_properties

    def get_constraints_summary(self, query: str) -> str:
        """Get a formatted summary of constraints found in the query"""
        constraints = self.constraint_parser.parse_all_constraints(query)
        return self.constraint_parser.format_constraints_summary(constraints)

    def filter_by_numerical_range(self,
                                properties: List[Dict],
                                field: str,
                                min_value: Optional[float] = None,
                                max_value: Optional[float] = None) -> List[Dict]:
        """Filter properties based on numerical range"""
        filtered_properties = []
        
        for property_data in properties:
            try:
                value = float(property_data.get(field, 0))
                
                if min_value is not None and value < min_value:
                    continue
                    
                if max_value is not None and value > max_value:
                    continue
                    
                filtered_properties.append(property_data)
                
            except (ValueError, TypeError) as e:
                logging.error(f"Error filtering {field}: {str(e)}")
                continue
                
        return filtered_properties

    def filter_by_status(self,
                        properties: List[Dict],
                        status: str) -> List[Dict]:
        """Filter properties by status"""
        return [p for p in properties 
                if p.get('Status', '').lower() == status.lower()]

    def filter_by_bhk(self,
                     properties: List[Dict],
                     bhk: Union[int, str]) -> List[Dict]:
        """Filter properties by BHK count"""
        try:
            bhk_value = int(bhk) if isinstance(bhk, str) else bhk
            return [p for p in properties 
                   if int(p.get('BHK', 0)) == bhk_value]
        except (ValueError, TypeError):
            return []

    def filter_by_bathrooms(self,
                          properties: List[Dict],
                          bathroom_count: Union[int, str]) -> List[Dict]:
        """Filter properties by bathroom count"""
        try:
            bath_value = int(bathroom_count) if isinstance(bathroom_count, str) else bathroom_count
            return [p for p in properties 
                   if int(p.get('Bathrooms', 0)) == bath_value]
        except (ValueError, TypeError):
            return []

    def filter_by_year_built(self,
                           properties: List[Dict],
                           min_year: Optional[int] = None,
                           max_year: Optional[int] = None) -> List[Dict]:
        """Filter properties by year built"""
        return self.filter_by_numerical_range(
            properties, 'Year_Built', min_year, max_year
        )

    def filter_by_square_footage(self,
                               properties: List[Dict],
                               min_sqft: Optional[float] = None,
                               max_sqft: Optional[float] = None) -> List[Dict]:
        """Filter properties by square footage"""
        return self.filter_by_numerical_range(
            properties, 'Square_Footage', min_sqft, max_sqft
        )

    def filter_by_market_value(self,
                             properties: List[Dict],
                             min_value: Optional[float] = None,
                             max_value: Optional[float] = None) -> List[Dict]:
        """Filter properties by market value"""
        return self.filter_by_numerical_range(
            properties, 'Market_Value', min_value, max_value
        )

    def get_property_embedding(self, property_data: Dict) -> np.ndarray:
        """Get embedding for property description"""
        description = self.format_property_details(property_data)
        return self.sentence_transformer.encode(description)

    def find_similar_properties(self,
                              reference_property: Dict,
                              candidate_properties: List[Dict],
                              top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Find properties similar to reference property"""
        ref_embedding = self.get_property_embedding(reference_property)
        
        similarities = []
        for property_data in candidate_properties:
            try:
                prop_embedding = self.get_property_embedding(property_data)
                similarity = np.dot(ref_embedding, prop_embedding) / (
                    np.linalg.norm(ref_embedding) * np.linalg.norm(prop_embedding)
                )
                similarities.append((property_data, float(similarity)))
            except Exception as e:
                logging.error(f"Error calculating similarity: {str(e)}")
                continue
                
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

    def format_zip_code(self, zip_code: Union[str, float, int]) -> str:
        """Format zip code as string"""
        try:
            return str(int(float(zip_code)))
        except (ValueError, TypeError):
            return str(zip_code)

    def process_property_data(self, property_data: Dict) -> Dict:
        """Process and clean property data"""
        processed_data = property_data.copy()
        
        # Format zip code
        if 'Zip_Code' in processed_data:
            processed_data['Zip_Code'] = self.format_zip_code(
                processed_data['Zip_Code']
            )
            
        # Convert numerical fields
        numerical_fields = [
            'Square_Footage', 'Market_Value', 'Year_Built',
            'BHK', 'Bathrooms', 'Latitude', 'Longitude'
        ]
        
        for field in numerical_fields:
            if field in processed_data:
                try:
                    processed_data[field] = float(processed_data[field])
                except (ValueError, TypeError):
                    processed_data[field] = 0.0
                    
        return processed_data 