import numpy as np
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from typing import Dict, List, Tuple, Optional
import logging
from sentence_transformers import SentenceTransformer
import torch
import math
from modules.models import get_cached_properties
from modules.global_models import get_jina_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocationProcessor:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="real_estate_app")
        try:
            self.sentence_transformer = get_jina_model()
        except Exception as e:
            logger.error(f"Error initializing LocationProcessor: {e}")
            self.sentence_transformer = None
        
    def get_location_details(self, latitude: float, longitude: float) -> Dict:
        """Get detailed location information from coordinates"""
        try:
            location = self.geolocator.reverse(f"{latitude}, {longitude}", language='en')
            if location and location.raw.get('address'):
                address = location.raw['address']
                return {
                    'city': address.get('city') or address.get('town') or address.get('suburb'),
                    'state': address.get('state'),
                    'country': address.get('country'),
                    'postcode': address.get('postcode'),
                    'road': address.get('road'),
                    'neighbourhood': address.get('neighbourhood'),
                    'suburb': address.get('suburb')
                }
        except Exception as e:
            logging.error(f"Error getting location details: {str(e)}")
        return {}

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        try:
            # Convert latitude and longitude from degrees to radians
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            r = 6371  # Radius of earth in kilometers
            
            return c * r
        except Exception as e:
            logging.error(f"Error calculating distance: {str(e)}")
            return float('inf')

    def find_nearby_properties(self, latitude: float, longitude: float, radius_km: float = 10.0) -> List[Dict]:
        """Find properties within specified radius of given coordinates"""
        print(f"\n=== Finding nearby properties ===")
        print(f"Searching within {radius_km}km of coordinates: {latitude}, {longitude}")
        
        try:
            properties = get_cached_properties()
            if not properties:
                print("No properties available in cache")
                return []
            
            # Debug: Check first property for ID fields
            if properties:
                first_prop = properties[0]
                print(f"🔍 Debug: First property keys: {list(first_prop.keys())}")
                # Check for any field that might contain an ID
                for key, value in first_prop.items():
                    if 'id' in key.lower() or 'property' in key.lower():
                        print(f"🔍 Debug: Potential ID field '{key}': {value}")
                
            nearby_properties = []
            for prop in properties:
                try:
                    # Get property location - handle both location object and direct lat/lon fields
                    prop_lat = None
                    prop_lon = None
                    
                    # Try to get location from location object
                    location = prop.get('location', {})
                    if location and isinstance(location, dict):
                        prop_lat = location.get('latitude')
                        prop_lon = location.get('longitude')
                    
                    # If not found in location object, try direct fields
                    if prop_lat is None or prop_lon is None:
                        prop_lat = prop.get('Latitude')
                        prop_lon = prop.get('Longitude')
                    
                    # Skip if coordinates are missing or invalid
                    if not prop_lat or not prop_lon:
                        continue
                        
                    try:
                        prop_lat = float(prop_lat)
                        prop_lon = float(prop_lon)
                    except (ValueError, TypeError):
                        continue
                    
                    # Skip if coordinates are zero or invalid
                    if prop_lat == 0 or prop_lon == 0:
                        continue
                    
                    # Calculate distance
                    distance = self.calculate_distance(latitude, longitude, prop_lat, prop_lon)
                    
                    # Add distance to property data
                    prop['Distance'] = round(distance, 2)
                    
                    # Check if property is within radius
                    if distance <= radius_km:
                        print(f"Found nearby property: {prop.get('propertyName', 'Unnamed Property')} at {distance}km")
                        nearby_properties.append(prop)
                        
                except Exception as e:
                    print(f"Error processing property: {str(e)}")
                    continue
                    
            # Sort by distance
            nearby_properties.sort(key=lambda x: x.get('Distance', float('inf')))
            print(f"Found {len(nearby_properties)} properties within {radius_km}km")
            return nearby_properties
            
        except Exception as e:
            print(f"Error finding nearby properties: {str(e)}")
            return []

    def set_location(self, latitude: float, longitude: float, session_id: str) -> Dict:
        """Set user location and find nearby properties"""
        print(f"\n=== Setting location ===")
        print(f"Latitude: {latitude}")
        print(f"Longitude: {longitude}")
        print(f"Session ID: {session_id}")
        
        try:
            # Get location details
            location_details = self.get_location_details(latitude, longitude)
            print(f"Location details: {location_details}")
            
            # Find nearby properties
            nearby_properties = self.find_nearby_properties(latitude, longitude)
            
            # Format the response
            response = {
                "status": "success",
                "message": f"Found {len(nearby_properties)} properties nearby",
                "location": location_details,
                "properties": nearby_properties
            }
            
            # Add more detailed information if properties were found
            if nearby_properties:
                response["nearest_property"] = {
                    "name": nearby_properties[0].get('propertyName', 'Unnamed Property'),
                    "distance": nearby_properties[0].get('Distance', 0),
                    "address": nearby_properties[0].get('Address', 'No address available')
                }
            
            return response
            
        except Exception as e:
            print(f"Error in set_location: {str(e)}")
            return {
                "status": "error",
                "message": "Error processing location",
                "properties": []
            }

    def calculate_distances(self, 
                          reference_point: Tuple[float, float], 
                          properties: List[Dict]) -> List[Dict]:
        """Calculate distances between reference point and properties"""
        distances = []
        for property_data in properties:
            try:
                prop_lat = float(property_data.get('Latitude', 0))
                prop_lon = float(property_data.get('Longitude', 0))
                
                if prop_lat and prop_lon:
                    distance = geodesic(reference_point, (prop_lat, prop_lon)).miles
                    property_data['Distance'] = round(distance, 2)
                    distances.append(property_data)
            except (ValueError, TypeError) as e:
                logging.error(f"Error calculating distance: {str(e)}")
                continue
                
        return sorted(distances, key=lambda x: x.get('Distance', float('inf')))

    def find_nearby_landmarks(self, 
                            latitude: float, 
                            longitude: float, 
                            radius_miles: float = 5.0) -> List[Dict]:
        """Find landmarks near a given location"""
        try:
            # Use Nominatim to search for nearby places
            query = f"amenity near {latitude}, {longitude}"
            places = self.geolocator.geocode(query, exactly_one=False, limit=10)
            
            landmarks = []
            if places:
                for place in places:
                    try:
                        place_lat = float(place.raw.get('lat', 0))
                        place_lon = float(place.raw.get('lon', 0))
                        
                        if place_lat and place_lon:
                            distance = geodesic((latitude, longitude), 
                                             (place_lat, place_lon)).miles
                            
                            if distance <= radius_miles:
                                landmarks.append({
                                    'name': place.raw.get('display_name', 'Unknown'),
                                    'type': place.raw.get('type', 'Unknown'),
                                    'distance': round(distance, 2)
                                })
                    except (ValueError, TypeError):
                        continue
                        
            return sorted(landmarks, key=lambda x: x['distance'])
            
        except Exception as e:
            logging.error(f"Error finding nearby landmarks: {str(e)}")
            return []

    def filter_by_location_criteria(self,
                                  properties: List[Dict],
                                  criteria: Dict) -> List[Dict]:
        """Filter properties based on location criteria"""
        filtered_properties = []
        
        for property_data in properties:
            try:
                # Check if property meets all criteria
                meets_criteria = True
                
                # Check distance if specified
                if 'max_distance' in criteria:
                    if property_data.get('Distance', float('inf')) > criteria['max_distance']:
                        meets_criteria = False
                
                # Check landmarks if specified
                if 'nearby_landmarks' in criteria:
                    property_landmarks = self.find_nearby_landmarks(
                        float(property_data.get('Latitude', 0)),
                        float(property_data.get('Longitude', 0))
                    )
                    landmark_names = [l['name'].lower() for l in property_landmarks]
                    if not any(landmark.lower() in landmark_names 
                             for landmark in criteria['nearby_landmarks']):
                        meets_criteria = False
                
                if meets_criteria:
                    filtered_properties.append(property_data)
                    
            except Exception as e:
                logging.error(f"Error filtering property: {str(e)}")
                continue
                
        return filtered_properties

    def get_location_embedding(self, location_text: str) -> np.ndarray:
        """Get embedding for location text"""
        return self.sentence_transformer.encode(location_text)

    def find_similar_locations(self,
                             reference_location: str,
                             candidate_locations: List[str],
                             top_k: int = 5) -> List[Tuple[str, float]]:
        """Find locations similar to reference location"""
        ref_embedding = self.get_location_embedding(reference_location)
        candidate_embeddings = self.sentence_transformer.encode(candidate_locations)
        
        similarities = []
        for location, embedding in zip(candidate_locations, candidate_embeddings):
            similarity = np.dot(ref_embedding, embedding) / (
                np.linalg.norm(ref_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((location, float(similarity)))
            
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

def find_nearby_properties(latitude: float, longitude: float, radius_km: float = 5.0) -> List[Dict]:
    """Find properties within specified radius of given coordinates"""
    print(f"\n=== Finding nearby properties ===")
    print(f"Searching within {radius_km}km of coordinates: {latitude}, {longitude}")
    
    try:
        properties = get_cached_properties()
        if not properties:
            print("No properties available in cache")
            return []
            
        nearby_properties = []
        for prop in properties:
            try:
                # Get property location
                location = prop.get('location', {})
                if not location:
                    continue
                    
                prop_lat = float(location.get('latitude', 0))
                prop_lon = float(location.get('longitude', 0))
                
                if prop_lat == 0 or prop_lon == 0:
                    continue
                
                # Calculate distance
                distance = calculate_distance(latitude, longitude, prop_lat, prop_lon)
                
                # Add distance to property data
                prop['Distance'] = round(distance, 2)
                
                # Check if property is within radius
                if distance <= radius_km:
                    print(f"Found nearby property: {prop.get('propertyName')} at {distance}km")
                    nearby_properties.append(prop)
                    
            except Exception as e:
                print(f"Error processing property: {str(e)}")
                continue
                
        # Sort by distance
        nearby_properties.sort(key=lambda x: x.get('Distance', float('inf')))
        print(f"Found {len(nearby_properties)} properties within {radius_km}km")
        return nearby_properties
        
    except Exception as e:
        print(f"Error finding nearby properties: {str(e)}")
        return []

def set_location(latitude: float, longitude: float, session_id: str) -> Dict:
    """Set user location and find nearby properties"""
    print(f"\n=== Setting location ===")
    print(f"Latitude: {latitude}")
    print(f"Longitude: {longitude}")
    print(f"Session ID: {session_id}")
    
    try:
        # Find nearby properties
        nearby_properties = find_nearby_properties(latitude, longitude)
        
        return {
            "status": "success",
            "message": f"Found {len(nearby_properties)} properties nearby",
            "properties": nearby_properties
        }
        
    except Exception as e:
        print(f"Error in set_location: {str(e)}")
        return {
            "status": "error",
            "message": "Error processing location",
            "properties": []
        } 