import geocoder
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import logging

def get_nearby_properties(latitude, longitude, df, top_k=5):
    """Get properties near a given location"""
    try:
        my_location = (latitude, longitude)

        # Filter out rows with invalid coordinates
        valid_properties = df[
            df['Latitude'].notna() &
            df['Longitude'].notna() &
            df['Latitude'].apply(lambda x: isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '').isdigit())) &
            df['Longitude'].apply(lambda x: isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '').isdigit()))
        ].copy()

        # Convert coordinates to float
        valid_properties['Latitude'] = valid_properties['Latitude'].astype(float)
        valid_properties['Longitude'] = valid_properties['Longitude'].astype(float)

        # Calculate distances
        valid_properties['Distance'] = valid_properties.apply(
            lambda row: geodesic(my_location, (row['Latitude'], row['Longitude'])).miles,
            axis=1
        )

        # Get nearest properties
        nearest_properties = valid_properties.nsmallest(top_k, 'Distance')
        return nearest_properties

    except Exception as e:
        logging.error(f"Error getting nearby properties: {str(e)}")
        return None

def get_location_details(latitude, longitude):
    """Get location details from coordinates"""
    try:
        geolocator = Nominatim(user_agent="hive_prop")
        location = geolocator.reverse(f"{latitude}, {longitude}", language='en')

        if location and location.raw.get('address'):
            address = location.raw['address']
            city = address.get('city') or address.get('town') or address.get('suburb') or address.get('county')
            state = address.get('state')
            country = address.get('country')

            return {
                'city': city,
                'state': state,
                'country': country
            }
        else:
            return None

    except Exception as e:
        logging.error(f"Error getting location details: {str(e)}")
        return None

def set_location(latitude, longitude, session_id, conversation_context):
    """Set location for a session"""
    try:
        location_details = get_location_details(latitude, longitude)
        if location_details:
            conversation_context[session_id] = {
                'location': (latitude, longitude),
                'city': location_details['city'],
                'state': location_details['state'],
                'country': location_details['country']
            }
            return True, location_details
        return False, None
    except Exception as e:
        logging.error(f"Error setting location: {str(e)}")
        return False, None 