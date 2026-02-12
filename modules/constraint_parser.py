import re
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class ConstraintParser:
    """
    Parser for extracting property constraints from natural language queries.
    Supports price ranges, bed/bath counts, and other numerical constraints.
    """
    
    def __init__(self):
        self.price_patterns = {
            'under': r'(?:under|below|less than|upto|up to)\s*([\d,]+)',
            'over': r'(?:over|above|more than|greater than)\s*([\d,]+)',
            'between': r'(?:between|from)\s*([\d,]+)\s*(?:and|to|-)\s*([\d,]+)',
            'exact': r'(?:price|cost|worth)\s*([\d,]+)',
            'range_with_currency': r'(?:rs\.?|₹|inr|rupees?)\s*([\d,]+)\s*(?:to|-)\s*([\d,]+)',
            'under_with_currency': r'(?:under|below|less than)\s*(?:rs\.?|₹|inr|rupees?)\s*([\d,]+)',
            'over_with_currency': r'(?:over|above|more than)\s*(?:rs\.?|₹|inr|rupees?)\s*([\d,]+)'
        }
        
        self.bed_patterns = {
            'exact': r'(?:beds?|bhk|bedrooms?)\s*(\d+)',
            'range': r'(?:beds?|bhk|bedrooms?)\s*(\d+)\s*(?:to|-)\s*(\d+)',
            'minimum': r'(?:minimum|at least)\s*(\d+)\s*(?:beds?|bhk|bedrooms?)',
            'maximum': r'(?:maximum|up to)\s*(\d+)\s*(?:beds?|bhk|bedrooms?)'
        }
        
        self.bath_patterns = {
            'exact': r'(?:baths?|bathrooms?)\s*(\d+)',
            'range': r'(?:baths?|bathrooms?)\s*(\d+)\s*(?:to|-)\s*(\d+)',
            'minimum': r'(?:minimum|at least)\s*(\d+)\s*(?:baths?|bathrooms?)',
            'maximum': r'(?:maximum|up to)\s*(\d+)\s*(?:baths?|bathrooms?)'
        }
        
        self.area_patterns = {
            'exact': r'(?:area|sq\s*ft|square\s*feet?)\s*([\d,]+)',
            'range': r'(?:area|sq\s*ft|square\s*feet?)\s*([\d,]+)\s*(?:to|-)\s*([\d,]+)',
            'minimum': r'(?:minimum|at least)\s*([\d,]+)\s*(?:sq\s*ft|square\s*feet?)',
            'maximum': r'(?:maximum|up to)\s*([\d,]+)\s*(?:sq\s*ft|square\s*feet?)'
        }

    def parse_num(self, s: str) -> float:
        """Parse number from string, handling commas and spaces"""
        try:
            return float(s.replace(",", "").replace(" ", "").replace("lakh", "00000").replace("crore", "0000000"))
        except (ValueError, TypeError):
            logger.warning(f"Could not parse number: {s}")
            return 0.0

    def parse_price_constraints(self, query: str) -> Dict[str, Optional[float]]:
        """Parse price constraints from query"""
        constraints = {
            'min_price': None,
            'max_price': None
        }
        
        query_lower = query.lower()
        
        # Check for between range
        between_match = re.search(self.price_patterns['between'], query_lower)
        if between_match:
            constraints['min_price'] = self.parse_num(between_match.group(1))
            constraints['max_price'] = self.parse_num(between_match.group(2))
            return constraints
        
        # Check for range with currency
        range_currency_match = re.search(self.price_patterns['range_with_currency'], query_lower)
        if range_currency_match:
            constraints['min_price'] = self.parse_num(range_currency_match.group(1))
            constraints['max_price'] = self.parse_num(range_currency_match.group(2))
            return constraints
        
        # Check for under/less than
        under_match = re.search(self.price_patterns['under'], query_lower)
        if under_match:
            constraints['max_price'] = self.parse_num(under_match.group(1))
        
        # Check for under with currency
        under_currency_match = re.search(self.price_patterns['under_with_currency'], query_lower)
        if under_currency_match:
            constraints['max_price'] = self.parse_num(under_currency_match.group(1))
        
        # Check for over/more than
        over_match = re.search(self.price_patterns['over'], query_lower)
        if over_match:
            constraints['min_price'] = self.parse_num(over_match.group(1))
        
        # Check for over with curr  ency
        over_currency_match = re.search(self.price_patterns['over_with_currency'], query_lower)
        if over_currency_match:
            constraints['min_price'] = self.parse_num(over_currency_match.group(1))
        
        # Check for exact price
        exact_match = re.search(self.price_patterns['exact'], query_lower)
        if exact_match:
            price = self.parse_num(exact_match.group(1))
            constraints['min_price'] = price
            constraints['max_price'] = price
        
        return constraints

    def parse_bed_constraints(self, query: str) -> Dict[str, Optional[int]]:
        """Parse bedroom constraints from query"""
        constraints = {
            'min_beds': None,
            'max_beds': None
        }
        
        query_lower = query.lower()
        
        # Check for range
        range_match = re.search(self.bed_patterns['range'], query_lower)
        if range_match:
            constraints['min_beds'] = int(range_match.group(1))
            constraints['max_beds'] = int(range_match.group(2))
            return constraints
        
        # Check for exact count
        exact_match = re.search(self.bed_patterns['exact'], query_lower)
        if exact_match:
            beds = int(exact_match.group(1))
            constraints['min_beds'] = beds
            constraints['max_beds'] = beds
            return constraints
        
        # Check for minimum
        min_match = re.search(self.bed_patterns['minimum'], query_lower)
        if min_match:
            constraints['min_beds'] = int(min_match.group(1))
        
        # Check for maximum
        max_match = re.search(self.bed_patterns['maximum'], query_lower)
        if max_match:
            constraints['max_beds'] = int(max_match.group(1))
        
        return constraints

    def parse_bath_constraints(self, query: str) -> Dict[str, Optional[int]]:
        """Parse bathroom constraints from query"""
        constraints = {
            'min_baths': None,
            'max_baths': None
        }
        
        query_lower = query.lower()
        
        # Check for range
        range_match = re.search(self.bath_patterns['range'], query_lower)
        if range_match:
            constraints['min_baths'] = int(range_match.group(1))
            constraints['max_baths'] = int(range_match.group(2))
            return constraints
        
        # Check for exact count
        exact_match = re.search(self.bath_patterns['exact'], query_lower)
        if exact_match:
            baths = int(exact_match.group(1))
            constraints['min_baths'] = baths
            constraints['max_baths'] = baths
            return constraints
        
        # Check for minimum
        min_match = re.search(self.bath_patterns['minimum'], query_lower)
        if min_match:
            constraints['min_baths'] = int(min_match.group(1))
        
        # Check for maximum
        max_match = re.search(self.bath_patterns['maximum'], query_lower)
        if max_match:
            constraints['max_baths'] = int(max_match.group(1))
        
        return constraints

    def parse_area_constraints(self, query: str) -> Dict[str, Optional[float]]:
        """Parse area constraints from query"""
        constraints = {
            'min_area': None,
            'max_area': None
        }
        
        query_lower = query.lower()
        
        # Check for range
        range_match = re.search(self.area_patterns['range'], query_lower)
        if range_match:
            constraints['min_area'] = self.parse_num(range_match.group(1))
            constraints['max_area'] = self.parse_num(range_match.group(2))
            return constraints
        
        # Check for exact area
        exact_match = re.search(self.area_patterns['exact'], query_lower)
        if exact_match:
            area = self.parse_num(exact_match.group(1))
            constraints['min_area'] = area
            constraints['max_area'] = area
            return constraints
        
        # Check for minimum
        min_match = re.search(self.area_patterns['minimum'], query_lower)
        if min_match:
            constraints['min_area'] = self.parse_num(min_match.group(1))
        
        # Check for maximum
        max_match = re.search(self.area_patterns['maximum'], query_lower)
        if max_match:
            constraints['max_area'] = self.parse_num(max_match.group(1))
        
        return constraints

    def parse_all_constraints(self, query: str) -> Dict[str, Dict[str, Optional[Union[float, int]]]]:
        """Parse all constraints from a query"""
        return {
            'price': self.parse_price_constraints(query),
            'beds': self.parse_bed_constraints(query),
            'baths': self.parse_bath_constraints(query),
            'area': self.parse_area_constraints(query)
        }

    def property_matches_constraints(self, property_data: Dict, constraints: Dict) -> bool:
        """Check if property matches the parsed constraints"""
        
        # Debug: Check for propertyId in the input data
        property_id = property_data.get('propertyId', property_data.get('id', property_data.get('PropertyID', 'NOT_FOUND')))
        if property_id != 'NOT_FOUND':
            print(f"🔍 Debug: propertyId in constraint check: {property_id}")
        
        # Check price constraints
        price_constraints = constraints.get('price', {})
        if price_constraints['min_price'] is not None or price_constraints['max_price'] is not None:
            price = property_data.get('marketValue')
            if price is not None:
                try:
                    price = float(price)
                except (ValueError, TypeError):
                    price = None
                
                if price is not None:
                    if price_constraints['min_price'] is not None and price < price_constraints['min_price']:
                        return False
                    if price_constraints['max_price'] is not None and price > price_constraints['max_price']:
                        return False
        
        # Check bed constraints
        bed_constraints = constraints.get('beds', {})
        if bed_constraints['min_beds'] is not None or bed_constraints['max_beds'] is not None:
            beds = property_data.get('numberOfRooms')
            if beds is not None:
                try:
                    beds = int(beds)
                except (ValueError, TypeError):
                    beds = None
                
                if beds is not None:
                    if bed_constraints['min_beds'] is not None and beds < bed_constraints['min_beds']:
                        return False
                    if bed_constraints['max_beds'] is not None and beds > bed_constraints['max_beds']:
                        return False
        
        # Check bath constraints
        bath_constraints = constraints.get('baths', {})
        if bath_constraints['min_baths'] is not None or bath_constraints['max_baths'] is not None:
            baths = None
            
            # Try to get from commercialPropertyDetails
            comm = property_data.get('commercialPropertyDetails', {})
            if comm and 'washrooms' in comm:
                try:
                    baths = int(comm['washrooms'])
                except (ValueError, TypeError):
                    baths = None
            
            # Try to get from pgPropertyDetails if available
            if baths is None:
                pg = property_data.get('pgPropertyDetails', {})
                if pg and 'washrooms' in pg:
                    try:
                        baths = int(pg['washrooms'])
                    except (ValueError, TypeError):
                        baths = None
            
            if baths is not None:
                if bath_constraints['min_baths'] is not None and baths < bath_constraints['min_baths']:
                    return False
                if bath_constraints['max_baths'] is not None and baths > bath_constraints['max_baths']:
                    return False
        
        # Check area constraints
        area_constraints = constraints.get('area', {})
        if area_constraints['min_area'] is not None or area_constraints['max_area'] is not None:
            area = property_data.get('totalSquareFeet')
            if area is not None:
                try:
                    area = float(area)
                except (ValueError, TypeError):
                    area = None
                
                if area is not None:
                    if area_constraints['min_area'] is not None and area < area_constraints['min_area']:
                        return False
                    if area_constraints['max_area'] is not None and area > area_constraints['max_area']:
                        return False
        
        return True

    def format_constraints_summary(self, constraints: Dict) -> str:
        """Format constraints into a readable summary"""
        summary_parts = []
        
        # Price summary
        price = constraints.get('price', {})
        if price['min_price'] is not None and price['max_price'] is not None:
            if price['min_price'] == price['max_price']:
                summary_parts.append(f"Price: ₹{price['min_price']:,.0f}")
            else:
                summary_parts.append(f"Price: ₹{price['min_price']:,.0f} - ₹{price['max_price']:,.0f}")
        elif price['min_price'] is not None:
            summary_parts.append(f"Price: Above ₹{price['min_price']:,.0f}")
        elif price['max_price'] is not None:
            summary_parts.append(f"Price: Below ₹{price['max_price']:,.0f}")
        
        # Bed summary
        beds = constraints.get('beds', {})
        if beds['min_beds'] is not None and beds['max_beds'] is not None:
            if beds['min_beds'] == beds['max_beds']:
                summary_parts.append(f"{beds['min_beds']} BHK")
            else:
                summary_parts.append(f"{beds['min_beds']}-{beds['max_beds']} BHK")
        elif beds['min_beds'] is not None:
            summary_parts.append(f"Minimum {beds['min_beds']} BHK")
        elif beds['max_beds'] is not None:
            summary_parts.append(f"Maximum {beds['max_beds']} BHK")
        
        # Bath summary
        baths = constraints.get('baths', {})
        if baths['min_baths'] is not None and baths['max_baths'] is not None:
            if baths['min_baths'] == baths['max_baths']:
                summary_parts.append(f"{baths['min_baths']} bathrooms")
            else:
                summary_parts.append(f"{baths['min_baths']}-{baths['max_baths']} bathrooms")
        elif baths['min_baths'] is not None:
            summary_parts.append(f"Minimum {baths['min_baths']} bathrooms")
        elif baths['max_baths'] is not None:
            summary_parts.append(f"Maximum {baths['max_baths']} bathrooms")
        
        # Area summary
        area = constraints.get('area', {})
        if area['min_area'] is not None and area['max_area'] is not None:
            if area['min_area'] == area['max_area']:
                summary_parts.append(f"{area['min_area']:,.0f} sq ft")
            else:
                summary_parts.append(f"{area['min_area']:,.0f} - {area['max_area']:,.0f} sq ft")
        elif area['min_area'] is not None:
            summary_parts.append(f"Minimum {area['min_area']:,.0f} sq ft")
        elif area['max_area'] is not None:
            summary_parts.append(f"Maximum {area['max_area']:,.0f} sq ft")
        
        return " | ".join(summary_parts) if summary_parts else "No specific constraints" 