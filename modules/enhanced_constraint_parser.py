import re
import logging
from typing import Dict, Optional, Union, List, Any
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np

logger = logging.getLogger(__name__)

class EnhancedConstraintParser:
    """
    Enhanced parser for extracting property constraints from natural language queries.
    Uses Google T5 model for better numerical understanding and dynamic constraint parsing.
    """
    
    def __init__(self, model_name: str = "google/t5-small"):
        """
        Initialize the enhanced constraint parser with T5 model
        
        Args:
            model_name: HuggingFace model name for T5 (default: google/t5-small for CPU efficiency)
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cpu"  # Use CPU for efficiency
        
        # Initialize T5 model for constraint parsing
        self._load_t5_model()
        
        # Enhanced patterns for better constraint extraction
        self.price_patterns = {
            'under': r'(?:under|below|less than|upto|up to|maximum|max)\s*([\d,]+(?:\s*(?:lakh|crore|cr|lk|thousand|k))?)',
            'over': r'(?:over|above|more than|greater than|minimum|min)\s*([\d,]+(?:\s*(?:lakh|crore|cr|lk|thousand|k))?)',
            'between': r'(?:between|from)\s*([\d,]+(?:\s*(?:lakh|crore|cr|lk|thousand|k))?)\s*(?:and|to|-)\s*([\d,]+(?:\s*(?:lakh|crore|cr|lk|thousand|k))?)',
            'exact': r'(?:price|cost|worth|exactly)\s*([\d,]+(?:\s*(?:lakh|crore|cr|lk|thousand|k))?)',
            'range_with_currency': r'(?:rs\.?|₹|inr|rupees?)\s*([\d,]+(?:\s*(?:lakh|crore|cr|lk|thousand|k))?)\s*(?:to|-)\s*([\d,]+(?:\s*(?:lakh|crore|cr|lk|thousand|k))?)',
            'under_with_currency': r'(?:under|below|less than)\s*(?:rs\.?|₹|inr|rupees?)\s*([\d,]+(?:\s*(?:lakh|crore|cr|lk|thousand|k))?)',
            'over_with_currency': r'(?:over|above|more than)\s*(?:rs\.?|₹|inr|rupees?)\s*([\d,]+(?:\s*(?:lakh|crore|cr|lk|thousand|k))?)'
        }
        
        self.bed_patterns = {
            'exact': r'(?:beds?|bhk|bedrooms?)\s*(\d+)',
            'range': r'(?:beds?|bhk|bedrooms?)\s*(\d+)\s*(?:to|-)\s*(\d+)',
            'minimum': r'(?:minimum|at least|min)\s*(\d+)\s*(?:beds?|bhk|bedrooms?)',
            'maximum': r'(?:maximum|up to|max)\s*(\d+)\s*(?:beds?|bhk|bedrooms?)'
        }
        
        self.bath_patterns = {
            'exact': r'(?:baths?|bathrooms?)\s*(\d+)',
            'range': r'(?:baths?|bathrooms?)\s*(\d+)\s*(?:to|-)\s*(\d+)',
            'minimum': r'(?:minimum|at least|min)\s*(\d+)\s*(?:baths?|bathrooms?)',
            'maximum': r'(?:maximum|up to|max)\s*(\d+)\s*(?:baths?|bathrooms?)'
        }
        
        self.area_patterns = {
            'exact': r'(?:area|sq\s*ft|square\s*feet?|sqft)\s*([\d,]+)',
            'range': r'(?:area|sq\s*ft|square\s*feet?|sqft)\s*([\d,]+)\s*(?:to|-)\s*([\d,]+)',
            'minimum': r'(?:minimum|at least|min)\s*([\d,]+)\s*(?:sq\s*ft|square\s*feet?|sqft)',
            'maximum': r'(?:maximum|up to|max)\s*([\d,]+)\s*(?:sq\s*ft|square\s*feet?|sqft)',
            'under': r'(?:under|below|less than|upto|up to|maximum|max)\s*([\d,]+)\s*(?:sq\s*ft|square\s*feet?|sqft)',
            'over': r'(?:over|above|more than|greater than|minimum|min)\s*([\d,]+)\s*(?:sq\s*ft|square\s*feet?|sqft)'
        }
        
        self.year_patterns = {
            'before': r'(?:built\s*before|before|older\s*than|constructed\s*before)\s*(\d{4})',
            'after': r'(?:built\s*after|after|newer\s*than|constructed\s*after)\s*(\d{4})',
            'between': r'(?:built\s*between|between)\s*(\d{4})\s*(?:and|to|-)\s*(\d{4})',
            'exact': r'(?:built\s*in|year|constructed\s*in)\s*(\d{4})'
        }
        
        self.property_type_patterns = {
            'villa': r'(?:villa|villas|independent\s*house|detached\s*house)',
            'apartment': r'(?:apartment|flat|condo|condominium)',
            'house': r'(?:house|home|residential)',
            'commercial': r'(?:commercial|office|shop|retail)',
            'pg': r'(?:pg|paying\s*guest|guest\s*house)'
        }

    def _load_t5_model(self):
        """Load T5 model for constraint parsing"""
        try:
            logger.info(f"Loading T5 model: {self.model_name}")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("✅ T5 model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load T5 model: {e}. Falling back to regex parsing.")
            self.tokenizer = None
            self.model = None

    def parse_num_with_t5(self, text: str) -> float:
        """Parse number from text using T5 model for better understanding"""
        if self.model is None:
            return self.parse_num_regex(text)
        
        try:
            # Create a prompt for number extraction
            prompt = f"Extract the numerical value from this text: {text}"
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = inputs.to(self.device)
            
            # Generate output
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=50,
                    num_beams=4,
                    early_stopping=True,
                    temperature=0.1
                )
            
            # Decode output
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract number from result
            number_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)', result)
            if number_match:
                return self.parse_num_regex(number_match.group(1))
            
            # Fallback to regex
            return self.parse_num_regex(text)
            
        except Exception as e:
            logger.warning(f"T5 parsing failed for '{text}': {e}. Falling back to regex.")
            return self.parse_num_regex(text)

    def parse_num_regex(self, s: str) -> float:
        """Parse number from string using regex, handling commas and Indian number system"""
        try:
            # Remove spaces and convert to lowercase
            s = s.replace(" ", "").lower()
            
            # Handle Indian number system (lakh, crore)
            if 'lakh' in s or 'lk' in s:
                s = s.replace('lakh', '').replace('lk', '')
                return float(s.replace(",", "")) * 100000
            elif 'crore' in s or 'cr' in s:
                s = s.replace('crore', '').replace('cr', '')
                return float(s.replace(",", "")) * 10000000
            elif 'thousand' in s or 'k' in s:
                s = s.replace('thousand', '').replace('k', '')
                return float(s.replace(",", "")) * 1000
            
            # Handle regular numbers
            return float(s.replace(",", ""))
        except (ValueError, TypeError):
            logger.warning(f"Could not parse number: {s}")
            return 0.0

    def extract_constraints_with_t5(self, query: str) -> Dict[str, Any]:
        """Extract constraints using T5 model for better understanding"""
        if self.model is None:
            return self.parse_all_constraints_regex(query)
        
        try:
            # Create a structured prompt for constraint extraction
            prompt = f"""
            Extract property constraints from this query: "{query}"
            
            Return constraints in this format:
            - Price: min/max/exact value
            - Beds: min/max/exact count
            - Baths: min/max/exact count
            - Area: min/max/exact sq ft
            - Year: before/after/exact year
            - Type: property type
            """
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = inputs.to(self.device)
            
            # Generate output
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=200,
                    num_beams=4,
                    early_stopping=True,
                    temperature=0.1
                )
            
            # Decode output
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse the structured output
            constraints = self._parse_t5_output(result, query)
            
            return constraints
            
        except Exception as e:
            logger.warning(f"T5 constraint extraction failed: {e}. Falling back to regex.")
            return self.parse_all_constraints_regex(query)

    def _parse_t5_output(self, t5_output: str, original_query: str) -> Dict[str, Any]:
        """Parse T5 model output to extract constraints"""
        constraints = {
            'price': {'min_price': None, 'max_price': None},
            'beds': {'min_beds': None, 'max_beds': None},
            'baths': {'min_baths': None, 'max_baths': None},
            'area': {'min_area': None, 'max_area': None},
            'year': {'min_year': None, 'max_year': None},
            'property_type': None
        }
        
        # Extract price constraints
        price_match = re.search(r'price:\s*(min|max|exact)\s*([\d,]+(?:\s*(?:lakh|crore|cr|lk))?)', t5_output.lower())
        if price_match:
            constraint_type = price_match.group(1)
            value = self.parse_num_regex(price_match.group(2))
            if constraint_type == 'min':
                constraints['price']['min_price'] = value
            elif constraint_type == 'max':
                constraints['price']['max_price'] = value
            elif constraint_type == 'exact':
                constraints['price']['min_price'] = value
                constraints['price']['max_price'] = value
        
        # Extract bed constraints
        bed_match = re.search(r'beds?:\s*(min|max|exact)\s*(\d+)', t5_output.lower())
        if bed_match:
            constraint_type = bed_match.group(1)
            value = int(bed_match.group(2))
            if constraint_type == 'min':
                constraints['beds']['min_beds'] = value
            elif constraint_type == 'max':
                constraints['beds']['max_beds'] = value
            elif constraint_type == 'exact':
                constraints['beds']['min_beds'] = value
                constraints['beds']['max_beds'] = value
        
        # Extract bath constraints
        bath_match = re.search(r'baths?:\s*(min|max|exact)\s*(\d+)', t5_output.lower())
        if bath_match:
            constraint_type = bath_match.group(1)
            value = int(bath_match.group(2))
            if constraint_type == 'min':
                constraints['baths']['min_baths'] = value
            elif constraint_type == 'max':
                constraints['baths']['max_baths'] = value
            elif constraint_type == 'exact':
                constraints['baths']['min_baths'] = value
                constraints['baths']['max_baths'] = value
        
        # Extract area constraints
        area_match = re.search(r'area:\s*(min|max|exact)\s*([\d,]+)', t5_output.lower())
        if area_match:
            constraint_type = area_match.group(1)
            value = self.parse_num_regex(area_match.group(2))
            if constraint_type == 'min':
                constraints['area']['min_area'] = value
            elif constraint_type == 'max':
                constraints['area']['max_area'] = value
            elif constraint_type == 'exact':
                constraints['area']['min_area'] = value
                constraints['area']['max_area'] = value
        
        # Extract year constraints
        year_match = re.search(r'year:\s*(before|after|exact)\s*(\d{4})', t5_output.lower())
        if year_match:
            constraint_type = year_match.group(1)
            value = int(year_match.group(2))
            if constraint_type == 'before':
                constraints['year']['max_year'] = value
            elif constraint_type == 'after':
                constraints['year']['min_year'] = value
            elif constraint_type == 'exact':
                constraints['year']['min_year'] = value
                constraints['year']['max_year'] = value
        
        # Extract property type
        type_match = re.search(r'type:\s*(villa|apartment|house|commercial|pg)', t5_output.lower())
        if type_match:
            constraints['property_type'] = type_match.group(1)
        
        return constraints

    def parse_price_constraints(self, query: str) -> Dict[str, Optional[float]]:
        """Parse price constraints from query using enhanced patterns"""
        constraints = {
            'min_price': None,
            'max_price': None
        }
        
        query_lower = query.lower()
        
        # Check for between range
        between_match = re.search(self.price_patterns['between'], query_lower)
        if between_match:
            constraints['min_price'] = self.parse_num_regex(between_match.group(1))
            constraints['max_price'] = self.parse_num_regex(between_match.group(2))
            return constraints
        
        # Check for range with currency
        range_currency_match = re.search(self.price_patterns['range_with_currency'], query_lower)
        if range_currency_match:
            constraints['min_price'] = self.parse_num_regex(range_currency_match.group(1))
            constraints['max_price'] = self.parse_num_regex(range_currency_match.group(2))
            return constraints
        
        # Check for under/less than
        under_match = re.search(self.price_patterns['under'], query_lower)
        if under_match:
            constraints['max_price'] = self.parse_num_regex(under_match.group(1))
        
        # Check for under with currency
        under_currency_match = re.search(self.price_patterns['under_with_currency'], query_lower)
        if under_currency_match:
            constraints['max_price'] = self.parse_num_regex(under_currency_match.group(1))
        
        # Check for over/more than
        over_match = re.search(self.price_patterns['over'], query_lower)
        if over_match:
            constraints['min_price'] = self.parse_num_regex(over_match.group(1))
        
        # Check for over with currency
        over_currency_match = re.search(self.price_patterns['over_with_currency'], query_lower)
        if over_currency_match:
            constraints['min_price'] = self.parse_num_regex(over_currency_match.group(1))
        
        # Check for exact price
        exact_match = re.search(self.price_patterns['exact'], query_lower)
        if exact_match:
            price = self.parse_num_regex(exact_match.group(1))
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
            constraints['min_area'] = self.parse_num_regex(range_match.group(1))
            constraints['max_area'] = self.parse_num_regex(range_match.group(2))
            return constraints
        
        # Check for exact area
        exact_match = re.search(self.area_patterns['exact'], query_lower)
        if exact_match:
            area = self.parse_num_regex(exact_match.group(1))
            constraints['min_area'] = area
            constraints['max_area'] = area
            return constraints
        
        # Check for minimum
        min_match = re.search(self.area_patterns['minimum'], query_lower)
        if min_match:
            constraints['min_area'] = self.parse_num_regex(min_match.group(1))
        
        # Check for maximum
        max_match = re.search(self.area_patterns['maximum'], query_lower)
        if max_match:
            constraints['max_area'] = self.parse_num_regex(max_match.group(1))
        
        # Check for under
        under_match = re.search(self.area_patterns['under'], query_lower)
        if under_match:
            constraints['max_area'] = self.parse_num_regex(under_match.group(1))
        
        # Check for over
        over_match = re.search(self.area_patterns['over'], query_lower)
        if over_match:
            constraints['min_area'] = self.parse_num_regex(over_match.group(1))
        
        return constraints

    def parse_year_constraints(self, query: str) -> Dict[str, Optional[int]]:
        """Parse year constraints from query"""
        constraints = {
            'min_year': None,
            'max_year': None
        }
        
        query_lower = query.lower()
        
        # Check for before
        before_match = re.search(self.year_patterns['before'], query_lower)
        if before_match:
            constraints['max_year'] = int(before_match.group(1))
        
        # Check for after
        after_match = re.search(self.year_patterns['after'], query_lower)
        if after_match:
            constraints['min_year'] = int(after_match.group(1))
        
        # Check for between
        between_match = re.search(self.year_patterns['between'], query_lower)
        if between_match:
            constraints['min_year'] = int(between_match.group(1))
            constraints['max_year'] = int(between_match.group(2))
        
        # Check for exact year
        exact_match = re.search(self.year_patterns['exact'], query_lower)
        if exact_match:
            year = int(exact_match.group(1))
            constraints['min_year'] = year
            constraints['max_year'] = year
        
        return constraints

    def parse_property_type(self, query: str) -> Optional[str]:
        """Parse property type from query - only for specific queries"""
        query_lower = query.lower()
        
        # Only parse property type if query contains numerical constraints
        has_numerical_constraints = (
            any(term in query_lower for term in ['under', 'over', 'above', 'below', 'between', 'before', 'after']) and
            any(term in query_lower for term in ['lakh', 'crore', 'cr', 'lk', 'sq ft', 'square feet', 'year', 'built'])
        )
        
        if not has_numerical_constraints:
            return None
        
        # Check for explicit patterns only
        for prop_type, pattern in self.property_type_patterns.items():
            if re.search(pattern, query_lower):
                return prop_type
        
        return None

    def parse_all_constraints_regex(self, query: str) -> Dict[str, Dict[str, Optional[Union[float, int]]]]:
        """Parse all constraints from a query using regex patterns"""
        return {
            'price': self.parse_price_constraints(query),
            'beds': self.parse_bed_constraints(query),
            'baths': self.parse_bath_constraints(query),
            'area': self.parse_area_constraints(query),
            'year': self.parse_year_constraints(query),
            'property_type': self.parse_property_type(query)
        }

    def parse_all_constraints(self, query: str) -> Dict[str, Dict[str, Optional[Union[float, int]]]]:
        """Parse all constraints from a query using T5 model with regex fallback"""
        # Try T5 first for better understanding
        t5_constraints = self.extract_constraints_with_t5(query)
        
        # If T5 found constraints, use them
        if any(t5_constraints.values()):
            return t5_constraints
        
        # Fallback to regex parsing
        return self.parse_all_constraints_regex(query)

    def property_matches_constraints(self, property_data: Dict, constraints: Dict) -> bool:
        """Check if property matches the parsed constraints"""
        
        # Check price constraints
        price_constraints = constraints.get('price', {})
        if price_constraints['min_price'] is not None or price_constraints['max_price'] is not None:
            price = property_data.get('marketValue') or property_data.get('MarketValue')
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
            beds = property_data.get('numberOfRooms') or property_data.get('Beds')
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
            baths = property_data.get('Baths')
            if baths is not None:
                try:
                    baths = int(baths)
                except (ValueError, TypeError):
                    baths = None
            
            # Try to get from commercialPropertyDetails
            if baths is None:
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
            area = property_data.get('totalSquareFeet') or property_data.get('LeasableSquareFeet')
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
        
        # Check year constraints
        year_constraints = constraints.get('year', {})
        if year_constraints['min_year'] is not None or year_constraints['max_year'] is not None:
            year = property_data.get('yearBuilt') or property_data.get('YearBuilt')
            if year is not None:
                try:
                    year = int(year)
                except (ValueError, TypeError):
                    year = None
                
                if year is not None:
                    if year_constraints['min_year'] is not None and year < year_constraints['min_year']:
                        return False
                    if year_constraints['max_year'] is not None and year > year_constraints['max_year']:
                        return False
        
        # Check property type constraints - only if explicitly specified with numerical constraints
        property_type = constraints.get('property_type')
        if property_type:
            prop_type = property_data.get('propertyType') or property_data.get('PropertyType', '').lower()
            # Only apply property type filtering if we have other numerical constraints
            has_numerical_constraints = (
                constraints.get('price', {}).get('min_price') is not None or 
                constraints.get('price', {}).get('max_price') is not None or
                constraints.get('year', {}).get('min_year') is not None or 
                constraints.get('year', {}).get('max_year') is not None or
                constraints.get('area', {}).get('min_area') is not None or 
                constraints.get('area', {}).get('max_area') is not None
            )
            
            if has_numerical_constraints:
                # Only then apply property type filtering
                if property_type not in prop_type:
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
        
        # Year summary
        year = constraints.get('year', {})
        if year['min_year'] is not None and year['max_year'] is not None:
            if year['min_year'] == year['max_year']:
                summary_parts.append(f"Built in {year['min_year']}")
            else:
                summary_parts.append(f"Built {year['min_year']}-{year['max_year']}")
        elif year['min_year'] is not None:
            summary_parts.append(f"Built after {year['min_year']}")
        elif year['max_year'] is not None:
            summary_parts.append(f"Built before {year['max_year']}")
        
        # Property type summary
        property_type = constraints.get('property_type')
        if property_type:
            summary_parts.append(f"Type: {property_type.title()}")
        
        return " | ".join(summary_parts) if summary_parts else "No specific constraints" 