import os
from dotenv import load_dotenv
from enum import Enum
import tempfile

# Load environment variables
load_dotenv()

# User Plan Enum
class UserPlan(Enum):
    BASIC = "basic"
    PLUS = "plus"
    PRO = "pro"

# API Keys and Credentials
API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://31a6-2409-4070-4201-830d-a9df-459f-9630-d3cb.ngrok-free.app/api/Property")
API_ENDPOINT = f"{API_BASE_URL}/allPropertieswithfulldetails"

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model Paths - Updated for local deployment
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "models", "saved_models"))
LLM_MODEL_DIR = os.getenv("LLM_MODEL_DIR", os.path.join(BASE_DIR, "models", "llm"))
FEATURE_MATCHER_DIR = os.path.join(MODEL_DIR, "feature_matcher")
BASE_MODEL_DIR = os.path.join(FEATURE_MATCHER_DIR, "base_model")

# Create necessary directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LLM_MODEL_DIR, exist_ok=True)
os.makedirs(FEATURE_MATCHER_DIR, exist_ok=True)
os.makedirs(BASE_MODEL_DIR, exist_ok=True)

# Use system temp directory instead of creating our own
TEMP_DIR = tempfile.gettempdir()

# Rate Limiting
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", 60))
MAX_REQUESTS_PER_WINDOW = int(os.getenv("MAX_REQUESTS_PER_WINDOW", 30))
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", 1000))

# Cache Settings
CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))

# Domain Classifier
DOMAIN_CLASSIFIER_MODEL = os.getenv("DOMAIN_CLASSIFIER_MODEL", "distilbert-base-uncased")

# Plan input limits
PLAN_INPUT_LIMITS = {
    UserPlan.BASIC: int(os.getenv("BASIC_PLAN_LIMIT", 5)),
    UserPlan.PLUS: int(os.getenv("PLUS_PLAN_LIMIT", 10)),
    UserPlan.PRO: int(os.getenv("PRO_PLAN_LIMIT", 20))
}

# Plan-specific fields
PLAN_FIELDS = {
    UserPlan.BASIC: {
        "PropertyName", "Address", "City", "State", "ZipCode",
        "LeasableSquareFeet", "NumberOfRooms", "Beds", "Baths",
        "PropertyStatus", "Description"
    },
    UserPlan.PLUS: {
        # Basic fields plus additional ones
        "PropertyName", "Address", "City", "State", "ZipCode",
        "LeasableSquareFeet", "NumberOfRooms", "Beds", "Baths",
        "PropertyStatus", "Description", "YearBuilt", "MarketValue",
        "PropertyType", "ParkingSpaces", "PropertyManager",
        "TaxAssessmentNumber", "Latitude", "Longitude", "CreateDate",
        "LastModifiedDate", "ViewNumber", "Contact", "TotalSquareFeet"
    },
    UserPlan.PRO: {
        # All fields
        "PropertyName", "Address", "City", "State", "ZipCode",
        "LeasableSquareFeet", "NumberOfRooms", "Beds", "Baths",
        "PropertyStatus", "Description", "YearBuilt", "MarketValue",
        "PropertyType", "ParkingSpaces", "PropertyManager",
        "TaxAssessmentNumber", "Latitude", "Longitude", "CreateDate",
        "LastModifiedDate", "ViewNumber", "Contact", "TotalSquareFeet",
        "AgentName", "AgentPhoneNumber", "AgentEmail", "KeyFeatures",
        "NearbyAmenities", "property_image",
        "Distance", "IsDeleted"
    }
} 