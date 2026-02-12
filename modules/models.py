import torch
import faiss
import pandas as pd
import requests
import json
import urllib3
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from modules.config import (
    MODEL_DIR, 
    LLM_MODEL_DIR, 
    FEATURE_MATCHER_DIR,
    BASE_MODEL_DIR,
    TEMP_DIR,
    BASE_DIR
)
from modules.parallel import ModelParallelizer, parallel_map, batch_process, get_device
import os
import pickle
import numpy as np
import logging
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Dict, Any, Tuple, List
import asyncio
import aiohttp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable SSL warnings for development
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Check device
device = get_device()
print(f"Using device: {device}")

# Global variables
model_embedding = None
model_parallelizer = None
properties_cache = None
property_embeddings = None

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://0f9dfdaf65e6.ngrok-free.app/api/Property")
API_ENDPOINT = f"{API_BASE_URL}/allPropertieswithfulldetails"

# Global cache for properties
_properties_cache = None

class OptimizedRAGLoader:
    """Loads optimized models and components from test_optimized_rag.py"""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = None
        self.index = None
        self.pca = None
        self.tokenizer = None
        self.feature_matcher = None

    def load_optimized_model(self, model_path: Path, device: str):
        """Load optimized model with proper error handling"""
        try:
            print(f"Loading model from {model_path}")
            
            # Check if it's a quantized model
            if model_path.suffix == '.pth':
                # Load quantized model
                print("Loading quantized model...")
                model = SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True, device=device)
                
                # Load the state dict
                state_dict = torch.load(model_path, map_location=device)
                
                # Handle the architecture mismatch by loading only compatible parts
                model_state_dict = model.state_dict()
                
                # Filter out incompatible keys
                compatible_state_dict = {}
                for key, value in state_dict.items():
                    if key in model_state_dict and model_state_dict[key].shape == value.shape:
                        compatible_state_dict[key] = value
                    else:
                        print(f"Skipping incompatible key: {key}")
                
                # Load compatible parts
                missing_keys, unexpected_keys = model.load_state_dict(compatible_state_dict, strict=False)
                
                if missing_keys:
                    print(f"Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"Unexpected keys: {len(unexpected_keys)}")
                
                print(f"✅ Loaded quantized model from {model_path}")
                return model
            else:
                # Load regular model
                model = SentenceTransformer(str(model_path), device=device)
                print(f"✅ Loaded model from {model_path}")
                return model
                
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to Jina model
            print("Using Jina fallback model...")
            try:
                model = SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True, device=device)
                print("✅ Loaded Jina fallback model")
                return model
            except Exception as fallback_error:
                print(f"Jina fallback model also failed: {fallback_error}")
                return None

    def load_all_components(self):  
        """Load all saved components"""
        print("\n📦 Loading optimized models and components...")

        # Load main model
        model_path = self.model_dir / "model_state_dict.pth"
        if model_path.exists():
            self.model = self.load_optimized_model(model_path, device)
            print(f"✅ Loaded quantized Jina model from {model_path}")
        else:
            print(f"❌ Model not found at {model_path}")
            return False
            
        # Load FAISS index
        index_path = self.model_dir / "property_index.faiss"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            print(f"✅ Loaded FAISS index from {index_path}")
        else:
            print(f"❌ FAISS index not found at {index_path}")
            return False
                
        # Load PCA model
        pca_path = self.model_dir / "pca_model.pkl"
        if pca_path.exists():
            with open(pca_path, 'rb') as f:
                self.pca = pickle.load(f)
            print(f"✅ Loaded PCA model from {pca_path}")

        # Load tokenizer
        try:
            self.tokenizer = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True).tokenizer
            print(f"✅ Loaded tokenizer")
        except Exception as e:
            print(f"⚠️ Could not load tokenizer: {e}")

        # Load system config
        config_path = self.model_dir / "search_system.pkl"
        if config_path.exists():
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
            print(f"✅ Loaded system configuration")

        # Load feature matcher
        feature_matcher_dir = self.model_dir / "feature_matcher"
        if feature_matcher_dir.exists():
            self.feature_matcher = self.load_feature_matcher(feature_matcher_dir)
            print(f"✅ Loaded feature matcher")

        return True

    def load_feature_matcher(self, feature_matcher_dir: Path):
        """Load optimized feature matcher"""
        try:
            # Load base model
            base_model_path = feature_matcher_dir / "base_model.pth"
            base_model = self.load_optimized_model(base_model_path, device)

            # Load semantic model
            semantic_model_path = feature_matcher_dir / "semantic_model.pth"
            semantic_model = self.load_optimized_model(semantic_model_path, device)

            # Initialize zero-shot model directly
            zero_shot_model = pipeline("zero-shot-classification", 
                                     model="facebook/bart-large-mnli",
                                     device=0 if torch.cuda.is_available() else -1)

            # Load patterns
            patterns_path = feature_matcher_dir / "feature_patterns.pkl"
            with open(patterns_path, 'rb') as f:
                patterns_data = pickle.load(f)

            return {
                'base_model': base_model,
                'semantic_model': semantic_model,
                'zero_shot_model': zero_shot_model,
                'patterns': patterns_data
            }
        except Exception as e:
            print(f"⚠️ Could not load feature matcher: {e}")
            return None

class ParallelPropertyFetcher:
    """Fetches properties using parallel processing from test_optimized_rag.py"""

    def __init__(self, api_endpoint: str):
        self.api_endpoint = api_endpoint

    def fetch_all_properties(self, total_properties: int = 600) -> List[Dict]:
        """Main function to fetch properties using parallel processing"""
        print("🚀 Starting parallel property fetch...")
        
        start_time = time.time()
        
        try:
            # Use ThreadPoolExecutor for parallel processing (works in all environments)
            properties = self._fetch_properties_parallel(total_properties)
            
            end_time = time.time()
            duration = end_time - start_time
            print(f"⏱️ Parallel fetch completed in {duration:.2f} seconds")
            print(f"📊 Average speed: {len(properties)/duration:.1f} properties/second")
            
            return properties
        except Exception as e:
            print(f"❌ Error in fetch_all_properties: {e}")
            return []

    def _fetch_properties_parallel(self, total_properties: int = 600, batch_size: int = 100) -> List[Dict]:
        """Fetch properties using ThreadPoolExecutor for parallel processing"""
        print(f"🔄 Fetching {total_properties} properties in parallel batches of {batch_size}...")
        
        properties = []
        num_batches = (total_properties + batch_size - 1) // batch_size
        
        def fetch_batch(page_number):
            """Fetch a single batch of properties"""
            try:
                headers = {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                    'ngrok-skip-browser-warning': 'true'
                }
                
                response = requests.get(
                    self.api_endpoint,
                    params={"pageNumber": page_number + 1, "pageSize": batch_size},
                    headers=headers,
                    verify=False
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("data", [])
                else:
                    print(f"⚠️ HTTP {response.status_code} for batch {page_number + 1}")
                    return []
                    
            except Exception as e:
                print(f"⚠️ Error fetching batch {page_number + 1}: {str(e)}")
                return []
        
        # Use ThreadPoolExecutor for parallel processing with optimized workers
        max_workers = min(20, num_batches)  # Use up to 20 workers, but not more than batches
        print(f"🚀 Using {max_workers} parallel workers for {num_batches} batches")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch tasks
            future_to_batch = {
                executor.submit(fetch_batch, page_number): page_number 
                for page_number in range(num_batches)
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=total_properties, desc="Properties fetched") as pbar:
                for future in as_completed(future_to_batch):
                    batch_number = future_to_batch[future]
                    try:
                        batch_data = future.result()
                        properties.extend(batch_data)
                        
                        new_count = len(batch_data)
                        pbar.update(new_count)
                        pbar.set_postfix({"Fetched": len(properties), "Batch": batch_number + 1})
                        
                        if len(properties) >= total_properties:
                            break
                            
                    except Exception as e:
                        print(f"❌ Error in batch {batch_number + 1}: {e}")
        
        print(f"\n✅ Total properties fetched: {len(properties)}")
        return properties[:total_properties]

class VectorDBManager:
    """Manages vector database operations from test_optimized_rag.py"""

    def __init__(self, model, pca=None):
        self.model = model
        self.pca = pca
        self.index = None
        self.properties_df = None

    def format_property(self, property_data):
        """Format property data into searchable text format"""
        # Note: propertyId is intentionally excluded from text to avoid sending it to LLM
        text = f"Property Name: {property_data.get('propertyName', 'N/A')} "
        text += f"Type: {property_data.get('typeName', 'N/A')} "
        text += f"Description: {property_data.get('description', 'N/A')} "
        text += f"Address: {property_data.get('address', 'N/A')} "

        location = property_data.get('location', {})
        if location:
            text += f"Full Address: {location.get('address', 'N/A')} "
            text += f"Latitude: {location.get('latitude', 'N/A')} "
            text += f"Longitude: {location.get('longitude', 'N/A')} "

        text += f"Total Square Feet: {property_data.get('totalSquareFeet', 'N/A')} "
        text += f"Number of Rooms: {property_data.get('numberOfRooms', 'N/A')} "
        text += f"Market Value: {property_data.get('marketValue', 'N/A')} "

        features = property_data.get('features', [])
        if features:
            text += f"Features: {', '.join(features)} "

        pg_details = property_data.get('pgPropertyDetails', {})
        if pg_details:
            text += f"PG Details: "
            text += f"Deposit: {pg_details.get('depositAmount', 'N/A')} "
            text += f"Food Included: {pg_details.get('foodIncluded', 'N/A')} "
            text += f"WiFi: {pg_details.get('wifiAvailable', 'N/A')} "
            text += f"AC: {pg_details.get('isACAvailable', 'N/A')} "
            text += f"Parking: {pg_details.get('isParkingAvailable', 'N/A')} "
            text += f"Power Backup: {pg_details.get('powerBackup', 'N/A')} "
            text += f"Total Beds: {pg_details.get('totalBeds', 'N/A')} "

        commercial_details = property_data.get('commercialPropertyDetails', {})
        if commercial_details:
            text += f"Commercial Details: "
            text += f"Washrooms: {commercial_details.get('washrooms', 'N/A')} "
            text += f"Floor Details: {commercial_details.get('floorDetails', 'N/A')} "
            text += f"Parking: {commercial_details.get('hasParking', 'N/A')} "
            text += f"Lift: {commercial_details.get('hasLift', 'N/A')} "
            text += f"Furnished: {commercial_details.get('isFurnished', 'N/A')} "

        agents = property_data.get('agents', [])
        if agents:
            for agent in agents:
                text += f"Agent: {agent.get('name', 'N/A')} "
                text += f"Phone: {agent.get('phoneNumber', 'N/A')} "
                text += f"Email: {agent.get('email', 'N/A')} "

        return text

    def create_vector_db(self, properties: List[Dict]):
        """Create vector database from properties"""
        print("\n🔧 Creating vector database...")

        # Convert to DataFrame
        self.properties_df = pd.DataFrame(properties)
        print(f"📊 DataFrame shape: {self.properties_df.shape}")

        if self.properties_df.empty:
            raise ValueError("DataFrame is empty. Cannot create vector database.")

        # Format properties for search (keep original data for embeddings)
        print("📝 Formatting properties for search...")
        formatted_properties = []
        for i, prop in enumerate(properties):
            formatted_prop = format_property_details(prop)
            if formatted_prop:
                formatted_properties.append(formatted_prop)
                if i < 3:  # Show first 3 properties for debugging
                    print(f"  Property {i+1}: {formatted_prop.get('PropertyName', 'N/A')} - {formatted_prop.get('PropertyType', 'N/A')}")
            else:
                # Fallback to original property if formatting fails
                formatted_properties.append(prop)
                print(f"  Property {i+1}: Formatting failed, using original")
        
        # Create new DataFrame with formatted properties
        self.properties_df = pd.DataFrame(formatted_properties)
        print(f"📊 Formatted DataFrame shape: {self.properties_df.shape}")
        print(f"📊 DataFrame columns: {list(self.properties_df.columns)}")
        
        # Debug: Show first row of DataFrame
        if not self.properties_df.empty:
            print("🔍 Debug: First row of DataFrame:")
            first_row = self.properties_df.iloc[0]
            print(f"  PropertyName: {first_row.get('PropertyName', 'NOT_FOUND')}")
            print(f"  PropertyType: {first_row.get('PropertyType', 'NOT_FOUND')}")
            print(f"  Address: {first_row.get('Address', 'NOT_FOUND')}")

        # Create text representation for embeddings
        print("📝 Creating text representations for embeddings...")
        text_representations = []
        for prop in properties:  # Use original properties for text
            text_rep = self.format_property(prop)
            text_representations.append(text_rep)

        # Generate embeddings
        print("🧠 Generating embeddings...")
        embeddings = self.model.encode(
            text_representations,
            convert_to_numpy=True,
            device=device,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=32
        )

        # Convert to numpy array
        if isinstance(embeddings, torch.Tensor):
            if embeddings.dtype == torch.bfloat16:
                embeddings = embeddings.float()
            property_embeddings = embeddings.cpu().numpy().astype(np.float32)
        else:
            property_embeddings = np.array(embeddings, dtype=np.float32)

        print(f"✅ Generated embeddings shape: {property_embeddings.shape}")

        # Apply PCA if needed
        if self.pca is not None:
            print("📉 Applying PCA transformation...")
            property_embeddings = self.pca.transform(property_embeddings).astype(np.float32)
            print(f"✅ PCA transformation completed. New shape: {property_embeddings.shape}")

        # Create FAISS index
        dimension = property_embeddings.shape[1]
        n_samples = property_embeddings.shape[0]

        print(f"🔍 Creating FAISS index with dimension {dimension}...")

        # Use simple index for testing (faster)
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(property_embeddings)

        print(f"✅ Added {n_samples} embeddings to FAISS index")
        return self.index

class OptimizedRAGRetriever:
    """Optimized RAG retriever from test_optimized_rag.py"""

    def __init__(self, model, index, tokenizer=None, pca=None, feature_matcher=None):
        self.model = model
        self.index = index
        self.tokenizer = tokenizer
        self.pca = pca
        self.feature_matcher = feature_matcher
        self.dimension = index.d
        self.properties_df = None

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve properties based on query"""
        try:
            # Get query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True, device=device)

            # Convert to proper type
            if isinstance(query_embedding, torch.Tensor):
                if query_embedding.dtype == torch.bfloat16:
                    query_embedding = query_embedding.float()
                query_embedding = query_embedding.cpu().numpy().astype(np.float32)
            else:
                query_embedding = np.array(query_embedding, dtype=np.float32)

            # Apply PCA if used during training
            if self.pca is not None:
                query_embedding = self.pca.transform(query_embedding)

            # Ensure correct shape
            if query_embedding.shape[1] != self.dimension:
                raise ValueError(f"Query embedding dimension {query_embedding.shape[1]} does not match index dimension {self.dimension}")

            # Search FAISS index
            distances, indices = self.index.search(query_embedding, top_k)

            # Return results
            retrieved_properties = []
            for idx, dist in zip(indices[0], distances[0]):
                if self.properties_df is not None and idx < len(self.properties_df):
                    property_data = self.properties_df.iloc[idx].to_dict()
                    retrieved_properties.append({
                        "property": property_data,
                        "distance": float(dist)
                    })

            return retrieved_properties
            
        except Exception as e:
            print(f"❌ Error in retrieval: {e}")
            return []

# Keep existing functions for backward compatibility
def fetch_and_cache_properties():
    """Fetch properties from API and cache them"""
    global properties_cache
    try:
        print("Fetching properties from API...")
        
        # Configure session with retry mechanism
        session = requests.Session()
        session.verify = False
        
        # Add headers for better API communication
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true'
        }
        
        # Make the API request with increased page size
        print(f"Making API request to: {API_ENDPOINT}")
        print(f"Request parameters: pageNumber=1, pageSize=600")
        response = session.get(
            API_ENDPOINT,
            params={"pageNumber": 1, "pageSize": 600},  # Increased page size to 600
            headers=headers
        )
        
        print(f"API Response Status: {response.status_code}")
        print(f"API Response Headers: {dict(response.headers)}")
        
        # Check for successful response
        response.raise_for_status()
        
        # Parse and validate response
        data = response.json()
        if not isinstance(data, dict) or 'data' not in data:
            raise ValueError("Invalid API response format")
            
        properties_cache = data["data"]
        if not properties_cache:
            raise ValueError("No properties found in API response")
            
        print(f"Successfully cached {len(properties_cache)} properties")
        print("\nSample Raw API Response (First Property):")
        print(json.dumps(properties_cache[0], indent=2))
        
        # Debug: Check for ID fields in the first property
        if properties_cache:
            first_prop = properties_cache[0]
            print(f"\n🔍 Debug: Checking for ID fields in first property...")
            print(f"🔍 Debug: All keys: {list(first_prop.keys())}")
            # Check for any field that might contain an ID
            for key, value in first_prop.items():
                if 'id' in key.lower() or 'property' in key.lower():
                    print(f"🔍 Debug: Potential ID field '{key}': {value}")
        
        return properties_cache
        
    except Exception as e:
        logger.error(f"Error fetching properties: {str(e)}")
        return []

def get_cached_properties():
    """Get cached properties or fetch if not available"""
    global properties_cache
    if properties_cache is None:
        properties_cache = fetch_and_cache_properties()
    return properties_cache or []

def create_property_embeddings(properties, model):
    """Create embeddings for properties and store in FAISS index"""
    global property_embeddings
    try:
        print("\n=== Creating property embeddings ===")
        
        # Prepare property texts for embedding
        property_texts = []
        for prop in properties:
            # Create a rich text representation of the property
            text = f"""
            Property Name: {prop.get('propertyName', 'N/A')}
            Type: {prop.get('typeName', 'N/A')}
            Description: {prop.get('description', 'N/A')}
            Address: {prop.get('address', 'N/A')}

            Location Information:
            Full Address: {prop.get('location', {}).get('address', 'N/A')}
            Latitude: {prop.get('location', {}).get('latitude', 'N/A')}
            Longitude: {prop.get('location', {}).get('longitude', 'N/A')}

            Property Details:
            Total Square Feet: {prop.get('totalSquareFeet', 'N/A')}
            Number of Rooms: {prop.get('numberOfRooms', 'N/A')}
            Market Value: {prop.get('marketValue', 'N/A')}

            Features: {', '.join(prop.get('features', []))}

            PG Property Details:
            {format_pg_details(prop.get('pgPropertyDetails', {}))}

            Commercial Property Details:
            {format_commercial_details(prop.get('commercialPropertyDetails', {}))}
            """
            property_texts.append(text)
        
        print(f"Created text representations for {len(property_texts)} properties")
        
        # Create embeddings in batches
        embeddings = []
        batch_size = 32
        for i in range(0, len(property_texts), batch_size):
            batch = property_texts[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(property_texts) + batch_size - 1)//batch_size}")
            batch_embeddings = model.encode(batch, convert_to_numpy=True)
            embeddings.extend(batch_embeddings)
            
        property_embeddings = np.array(embeddings).astype('float32')
        print(f"Created embeddings for {len(property_embeddings)} properties")
        
        # Create and save FAISS index
        dimension = property_embeddings.shape[1]
        print(f"Creating FAISS index with dimension {dimension}")
        index = faiss.IndexFlatL2(dimension)
        index.add(property_embeddings)
        
        # Save the index
        index_path = os.path.join(MODEL_DIR, "property_index.faiss")
        faiss.write_index(index, index_path)
        print(f"Saved FAISS index to {index_path}")
        
        return index
        
    except Exception as e:
        print(f"Error creating property embeddings: {str(e)}")
        raise

def format_pg_details(pg_details):
    """Format PG property details into text"""
    if not pg_details:
        return "N/A"
    
    return f"""
    Deposit: {pg_details.get('depositAmount', 'N/A')}
    Food Included: {pg_details.get('foodIncluded', 'N/A')}
    Food Type: {pg_details.get('foodAvailability', 'N/A')}
    WiFi: {pg_details.get('wifiAvailable', 'N/A')}
    AC: {pg_details.get('isACAvailable', 'N/A')}
    Parking: {pg_details.get('isParkingAvailable', 'N/A')}
    Power Backup: {pg_details.get('powerBackup', 'N/A')}
    Available For: {pg_details.get('availableFor', 'N/A')}
    Total Beds: {pg_details.get('totalBeds', 'N/A')}
    """

def format_commercial_details(commercial_details):
    """Format commercial property details into text"""
    if not commercial_details:
        return "N/A"
    
    return f"""
    Washrooms: {commercial_details.get('washrooms', 'N/A')}
    Floor Details: {commercial_details.get('floorDetails', 'N/A')}
    Parking: {commercial_details.get('hasParking', 'N/A')}
    Parking Capacity: {commercial_details.get('parkingCapacity', 'N/A')}
    Facing: {commercial_details.get('facing', 'N/A')}
    Lift: {commercial_details.get('hasLift', 'N/A')}
    Furnished: {commercial_details.get('isFurnished', 'N/A')}
    """

def load_sentence_transformer():
    global model_embedding, model_parallelizer
    print("\n=== Loading SentenceTransformer model ===")
    try:
        # Create cache directories in the code directory
        cache_base = os.path.join(BASE_DIR, '.cache')
        os.makedirs(cache_base, exist_ok=True)
        
        # Set cache directories
        hf_home = os.path.join(cache_base, 'huggingface')
        datasets_cache = os.path.join(cache_base, 'datasets')
        
        # Create all cache directories
        os.makedirs(hf_home, exist_ok=True)
        os.makedirs(datasets_cache, exist_ok=True)
        
        # Set environment variables
        os.environ['HF_HOME'] = hf_home
        os.environ['HF_DATASETS_CACHE'] = datasets_cache
        
        print(f"Using cache directories:")
        print(f"HF_HOME: {hf_home}")
        print(f"HF_DATASETS_CACHE: {datasets_cache}")
        
        # Initialize with Jina model
        print("Loading Jina embeddings model...")
        model_embedding = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
        
        # Move model to device after initialization with proper handling
        try:
            if torch.cuda.is_available():
                model_embedding = model_embedding.to('cuda')
            else:
                model_embedding = model_embedding.to('cpu')
        except Exception as e:
            print(f"Warning: Could not move model to device: {e}")
            # Keep model on current device
            
        print("Jina model loaded successfully")
        
        # Initialize parallelizer
        model_parallelizer = ModelParallelizer(model_embedding)
        print("Model parallelizer initialized")
        return model_embedding
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def load_faiss_index():
    print("\n=== Loading FAISS index ===")
    try:
        index_path = os.path.join(MODEL_DIR, "property_index.faiss")
        print(f"Looking for FAISS index at: {index_path}")
        
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            print("FAISS index loaded successfully")
            return index
        else:
            print("FAISS index not found, creating new index...")
            # Fetch properties and create new index
            properties = get_cached_properties()
            if not properties:
                raise ValueError("No properties available to create index")
            model = load_sentence_transformer()
            return create_property_embeddings(properties, model)
    except Exception as e:
        print(f"Error loading FAISS index: {str(e)}")
        raise

def load_pca_model():
    print("Loading PCA model...")
    try:
        pca_path = os.path.join(MODEL_DIR, "pca_model.pkl")
        if os.path.exists(pca_path):
            with open(pca_path, 'rb') as f:
                pca = pickle.load(f)
            print("PCA model loaded successfully.")
            return pca
        return None
    except Exception as e:
        logger.error(f"Error loading PCA model: {str(e)}")
        return None

def load_search_system(model_path: str = None, index_path: str = None, pca_path: str = None):
    """Load the entire search system using OptimizedRAGLoader"""
    try:
        model_dir = Path(MODEL_DIR)
        loader = OptimizedRAGLoader(model_dir)
        
        if loader.load_all_components():
            # Create VectorDBManager for vector operations
            vector_db = VectorDBManager(loader.model, loader.pca)
            
            # Create OptimizedRAGRetriever
            retriever = OptimizedRAGRetriever(
                model=loader.model,
                index=loader.index,
                tokenizer=loader.tokenizer,
                pca=loader.pca,
                feature_matcher=loader.feature_matcher
            )
            
            return retriever, vector_db
        else:
            raise ValueError("Failed to load all components")
            
    except Exception as e:
        logger.error(f"Error loading search system: {str(e)}")
        raise

def format_property_details(property_data):
    """Format property details with all available information"""
    try:
        # Debug: Print available keys to understand the API response structure
        if 'propertyId' not in property_data and 'id' not in property_data and 'PropertyID' not in property_data:
            print(f"🔍 Debug: Property data keys: {list(property_data.keys())}")
            print(f"🔍 Debug: Sample property data: {str(property_data)[:200]}...")
            # Check for any field that might contain an ID
            for key, value in property_data.items():
                if 'id' in key.lower() or 'property' in key.lower():
                    print(f"🔍 Debug: Potential ID field '{key}': {value}")
        
        # Extract location details safely
        location = property_data.get('location', {})
        address_parts = location.get('address', '').split(',') if location.get('address') else []
        
        # Ensure we have enough parts for address parsing
        while len(address_parts) < 4:
            address_parts.append('N/A')
            
        formatted_property = {
            # Basic Information
            "propertyId": property_data.get('id', property_data.get('propertyId', property_data.get('PropertyID', 'N/A'))),
            "PropertyName": property_data.get('propertyName', 'Unknown'),
            "Address": property_data.get('address', 'N/A'),
            "ZipCode": address_parts[-1].strip() if address_parts else 'N/A',
            "LeasableSquareFeet": float(property_data.get('totalSquareFeet', 0)),
            "YearBuilt": property_data.get('yearBuilt', None),
            "NumberOfRooms": int(property_data.get('numberOfRooms', 0)),
            "ParkingSpaces": int(property_data.get('commercialPropertyDetails', {}).get('parkingCapacity', 0)) if property_data.get('commercialPropertyDetails') else 0,
            "PropertyManager": property_data.get('agents', [{}])[0].get('name', 'N/A') if property_data.get('agents') else 'N/A',
            "MarketValue": float(property_data.get('marketValue', 0)),
            "TaxAssessmentNumber": None,  # Not available in API
            "Latitude": float(location.get('latitude', 0)) if location.get('latitude') is not None else 0.0,
            "Longitude": float(location.get('longitude', 0)) if location.get('longitude') is not None else 0.0,
            "CreateDate": property_data.get('date', 'N/A'),
            "LastModifiedDate": property_data.get('date', 'N/A'),
            "City": address_parts[1].strip() if len(address_parts) > 1 else 'N/A',
            "State": address_parts[2].strip() if len(address_parts) > 2 else 'N/A',
            "Country": address_parts[3].strip() if len(address_parts) > 3 else 'N/A',
            "PropertyType": property_data.get('typeName', 'N/A'),
            "PropertyStatus": property_data.get('parentCategoryName', 'N/A'),
            "Description": property_data.get('description', 'N/A'),
            "ViewNumber": 0,  # Not available in API
            "Contact": property_data.get('agents', [{}])[0].get('phoneNumber', 'N/A') if property_data.get('agents') else 'N/A',
            "TotalSquareFeet": float(property_data.get('totalSquareFeet', 0)),
            "IsDeleted": False,  # Not available in API
            "Beds": int(property_data.get('beds', 0)),  # Updated to use beds instead of numberOfRooms
            "Baths": int(property_data.get('baths', 0)),  # Updated to use baths directly
            "AgentName": property_data.get('agents', [{}])[0].get('name', 'N/A') if property_data.get('agents') else 'N/A',
            "AgentPhoneNumber": property_data.get('agents', [{}])[0].get('phoneNumber', 'N/A') if property_data.get('agents') else 'N/A',
            "AgentEmail": property_data.get('agents', [{}])[0].get('email', 'N/A') if property_data.get('agents') else 'N/A',
            "KeyFeatures": ', '.join(property_data.get('features', [])) if property_data.get('features') else 'N/A',
            "NearbyAmenities": property_data.get('description', 'N/A'),
            "propertyImages": property_data.get('propertyImages', []),
            
            # PG Property Details
            "PGDetails": {
                "DepositAmount": property_data.get('pgPropertyDetails', {}).get('depositAmount', 'N/A'),
                "FoodIncluded": property_data.get('pgPropertyDetails', {}).get('foodIncluded', 'N/A'),
                "FoodType": property_data.get('pgPropertyDetails', {}).get('foodAvailability', 'N/A'),
                "WifiAvailable": property_data.get('pgPropertyDetails', {}).get('wifiAvailable', 'N/A'),
                "ACAvailable": property_data.get('pgPropertyDetails', {}).get('isACAvailable', 'N/A'),
                "ParkingAvailable": property_data.get('pgPropertyDetails', {}).get('isParkingAvailable', 'N/A'),
                "PowerBackup": property_data.get('pgPropertyDetails', {}).get('powerBackup', 'N/A'),
                "AvailableFor": property_data.get('pgPropertyDetails', {}).get('availableFor', 'N/A'),
                "TotalBeds": property_data.get('pgPropertyDetails', {}).get('totalBeds', 'N/A'),
                "OperatingSince": property_data.get('pgPropertyDetails', {}).get('operatingSince', 'N/A'),
                "NoticePeriod": property_data.get('pgPropertyDetails', {}).get('noticePeriod', 'N/A'),
                "PreferredTenants": property_data.get('pgPropertyDetails', {}).get('preferredTenants', 'N/A')
            } if property_data.get('pgPropertyDetails') else None,
            
            # Commercial Property Details
            "CommercialDetails": {
                "Washrooms": property_data.get('commercialPropertyDetails', {}).get('washrooms', 'N/A'),
                "FloorDetails": property_data.get('commercialPropertyDetails', {}).get('floorDetails', 'N/A'),
                "HasParking": property_data.get('commercialPropertyDetails', {}).get('hasParking', 'N/A'),
                "ParkingCapacity": property_data.get('commercialPropertyDetails', {}).get('parkingCapacity', 'N/A'),
                "Facing": property_data.get('commercialPropertyDetails', {}).get('facing', 'N/A'),
                "HasLift": property_data.get('commercialPropertyDetails', {}).get('hasLift', 'N/A'),
                "IsFurnished": property_data.get('commercialPropertyDetails', {}).get('isFurnished', 'N/A'),
                "Overlooking": property_data.get('commercialPropertyDetails', {}).get('overlooking', 'N/A'),
                "MonthlyRent": property_data.get('commercialPropertyDetails', {}).get('monthlyRent', 'N/A'),
                "LeaseTerms": property_data.get('commercialPropertyDetails', {}).get('leaseTerms', 'N/A')
            } if property_data.get('commercialPropertyDetails') else None
        }
        return formatted_property
    except Exception as e:
        logger.error(f"Error formatting property details: {str(e)}")
        return None

def load_tokenizer_and_model():
    print("Loading tokenizer and LLM model...")
    try:
        # Set Triton cache directory
        os.environ['TRITON_CACHE_DIR'] = os.path.join(BASE_DIR, '.cache', 'triton')
        os.makedirs(os.environ['TRITON_CACHE_DIR'], exist_ok=True)
        
        # Load tokenizer with trust_remote_code=True
        tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_DIR,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Load model with trust_remote_code=True and proper configuration
        model_llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_DIR,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_cache=False,
            load_in_4bit=True
        )
        
        # Move model to device and set to evaluation mode with proper handling
        try:
            if device_map == "auto":
                # Model is already on correct device due to device_map="auto"
                pass
            else:
                model_llm = model_llm.to(device)
            model_llm.eval()
        except Exception as e:
            print(f"Warning: Could not move LLM model to device: {e}")
            # Keep model on current device
            model_llm.eval()
        
        print("Tokenizer and LLM model loaded successfully.")
        return tokenizer, model_llm
    except Exception as e:
        logger.error(f"Error loading tokenizer/model: {str(e)}")
        raise 

# Initialize global retriever instance
_global_retriever = None
_global_vector_db = None

def get_global_retriever():
    """Skip ultimate RAG system - we're using only the original retriever"""
    print("⚠️ Skipping ultimate RAG system - using original retriever only")
    return None, None

def initialize_rag_system():
    """Initialize the RAG system with optimized components"""
    try:
        print("🚀 Initializing optimized RAG system...")
        
        # Get global retriever
        retriever, vector_db = get_global_retriever()
        
        if retriever is None:
            print("❌ Failed to initialize RAG system")
            return None
            
        print("✅ RAG system initialized successfully")
        return retriever
        
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return None

def cache_properties(properties: List[Dict]):
    """Cache properties globally"""
    global _properties_cache
    _properties_cache = properties
    print(f"✅ Cached {len(properties)} properties")

def get_cached_properties() -> List[Dict]:
    """Get cached properties"""
    global _properties_cache
    return _properties_cache if _properties_cache else []

def get_properties():
    """Get properties from cache or fetch if not available"""
    properties = get_cached_properties()
    if not properties:
        # Try to fetch properties if not cached
        try:
            properties = fetch_and_cache_properties()
        except Exception as e:
            logger.error(f"Error fetching properties: {e}")
            properties = []
    return properties 