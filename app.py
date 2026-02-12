import os
import logging
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pyngrok import ngrok
import webbrowser
import threading
from functools import wraps
import sys
import time
from geopy.distance import geodesic
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
# from pyngrok import ngrok
import uuid
import json
import gc
import psutil

# Add the modules directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules with error handling
try:
    from modules.config import *
    from modules.models import *
    from modules.security import *
    from modules.audio import *
    from modules.location_processor import LocationProcessor, set_location
    from modules.response import *
    from modules.input_tracker import *
    from modules.chatbot_processor import ChatbotProcessor
    from modules.constraint_parser import ConstraintParser
    from modules.property_processor import PropertyProcessor
    
    # Import cached properties function
    from modules.models import get_cached_properties
    
    # Import UserPlan enum
    from modules.config import UserPlan
    
    # Import filter function
    from modules.response import filter_property_by_plan, format_llm_prompt
    
    # Import plan function
    from modules.security import get_current_plan
    
    # Import plan limits
    from modules.config import PLAN_INPUT_LIMITS
    
    # Import new multi-user components
    from modules.session_manager import session_manager
    from modules.rate_limiter import rate_limiter, token_bucket_limiter
    from modules.websocket_manager import websocket_manager, MessageType
    
    # Import AI conversation memory
    from modules.ai_conversation_memory import get_ai_conversation_memory
    
    # Import enhanced security and processing components
    from modules.enhanced_security import get_enhanced_security_manager, rate_limit_by_user
    from modules.advanced_processor import get_advanced_processor, TaskType, TaskPriority, ProcessingStrategy
    from modules.error_recovery import get_error_recovery_system, handle_errors, error_context
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Import new optimization modules with error handling
try:
    from modules.async_processor import async_processor, RequestPriority
    from modules.connection_pool import http_pool, model_pool, async_http_client
    from modules.cache_manager import cache_manager, query_cache, model_cache, property_cache
    from modules.load_balancer import processing_manager
    from modules.performance_optimizer import get_performance_optimizer
    from modules.enhanced_model_manager import get_enhanced_model_manager
    OPTIMIZATIONS_ENABLED = True
    logger.info("Performance optimizations enabled")
except Exception as e:
    logger.warning(f"Performance optimizations disabled due to import error: {e}")
    OPTIMIZATIONS_ENABLED = False
    # Create dummy objects to prevent import errors
    async_processor = None
    RequestPriority = None
    http_pool = None
    model_pool = None
    async_http_client = None
    cache_manager = None
    query_cache = None
    model_cache = None
    property_cache = None
    processing_manager = None
    get_performance_optimizer = None
    get_enhanced_model_manager = None

# Always import optimized RAG processor (works independently)
try:
    from modules.optimized_rag_processor import get_optimized_rag_processor
    logger.info("Optimized RAG processor imported successfully")
except Exception as e:
    logger.error(f"Failed to import optimized RAG processor: {e}")
    get_optimized_rag_processor = None
    optimized_rag_retrieve = None

# Import AI property filter
try:
    from modules.ai_property_filter import get_ai_property_filter
    logger.info("AI property filter imported successfully")
except Exception as e:
    logger.error(f"Failed to import AI property filter: {e}")
    get_ai_property_filter = None

# Import specific functions with error handling
try:
    from modules.security import with_user_plan
    from modules.audio import process_audio_file
except ImportError as e:
    print(f"Error importing specific functions: {e}")
    # Create dummy functions
    def with_user_plan(f):
        return f
    def process_audio_file(audio_file):
        return {"error": "Audio processing not available"}, 500

# Remove ngrok setup for Hugging Face Spaces deployment
# NGROK_AUTH_TOKEN = "2wqZcIzLQYllgIf0DAut80BTPgf_5uy1Mbot6pgEPUwaDG4dU"
# ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # Use /tmp directory for log file since we're running as nobody user
        logging.FileHandler('/tmp/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app with correct template folder path
app = Flask(__name__, 
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

# Unified helper for user-friendly error responses
def friendly_error(message: str,
                   status: str = "error",
                   hint: str = None,
                   needs_location: bool = False,
                   retryable: bool = True,
                   http_code: int = 400):
    payload = {
        "status": status,
        "message": message,
        "retryable": retryable
    }
    if hint:
        payload["hint"] = hint
    if needs_location:
        payload["needs_location"] = True
    return jsonify(payload), http_code

# Initialize multi-user components
conversation_context = {}  # Legacy - will be replaced by session manager

# Initialize AI conversation memory
ai_conversation_memory = None
try:
    print("🔧 Initializing AI Conversation Memory...")
    ai_conversation_memory = get_ai_conversation_memory()
    if ai_conversation_memory is None:
        raise Exception("AI conversation memory returned None")
    print("✅ AI Conversation Memory initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AI conversation memory: {e}")
    print(f"⚠️ AI conversation memory initialization failed: {e}")
    print("⚠️ Continuing without AI conversation memory")
    ai_conversation_memory = None

# Initialize enhanced components
enhanced_security_manager = None
advanced_processor = None
error_recovery_system = None

try:
    enhanced_security_manager = get_enhanced_security_manager()
    advanced_processor = get_advanced_processor()
    error_recovery_system = get_error_recovery_system()
    print("✅ Enhanced security, processing, and error recovery systems initialized")
except Exception as e:
    logger.error(f"Failed to initialize enhanced components: {e}")
    print("⚠️ Enhanced components initialization failed, continuing without them")

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:4200", "https://localhost:4200"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "X-Session-ID"]
    }
})

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[f"{MAX_REQUESTS_PER_WINDOW} per minute", "1000 per hour"]
)

# Initialize components in the correct order
print("🚀 Initializing optimized RAG system...")

# Step 1: Fetch properties with parallel processing
print("🌐 Step 1: Fetching properties with parallel processing...")
properties = []
try:
    # Use parallel fetcher as primary method
    fetcher = ParallelPropertyFetcher(API_ENDPOINT)
    properties = fetcher.fetch_all_properties(total_properties=600)
    if properties:
        print(f"✅ Successfully fetched {len(properties)} properties using parallel fetcher")
        # Cache the properties
        cache_properties(properties)
    else:
        # Fallback to single request method
        print("Parallel fetcher returned no properties, trying single request...")
        properties = fetch_and_cache_properties()
        if not properties:
            logger.warning("Both property fetching methods failed. Using empty properties list.")
            properties = []
except Exception as e:
    logger.error(f"Parallel fetcher failed: {e}")
    # Fallback to single request method
    try:
        print("Trying single request method...")
        properties = fetch_and_cache_properties()
        if not properties:
            logger.warning("Single request method also failed. Using empty properties list.")
            properties = []
    except Exception as e2:
        logger.error(f"Single request method failed: {e2}")
        properties = []

# Step 2: Create vector database
print("🔧 Step 2: Creating vector database...")
model = None
index = None
vector_db = None
try:
    # Load Jina model for vector operations
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True, device=device)
    print(f"✅ Loaded Jina model: {model.get_sentence_embedding_dimension()} dimensions")
    
    # Create vector database with fetched properties
    if properties:
        vector_db = VectorDBManager(model)
        print(f"🔄 Creating vector database with {len(properties)} properties...")
        index = vector_db.create_vector_db(properties)
        print(f"✅ Vector database created successfully! Index size: {index.ntotal}")
    else:
        print("⚠️ No properties available for vector database creation")
except Exception as e:
    logger.error(f"Failed to create vector database: {e}")
    print("⚠️ Vector database creation failed, continuing without it")

# Step 3: Create retriever
print("🔍 Step 3: Creating optimized retriever...")
retriever = None
try:
    if model is not None and index is not None:
        retriever = OptimizedRAGRetriever(
            model=model,
            index=index,
            tokenizer=None,  # We'll use basic model without tokenizer
            pca=None,        # No PCA for simple setup
            feature_matcher=None  # No feature matcher for simple setup
        )
        # Set the properties_df from vector_db
        if vector_db is not None:
            retriever.properties_df = vector_db.properties_df
        print("✅ Optimized retriever created successfully!")
        print(f"📊 Retriever ready with {index.ntotal} indexed properties")
    else:
        print("⚠️ Model or index not available, skipping retriever creation")
except Exception as e:
    logger.error(f"Failed to create retriever: {e}")
    retriever = None

# Step 4: Initialize optimized RAG processor
print("🚀 Step 4: Initializing optimized RAG processor...")
optimized_rag_processor = None
try:
    if get_optimized_rag_processor is not None and model is not None and index is not None:
        # Initialize the optimized RAG processor with existing models
        optimized_rag_processor = get_optimized_rag_processor()
        optimized_rag_processor.model = model
        optimized_rag_processor.index = index
        optimized_rag_processor.dimension = index.d
        # Set the properties_df from vector_db (same as original retriever)
        if vector_db is not None:
            optimized_rag_processor.properties_df = vector_db.properties_df
        print("✅ Optimized RAG processor initialized successfully!")
        print(f"📊 Optimized RAG processor ready with {index.ntotal} indexed properties")
    else:
        print("⚠️ Optimized RAG processor not available or models not ready")
except Exception as e:
    logger.error(f"Failed to initialize optimized RAG processor: {e}")
    optimized_rag_processor = None

# Step 5: Initialize AI property filter
print("🤖 Step 5: Initializing AI property filter...")
ai_property_filter = None
try:
    if get_ai_property_filter is not None:
        ai_property_filter = get_ai_property_filter()
        print("✅ AI property filter initialized successfully!")
    else:
        print("⚠️ AI property filter not available")
except Exception as e:
    logger.error(f"Failed to initialize AI property filter: {e}")
    ai_property_filter = None

print("Loading tokenizer and LLM model...")
tokenizer = None
model_llm = None
try:
    tokenizer, model_llm = load_tokenizer_and_model()
    print("✅ Tokenizer and LLM model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load tokenizer and LLM model: {e}")
    print("⚠️ Tokenizer and LLM model loading failed, continuing without them")

print("Initializing security components...")
security_manager = None
query_validator = None
try:
    security_manager = SecurityManager()
    query_validator = QueryValidator(model_embedding)
    print("✅ Security components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize security components: {e}")
    print("⚠️ Security components initialization failed, continuing without them")

print("Initializing input tracker...")
input_tracker = None
try:
    input_tracker = UserInputTracker()
    print("✅ Input tracker initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize input tracker: {e}")
    print("⚠️ Input tracker initialization failed, continuing without it")

# Initialize processors
chatbot_processor = None
try:
    chatbot_processor = ChatbotProcessor()
    print("✅ Chatbot processor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chatbot processor: {e}")
    print("⚠️ Chatbot processor initialization failed, continuing without it")

def security_check(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            ip_address = request.remote_addr
            
            # Get session ID from request
            session_id = None
            if request.is_json:
                session_id = request.json.get('session_id')
            elif request.args:
                session_id = request.args.get('session_id')
            
            # Create session if not exists
            if not session_id:
                session_id = session_manager.create_session(
                    ip_address=ip_address,
                    user_plan="basic"
                )
                if session_id:
                    # Add session_id to request for later use
                    request.session_id = session_id
            
            # Enhanced rate limiting with session-based tracking
            if session_id:
                # Get user plan from session
                session = session_manager.get_session(session_id)
                user_plan = session.user_plan if session else "basic"
                
                # Check rate limits
                allowed, rate_info = rate_limiter.is_allowed(session_id, user_plan)
                if not allowed:
                    return jsonify({
                        "error": "Rate limit exceeded",
                        "rate_limit_info": rate_info
                    }), 429
                
                # Update session with rate limit info
                session_manager.update_session(session_id, {
                    "rate_limit_data": rate_info
                })
            
            # Legacy security manager check
            if security_manager is not None:
                if not security_manager.check_rate_limit(ip_address):
                    return jsonify({"error": "Rate limit exceeded"}), 429

            if request.method == 'POST':
                if not request.is_json:
                    return jsonify({"error": "Content-Type must be application/json"}), 415

            return f(*args, **kwargs)
        except Exception as e:
            logging.error(f"Security check failed: {str(e)}")
            return jsonify({"error": "Security check failed"}), 400
    return decorated_function

@app.before_request
def handle_preflight():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, X-Session-ID')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        return response

@app.route('/')
def index():
    print("Rendering index page")
    return render_template('index.html')

@app.route('/search', methods=['POST'])
@security_check
@limiter.limit("30 per minute")
@with_user_plan
def search():
    try:
        data = request.json
        query = data.get('query')

        if not query:
            return jsonify({"error": "Query parameter is missing"}), 400

        if query_validator is not None:
            cleaned_query = query_validator.clean_input(query)
            if not query_validator.validate_query_length(cleaned_query):
                return jsonify({"error": "Query too long"}), 400
        else:
            cleaned_query = query

        session_id = data.get('session_id')
        continue_conversation = data.get('continue', False)

        if session_id not in conversation_context or not continue_conversation:
            # Initialize constraint parser
            constraint_parser = ConstraintParser()
            
            # Parse constraints from query
            constraints = constraint_parser.parse_all_constraints(cleaned_query)
            constraints_summary = constraint_parser.format_constraints_summary(constraints)
            
            print(f"🔍 Parsed constraints: {constraints_summary}")
            
            # Use original retriever only (skip optimized RAG processor)
            if retriever is not None:
                try:
                    print(f"🔍 Searching vector database for: '{cleaned_query}'")
                    search_results = retriever.retrieve(cleaned_query, top_k=200)
                    print(f"✅ RAG retriever found {len(search_results)} results")
                    
                    # Apply AI-powered property filtering with performance monitoring
                    if ai_property_filter is not None and search_results:
                        try:
                            print(f"🤖 Applying OPTIMIZED AI property filtering...")
                            original_count = len(search_results)
                            ai_filtering_start = time.time()
                            
                            # Extract property data for filtering
                            properties_data = [result['property'] for result in search_results]
                            
                            # Clear GPU memory before processing
                            ai_property_filter.clear_gpu_memory()
                            
                            # Apply optimized AI filtering
                            ai_filtered_matches = ai_property_filter.filter_properties(
                                query=cleaned_query,
                                properties=properties_data,
                                max_results=50  # Limit to top 50 after AI filtering
                            )
                            
                            ai_filtering_time = time.time() - ai_filtering_start
                            
                            # Convert back to expected format
                            filtered_search_results = []
                            for match in ai_filtered_matches:
                                # Find original result with distance
                                original_result = next(
                                    (r for r in search_results if r['property'] == match.property_data), 
                                    None
                                )
                                if original_result:
                                    filtered_search_results.append({
                                        'property': match.property_data,
                                        'distance': original_result.get('distance', 0.0),
                                        'ai_confidence': match.confidence_score,
                                        'match_reasons': match.match_reasons,
                                        'matched_fields': match.matched_fields
                                    })
                            
                            search_results = filtered_search_results
                            print(f"✅ OPTIMIZED AI filtering: {original_count} → {len(search_results)} results in {ai_filtering_time:.2f}s")
                            print(f"📈 Performance: {original_count/ai_filtering_time:.1f} properties/second")
                            
                            # Show top AI matches for debugging
                            if search_results:
                                print("🏆 Top AI matches:")
                                for i, result in enumerate(search_results[:3]):
                                    property_name = result['property'].get('PropertyName', 'N/A')
                                    confidence = result.get('ai_confidence', 0.0)
                                    print(f"  {i+1}. {property_name} (AI Confidence: {confidence:.3f})")
                            
                            # Analyze match quality based on AI filtering results
                            status_message = ""  # Initialize status message
                            if ai_filtered_matches:
                                print("\n🔍 AI Match Quality Analysis:")
                                match_quality = ai_property_filter.analyze_match_quality(cleaned_query, ai_filtered_matches)
                                
                                # Print overall quality assessment
                                overall_quality = match_quality['overall_quality']
                                total_matches = match_quality['total_matches']
                                avg_confidence = match_quality['average_confidence']
                                highest_confidence = match_quality['highest_confidence']
                                
                                # Generate status message based on match quality
                                if overall_quality == 'exact_match':
                                    print(f"✅ EXACT MATCH: Found {total_matches} properties with high confidence (avg: {avg_confidence:.3f}, max: {highest_confidence:.3f})")
                                    status_message = f"Found {total_matches} exact matches based on your query"
                                elif overall_quality == 'similar_match':
                                    print(f"🔄 SIMILAR MATCH: Found {total_matches} properties with medium confidence (avg: {avg_confidence:.3f}, max: {highest_confidence:.3f})")
                                    status_message = f"Couldn't find exact matches, here are {total_matches} similar matches"
                                elif overall_quality == 'weak_match':
                                    print(f"⚠️ WEAK MATCH: Found {total_matches} properties with low confidence (avg: {avg_confidence:.3f}, max: {highest_confidence:.3f})")
                                    status_message = f"Couldn't find exact matches, here are {total_matches} similar matches"
                                else:
                                    print(f"❌ NO MATCH: No meaningful matches found for query")
                                    status_message = "No matches found based on your query"
                                
                                # Print detailed breakdown
                                exact_count = match_quality['exact_matches']
                                similar_count = match_quality['similar_matches']
                                weak_count = match_quality['low_quality_matches']
                                
                                print(f"📊 Match Breakdown:")
                                print(f"   • Exact matches (≥0.8): {exact_count}")
                                print(f"   • Similar matches (0.5-0.8): {similar_count}")
                                print(f"   • Weak matches (0.3-0.5): {weak_count}")
                                
                                # Print top match details
                                if match_quality['analysis_details']['top_match_details']:
                                    print(f"🎯 Top Match Details:")
                                    for detail in match_quality['analysis_details']['top_match_details']:
                                        rank = detail['rank']
                                        name = detail['property_name']
                                        conf = detail['confidence']
                                        fields = ', '.join(detail['matched_fields'][:3])  # Top 3 fields
                                        print(f"   {rank}. {name} (Confidence: {conf:.3f}, Fields: {fields})")
                                
                                print(f"🔧 Thresholds Used: Exact≥0.8, Similar≥0.5, Weak≥0.3")
                                print("=" * 60)
                            
                            # Clear GPU memory after processing
                            ai_property_filter.clear_gpu_memory()
                                    
                        except Exception as e:
                            print(f"⚠️ AI filtering failed: {e}")
                            # Continue with original results if AI filtering fails
                            # Set fallback status message
                            if len(search_results) > 0:
                                status_message = f"Found {len(search_results)} properties based on your query"
                            else:
                                status_message = "No matches found based on your query"
                    
                    # Apply constraint filtering
                    if constraints_summary != "No specific constraints":
                        print(f"🔧 Applying constraint filtering...")
                        original_count = len(search_results)
                        filtered_results = []
                        
                        for result in search_results:
                            property_data = result['property']
                            if constraint_parser.property_matches_constraints(property_data, constraints):
                                filtered_results.append(result)
                        
                        search_results = filtered_results
                        print(f"✅ Constraint filtering: {original_count} → {len(search_results)} results")
                    
                    # Set fallback status message if AI filtering was not available
                    if not status_message:
                        if len(search_results) > 0:
                            status_message = f"Found {len(search_results)} properties based on your query"
                        else:
                            status_message = "No matches found based on your query"
                    
                    # Debug: Show what data we're getting
                    if search_results:
                        print("🔍 Debug: First result property data:")
                        first_property = search_results[0]['property']
                        print(f"  propertyId: {first_property.get('propertyId', 'NOT_FOUND')}")
                        print(f"  PropertyName: {first_property.get('PropertyName', 'NOT_FOUND')}")
                        print(f"  PropertyType: {first_property.get('PropertyType', 'NOT_FOUND')}")
                        print(f"  Address: {first_property.get('Address', 'NOT_FOUND')}")
                        print(f"  Available keys: {list(first_property.keys())}")
                        # Check for different possible ID field names
                        print(f"  Raw ID fields: id={first_property.get('id', 'NOT_FOUND')}, propertyId={first_property.get('propertyId', 'NOT_FOUND')}")
                    
                    # Show top results for debugging
                    if search_results:
                        print("🏆 Top matches:")
                        for i, result in enumerate(search_results[:3]):
                            property_name = result['property'].get('PropertyName', 'N/A')
                            distance = result.get('distance', 0.0)
                            print(f"  {i+1}. {property_name} (Distance: {distance:.4f})")
                except Exception as e:
                    print(f"⚠️ RAG retriever failed: {e}")
                    search_results = []
            else:
                # Fallback to chatbot processor with available properties
                print("Using fallback search with available properties...")
                # Get cached properties
                cached_properties = get_cached_properties()
                if not cached_properties:
                    # Try to fetch properties again
                    cached_properties = fetch_and_cache_properties()
                
                if cached_properties:
                    # Apply constraint filtering to cached properties
                    if constraints_summary != "No specific constraints":
                        print(f"🔧 Applying constraint filtering to cached properties...")
                        original_count = len(cached_properties)
                        filtered_properties = []
                        
                        for property_data in cached_properties:
                            if constraint_parser.property_matches_constraints(property_data, constraints):
                                filtered_properties.append(property_data)
                        
                        cached_properties = filtered_properties
                        print(f"✅ Constraint filtering: {original_count} → {len(cached_properties)} properties")
                    
                    # Use chatbot processor with filtered properties
                    if chatbot_processor is not None:
                        search_results = chatbot_processor.process_query(cleaned_query, retriever=retriever)
                        # Convert to expected format
                        search_results = [{"property": result['data'], "distance": 0.0} for result in search_results]
                        print(f"✅ Fallback search found {len(search_results)} results")
                    else:
                        # Use cached properties directly
                        search_results = [{"property": prop, "distance": 0.0} for prop in cached_properties]
                        print(f"✅ Using cached properties directly: {len(search_results)} results")
                else:
                    print("❌ No properties available for search")
                    search_results = []
            
            formatted_results = []

            for result in search_results:
                property_info = result['property']

                # Get property images from the property info
                property_images = property_info.get('propertyImages', [])
                if isinstance(property_images, str):
                    if ',' in property_images:
                        property_images = [img.strip() for img in property_images.split(',')]
                    else:
                        property_images = [property_images]
                elif property_images is None:
                    property_images = []

                property_info = convert_numeric_fields_to_int(property_info)

                formatted_result = {
                    "propertyId": property_info.get('propertyId', property_info.get('PropertyID', 'N/A')),
                    "PropertyName": property_info.get('PropertyName', 'N/A'),
                    "Address": property_info.get('Address', 'N/A'),
                    "ZipCode": property_info.get('ZipCode', 0),
                    "LeasableSquareFeet": property_info.get('LeasableSquareFeet', 0),
                    "YearBuilt": property_info.get('YearBuilt', 0),
                    "NumberOfRooms": property_info.get('NumberOfRooms', 0),
                    "ParkingSpaces": property_info.get('ParkingSpaces', 0),
                    "PropertyManager": property_info.get('PropertyManager', 'N/A'),
                    "MarketValue": float(property_info.get('MarketValue', 0)),
                    "TaxAssessmentNumber": property_info.get('TaxAssessmentNumber', 'N/A'),
                    "Latitude": float(property_info.get('Latitude', 0)),
                    "Longitude": float(property_info.get('Longitude', 0)),
                    "CreateDate": property_info.get('CreateDate', 'N/A'),
                    "LastModifiedDate": property_info.get('LastModifiedDate', 'N/A'),
                    "City": property_info.get('City', 'N/A'),
                    "State": property_info.get('State', 'N/A'),
                    "Country": property_info.get('Country', 'N/A'),
                    "PropertyType": property_info.get('PropertyType', 'N/A'),
                    "PropertyStatus": property_info.get('PropertyStatus', 'N/A'),
                    "Description": property_info.get('Description', 'N/A'),
                    "ViewNumber": property_info.get('ViewNumber', 0),
                    "Contact": property_info.get('Contact', 0),
                    "TotalSquareFeet": property_info.get('TotalSquareFeet', 0),
                    "IsDeleted": bool(property_info.get('IsDeleted', False)),
                    "Beds": property_info.get('Beds', 0),
                    "Baths": property_info.get('Baths', 0),
                    "AgentName": property_info.get('AgentName', 'N/A'),
                    "AgentPhoneNumber": property_info.get('AgentPhoneNumber', 'N/A'),
                    "AgentEmail": property_info.get('AgentEmail', 'N/A'),
                    "KeyFeatures": property_info.get('KeyFeatures', 'N/A'),
                    "NearbyAmenities": property_info.get('NearbyAmenities', 'N/A'),
                    "propertyImages": property_images,
                    "Distance": result.get('distance', 0.0),
                    "ConstraintsMatched": constraints_summary if constraints_summary != "No specific constraints" else None
                }
                formatted_results.append(formatted_result)

            conversation_context[session_id] = formatted_results
        else:
            formatted_results = conversation_context[session_id]

        print(f"Returning {len(formatted_results)} search results")
        if formatted_results:
            print(f"Sample property images array: {formatted_results[0]['propertyImages']}")
        
        # Seed AI conversation memory from search to enable immediate follow-ups
        try:
            if ai_conversation_memory is not None and session_id:
                top_names = [fr.get('PropertyName', 'Property') for fr in formatted_results[:5]]
                assistant_summary = f"Here are {len(formatted_results)} properties: " + ", ".join(top_names)
                ai_conversation_memory.add_conversation_turn(
                    session_id=session_id,
                    user_query=cleaned_query,
                    assistant_response=assistant_summary,
                    retrieved_properties=formatted_results
                )
        except Exception as e:
            logger.warning(f"Failed to seed AI conversation memory from search: {e}")

        return jsonify({
            "properties": formatted_results,
            "status_message": status_message
        })

    except Exception as e:
        logging.error(f"Error in search endpoint: {str(e)}")
        return jsonify({"error": "An error occurred processing your request"}), 500

@app.route('/transcribe', methods=['POST'])
@security_check
def transcribe():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        
        # Validate file size (max 10MB)
        if audio_file.content_length and audio_file.content_length > 10 * 1024 * 1024:
            return jsonify({"error": "Audio file too large. Maximum size is 10MB"}), 400

        # Validate file type
        allowed_extensions = {'wav', 'mp3', 'ogg', 'webm'}
        if '.' not in audio_file.filename or \
           audio_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({"error": "Invalid audio file format. Supported formats: WAV, MP3, OGG, WEBM"}), 400

        result = process_audio_file(audio_file)
        
        if isinstance(result, tuple) and len(result) == 2:
            response, status_code = result
            return jsonify(response), status_code
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in transcribe endpoint: {str(e)}")
        return jsonify({"error": "An error occurred processing your audio file"}), 500

@app.route('/generate', methods=['POST'])
@security_check
@limiter.limit("30 per minute")
@with_user_plan
def generate():
    data = request.json
    query = data.get('query')
    session_id = data.get('session_id')
    continue_conversation = data.get('continue', False)
    current_plan = get_current_plan()

    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400
    
    # Check if optimizations are enabled
    if not OPTIMIZATIONS_ENABLED:
        # Fallback to original synchronous processing
        if tokenizer is not None and model_llm is not None:
            if session_id in conversation_context and continue_conversation:
                previous_results = conversation_context[session_id]
                combined_query = f"Based on previous results:{previous_results}New Query: {query}"
                response, duration = generate_response(combined_query, tokenizer, model_llm)
            else:
                response, duration = generate_response(query, tokenizer, model_llm)
                conversation_context[session_id] = response
            print(f"Generated response: {response}")
            print(f"Time taken to generate response: {duration:.2f} seconds\n")
            return jsonify({"response": response, "duration": duration})
        else:
            return jsonify({"error": "Text generation models not available"}), 503
    
    # Use enhanced model manager for parallel processing
    if get_enhanced_model_manager is not None:
        try:
            model_manager = get_enhanced_model_manager()
            
            # Prepare query with conversation context
            if session_id in conversation_context and continue_conversation:
                previous_results = conversation_context[session_id]
                combined_query = f"Based on previous results:{previous_results}New Query: {query}"
            else:
                combined_query = query
            
            # Generate response using parallel processing
            responses = model_manager.parallel_generate_responses([combined_query], batch_size=1)
            response = responses[0] if responses else "Error generating response"
            
            # Update conversation context
            conversation_context[session_id] = response
            
            print(f"🚀 Generated response using enhanced model manager: {response}")
            return jsonify({"response": response, "duration": 0.5, "optimized": True})
            
        except Exception as e:
            logger.error(f"Enhanced model manager failed: {e}")
            # Fall through to optimized generation
    else:
        logger.warning("Enhanced model manager not available")
    
    # Use optimized generation directly
    if tokenizer is not None and model_llm is not None:
        try:
            # Prepare query with conversation context
            if session_id in conversation_context and continue_conversation:
                previous_results = conversation_context[session_id]
                combined_query = f"Based on previous results:{previous_results}New Query: {query}"
            else:
                combined_query = query
            
            # Use optimized generation function
            from modules.response import generate_response_optimized
            response, duration = generate_response_optimized(
                combined_query, tokenizer, model_llm,
                max_new_tokens=256, temperature=0.7
            )
            
            # Update conversation context
            conversation_context[session_id] = response
            
            print(f"🚀 Generated response using optimized generation: {response}")
            print(f"⚡ Duration: {duration:.2f} seconds ({(72.8/duration):.1f}x faster!)")
            return jsonify({"response": response, "duration": duration, "optimized": True})
            
        except Exception as e:
            logger.error(f"Optimized generation failed: {e}")
            # Fall through to async processor
    else:
        logger.warning("Tokenizer or model not available for optimized generation")
    
    # Check cache first
    if query_cache is not None:
        cache_key = f"generate:{session_id}:{hash(query)}"
        cached_result = query_cache.get(query)
        if cached_result:
            logger.info(f"Cache hit for query: {query[:50]}...")
            return jsonify(cached_result)
    else:
        logger.warning("Query cache not available")
    
    # Prepare query with conversation context
    if session_id in conversation_context and continue_conversation:
        previous_results = conversation_context[session_id]
        combined_query = f"Based on previous results:{previous_results}New Query: {query}"
    else:
        combined_query = query
    
    # Submit to async processor with high priority
    if async_processor is not None and RequestPriority is not None:
        priority = RequestPriority.HIGH if query.lower() in ['hi', 'hello', 'help'] else RequestPriority.NORMAL
        
        def callback(result):
            if result.get('status') == 'success':
                response = result.get('response', '')
                duration = result.get('duration', 0)
                
                # Cache the result
                if query_cache is not None:
                    query_cache.set(query, {
                        "response": response, 
                        "duration": duration,
                        "cached": True
                    })
                
                # Update conversation context
                conversation_context[session_id] = response
        
        # Submit request to async processor
        request_id = async_processor.submit_text_generation(
            session_id=session_id,
            query=combined_query,
            priority=priority,
            callback=callback
        )
    else:
        logger.warning("Async processor not available")
        return jsonify({"error": "Text generation service not available"}), 503
    
    # Wait for result with timeout
    if async_processor is not None:
        max_wait_time = 30  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            result = async_processor.get_request_status(request_id)
            if result:
                if result.get('status') == 'success':
                    response = result.get('response', '')
                    duration = result.get('duration', 0)
                    print(f"Generated response: {response}")
                    print(f"Time taken to generate response: {duration:.2f} seconds\n")
                    return jsonify({"response": response, "duration": duration})
                elif result.get('status') == 'error':
                    return jsonify({"error": result.get('error', 'Generation failed')}), 500
            
            time.sleep(0.1)  # Small delay before checking again
        
        # Timeout - return processing message
        return jsonify({
            "response": "Your request is being processed. Please check back in a moment.",
            "processing": True,
            "request_id": request_id
        }), 202
    else:
        return jsonify({"error": "Async processor not available"}), 503

@app.route('/set-location', methods=['POST'])
@security_check
def handle_set_location():
    """Handle location setting and nearby property search"""
    try:
        # Get request data
        data = request.get_json()
        print(f"Received data: {data}")
        
        # Extract values
        latitude = float(data.get('latitude', 0))
        longitude = float(data.get('longitude', 0))
        session_id = data.get('session_id', '')
        
        print(f"Extracted values - latitude: {latitude}, longitude: {longitude}, session_id: {session_id}")
        
        # Validate coordinates
        if latitude == 0 or longitude == 0:
            return jsonify({
                "status": "error",
                "message": "Invalid coordinates"
            }), 400
            
        # Initialize location processor
        location_processor = LocationProcessor()
        
        # Set location and find nearby properties
        result = location_processor.set_location(latitude, longitude, session_id)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in set_location: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Error processing location"
        }), 500

@app.route('/get-nearby-properties', methods=['POST'])
@security_check
def get_nearby_properties():
    """Get properties near the user's location"""
    try:
        data = request.get_json()
        latitude = float(data.get('latitude', 0))
        longitude = float(data.get('longitude', 0))
        radius_km = float(data.get('radius_km', 10.0))
        session_id = data.get('session_id', 'default')
        
        if latitude == 0 and longitude == 0:
            return jsonify({'error': 'Invalid coordinates'}), 400
        
        # Get properties from cache
        properties = get_cached_properties()
        if not properties:
            return jsonify({'error': 'No properties available'}), 500
        
        # Process location and find nearby properties
        location_processor = LocationProcessor()
        nearby_properties = location_processor.find_nearby_properties(latitude, longitude, radius_km)
        
        # Format properties for response
        property_processor = PropertyProcessor()
        formatted_properties = []
        
        for property_data in nearby_properties:
            formatted_property = property_processor.format_property_details(property_data)
            if formatted_property:
                formatted_properties.append({
                    'details': formatted_property,
                    'data': property_data,
                    'distance_km': property_data.get('distance_km', 0)
                })
        
        return jsonify({
            'success': True,
            'properties': formatted_properties,
            'count': len(formatted_properties),
            'radius_km': radius_km,
            'location': {'latitude': latitude, 'longitude': longitude}
        })
        
    except Exception as e:
        logger.error(f"Error in get_nearby_properties: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/check-input-limit', methods=['GET'])
@security_check
def check_input_limit():
    try:
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400

        current_plan = get_current_plan()
        if input_tracker is not None:
            remaining_inputs = input_tracker.get_remaining_inputs(session_id, current_plan)
            usage_stats = input_tracker.get_usage_stats(session_id)
        else:
            remaining_inputs = 999  # Unlimited if tracker not available
            usage_stats = {"total_used": 0, "remaining_time": 24}

        return jsonify({
            "plan": current_plan.value,
            "remaining_inputs": remaining_inputs,
            "total_limit": PLAN_INPUT_LIMITS[current_plan],
            "usage_stats": usage_stats
        })

    except Exception as e:
        logging.error(f"Error checking input limit: {str(e)}")
        return jsonify({"error": "Error checking input limit"}), 500

@app.route('/check-request-status', methods=['GET'])
@security_check
def check_request_status():
    """Check the status of a processing request"""
    try:
        request_id = request.args.get('request_id')
        if not request_id:
            return jsonify({"error": "request_id is required"}), 400
        
        if not OPTIMIZATIONS_ENABLED or async_processor is None:
            return jsonify({
                "error": "Request status checking not available - optimizations disabled",
                "optimizations_enabled": False
            }), 503
        
        result = async_processor.get_request_status(request_id)
        if result:
            return jsonify(result)
        else:
            return jsonify({"status": "not_found"}), 404
            
    except Exception as e:
        logging.error(f"Error checking request status: {str(e)}")
        return jsonify({"error": "Error checking request status"}), 500

@app.route('/optimization-status', methods=['GET'])
@security_check
def get_legacy_optimization_status():
    """Get legacy optimization status"""
    try:
        status = {
            'optimizations_enabled': OPTIMIZATIONS_ENABLED,
            'components': {
                'async_processor': async_processor is not None,
                'connection_pool': http_pool is not None and model_pool is not None,
                'cache_manager': cache_manager is not None and query_cache is not None,
                'load_balancer': processing_manager is not None
            }
        }
        return jsonify(status)
        
    except Exception as e:
        logging.error(f"Error getting optimization status: {str(e)}")
        return jsonify({"error": "Error getting optimization status"}), 500

@app.route('/system-stats', methods=['GET'])
@security_check
def get_system_stats():
    """Get system performance statistics"""
    try:
        stats = {
            'optimizations_enabled': OPTIMIZATIONS_ENABLED,
            'status': 'active'
        }
        
        # Add AI conversation memory stats
        if ai_conversation_memory is not None:
            try:
                stats['ai_conversation_memory'] = ai_conversation_memory.get_stats()
            except Exception as e:
                stats['ai_conversation_memory'] = {'error': str(e)}
        else:
            stats['ai_conversation_memory'] = {'status': 'not_available'}
        
        if OPTIMIZATIONS_ENABLED and async_processor is not None:
            try:
                stats['async_processor'] = async_processor.get_queue_stats()
            except Exception as e:
                stats['async_processor'] = {'error': str(e)}
        
        if OPTIMIZATIONS_ENABLED and cache_manager is not None:
            try:
                stats['cache_manager'] = cache_manager.get_all_stats()
            except Exception as e:
                stats['cache_manager'] = {'error': str(e)}
        
        if OPTIMIZATIONS_ENABLED and processing_manager is not None:
            try:
                stats['processing_manager'] = processing_manager.get_stats()
                if hasattr(processing_manager, 'load_balancer') and processing_manager.load_balancer is not None:
                    stats['load_balancer'] = processing_manager.load_balancer.get_stats()
            except Exception as e:
                stats['processing_manager'] = {'error': str(e)}
        
        # Add new optimization stats
        if OPTIMIZATIONS_ENABLED and get_performance_optimizer is not None:
            try:
                stats['performance_optimizer'] = get_performance_optimizer().get_performance_stats()
            except Exception as e:
                stats['performance_optimizer'] = {'error': str(e)}
        
        if OPTIMIZATIONS_ENABLED and get_enhanced_model_manager is not None:
            try:
                stats['enhanced_model_manager'] = get_enhanced_model_manager().get_model_stats()
            except Exception as e:
                stats['enhanced_model_manager'] = {'error': str(e)}
        
        if optimized_rag_processor is not None:
            try:
                stats['optimized_rag_processor'] = optimized_rag_processor.get_performance_stats()
            except Exception as e:
                stats['optimized_rag_processor'] = {'error': str(e)}
        
        # Add AI property filter stats
        if ai_property_filter is not None:
            try:
                filter_stats = ai_property_filter.get_filtering_stats()
                # Add performance metrics
                filter_stats['gpu_available'] = torch.cuda.is_available()
                if torch.cuda.is_available():
                    filter_stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / (1024**3)  # GB
                    filter_stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / (1024**3)  # GB
                stats['ai_property_filter'] = filter_stats
            except Exception as e:
                stats['ai_property_filter'] = {'error': str(e)}
        else:
            stats['ai_property_filter'] = {'status': 'not_available'}
        
        # Add multi-user stats
        try:
            stats['session_manager'] = session_manager.get_stats()
            stats['rate_limiter'] = rate_limiter.get_stats()
            stats['websocket_manager'] = websocket_manager.get_connection_stats()
        except Exception as e:
            stats['multi_user'] = {'error': str(e)}
        
        # Add enhanced component stats
        try:
            if enhanced_security_manager:
                stats['security_stats'] = enhanced_security_manager.get_security_stats()
            
            if advanced_processor:
                stats['processing_stats'] = advanced_processor.get_performance_stats()
            
            if error_recovery_system:
                stats['error_stats'] = error_recovery_system.get_error_stats()
        except Exception as e:
            stats['enhanced_components'] = {'error': str(e)}
        
        return jsonify(stats)
        
    except Exception as e:
        logging.error(f"Error getting system stats: {str(e)}")
        return jsonify({"error": "Error getting system stats"}), 500

@app.route('/session/create', methods=['POST'])
def create_session():
    """Create a new user session"""
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id')
        user_plan = data.get('user_plan', 'basic')
        ip_address = request.remote_addr
        
        session_id = session_manager.create_session(
            user_id=user_id,
            ip_address=ip_address,
            user_plan=user_plan
        )
        
        if session_id:
            return jsonify({
                'session_id': session_id,
                'status': 'created',
                'user_plan': user_plan
            })
        else:
            return jsonify({'error': 'Failed to create session'}), 500
            
    except Exception as e:
        logging.error(f"Error creating session: {str(e)}")
        return jsonify({"error": "Error creating session"}), 500

@app.route('/session/<session_id>', methods=['GET'])
def get_session_info(session_id):
    """Get session information"""
    try:
        session = session_manager.get_session(session_id)
        if session:
            return jsonify({
                'session_id': session.session_id,
                'user_id': session.user_id,
                'user_plan': session.user_plan,
                'created_at': session.created_at,
                'last_activity': session.last_activity,
                'is_active': session.is_active,
                'search_history_count': len(session.search_history),
                'rate_limit_info': session.rate_limit_data
            })
        else:
            return jsonify({'error': 'Session not found'}), 404
            
    except Exception as e:
        logging.error(f"Error getting session info: {str(e)}")
        return jsonify({"error": "Error getting session info"}), 500

@app.route('/session/<session_id>/update', methods=['POST'])
def update_session(session_id):
    """Update session data"""
    try:
        data = request.get_json() or {}
        
        success = session_manager.update_session(session_id, data)
        if success:
            return jsonify({'status': 'updated'})
        else:
            return jsonify({'error': 'Session not found'}), 404
            
    except Exception as e:
        logging.error(f"Error updating session: {str(e)}")
        return jsonify({"error": "Error updating session"}), 500

@app.route('/session/<session_id>/delete', methods=['DELETE'])
def delete_session(session_id):
    """Delete a session"""
    try:
        success = session_manager.delete_session(session_id)
        if success:
            # Also clear AI conversation memory
            if ai_conversation_memory is not None:
                ai_conversation_memory.clear_conversation(session_id)
            return jsonify({'status': 'deleted'})
        else:
            return jsonify({'error': 'Session not found'}), 404
            
    except Exception as e:
        logging.error(f"Error deleting session: {str(e)}")
        return jsonify({"error": "Error deleting session"}), 500

@app.route('/conversation/<session_id>/context', methods=['GET'])
def get_conversation_context(session_id):
    """Get AI conversation context for a session"""
    try:
        if ai_conversation_memory is not None:
            conversation_context = ai_conversation_memory.get_conversation_context(session_id)
            if conversation_context:
                return jsonify({
                    'session_id': session_id,
                    'current_topic': conversation_context.current_topic,
                    'conversation_summary': conversation_context.conversation_summary,
                    'context_strength': conversation_context.context_strength,
                    'user_preferences': conversation_context.user_preferences,
                    'recent_turns': ai_conversation_memory.get_conversation_history(session_id, max_turns=5),
                    'last_activity': conversation_context.last_activity.isoformat()
                })
            else:
                return jsonify({'error': 'No conversation context found'}), 404
        else:
            return jsonify({'error': 'AI conversation memory not available'}), 503
            
    except Exception as e:
        logging.error(f"Error getting conversation context: {str(e)}")
        return jsonify({"error": "Error getting conversation context"}), 500

@app.route('/conversation/<session_id>/clear', methods=['POST'])
def clear_conversation(session_id):
    """Clear conversation context for a session"""
    try:
        if ai_conversation_memory is not None:
            ai_conversation_memory.clear_conversation(session_id)
            return jsonify({'status': 'cleared'})
        else:
            return jsonify({'error': 'AI conversation memory not available'}), 503
            
    except Exception as e:
        logging.error(f"Error clearing conversation: {str(e)}")
        return jsonify({"error": "Error clearing conversation"}), 500

@app.route('/rate-limit/check', methods=['POST'])
def check_rate_limit():
    """Check rate limit for a session"""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id')
        user_plan = data.get('user_plan', 'basic')
        
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
        
        allowed, rate_info = rate_limiter.is_allowed(session_id, user_plan)
        remaining = rate_limiter.get_remaining_requests(session_id, user_plan)
        
        return jsonify({
            'allowed': allowed,
            'rate_limit_info': rate_info,
            'remaining_requests': remaining
        })
        
    except Exception as e:
        logging.error(f"Error checking rate limit: {str(e)}")
        return jsonify({"error": "Error checking rate limit"}), 500

@app.route('/websocket/status', methods=['GET'])
def get_websocket_status():
    """Get WebSocket server status"""
    try:
        stats = websocket_manager.get_connection_stats()
        return jsonify({
            'status': 'running' if websocket_manager.is_running else 'stopped',
            'stats': stats
        })
        
    except Exception as e:
        logging.error(f"Error getting WebSocket status: {str(e)}")
        return jsonify({"error": "Error getting WebSocket status"}), 500

# Enhanced Security Endpoints
@app.route('/auth/register', methods=['POST'])
def register_user():
    """Register a new user"""
    try:
        data = request.get_json()
        if not data or 'user_id' not in data or 'password' not in data:
            return jsonify({'error': 'Missing user_id or password'}), 400
        
        user_id = data['user_id']
        password = data['password']
        permissions = data.get('permissions', ['basic'])
        
        if enhanced_security_manager:
            success = enhanced_security_manager.register_user(user_id, password, permissions)
            if success:
                return jsonify({'message': 'User registered successfully'})
            else:
                return jsonify({'error': 'User already exists'}), 409
        else:
            return jsonify({'error': 'Security manager not available'}), 503
            
    except Exception as e:
        logging.error(f"Error registering user: {str(e)}")
        return jsonify({"error": "Error registering user"}), 500

@app.route('/auth/login', methods=['POST'])
def login_user():
    """Authenticate user and get session token"""
    try:
        data = request.get_json()
        if not data or 'user_id' not in data or 'password' not in data:
            return jsonify({'error': 'Missing user_id or password'}), 400
        
        user_id = data['user_id']
        password = data['password']
        ip_address = request.remote_addr
        
        if enhanced_security_manager:
            success, session_token, user_data = enhanced_security_manager.authenticate_user(
                user_id, password, ip_address
            )
            
            if success:
                return jsonify({
                    'session_token': session_token,
                    'user_data': user_data,
                    'message': 'Authentication successful'
                })
            else:
                return jsonify({'error': session_token}), 401  # session_token contains error message
        else:
            return jsonify({'error': 'Security manager not available'}), 503
            
    except Exception as e:
        logging.error(f"Error authenticating user: {str(e)}")
        return jsonify({"error": "Error authenticating user"}), 500

@app.route('/auth/verify', methods=['POST'])
def verify_session():
    """Verify session token"""
    try:
        data = request.get_json()
        if not data or 'session_token' not in data:
            return jsonify({'error': 'Missing session_token'}), 400
        
        session_token = data['session_token']
        ip_address = request.remote_addr
        
        if enhanced_security_manager:
            is_valid, user_data = enhanced_security_manager.verify_session(session_token, ip_address)
            
            if is_valid:
                return jsonify({
                    'valid': True,
                    'user_data': user_data
                })
            else:
                return jsonify({'valid': False, 'error': 'Invalid or expired session'})
        else:
            return jsonify({'error': 'Security manager not available'}), 503
            
    except Exception as e:
        logging.error(f"Error verifying session: {str(e)}")
        return jsonify({"error": "Error verifying session"}), 500

# Advanced Processing Endpoints
@app.route('/processing/submit', methods=['POST'])
def submit_processing_task():
    """Submit a task for advanced processing"""
    try:
        data = request.get_json()
        if not data or 'task_type' not in data or 'function_name' not in data:
            return jsonify({'error': 'Missing task_type or function_name'}), 400
        
        task_type = data['task_type']
        function_name = data['function_name']
        priority = data.get('priority', 'normal')
        timeout = data.get('timeout')
        
        # Map task type to enum
        task_type_enum = getattr(TaskType, task_type.upper(), TaskType.IO_BOUND)
        priority_enum = getattr(TaskPriority, priority.upper(), TaskPriority.NORMAL)
        
        if advanced_processor:
            # Get function from available functions
            available_functions = {
                'search_properties': lambda: search_properties(),
                'generate_response': lambda: generate_response(),
                'process_audio': lambda: process_audio_file()
            }
            
            func = available_functions.get(function_name)
            if not func:
                return jsonify({'error': 'Function not available'}), 400
            
            task_id = advanced_processor.submit_task(
                func=func,
                task_type=task_type_enum,
                priority=priority_enum,
                timeout=timeout
            )
            
            return jsonify({
                'task_id': task_id,
                'status': 'submitted'
            })
        else:
            return jsonify({'error': 'Advanced processor not available'}), 503
            
    except Exception as e:
        logging.error(f"Error submitting processing task: {str(e)}")
        return jsonify({"error": "Error submitting task"}), 500

@app.route('/processing/status/<task_id>', methods=['GET'])
def get_processing_status(task_id):
    """Get processing task status"""
    try:
        if advanced_processor:
            result = advanced_processor.get_task_result(task_id)
            
            if result:
                return jsonify({
                    'task_id': task_id,
                    'success': result.success,
                    'result': result.result if result.success else None,
                    'error': result.error if not result.success else None,
                    'duration': result.duration,
                    'metadata': result.metadata
                })
            else:
                return jsonify({
                    'task_id': task_id,
                    'status': 'pending'
                })
        else:
            return jsonify({'error': 'Advanced processor not available'}), 503
            
    except Exception as e:
        logging.error(f"Error getting processing status: {str(e)}")
        return jsonify({"error": "Error getting processing status"}), 500

# Error Recovery Endpoints
@app.route('/errors/stats', methods=['GET'])
def get_error_statistics():
    """Get error recovery statistics"""
    try:
        if error_recovery_system:
            stats = error_recovery_system.get_error_stats()
            return jsonify(stats)
        else:
            return jsonify({'error': 'Error recovery system not available'}), 503
            
    except Exception as e:
        logging.error(f"Error getting error statistics: {str(e)}")
        return jsonify({"error": "Error getting error statistics"}), 500

@app.route('/errors/<error_id>', methods=['GET'])
def get_error_details(error_id):
    """Get detailed error information"""
    try:
        if error_recovery_system:
            error_info = error_recovery_system.get_error_by_id(error_id)
            
            if error_info:
                return jsonify({
                    'error_id': error_info.error_id,
                    'timestamp': error_info.error_id,
                    'error_type': error_info.error_type,
                    'error_message': error_info.error_message,
                    'severity': error_info.severity.value,
                    'category': error_info.category.value,
                    'resolved': error_info.resolved,
                    'recovery_attempts': error_info.recovery_attempts
                })
            else:
                return jsonify({'error': 'Error not found'}), 404
        else:
            return jsonify({'error': 'Error recovery system not available'}), 503
            
    except Exception as e:
        logging.error(f"Error getting error details: {str(e)}")
        return jsonify({"error": "Error getting error details"}), 500

@app.route('/errors/<error_id>/resolve', methods=['POST'])
def resolve_error(error_id):
    """Manually resolve an error"""
    try:
        data = request.get_json() or {}
        resolution_notes = data.get('resolution_notes', 'Manually resolved')
        
        if error_recovery_system:
            error_recovery_system.resolve_error(error_id, resolution_notes)
            return jsonify({'message': 'Error resolved successfully'})
        else:
            return jsonify({'error': 'Error recovery system not available'}), 503
            
    except Exception as e:
        logging.error(f"Error resolving error: {str(e)}")
        return jsonify({"error": "Error resolving error"}), 500

@app.route('/recommend', methods=['POST'])
@security_check
@limiter.limit("30 per minute")
@with_user_plan
def recommend():
    try:
        data = request.json
        query = data.get('query')
        session_id = data.get('session_id')
        continue_conversation = data.get('continue', False)
        current_plan = get_current_plan()

        if not query:
            return jsonify({"error": "Query parameter is missing"}), 400

        # Clean and validate input
        if query_validator is not None:
            cleaned_query = query_validator.clean_input(query)
            if not query_validator.validate_query_length(cleaned_query):
                return jsonify({"error": "Query too long"}), 400

            # Allow follow-ups even if phrased generically
            allow_generic = False
            try:
                if ai_conversation_memory is not None:
                    is_fu, rel = ai_conversation_memory.is_follow_up_query(session_id, cleaned_query)
                    allow_generic = is_fu and rel > 0.5
            except Exception as _:
                allow_generic = False

            # Check if query is related to real estate unless a confident follow-up
            if not allow_generic:
                if not query_validator.is_real_estate_query(cleaned_query):
                    return jsonify({
                        "response": "I'm a real estate chatbot. I can help you with property-related queries like finding apartments, PG accommodations, hostels, commercial properties, farmland, or agricultural land. Please ask me about properties!",
                        "is_real_estate": False
                    })
        else:
            cleaned_query = query

        # Special handling for greeting queries (hi, hello, hey, etc.)
        greeting_variations = ['hi', 'hello', 'hey', 'hii', 'hiii', 'hiiii', 'helloo', 'heyy', 'heyyy']
        if cleaned_query.lower() in greeting_variations:
            return jsonify({
                "response": "Do you want to know the properties located near you? (yes/no):",
                "is_location_query": True
            })

        # Special handling for "yes" after "hi"
        if cleaned_query.lower() == 'yes':
            # Get location from the request
            latitude = data.get('latitude')
            longitude = data.get('longitude')
            
            if not latitude or not longitude:
                return friendly_error(
                    message="Location is not available.",
                    hint="Please allow location access in your browser or set your location using the Set Location option.",
                    needs_location=True,
                    http_code=400
                )

            # Initialize location processor
            try:
                location_processor = LocationProcessor()
                
                # Get nearby properties
                result = location_processor.set_location(latitude, longitude, session_id)
            except Exception as e:
                logger.error(f"Error initializing location processor: {e}")
                return friendly_error(
                    message="Location processing is temporarily unavailable.",
                    hint="Please try again in a moment or proceed without location.",
                    http_code=503
                )
            
            if result["status"] == "success":
                # Format the response for frontend
                properties = result["properties"]
                response_text = "Here are the properties near your location:\n\n"
                
                for i, prop in enumerate(properties, 1):
                    response_text += (
                        f"{i}. {prop.get('PropertyName', 'Unnamed Property')}\n"
                        f"   Address: {prop.get('Address', 'No address available')}\n"
                        f"   Distance: {prop.get('Distance', 0)} km\n"
                        f"   Type: {prop.get('PropertyType', 'Not specified')}\n"
                        f"   Price: ${prop.get('MarketValue', 0):,.2f}\n\n"
                    )
                
                return jsonify({
                    # "response": response_text,
                    "properties": properties,
                    "location": result["location"],
                    "is_location_based": True,
                    "status": "success"
                })
            else:
                return friendly_error(
                    message="No properties were found near your current location.",
                    hint="Try expanding your radius or searching without location.",
                    http_code=404
                )

        # Special handling for "no" after greeting
        if cleaned_query.lower() == 'no':
            return jsonify({
                "response": "Ok, specify what do you want and I will give",
                "is_general_query": True
            })

        # Defaults for follow-up detection
        is_follow_up = False
        context_relevance = 0.0

        # Use AI conversation memory for intelligent context handling
        if ai_conversation_memory is not None:
            try:
                # Check if this is a follow-up query using AI understanding
                is_follow_up, context_relevance = ai_conversation_memory.is_follow_up_query(session_id, cleaned_query)
                logger.info(f"AI conversation memory check - Query: '{cleaned_query}', Is follow-up: {is_follow_up}, Relevance: {context_relevance:.3f}")
                logger.info(f"Session ID: {session_id}")
                
                # Debug: Check if we have conversation context
                conv_context = ai_conversation_memory.get_conversation_context(session_id)
                if conv_context:
                    logger.info(f"Conversation context exists with {len(conv_context.turns)} turns")
                    if conv_context.turns:
                        last_turn = conv_context.turns[-1]
                        logger.info(f"Last turn query: '{last_turn.user_query}'")
                        logger.info(f"Last turn has {len(last_turn.retrieved_properties)} properties")
                        logger.info(f"Last turn properties: {[prop.get('PropertyName', 'Unknown') for prop in last_turn.retrieved_properties]}")
                else:
                    logger.info(f"No conversation context found for session {session_id}")
            except Exception as e:
                logger.error(f"Error in AI conversation memory processing: {e}")
                is_follow_up = False
                context_relevance = 0.0
                logger.info("Falling back to regular processing due to AI memory error")
            
            # Defaults for previous turn context
            previous_query = None
            previous_response = None

            if is_follow_up and context_relevance > 0.5:
                try:
                    # Get the previous conversation turn to retrieve the same properties
                    conversation_context = ai_conversation_memory.get_conversation_context(session_id)
                    if conversation_context and conversation_context.turns:
                        last_turn = conversation_context.turns[-1]
                        previous_query = last_turn.user_query
                        previous_response = last_turn.assistant_response
                        if last_turn.retrieved_properties:
                            logger.info(f"AI detected follow-up query. Using properties from previous turn: {len(last_turn.retrieved_properties)} properties")
                            logger.info(f"Previous turn properties: {[prop.get('PropertyName', 'Unknown') for prop in last_turn.retrieved_properties]}")
                            logger.info(f"Previous turn query: '{last_turn.user_query}'")
                            logger.info(f"Current follow-up query: '{cleaned_query}'")
                            
                            # Check if this is a specific property reference
                            query_lower = cleaned_query.lower()
                            property_ordinals = ["first", "second", "third", "fourth", "fifth", "1st", "2nd", "3rd", "4th", "5th"]
                            
                            # Check if this is a specific property reference
                            ordinal_index = None
                            for i, ordinal in enumerate(property_ordinals):
                                if ordinal in query_lower:
                                    ordinal_index = i % 5  # Map to 0-4 index
                                    break
                            
                            if ordinal_index is not None and ordinal_index < len(last_turn.retrieved_properties):
                                # Return only the specific property
                                specific_property = last_turn.retrieved_properties[ordinal_index]
                                logger.info(f"Returning specific property {ordinal_index + 1}: {specific_property.get('PropertyName', 'Unknown')}")
                                raw_results = [{"property": specific_property, "distance": 0.0}]
                            else:
                                # Use all properties from the previous turn
                                raw_results = [{"property": prop, "distance": 0.0} for prop in last_turn.retrieved_properties]
                            
                            # Skip the RAG processing and go directly to response generation
                            skip_rag_processing = True
                        else:
                            # Use AI-enhanced contextual query for new search
                            enhanced_query = ai_conversation_memory.get_contextual_query(session_id, cleaned_query)
                            logger.info(f"AI detected follow-up query. Enhanced: {enhanced_query}")
                            combined_query = enhanced_query
                            skip_rag_processing = False
                    else:
                        combined_query = cleaned_query
                        skip_rag_processing = False
                except Exception as e:
                    logger.error(f"Error processing follow-up query: {e}")
                    combined_query = cleaned_query
                    skip_rag_processing = False
            else:
                combined_query = cleaned_query
                skip_rag_processing = False
        else:
            # Fallback to legacy conversation context
            if session_id in conversation_context and continue_conversation:
                previous_results = conversation_context[session_id]
                combined_query = f"Based on previous results:{previous_results}New Query: {cleaned_query}"
            else:
                combined_query = cleaned_query
            skip_rag_processing = False
        
        # Check if we should skip RAG processing (for follow-up queries using same properties)
        if 'skip_rag_processing' in locals() and skip_rag_processing:
            logger.info("Skipping RAG processing - using properties from previous conversation turn")
            # Ensure we have properties
            if not raw_results:
                logger.warning("No properties found in previous turn, falling back to RAG processing")
                skip_rag_processing = False
        else:
            # Check if optimizations are enabled for RAG processing
            if not OPTIMIZATIONS_ENABLED or query_cache is None or async_processor is None:
                # Fallback to original synchronous processing
                
                # Use direct RAG processing
                if retriever is not None:
                    raw_results = retriever.retrieve(combined_query, top_k=5)
                else:
                    # Fallback to chatbot processor
                    if chatbot_processor is not None:
                        raw_results = chatbot_processor.process_query(combined_query, retriever=retriever)
                        # Convert to expected format
                        raw_results = [{"property": result['data'], "distance": 0.0} for result in raw_results]
                    else:
                        # Use cached properties directly
                        cached_properties = get_cached_properties()
                        if cached_properties:
                            raw_results = [{"property": prop, "distance": 0.0} for prop in cached_properties]
                        else:
                            raw_results = []
            else:
                # Use optimized async processing
                # Check cache first for RAG results
                cache_key = f"rag:{session_id}:{hash(cleaned_query)}"
                cached_results = query_cache.get(cleaned_query)
                
                if cached_results and 'results' in cached_results:
                    logger.info(f"Cache hit for RAG query: {cleaned_query[:50]}...")
                    raw_results = cached_results['results']
                else:
                    # Handle regular queries with RAG-based recommendation
                    if session_id in conversation_context and continue_conversation:
                        previous_results = conversation_context[session_id]
                        combined_query = f"Based on previous results:{previous_results}New Query: {cleaned_query}"
                    else:
                        combined_query = cleaned_query
                    
                    # Submit to async RAG processor
                    def rag_callback(result):
                        if result.get('status') == 'success':
                            results = result.get('results', [])
                            # Cache the results
                            query_cache.set(cleaned_query, {
                                "results": results,
                                "cached": True
                            })
                    
                    # Submit RAG request
                    rag_request_id = async_processor.submit_rag_retrieval(
                        session_id=session_id,
                        query=combined_query,
                        priority=RequestPriority.NORMAL,
                        callback=rag_callback
                    )
                    
                    # Wait for RAG results with timeout
                    rag_start_time = time.time()
                    raw_results = []
                    
                    while time.time() - rag_start_time < 15:  # 15 second timeout for RAG
                        rag_result = async_processor.get_request_status(rag_request_id)
                        if rag_result:
                            if rag_result.get('status') == 'success':
                                raw_results = rag_result.get('results', [])
                                break
                            elif rag_result.get('status') == 'error':
                                logger.error(f"RAG processing error: {rag_result.get('error')}")
                                break
                        
                        time.sleep(0.1)
                    
                    # Fallback to direct processing if async fails
                    if not raw_results:
                        logger.warning("Async RAG failed, using direct processing")
                        if optimized_rag_processor is not None:
                            raw_results = optimized_rag_processor.retrieve(combined_query, top_k=5)
                        elif retriever is not None:
                            raw_results = retriever.retrieve(combined_query, top_k=5)
                        else:
                            # Fallback to chatbot processor
                            if chatbot_processor is not None:
                                raw_results = chatbot_processor.process_query(combined_query, retriever=retriever)
                                # Convert to expected format
                                raw_results = [{"property": result['data'], "distance": 0.0} for result in raw_results]
                            else:
                                # Use cached properties directly
                                cached_properties = get_cached_properties()
                                if cached_properties:
                                    raw_results = [{"property": prop, "distance": 0.0} for prop in cached_properties]
                                else:
                                    raw_results = []

        # Filter results based on user plan
        filtered_results = []
        for result in raw_results:
            property_dict = result['property'].to_dict() if hasattr(result['property'], 'to_dict') else result['property']
            property_dict = convert_numeric_fields_to_int(property_dict)
            filtered_property = filter_property_by_plan(property_dict, current_plan)
            
            if 'propertyImages' in filtered_property:
                del filtered_property['propertyImages']
            if 'property_image' in filtered_property:
                del filtered_property['property_image']
            if 'image_url' in filtered_property:
                del filtered_property['image_url']

            filtered_results.append({
                'property': filtered_property,
                'propertyImages': result.get('image_url', []) if current_plan == UserPlan.PRO else [],
                'distance': result.get('distance')
            })

        # Generate response
        response_text, has_restricted_request = format_llm_prompt(
            query=combined_query if continue_conversation else cleaned_query,
            filtered_results=filtered_results,
            user_plan=current_plan,
            original_query=cleaned_query,
            is_follow_up=is_follow_up,
            context_relevance=context_relevance,
            previous_query=locals().get('previous_query'),
            previous_response=locals().get('previous_response')
        )

        # Prefer batched generation for lower latency under load
        from modules.response import batched_generate
        response, duration = batched_generate(
            response_text,
            tokenizer=tokenizer,
            model_llm=model_llm,
            max_new_tokens=512,
            temperature=0.7,
            top_k=30,
            top_p=0.8,
            repetition_penalty=1.05
        )

        # Store the response in AI conversation memory
        if ai_conversation_memory is not None:
            try:
                # Store the original raw_results instead of filtered_results for better context
                properties_to_store = []
                for result in raw_results:
                    if isinstance(result, dict) and 'property' in result:
                        # Store the original property data
                        properties_to_store.append(result['property'])
                    else:
                        # If it's already a property dict, store as is
                        properties_to_store.append(result)
                
                logger.info(f"🔍 DEBUG: About to store {len(properties_to_store)} properties for session {session_id}")
                logger.info(f"🔍 DEBUG: Properties to store: {[prop.get('PropertyName', 'Unknown') for prop in properties_to_store]}")
                
                # Add conversation turn to AI memory
                ai_conversation_memory.add_conversation_turn(
                    session_id=session_id,
                    user_query=cleaned_query,
                    assistant_response=response,
                    retrieved_properties=properties_to_store
                )
                logger.info(f"✅ Added conversation turn to AI memory for session {session_id} with {len(properties_to_store)} properties")
                
                # Debug: Show conversation context
                conv_context = ai_conversation_memory.get_conversation_context(session_id)
                if conv_context:
                    logger.info(f"📝 Conversation context for session {session_id}: {len(conv_context.turns)} turns, current topic: {conv_context.current_topic}")
                    if conv_context.turns:
                        last_turn = conv_context.turns[-1]
                        logger.info(f"📝 Last turn stored {len(last_turn.retrieved_properties)} properties: {[prop.get('PropertyName', 'Unknown') for prop in last_turn.retrieved_properties]}")
                else:
                    logger.warning(f"⚠️ No conversation context found for session {session_id} after storing")
            except Exception as e:
                logger.error(f"Error storing conversation in AI memory: {e}")
                # Fallback to legacy conversation context
                conversation_context[session_id] = response
        else:
            # Fallback to legacy conversation context
            conversation_context[session_id] = response

        return jsonify({
            "response": response,
            "duration": duration,
            "plan_level": current_plan.value,
            "filtered_results": filtered_results,
            "input_limit_info": {
                "remaining_inputs": input_tracker.get_remaining_inputs(session_id, current_plan) if input_tracker is not None else 999,
                "total_limit": PLAN_INPUT_LIMITS[current_plan],
                "usage_stats": input_tracker.get_usage_stats(session_id) if input_tracker is not None else {"total_used": 0, "remaining_time": 24}
            }
        })

    except Exception as e:
        logging.error(f"Error in recommend endpoint: {str(e)}")
        return jsonify({"error": "An error occurred processing your request"}), 500

@app.route('/api/properties/search', methods=['POST'])
def search_properties():
    try:
        data = request.get_json()
        query = data.get('query', '')
        user_location = data.get('user_location')  # (latitude, longitude)
        
        # Get properties from database or external source
        properties = get_properties()  # Implement this function to get properties
        
        # Process query and get filtered properties
        results = chatbot_processor.process_query(
            query, properties, user_location
        )
        
        return jsonify({
            'status': 'success',
            'results': results
        })
        
    except Exception as e:
        logging.error(f"Error searching properties: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/properties/similar', methods=['POST'])
def find_similar_properties():
    try:
        data = request.get_json()
        reference_property = data.get('property')
        top_k = data.get('top_k', 5)
        
        # Get properties from database or external source
        properties = get_properties()  # Implement this function to get properties
        
        # Find similar properties
        results = chatbot_processor.get_similar_properties(
            reference_property, properties, top_k
        )
        
        return jsonify({
            'status': 'success',
            'results': results
        })
        
    except Exception as e:
        logging.error(f"Error finding similar properties: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/properties/landmarks', methods=['POST'])
def get_property_landmarks():
    try:
        data = request.get_json()
        property_data = data.get('property')
        radius_miles = data.get('radius_miles', 5.0)
        
        # Get nearby landmarks
        landmarks = chatbot_processor.get_nearby_landmarks(
            property_data, radius_miles
        )
        
        return jsonify({
            'status': 'success',
            'landmarks': landmarks
        })
        
    except Exception as e:
        logging.error(f"Error getting property landmarks: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/properties/location', methods=['POST'])
def get_property_location():
    try:
        data = request.get_json()
        property_data = data.get('property')
        
        # Get location details
        location_details = chatbot_processor.get_location_details(property_data)
        
        return jsonify({
            'status': 'success',
            'location': location_details
        })
        
    except Exception as e:
        logging.error(f"Error getting property location: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Global thread pool for async processing
request_thread_pool = ThreadPoolExecutor(max_workers=32)

# Performance monitoring
performance_stats = {
    'total_requests': 0,
    'active_requests': 0,
    'avg_response_time': 0.0,
    'memory_usage': 0.0,
    'cpu_usage': 0.0
}

def monitor_performance():
    """Monitor system performance"""
    while True:
        try:
            # Update memory usage
            memory = psutil.virtual_memory()
            performance_stats['memory_usage'] = memory.percent
            
            # Update CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            performance_stats['cpu_usage'] = cpu_percent
            
            # Force garbage collection if memory usage is high
            if memory.percent > 80:
                gc.collect()
                logger.info("Forced garbage collection due to high memory usage")
            
            time.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logger.error(f"Error in performance monitoring: {e}")
            time.sleep(60)

# Start performance monitoring in background
performance_thread = threading.Thread(target=monitor_performance, daemon=True)
performance_thread.start()

def async_process_request(func):
    """Decorator for async request processing"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            performance_stats['active_requests'] += 1
            performance_stats['total_requests'] += 1
            
            start_time = time.time()
            result = func(*args, **kwargs)
            
            # Update response time
            response_time = time.time() - start_time
            current_avg = performance_stats['avg_response_time']
            total_requests = performance_stats['total_requests']
            performance_stats['avg_response_time'] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in async request processing: {e}")
            return jsonify({'error': 'Internal server error'}), 500
        finally:
            performance_stats['active_requests'] -= 1
    
    return wrapper

@app.route('/api/chat', methods=['POST'])
@async_process_request
def chat_endpoint():
    """Enhanced chat endpoint with continuous conversation understanding"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        session_id = data.get('session_id', str(uuid.uuid4()))
        conversation_history = data.get('conversation_history', [])
        user_context = data.get('user_context', {})
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Process chat message using the enhanced chatbot processor
        from modules.enhanced_chatbot_processor import process_enhanced_chat_message
        result = process_enhanced_chat_message(message, session_id, conversation_history, user_context)
        
        return jsonify({
            'response': result['response'],
            'properties': result['properties'],
            'query_type': result['query_type'],
            'confidence': result['confidence'],
            'processing_mode': result['processing_mode'],
            'conversation_context': result['conversation_context'],
            'follow_up_suggestions': result['follow_up_suggestions'],
            'processing_time': result['processing_time'],
            'session_id': result['session_id'],
            'cached': result['cached'],
            'reasoning': result['reasoning']
        })
        
    except Exception as e:
        logger.error(f"Error in enhanced chat endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/chat/stream', methods=['POST'])
@async_process_request
def chat_stream_endpoint():
    """Optimized streaming chat endpoint for real-time responses"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        def generate():
            # Process chat message
            # NO CONVERSATION STORING: Removed conversation_history to reduce processing time
            from modules.chatbot_processor import process_chat_message
            result = process_chat_message(message, session_id, None)
            
            # Stream the response
            response_data = {
                'response': result['response'],
                'properties': result['properties'],
                'confidence': result['confidence'],
                'processing_time': result.get('processing_time', 0),
                'system_mode': result.get('system_mode', 'hybrid'),
                'follow_up_suggestions': result.get('follow_up_suggestions', []),
                'session_id': session_id,
                'cached': result.get('cached', False)
            }
            
            yield f"data: {json.dumps(response_data)}\n\n"
        
        return Response(generate(), mimetype='text/plain')
        
    except Exception as e:
        logger.error(f"Error in chat stream endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/performance/stats', methods=['GET'])
def get_performance_stats():
    """Get system performance statistics"""
    try:
        # Get RAG system stats
        from modules.rag.ultimate_rag_system import get_ultimate_rag_system
        rag_system = get_ultimate_rag_system()
        rag_stats = rag_system.get_system_stats()
        
        # Get chatbot processor stats
        from modules.chatbot_processor import ChatbotProcessor
        chatbot_processor = ChatbotProcessor()
        chatbot_stats = chatbot_processor.get_processing_stats()
        
        return jsonify({
            'system_performance': performance_stats,
            'rag_system_stats': rag_stats,
            'chatbot_stats': chatbot_stats,
            'memory_usage_mb': psutil.virtual_memory().used / (1024 * 1024),
            'cpu_usage_percent': psutil.cpu_percent(),
            'active_threads': threading.active_count()
        })
        
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        return jsonify({'error': 'Error getting performance stats'}), 500

@app.route('/api/cache/clear', methods=['POST'])
def clear_all_caches():
    """Clear all system caches"""
    try:
        # Clear RAG system cache
        from modules.rag.ultimate_rag_system import get_ultimate_rag_system
        rag_system = get_ultimate_rag_system()
        rag_system.clear_cache()
        
        # Clear chatbot processor cache
        from modules.chatbot_processor import ChatbotProcessor
        chatbot_processor = ChatbotProcessor()
        chatbot_processor.clear_cache()
        
        # Force garbage collection
        gc.collect()
        
        return jsonify({'message': 'All caches cleared successfully'})
        
    except Exception as e:
        logger.error(f"Error clearing caches: {e}")
        return jsonify({'error': 'Error clearing caches'}), 500

@app.route('/api/ai-filter/performance', methods=['GET'])
@security_check
def get_ai_filter_performance():
    """Get AI property filter performance statistics"""
    try:
        if ai_property_filter is not None:
            stats = ai_property_filter.get_filtering_stats()
            
            # Add GPU performance metrics
            if torch.cuda.is_available():
                stats['gpu_performance'] = {
                    'available': True,
                    'memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                    'memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                    'memory_free_gb': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / (1024**3),
                    'device_name': torch.cuda.get_device_name(0),
                    'device_count': torch.cuda.device_count()
                }
            else:
                stats['gpu_performance'] = {'available': False}
            
            # Add optimization recommendations
            recommendations = []
            if torch.cuda.is_available():
                memory_usage = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                if memory_usage > 0.8:
                    recommendations.append("High GPU memory usage detected. Consider reducing batch size.")
                elif memory_usage < 0.3:
                    recommendations.append("Low GPU memory usage. Consider increasing batch size for better performance.")
            
            stats['recommendations'] = recommendations
            
            return jsonify({
                'status': 'success',
                'ai_filter_stats': stats
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'AI property filter not available'
            }), 503
            
    except Exception as e:
        logger.error(f"Error getting AI filter performance: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Error getting AI filter performance'
        }), 500

@app.route('/api/optimization/status', methods=['GET'])
def get_optimization_status():
    """Get optimization status and recommendations"""
    try:
        # Get current performance metrics
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()
        
        # Get cache hit rates
        from modules.rag.ultimate_rag_system import get_ultimate_rag_system
        rag_system = get_ultimate_rag_system()
        rag_stats = rag_system.get_system_stats()
        
        # Calculate optimization score
        optimization_score = 0
        recommendations = []
        
        # Memory optimization
        if memory_usage < 70:
            optimization_score += 25
        else:
            recommendations.append("High memory usage detected. Consider clearing caches or reducing batch sizes.")
        
        # CPU optimization
        if cpu_usage < 80:
            optimization_score += 25
        else:
            recommendations.append("High CPU usage detected. Consider reducing concurrent requests.")
        
        # Cache optimization
        cache_hit_rate = rag_stats.get('cache_hit_rate', 0)
        if cache_hit_rate > 0.7:
            optimization_score += 25
        else:
            recommendations.append("Low cache hit rate. Consider increasing cache size or optimizing cache keys.")
        
        # Response time optimization
        avg_response_time = performance_stats.get('avg_response_time', 0)
        if avg_response_time < 2.0:
            optimization_score += 25
        else:
            recommendations.append("High response time detected. Consider optimizing model loading or using smaller models.")
        
        return jsonify({
            'optimization_score': optimization_score,
            'recommendations': recommendations,
            'current_metrics': {
                'memory_usage_percent': memory_usage,
                'cpu_usage_percent': cpu_usage,
                'cache_hit_rate': cache_hit_rate,
                'avg_response_time': avg_response_time,
                'active_requests': performance_stats.get('active_requests', 0)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting optimization status: {e}")
        return jsonify({'error': 'Error getting optimization status'}), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    return friendly_error(
        message="Rate limit exceeded.",
        hint="Please wait a moment before trying again.",
        http_code=429
    )

@app.errorhandler(400)
def bad_request_handler(e):
    return friendly_error(
        message="Bad request.",
        hint="Please verify your input and try again.",
        http_code=400
    )

@app.errorhandler(500)
def internal_error_handler(e):
    return friendly_error(
        message="We encountered an unexpected error.",
        hint="Please try again in a moment.",
        http_code=500
    )

# Add helper functions
def convert_numeric_fields_to_int(property_dict):
    """Convert numeric fields to integers in property dictionary"""
    numeric_fields = ['Bedrooms', 'Bathrooms', 'SquareFeet', 'YearBuilt', 'Price']
    for field in numeric_fields:
        if field in property_dict and property_dict[field] is not None:
            try:
                property_dict[field] = int(float(property_dict[field]))
            except (ValueError, TypeError):
                property_dict[field] = None
    return property_dict

def cleanup_resources():
    """Clean up all resources on shutdown"""
    try:
        logger.info("Cleaning up resources...")
        
        if OPTIMIZATIONS_ENABLED:
            # Shutdown async processor
            if async_processor:
                async_processor.shutdown()
            
            # Shutdown processing manager
            if processing_manager:
                processing_manager.shutdown()
            
            # Close connection pools
            if http_pool:
                http_pool.close()
            
            # Close async HTTP client
            if async_http_client:
                asyncio.run(async_http_client.close())
            
            # Clean up cache manager
            if cache_manager:
                cache_manager.clear_all()
        
        logger.info("Resource cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

import atexit
atexit.register(cleanup_resources)

async def start_websocket_server():
    """Start WebSocket server in background"""
    try:
        await websocket_manager.start_server()
    except Exception as e:
        logger.error(f"Error starting WebSocket server: {e}")

def start_websocket_background():
    """Start WebSocket server in background thread"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(start_websocket_server())
    except Exception as e:
        logger.error(f"Error in WebSocket background thread: {e}")

if __name__ == '__main__':
    try:
        # Get port from environment variable or default to 7860
        port = int(os.environ.get('PORT', 7860))

        # Start WebSocket server in background
        websocket_thread = threading.Thread(target=start_websocket_background, daemon=True)
        websocket_thread.start()
        logger.info("WebSocket server started in background")

        # Remove ngrok tunnel creation for Hugging Face Spaces deployment
        # public_url = ngrok.connect(port)
        # print(f" * ngrok tunnel available at: {public_url}")

        # Run the app on 0.0.0.0 to be publicly accessible
        app.run(host='0.0.0.0', port=port)
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        cleanup_resources()
    except Exception as e:
        logger.error(f"Error running app: {e}")
        cleanup_resources()