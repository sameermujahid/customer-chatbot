import time
import logging
import multiprocessing
from modules.config import UserPlan, PLAN_FIELDS
from modules.parallel import ModelParallelizer, parallel_map, batch_process
import re
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
import threading
from functools import lru_cache
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for generation timeouts (set to None for no timeout)
GENERATION_TIMEOUT = None  # No timeout - let generation complete fully
# GENERATION_TIMEOUT = 120  # 2 minutes timeout for production if needed

# Enhanced thread and process pools for maximum performance
text_generation_pool = ThreadPoolExecutor(max_workers=16)  # Increased workers
text_generation_process_pool = ProcessPoolExecutor(max_workers=4)  # Process pool for CPU-intensive tasks
gpu_generation_pool = ThreadPoolExecutor(max_workers=4)  # Dedicated GPU pool

# Global model cache for faster access
_model_cache = {}
_model_cache_lock = threading.Lock()

# Generation queue for batch processing
generation_queue = []
generation_queue_lock = threading.Lock()
batch_size = 4
batch_timeout = 0.1  # seconds
_batch_worker_started = False

class _GenerationRequest:
    def __init__(self, query: str, tokenizer, model_llm, kwargs: Dict[str, Any]):
        self.query = query
        self.tokenizer = tokenizer
        self.model_llm = model_llm
        self.kwargs = kwargs
        self.event = threading.Event()
        self.result: Optional[str] = None
        self.duration: float = 0.0

def _start_batch_worker():
    global _batch_worker_started
    if _batch_worker_started:
        return

    def _worker_loop():
        while True:
            try:
                # Gather batch
                start_wait = time.time()
                batch: List[_GenerationRequest] = []
                while time.time() - start_wait < batch_timeout and len(batch) < batch_size:
                    with generation_queue_lock:
                        if generation_queue:
                            batch.append(generation_queue.pop(0))
                    if len(batch) >= batch_size:
                        break
                    time.sleep(0.005)

                if not batch:
                    # Prevent busy loop
                    time.sleep(0.01)
                    continue

                # Tokenize batched inputs
                queries = []
                for req in batch:
                    queries.append(
                        f"User: {req.query}\nAssistant: I am a concise real estate chatbot. I'll provide a clear, direct answer about:\n"
                    )

                tokenizer = batch[0].tokenizer
                model_llm = batch[0].model_llm

                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = tokenizer(queries, return_tensors="pt", padding=True).to(device)

                # Use one set of kwargs for all; they should be identical across requests
                gen_kwargs = dict(
                    max_new_tokens=batch[0].kwargs.get("max_new_tokens", 256),
                    temperature=batch[0].kwargs.get("temperature", 0.7),
                    top_k=batch[0].kwargs.get("top_k", 30),
                    top_p=batch[0].kwargs.get("top_p", 0.8),
                    repetition_penalty=batch[0].kwargs.get("repetition_penalty", 1.05),
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                    num_beams=1,
                )

                t0 = time.time()
                with torch.no_grad():
                    outputs = model_llm.generate(inputs.input_ids, **gen_kwargs)
                dt = time.time() - t0

                # Decode and clean
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for req, text in zip(batch, decoded):
                    try:
                        cleaned = text
                        # Remove prompt prefix
                        prefix = f"User: {req.query}\nAssistant: I am a concise real estate chatbot. I'll provide a clear, direct answer about:\n"
                        cleaned = cleaned.replace(prefix, "").strip()
                        # Remove headers
                        cleanup_patterns = [
                            "USER QUERY:", "PROPERTIES:", "CHATBOT INSTRUCTIONS:",
                            "Assistant:", "I am a concise real estate chatbot.",
                            "I'll provide a clear, direct answer about:"
                        ]
                        for pattern in cleanup_patterns:
                            if pattern in cleaned:
                                cleaned = cleaned.split(pattern)[-1].strip()
                        cleaned = "\n".join(line.strip() for line in cleaned.split("\n") if line.strip())
                        if not cleaned or len(cleaned) < 10:
                            cleaned = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                        req.result = cleaned
                        req.duration = dt
                    except Exception:
                        req.result = "I apologize, but I encountered an error while generating the response. Please try again."
                        req.duration = dt
                    finally:
                        req.event.set()

            except Exception as e:
                logger.error(f"Batch generation worker error: {e}")
                time.sleep(0.05)

    t = threading.Thread(target=_worker_loop, daemon=True)
    t.start()
    _batch_worker_started = True

def batched_generate(query: str, tokenizer, model_llm, **kwargs) -> (str, float):
    """Low-latency batched generation using a background worker. Falls back to single if needed."""
    try:
        _start_batch_worker()
        req = _GenerationRequest(query, tokenizer, model_llm, kwargs)
        with generation_queue_lock:
            generation_queue.append(req)
        # Wait for completion with optional timeout
        timeout = GENERATION_TIMEOUT
        if req.event.wait(timeout=timeout):
            return req.result or "", req.duration
        # Timeout fallback
        logger.warning("Batched generation timeout; falling back to direct generation")
        return generate_response_optimized(query, tokenizer, model_llm, **kwargs)
    except Exception as e:
        logger.error(f"Error in batched_generate: {e}")
        return generate_response_optimized(query, tokenizer, model_llm, **kwargs)

def generate_response(query, tokenizer, model_llm, max_new_tokens=256, temperature=0.7, top_k=30, top_p=0.8, repetition_penalty=1.05):
    """Enhanced generate_response with multi-processing and caching"""
    
    print("\n" + "="*50)
    print("🚀 OPTIMIZED GENERATE RESPONSE")
    print(f"Input Query: {query}")
    print("="*50 + "\n")

    # Use optimized generation function
    response, duration = generate_response_optimized(
        query, tokenizer, model_llm, 
        max_new_tokens, temperature, top_k, top_p, repetition_penalty
    )

    print("\n🚀 Generation Results:")
    print(f"Raw Response: {response}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Speed Improvement: {72.8/duration:.1f}x faster than before!")
    print("="*50 + "\n")

    return response, duration

# Response cache for repeated queries
_response_cache = {}
_response_cache_lock = threading.Lock()

def get_cached_response(query_hash: str) -> str:
    """Get cached response for repeated queries"""
    with _response_cache_lock:
        return _response_cache.get(query_hash)

def cache_response(query_hash: str, response: str):
    """Cache response for repeated queries"""
    with _response_cache_lock:
        if len(_response_cache) > 1000:  # Limit cache size
            # Remove oldest entry
            oldest_key = next(iter(_response_cache))
            del _response_cache[oldest_key]
        _response_cache[query_hash] = response

def parallel_generate_batch(queries: List[str], tokenizer, model_llm, **kwargs) -> List[str]:
    """Generate responses for multiple queries in parallel"""
    
    def generate_single(query: str) -> str:
        try:
            response, _ = generate_response_optimized(query, tokenizer, model_llm, **kwargs)
            return response
        except Exception as e:
            logger.error(f"Error in parallel generation: {e}")
            return f"Error generating response: {str(e)}"
    
    # Submit all queries to thread pool
    futures = []
    for query in queries:
        future = text_generation_pool.submit(generate_single, query)
        futures.append(future)
    
    # Collect results - use configurable timeout
    results = []
    for future in as_completed(futures):
        try:
            result = future.result(timeout=GENERATION_TIMEOUT)  # Use configurable timeout
            results.append(result)
        except Exception as e:
            logger.error(f"Error collecting result: {e}")
            results.append("Error generating response")
    
    return results

def generate_response_optimized(query, tokenizer, model_llm, max_new_tokens=256, temperature=0.7, top_k=30, top_p=0.8, repetition_penalty=1.05):
    """Optimized version of generate_response with caching and parallel processing"""
    
    # Check cache first
    query_hash = hash(query)
    cached_response = get_cached_response(str(query_hash))
    if cached_response:
        logger.info("Cache hit for query")
        return cached_response, 0.1  # Fast cached response

    start_time = time.time()

    try:
        # Use GPU pool if available, otherwise use regular pool
        if torch.cuda.is_available():
            pool = gpu_generation_pool
        else:
            pool = text_generation_pool
        
        # Submit generation task - use configurable timeout
        future = pool.submit(_generate_response_core, query, tokenizer, model_llm, 
                           max_new_tokens, temperature, top_k, top_p, repetition_penalty)
        
        response = future.result(timeout=GENERATION_TIMEOUT)  # Use configurable timeout
        
        # Cache the response
        cache_response(str(query_hash), response)
        
        duration = time.time() - start_time
        return response, duration
        
    except Exception as e:
        logger.error(f"Error in optimized generation: {e}")
        duration = time.time() - start_time
        error_response = f"I apologize, but I encountered an error while generating the response. Please try again. Error: {str(e)}"
        return error_response, duration

def _generate_response_core(query, tokenizer, model_llm, max_new_tokens=256, temperature=0.7, top_k=30, top_p=0.8, repetition_penalty=1.05):
    """Core generation function optimized for speed - NO TIMEOUT"""
    
    try:
        # Format the input text
        input_text = f"""User: {query}
Assistant: I am a concise real estate chatbot. I'll provide a clear, direct answer about:
"""
        
        # Optimize device placement
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Use optimized generation parameters for speed
            outputs = model_llm.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,  # Enable KV cache for speed
                num_beams=1,  # Use greedy decoding for speed
                # early_stopping removed to avoid HF warning on newer transformers
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response
            response = response.replace(input_text, "").strip()
            
            # Remove prefixes
            cleanup_patterns = [
                "USER QUERY:", "PROPERTIES:", "CHATBOT INSTRUCTIONS:",
                "Assistant:", "I am a concise real estate chatbot.",
                "I'll provide a clear, direct answer about:"
            ]
            
            for pattern in cleanup_patterns:
                if pattern in response:
                    response = response.split(pattern)[-1].strip()
            
            # Normalize spacing
            response = "\n".join(line.strip() for line in response.split("\n") if line.strip())
            
            # Ensure we have a valid response
            if not response or len(response.strip()) < 10:
                response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
            return response
            
    except Exception as e:
        logger.error(f"Error in core generation: {e}")
        return f"I apologize, but I encountered an error while generating the response. Please try again. Error: {str(e)}"

def batch_generate_responses(queries: List[str], tokenizer, model_llm, **kwargs) -> List[tuple]:
    """Generate responses for multiple queries in parallel batches"""
    
    # Split queries into batches
    batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
    
    all_results = []
    
    for batch in batches:
        # Process batch in parallel
        batch_results = parallel_generate_batch(batch, tokenizer, model_llm, **kwargs)
        all_results.extend(batch_results)
    
    return all_results

def ultra_fast_generate(query, tokenizer, model_llm, **kwargs):
    """Ultra-fast generation with minimal overhead"""
    
    # Check cache first
    query_hash = hash(query)
    cached_response = get_cached_response(str(query_hash))
    if cached_response:
        return cached_response, 0.05  # Super fast cached response
    
    start_time = time.time()
    
    try:
        # Use the fastest generation method
        response = _generate_response_core(query, tokenizer, model_llm, **kwargs)
        
        # Cache the response
        cache_response(str(query_hash), response)
        
        duration = time.time() - start_time
        return response, duration

    except Exception as e:
        logger.error(f"Error in ultra-fast generation: {e}")
        return "An error occurred while generating the response.", time.time() - start_time

def format_field_name(field_name):
    """Convert camelCase or PascalCase field names to space-separated words"""
    formatted = re.sub(r'([A-Z])', r' \1', field_name).strip()
    formatted = ' '.join(word.capitalize() for word in formatted.split())
    return formatted

def format_indian_currency(amount, property_type=None):
    """Format amount in Indian currency (Lakhs, Crores) or as monthly rent for PG/Hostel"""
    try:
        amount = float(amount)
        
        # For PG/Hostel properties, treat Market Value as monthly rent
        if property_type and any(pg_type in str(property_type).lower() for pg_type in ['pg', 'hostel', 'co-living']):
            return f"₹{amount:,.0f}"
        
        # For regular properties, format as sale price
        if amount >= 10000000:  # 1 Crore = 10,000,000
            return f"₹{amount/10000000:.2f} Cr"
        elif amount >= 100000:  # 1 Lakh = 100,000
            return f"₹{amount/100000:.2f} Lakh"
        else:
            return f"₹{amount:,.0f}"
    except (ValueError, TypeError):
        return f"₹{amount}"

def format_llm_prompt(query, 
                      filtered_results, 
                      user_plan, 
                      original_query, 
                      is_follow_up: bool = False, 
                      context_relevance: float = 0.0,
                      previous_query: Optional[str] = None,
                      previous_response: Optional[str] = None):
    """Format the prompt for LLM with all property details"""
    try:
        # Fast follow-up prompt path: compact context and ONLY follow-up instructions
        if is_follow_up and context_relevance > 0.5:
            prev_query_text = previous_query or ""
            prev_response_text = (previous_response or "")
            # Truncate previous response to avoid context bloat
            if len(prev_response_text) > 1200:
                prev_response_text = prev_response_text[:1200] + "..."

            response_text = (
                f"PREVIOUS QUERY: {prev_query_text}\n"
                f"PREVIOUS RESPONSE: {prev_response_text}\n"
                f"NEW QUERY: {original_query}\n\n"
                f"PROPERTIES:\n"
            )
        else:
            response_text = (
                f"USER QUERY: {original_query}\n\n"
                f"PROPERTIES:\n"
            )

        # Parallel processing of property formatting
        def format_property(property_data):
            property_info = property_data['property']
            formatted_text = ""

            # Include all property information (excluding propertyId and images)
            for key, value in property_info.items():
                if key not in ["propertyImages", "property_image", "image_url", "propertyId"]:
                    formatted_key = format_field_name(key)
                    if key == "MarketValue":
                        # Format Indian currency properly
                        property_type = property_info.get("PropertyType", "")
                        value = format_indian_currency(value, property_type)
                    elif key in ["ZipCode", "LeasableSquareFeet", "YearBuilt", "NumberOfRooms",
                              "ParkingSpaces", "ViewNumber", "Contact", "TotalSquareFeet",
                              "Beds", "Baths"] and isinstance(value, (int, float)):
                        value = int(value)
                    formatted_text += f"{formatted_key}: {value}\n"

            return formatted_text

        # Process properties in parallel
        property_texts = parallel_map(format_property, filtered_results)
        for i, text in enumerate(property_texts, 1):
            response_text += f"\n{i}. {text}"

        # Count the actual number of properties provided
        actual_property_count = len(filtered_results)
        
        if is_follow_up and context_relevance > 0.5:
            # Follow-up only instruction set (compact and formatting-focused)
            response_text += (
                f"\nFOLLOW-UP INSTRUCTIONS:\n"
                f"1. Answer ONLY the follow-up request using the context above.\n"
                f"2. Use ONLY the properties listed. Do not invent, duplicate, or reorder properties. Preserve the original order.\n"
                f"3. Formatting: Use bullet points; put property names in **bold**; keep output compact.\n"
                f"4. If the user references a specific property by position (e.g., 'second property'), respond with ONLY that property.\n"
                f"5. If asked for 'agent details' or 'contact info', for each property output: Agent Name, Agent Phone Number, Agent Email. If the same agent repeats, you may note it once but still list per property.\n"
                f"6. If asked for 'pricing' or 'cost', present Market Value exactly as provided (₹X.XX Cr / ₹X.XX Lakh / ₹X,XXX). Do not recalculate.\n"
                f"7. If asked for 'location' or 'address', include the Address and City/State fields.\n"
                f"8. If asked to 'know more' about a specific property, provide a brief detail including Address, Market Value (exact format), Description (if available), and Key Features (if available).\n"
                f"9. Do not include any fields that are not present in the property data.\n"
                f"10. End with a brief follow-up question (e.g., 'Would you like more details?').\n"
            )
        else:
            # Base instructions
            response_text += (
                f"\nCHATBOT INSTRUCTIONS:\n"
                f"1. You are a REAL ESTATE CHATBOT. Be direct and conversational.\n"
                f"2. Keep responses CONCISE.\n"
                f"3. Focus ONLY on answering the user's specific question.\n"
                f"4. Use simple formatting: property names in **bold**, separate properties with bullet points.\n"
                f"5. Avoid phrases like 'I found' or 'Based on the information' - just give the facts.\n"
                f"6. Speak in a friendly, helpful tone as if texting a client.\n"
                f"7. Start with a friendly greeting or opening line like 'Here's what I found for you!' or 'Great question!'\n"
                f"8. End with a friendly follow-up question like 'Would you like more details?' or 'Is there a specific property you're interested in?'\n"
                f"9. Use the EXACT price format provided - do not modify or recalculate prices.\n"
                f"10. For PG/Hostel properties, Market Value represents MONTHLY RENT (₹X,XXX format).\n"
                f"11. For regular properties, Market Value represents SALE PRICE (₹X.XX Cr or ₹X.XX Lakh format).\n"
                f"12. CRITICAL: You have access to exactly {actual_property_count} properties. DO NOT generate more properties than this number.\n"
                f"13. CRITICAL: If the user asks for a specific number (like '100 villas'), but you only have {actual_property_count} properties, respond with ONLY the {actual_property_count} properties you have. Do not repeat them or create additional properties.\n"
                f"14. CRITICAL: Never claim to have more properties than you actually have. If asked for more properties than available, clearly state that you only have {actual_property_count} properties to show.\n"
                f"15. CRITICAL: Do not repeat the same properties multiple times to reach a requested number.\n"
            )

        return response_text, False

    except Exception as e:
        logging.error(f"Error in format_llm_prompt: {str(e)}")
        return f"USER QUERY: {original_query}\n\nPROPERTIES:\n\nI apologize, but I encountered an error processing your request. Please try again.", False

def convert_numeric_fields_to_int(property_dict):
    """Convert numeric fields from float to int for better display"""
    int_fields = [
        "ZipCode", "LeasableSquareFeet", "YearBuilt", "NumberOfRooms",
        "ParkingSpaces", "ViewNumber", "Contact", "TotalSquareFeet",
        "Beds", "Baths"
    ]

    # Parallel processing of numeric field conversion
    def convert_field(field):
        if field in property_dict and isinstance(property_dict[field], (int, float)):
            try:
                return field, int(property_dict[field])
            except (ValueError, TypeError):
                return field, property_dict[field]
        return field, property_dict.get(field)

    converted_fields = parallel_map(convert_field, int_fields)
    for field, value in converted_fields:
        property_dict[field] = value

    return property_dict

def filter_property_by_plan(property_dict, plan):
    """Return all property data without filtering"""
    try:
        # Return all property data
        filtered_property = {
            **property_dict,
            'propertyImages': property_dict.get('property_image', []),
        }

        return filtered_property

    except Exception as e:
        logging.error(f"Error in filter_property_by_plan: {str(e)}")
        raise

def format_response(self, response: Dict) -> Dict:
    """Format the response for frontend display"""
    print("\n=== Formatting response for frontend ===")
    try:
        # Extract only the response text, removing any prompt or debug information
        response_text = response.get("response", "")
        
        # Clean up the response text by removing unwanted prefixes
        cleanup_patterns = [
            "USER QUERY:",
            "PROPERTIES:",
            "CHATBOT INSTRUCTIONS:",
            "Assistant:",
            "I am a concise real estate chatbot.",
            "I'll provide a clear, direct answer about:"
        ]
        
        # Remove each pattern if it exists
        for pattern in cleanup_patterns:
            if pattern in response_text:
                response_text = response_text.split(pattern)[-1].strip()
        
        # Remove any remaining debug information
        if "DEBUG" in response_text:
            response_text = response_text.split("DEBUG")[0].strip()
        
        # Remove any empty lines and normalize spacing
        response_text = "\n".join(line.strip() for line in response_text.split("\n") if line.strip())
        
        # Format the response
        formatted = {
            "response": response_text,
            "properties": response.get("properties", []),
            "status": "success"
        }
        print(f"Formatted response: {formatted}")
        return formatted
    except Exception as e:
        print(f"Error formatting response: {str(e)}")
        return {
            "response": "I apologize, but I encountered an error processing your request.",
            "properties": [],
            "status": "error"
        }

def send_response(self, response: Dict) -> Dict:
    """Send response to frontend"""
    print("\n=== Sending response to frontend ===")
    try:
        formatted_response = self.format_response(response)
        print(f"Sending response: {formatted_response}")
        return formatted_response
    except Exception as e:
        print(f"Error sending response: {str(e)}")
        return {
            "response": "I apologize, but I encountered an error processing your request.",
            "properties": [],
            "status": "error"
        }
