import threading
import queue
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import torch
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import aiohttp
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PooledConnection:
    """Represents a pooled connection"""
    id: str
    created_at: float
    last_used: float
    is_active: bool
    connection: Any
    connection_type: str

class ConnectionPool:
    """Generic connection pool for managing various types of connections"""
    
    def __init__(self, max_connections: int = 10, max_idle_time: float = 300):
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time
        
        self.connections = {}
        self.available_connections = queue.Queue()
        self.lock = threading.Lock()
        
        # Cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_worker(self):
        """Background worker to clean up idle connections"""
        while True:
            try:
                current_time = time.time()
                with self.lock:
                    connections_to_remove = []
                    
                    for conn_id, conn in self.connections.items():
                        if (current_time - conn.last_used) > self.max_idle_time:
                            connections_to_remove.append(conn_id)
                    
                    for conn_id in connections_to_remove:
                        self._close_connection(conn_id)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
                time.sleep(60)
    
    def _close_connection(self, conn_id: str):
        """Close a specific connection"""
        if conn_id in self.connections:
            conn = self.connections[conn_id]
            try:
                if hasattr(conn.connection, 'close'):
                    conn.connection.close()
                elif hasattr(conn.connection, 'shutdown'):
                    conn.connection.shutdown()
            except Exception as e:
                logger.error(f"Error closing connection {conn_id}: {e}")
            
            del self.connections[conn_id]
    
    def get_connection(self, connection_type: str, create_func: Callable) -> Optional[Any]:
        """Get a connection from the pool or create a new one"""
        try:
            # Try to get an available connection
            while not self.available_connections.empty():
                conn_id = self.available_connections.get_nowait()
                if conn_id in self.connections:
                    conn = self.connections[conn_id]
                    if conn.connection_type == connection_type and conn.is_active:
                        conn.last_used = time.time()
                        return conn.connection
            
            # Create new connection if under limit
            with self.lock:
                if len(self.connections) < self.max_connections:
                    conn_id = f"{connection_type}_{len(self.connections)}_{int(time.time())}"
                    connection = create_func()
                    
                    pooled_conn = PooledConnection(
                        id=conn_id,
                        created_at=time.time(),
                        last_used=time.time(),
                        is_active=True,
                        connection=connection,
                        connection_type=connection_type
                    )
                    
                    self.connections[conn_id] = pooled_conn
                    return connection
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting connection: {e}")
            return None
    
    def return_connection(self, connection: Any):
        """Return a connection to the pool"""
        try:
            with self.lock:
                for conn_id, pooled_conn in self.connections.items():
                    if pooled_conn.connection == connection:
                        pooled_conn.last_used = time.time()
                        self.available_connections.put(conn_id)
                        break
        except Exception as e:
            logger.error(f"Error returning connection: {e}")
    
    def close_all(self):
        """Close all connections in the pool"""
        with self.lock:
            for conn_id in list(self.connections.keys()):
                self._close_connection(conn_id)

class HTTPConnectionPool:
    """Specialized connection pool for HTTP connections"""
    
    def __init__(self, max_connections: int = 20, max_retries: int = 3):
        self.max_connections = max_connections
        self.max_retries = max_retries
        
        # Create session with connection pooling
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=1
        )
        
        # Configure adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=max_connections,
            pool_maxsize=max_connections,
            max_retries=retry_strategy
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Thread pool for async-like requests
        self.thread_pool = ThreadPoolExecutor(max_workers=max_connections)
    
    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make an HTTP request using the pooled session"""
        return self.session.request(method, url, **kwargs)
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """Make a GET request"""
        return self.session.get(url, **kwargs)
    
    def post(self, url: str, **kwargs) -> requests.Response:
        """Make a POST request"""
        return self.session.post(url, **kwargs)
    
    def request_async(self, method: str, url: str, **kwargs):
        """Make an async HTTP request"""
        return self.thread_pool.submit(self.session.request, method, url, **kwargs)
    
    def close(self):
        """Close the session and thread pool"""
        self.session.close()
        self.thread_pool.shutdown(wait=True)

class ModelConnectionPool:
    """Connection pool for managing model instances"""
    
    def __init__(self, max_models: int = 4):
        self.max_models = max_models
        self.models = {}
        self.model_queue = queue.Queue()
        self.lock = threading.Lock()
        
        # Model loading functions
        self.model_loaders = {}
    
    def register_model_loader(self, model_name: str, loader_func: Callable):
        """Register a model loader function"""
        self.model_loaders[model_name] = loader_func
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a model instance"""
        try:
            # Check if model is already loaded
            if model_name in self.models:
                model_info = self.models[model_name]
                model_info['last_used'] = time.time()
                return model_info['model']
            
            # Load new model if under limit
            with self.lock:
                if len(self.models) < self.max_models:
                    if model_name in self.model_loaders:
                        model = self.model_loaders[model_name]()
                        
                        self.models[model_name] = {
                            'model': model,
                            'created_at': time.time(),
                            'last_used': time.time()
                        }
                        
                        return model
            
            # If at limit, return the least recently used model
            if self.models:
                lru_model = min(self.models.items(), 
                              key=lambda x: x[1]['last_used'])
                model_name_lru, model_info = lru_model
                
                # Unload the LRU model
                del self.models[model_name_lru]
                
                # Load the requested model
                if model_name in self.model_loaders:
                    model = self.model_loaders[model_name]()
                    
                    self.models[model_name] = {
                        'model': model,
                        'created_at': time.time(),
                        'last_used': time.time()
                    }
                    
                    return model
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting model {model_name}: {e}")
            return None
    
    def preload_models(self, model_names: List[str]):
        """Preload models in background"""
        def preload_worker():
            for model_name in model_names:
                try:
                    self.get_model(model_name)
                    time.sleep(1)  # Small delay between loads
                except Exception as e:
                    logger.error(f"Error preloading model {model_name}: {e}")
        
        thread = threading.Thread(target=preload_worker, daemon=True)
        thread.start()
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded models"""
        with self.lock:
            return {
                'loaded_models': list(self.models.keys()),
                'model_count': len(self.models),
                'max_models': self.max_models,
                'model_info': {
                    name: {
                        'created_at': info['created_at'],
                        'last_used': info['last_used'],
                        'age': time.time() - info['created_at']
                    }
                    for name, info in self.models.items()
                }
            }

class AsyncHTTPClient:
    """Async HTTP client using aiohttp with connection pooling"""
    
    def __init__(self, max_connections: int = 20):
        self.max_connections = max_connections
        self.connector = None
        self.session = None
        self._setup_session()
    
    def _setup_session(self):
        """Setup aiohttp session with connection pooling"""
        self.connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout
        )
    
    async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make an async HTTP request"""
        if self.session is None:
            self._setup_session()
        
        return await self.session.request(method, url, **kwargs)
    
    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make an async GET request"""
        return await self.request('GET', url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make an async POST request"""
        return await self.request('POST', url, **kwargs)
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()
            self.session = None

# Global instances
http_pool = HTTPConnectionPool()
model_pool = ModelConnectionPool()
async_http_client = AsyncHTTPClient()

# Register model loaders
def register_default_model_loaders():
    """Register default model loaders"""
    try:
        from modules.models import load_tokenizer_and_model, load_sentence_transformer
        
        def load_llm_model():
            try:
                tokenizer, model = load_tokenizer_and_model()
                return {'tokenizer': tokenizer, 'model': model}
            except Exception as e:
                logger.error(f"Error loading LLM model: {e}")
                return None
        
        def load_embedding_model():
            try:
                return load_sentence_transformer()
            except Exception as e:
                logger.error(f"Error loading embedding model: {e}")
                return None
        
        model_pool.register_model_loader('llm', load_llm_model)
        model_pool.register_model_loader('embedding', load_embedding_model)
        
        logger.info("Default model loaders registered")
        
    except Exception as e:
        logger.error(f"Error registering model loaders: {e}")

# Initialize on import with error handling
try:
    register_default_model_loaders()
except Exception as e:
    logger.error(f"Failed to initialize model loaders: {e}") 