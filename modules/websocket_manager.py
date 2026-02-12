import asyncio
import json
import logging
import threading
import time
from typing import Dict, Set, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import websockets
from websockets.server import WebSocketServerProtocol
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of WebSocket messages"""
    SEARCH_REQUEST = "search_request"
    SEARCH_RESPONSE = "search_response"
    GENERATE_REQUEST = "generate_request"
    GENERATE_RESPONSE = "generate_response"
    LOCATION_UPDATE = "location_update"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"

@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    type: MessageType
    session_id: str
    user_id: Optional[str]
    data: Dict[str, Any]
    timestamp: float
    message_id: str

class WebSocketManager:
    """Manages WebSocket connections for real-time communication"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.connections: Dict[str, WebSocketServerProtocol] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> set of connection_ids
        self.session_connections: Dict[str, str] = {}  # session_id -> connection_id
        self.lock = threading.RLock()
        
        # Message handlers
        self.message_handlers: Dict[MessageType, Callable] = {}
        
        # Server state
        self.server = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "total_messages": 0,
            "start_time": time.time()
        }
    
    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler"""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for {message_type}")
    
    async def start_server(self):
        """Start the WebSocket server"""
        try:
            self.server = await websockets.serve(
                self._handle_connection,
                self.host,
                self.port
            )
            self.is_running = True
            logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            
            # Keep server running
            await self.server.wait_closed()
            
        except Exception as e:
            logger.error(f"Error starting WebSocket server: {e}")
    
    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a new WebSocket connection"""
        connection_id = str(uuid.uuid4())
        
        try:
            # Store connection
            with self.lock:
                self.connections[connection_id] = websocket
                self.stats["total_connections"] += 1
                self.stats["active_connections"] += 1
            
            logger.info(f"New WebSocket connection: {connection_id}")
            
            # Handle messages
            async for message in websocket:
                await self._handle_message(connection_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {connection_id}")
        except Exception as e:
            logger.error(f"Error handling WebSocket connection {connection_id}: {e}")
        finally:
            # Clean up connection
            await self._cleanup_connection(connection_id)
    
    async def _handle_message(self, connection_id: str, message: str):
        """Handle incoming WebSocket message"""
        try:
            # Parse message
            data = json.loads(message)
            message_type = MessageType(data.get("type"))
            session_id = data.get("session_id")
            user_id = data.get("user_id")
            
            # Create message object
            ws_message = WebSocketMessage(
                type=message_type,
                session_id=session_id,
                user_id=user_id,
                data=data.get("data", {}),
                timestamp=time.time(),
                message_id=str(uuid.uuid4())
            )
            
            # Update statistics
            with self.lock:
                self.stats["total_messages"] += 1
            
            # Route to appropriate handler
            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                response = await handler(ws_message)
                
                if response:
                    await self._send_message(connection_id, response)
            else:
                logger.warning(f"No handler registered for message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message from {connection_id}")
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")
    
    async def _send_message(self, connection_id: str, message: Dict[str, Any]):
        """Send message to specific connection"""
        try:
            if connection_id in self.connections:
                websocket = self.connections[connection_id]
                await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message to {connection_id}: {e}")
    
    async def broadcast_message(self, message: Dict[str, Any], user_id: Optional[str] = None):
        """Broadcast message to all connections or specific user"""
        try:
            with self.lock:
                if user_id:
                    # Send to specific user's connections
                    user_connections = self.user_connections.get(user_id, set())
                    for connection_id in user_connections:
                        if connection_id in self.connections:
                            await self._send_message(connection_id, message)
                else:
                    # Send to all connections
                    for connection_id in self.connections:
                        await self._send_message(connection_id, message)
                        
        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")
    
    async def send_to_session(self, session_id: str, message: Dict[str, Any]):
        """Send message to specific session"""
        try:
            with self.lock:
                connection_id = self.session_connections.get(session_id)
                if connection_id and connection_id in self.connections:
                    await self._send_message(connection_id, message)
                    
        except Exception as e:
            logger.error(f"Error sending message to session {session_id}: {e}")
    
    async def _cleanup_connection(self, connection_id: str):
        """Clean up a closed connection"""
        try:
            with self.lock:
                if connection_id in self.connections:
                    del self.connections[connection_id]
                    self.stats["active_connections"] -= 1
                
                # Remove from user connections
                for user_id, connections in self.user_connections.items():
                    if connection_id in connections:
                        connections.remove(connection_id)
                        if not connections:
                            del self.user_connections[user_id]
                
                # Remove from session connections
                for session_id, conn_id in list(self.session_connections.items()):
                    if conn_id == connection_id:
                        del self.session_connections[session_id]
                        
        except Exception as e:
            logger.error(f"Error cleaning up connection {connection_id}: {e}")
    
    def register_user_connection(self, user_id: str, connection_id: str):
        """Register a connection for a user"""
        try:
            with self.lock:
                if user_id not in self.user_connections:
                    self.user_connections[user_id] = set()
                self.user_connections[user_id].add(connection_id)
                
        except Exception as e:
            logger.error(f"Error registering user connection: {e}")
    
    def register_session_connection(self, session_id: str, connection_id: str):
        """Register a connection for a session"""
        try:
            with self.lock:
                self.session_connections[session_id] = connection_id
                
        except Exception as e:
            logger.error(f"Error registering session connection: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        try:
            with self.lock:
                return {
                    **self.stats,
                    "unique_users": len(self.user_connections),
                    "active_sessions": len(self.session_connections),
                    "uptime": time.time() - self.stats["start_time"]
                }
        except Exception as e:
            logger.error(f"Error getting connection stats: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown the WebSocket server"""
        try:
            self.is_running = False
            
            # Close all connections
            with self.lock:
                for connection_id, websocket in self.connections.items():
                    try:
                        await websocket.close()
                    except Exception as e:
                        logger.error(f"Error closing connection {connection_id}: {e}")
                
                self.connections.clear()
                self.user_connections.clear()
                self.session_connections.clear()
            
            # Stop server
            if self.server:
                self.server.close()
                await self.server.wait_closed()
                
            logger.info("WebSocket server shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down WebSocket server: {e}")

# Default message handlers
async def handle_search_request(message: WebSocketMessage) -> Optional[Dict[str, Any]]:
    """Default handler for search requests"""
    try:
        # This would integrate with your existing search functionality
        return {
            "type": MessageType.SEARCH_RESPONSE.value,
            "session_id": message.session_id,
            "user_id": message.user_id,
            "data": {
                "status": "processing",
                "message": "Search request received"
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error handling search request: {e}")
        return None

async def handle_generate_request(message: WebSocketMessage) -> Optional[Dict[str, Any]]:
    """Default handler for generation requests"""
    try:
        # This would integrate with your existing generation functionality
        return {
            "type": MessageType.GENERATE_RESPONSE.value,
            "session_id": message.session_id,
            "user_id": message.user_id,
            "data": {
                "status": "processing",
                "message": "Generation request received"
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error handling generation request: {e}")
        return None

# Global WebSocket manager instance
websocket_manager = WebSocketManager()

# Register default handlers
websocket_manager.register_handler(MessageType.SEARCH_REQUEST, handle_search_request)
websocket_manager.register_handler(MessageType.GENERATE_REQUEST, handle_generate_request) 