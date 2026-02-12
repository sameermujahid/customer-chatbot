import asyncio
import json
import logging
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import uuid
from websockets.server import serve, WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    message_id: str
    session_id: str
    user_id: Optional[str]
    message: str
    timestamp: datetime
    message_type: str = "text"
    metadata: Dict[str, Any] = None

@dataclass
class ChatSession:
    session_id: str
    user_id: Optional[str]
    websocket: WebSocketServerProtocol
    created_at: datetime
    last_activity: datetime
    conversation_history: List[ChatMessage]
    user_preferences: Dict[str, Any]

class WebSocketManager:
    """
    Real-time WebSocket manager for chat functionality
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.active_connections: Dict[str, ChatSession] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> set of session_ids
        self.server = None
        
        # Performance tracking
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'total_messages': 0,
            'avg_response_time': 0.0
        }
        
        logger.info(f"WebSocket Manager initialized on {host}:{port}")
    
    async def start_server(self):
        """Start the WebSocket server"""
        try:
            self.server = await serve(
                self.handle_connection,
                self.host,
                self.port
            )
            logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            
            # Keep server running
            await self.server.wait_closed()
            
        except Exception as e:
            logger.error(f"Error starting WebSocket server: {e}")
            raise
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection"""
        session_id = str(uuid.uuid4())
        user_id = None
        
        try:
            # Extract user info from query parameters
            query_params = self._parse_query_params(path)
            user_id = query_params.get('user_id')
            
            # Create chat session
            session = ChatSession(
                session_id=session_id,
                user_id=user_id,
                websocket=websocket,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                conversation_history=[],
                user_preferences={}
            )
            
            # Store connection
            self.active_connections[session_id] = session
            
            # Track user sessions
            if user_id:
                if user_id not in self.user_sessions:
                    self.user_sessions[user_id] = set()
                self.user_sessions[user_id].add(session_id)
            
            # Update stats
            self.stats['total_connections'] += 1
            self.stats['active_connections'] = len(self.active_connections)
            
            # Send welcome message
            await self.send_message(session_id, {
                'type': 'connection_established',
                'session_id': session_id,
                'message': 'Connected to AI Real Estate Assistant',
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"New WebSocket connection: {session_id} (user: {user_id})")
            
            # Handle incoming messages
            async for message in websocket:
                await self.handle_message(session_id, message)
                
        except ConnectionClosed:
            logger.info(f"WebSocket connection closed: {session_id}")
        except Exception as e:
            logger.error(f"Error handling WebSocket connection: {e}")
        finally:
            # Clean up connection
            await self.cleanup_connection(session_id, user_id)
    
    async def handle_message(self, session_id: str, message: str):
        """Handle incoming WebSocket message"""
        try:
            # Parse message
            data = json.loads(message)
            message_type = data.get('type', 'chat')
            
            # Update last activity
            if session_id in self.active_connections:
                self.active_connections[session_id].last_activity = datetime.now()
            
            # Handle different message types
            if message_type == 'chat':
                await self.handle_chat_message(session_id, data)
            elif message_type == 'typing':
                await self.handle_typing_indicator(session_id, data)
            elif message_type == 'user_preferences':
                await self.handle_user_preferences(session_id, data)
            elif message_type == 'ping':
                await self.send_message(session_id, {'type': 'pong'})
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message from {session_id}")
        except Exception as e:
            logger.error(f"Error handling message from {session_id}: {e}")
    
    async def handle_chat_message(self, session_id: str, data: Dict[str, Any]):
        """Handle chat message"""
        try:
            session = self.active_connections.get(session_id)
            if not session:
                return
            
            # Create chat message
            chat_message = ChatMessage(
                message_id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=session.user_id,
                message=data.get('message', ''),
                timestamp=datetime.now(),
                message_type=data.get('message_type', 'text'),
                metadata=data.get('metadata', {})
            )
            
            # Add to conversation history
            session.conversation_history.append(chat_message)
            
            # Send typing indicator
            await self.send_typing_indicator(session_id, True)
            
            # Process with RAG system
            response = await self.process_with_rag(chat_message)
            
            # Stop typing indicator
            await self.send_typing_indicator(session_id, False)
            
            # Send response
            await self.send_message(session_id, {
                'type': 'chat_response',
                'message_id': str(uuid.uuid4()),
                'response': response['response'],
                'properties': response.get('properties', []),
                'confidence': response.get('confidence', 0.0),
                'follow_up_suggestions': response.get('follow_up_suggestions', []),
                'timestamp': datetime.now().isoformat()
            })
            
            # Update stats
            self.stats['total_messages'] += 1
            
        except Exception as e:
            logger.error(f"Error handling chat message: {e}")
            await self.send_error(
                session_id,
                "We encountered an issue processing your message. Please try again in a moment."
            )
    
    async def handle_typing_indicator(self, session_id: str, data: Dict[str, Any]):
        """Handle typing indicator"""
        try:
            is_typing = data.get('is_typing', False)
            user_id = data.get('user_id')
            
            # Broadcast typing indicator to other users in same session
            await self.broadcast_typing_indicator(session_id, user_id, is_typing)
            
        except Exception as e:
            logger.error(f"Error handling typing indicator: {e}")
    
    async def handle_user_preferences(self, session_id: str, data: Dict[str, Any]):
        """Handle user preferences update"""
        try:
            session = self.active_connections.get(session_id)
            if session:
                session.user_preferences.update(data.get('preferences', {}))
                logger.info(f"Updated preferences for session {session_id}")
                
        except Exception as e:
            logger.error(f"Error handling user preferences: {e}")
    
    async def process_with_rag(self, chat_message: ChatMessage) -> Dict[str, Any]:
        """Process message with RAG system"""
        try:
            # Import RAG system
            from ..chatbot_processor import process_chat_message
            
            # Get conversation history
            session = self.active_connections.get(chat_message.session_id)
            conversation_history = []
            
            if session:
                # Convert chat messages to conversation history format
                for msg in session.conversation_history[-10:]:  # Last 10 messages
                    conversation_history.append({
                        'role': 'user' if msg.user_id else 'assistant',
                        'content': msg.message
                    })
            
            # Process with RAG system
            result = process_chat_message(
                message=chat_message.message,
                session_id=chat_message.session_id,
                conversation_history=conversation_history
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing with RAG: {e}")
            return {
                'response': 'I apologize, but I encountered an error processing your request.',
                'properties': [],
                'confidence': 0.0,
                'follow_up_suggestions': ['Please try rephrasing your question']
            }
    
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """Send message to specific session"""
        try:
            session = self.active_connections.get(session_id)
            if session and not session.websocket.closed:
                await session.websocket.send(json.dumps(message))
                
        except Exception as e:
            logger.error(f"Error sending message to {session_id}: {e}")
    
    async def send_typing_indicator(self, session_id: str, is_typing: bool):
        """Send typing indicator"""
        await self.send_message(session_id, {
            'type': 'typing_indicator',
            'is_typing': is_typing,
            'timestamp': datetime.now().isoformat()
        })
    
    async def send_error(self, session_id: str, error_message: str):
        """Send error message"""
        await self.send_message(session_id, {
            'type': 'error',
            'message': error_message,
            'hint': 'Please retry shortly. If the issue persists, try simplifying your request.',
            'timestamp': datetime.now().isoformat()
        })
    
    async def broadcast_typing_indicator(self, session_id: str, user_id: str, is_typing: bool):
        """Broadcast typing indicator to other users"""
        try:
            # For now, just send to the same session
            # In a multi-user scenario, you'd broadcast to other users
            await self.send_typing_indicator(session_id, is_typing)
            
        except Exception as e:
            logger.error(f"Error broadcasting typing indicator: {e}")
    
    async def broadcast_message(self, message: Dict[str, Any], exclude_session: str = None):
        """Broadcast message to all connected clients"""
        try:
            for session_id, session in self.active_connections.items():
                if session_id != exclude_session and not session.websocket.closed:
                    await self.send_message(session_id, message)
                    
        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")
    
    async def cleanup_connection(self, session_id: str, user_id: Optional[str]):
        """Clean up connection when client disconnects"""
        try:
            # Remove from active connections
            if session_id in self.active_connections:
                del self.active_connections[session_id]
            
            # Remove from user sessions
            if user_id and user_id in self.user_sessions:
                self.user_sessions[user_id].discard(session_id)
                if not self.user_sessions[user_id]:
                    del self.user_sessions[user_id]
            
            # Update stats
            self.stats['active_connections'] = len(self.active_connections)
            
            logger.info(f"Cleaned up connection: {session_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up connection: {e}")
    
    def _parse_query_params(self, path: str) -> Dict[str, str]:
        """Parse query parameters from WebSocket path"""
        try:
            if '?' in path:
                query_string = path.split('?')[1]
                params = {}
                for param in query_string.split('&'):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        params[key] = value
                return params
            return {}
        except Exception as e:
            logger.error(f"Error parsing query params: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics"""
        return {
            **self.stats,
            'user_sessions_count': len(self.user_sessions),
            'total_users': len(self.user_sessions)
        }
    
    async def shutdown(self):
        """Shutdown WebSocket server"""
        try:
            if self.server:
                self.server.close()
                await self.server.wait_closed()
            
            # Close all connections
            for session_id in list(self.active_connections.keys()):
                session = self.active_connections[session_id]
                if not session.websocket.closed:
                    await session.websocket.close()
            
            logger.info("WebSocket server shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down WebSocket server: {e}")

# Global WebSocket manager instance
_websocket_manager = None

def get_websocket_manager() -> WebSocketManager:
    """Get global WebSocket manager instance"""
    global _websocket_manager
    if _websocket_manager is None:
        _websocket_manager = WebSocketManager()
    return _websocket_manager
