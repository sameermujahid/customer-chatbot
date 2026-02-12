import threading
import time
import uuid
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import redis
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserSession:
    """Represents a user session with all necessary data"""
    session_id: str
    user_id: Optional[str]
    ip_address: str
    created_at: float
    last_activity: float
    is_active: bool
    user_plan: str
    conversation_context: Dict[str, Any]
    search_history: List[Dict[str, Any]]
    location_data: Optional[Dict[str, Any]]
    rate_limit_data: Dict[str, Any]
    preferences: Dict[str, Any]
    metadata: Dict[str, Any]

class SessionManager:
    """Manages user sessions for concurrent access"""
    
    def __init__(self, redis_url: Optional[str] = None, session_timeout: int = 3600):
        self.session_timeout = session_timeout
        self.sessions: Dict[str, UserSession] = {}
        self.user_sessions: Dict[str, List[str]] = defaultdict(list)  # user_id -> session_ids
        self.lock = threading.RLock()
        
        # Redis for distributed sessions (optional)
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                logger.info("Connected to Redis for distributed session management")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
        
        # Cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def create_session(self, user_id: Optional[str] = None, ip_address: str = "", 
                      user_plan: str = "basic") -> str:
        """Create a new user session"""
        try:
            session_id = str(uuid.uuid4())
            
            session = UserSession(
                session_id=session_id,
                user_id=user_id,
                ip_address=ip_address,
                created_at=time.time(),
                last_activity=time.time(),
                is_active=True,
                user_plan=user_plan,
                conversation_context={},
                search_history=[],
                location_data=None,
                rate_limit_data={"requests": 0, "window_start": time.time()},
                preferences={},
                metadata={"created_by": "session_manager"}
            )
            
            with self.lock:
                self.sessions[session_id] = session
                if user_id:
                    self.user_sessions[user_id].append(session_id)
                
                # Store in Redis if available
                if self.redis_client:
                    self._store_session_redis(session)
            
            logger.info(f"Created session {session_id} for user {user_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return None
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get a session by ID"""
        try:
            # Try local cache first
            with self.lock:
                if session_id in self.sessions:
                    session = self.sessions[session_id]
                    session.last_activity = time.time()
                    return session
            
            # Try Redis if available
            if self.redis_client:
                session = self._get_session_redis(session_id)
                if session:
                    with self.lock:
                        self.sessions[session_id] = session
                    return session
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")
            return None
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session data"""
        try:
            session = self.get_session(session_id)
            if not session:
                return False
            
            with self.lock:
                # Update session attributes
                for key, value in updates.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
                
                session.last_activity = time.time()
                
                # Store in Redis if available
                if self.redis_client:
                    self._store_session_redis(session)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating session {session_id}: {e}")
            return False
    
    def add_to_conversation_context(self, session_id: str, key: str, value: Any) -> bool:
        """Add data to conversation context"""
        try:
            session = self.get_session(session_id)
            if not session:
                return False
            
            with self.lock:
                session.conversation_context[key] = value
                session.last_activity = time.time()
                
                if self.redis_client:
                    self._store_session_redis(session)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding to conversation context: {e}")
            return False
    
    def add_search_history(self, session_id: str, search_data: Dict[str, Any]) -> bool:
        """Add search to history"""
        try:
            session = self.get_session(session_id)
            if not session:
                return False
            
            with self.lock:
                search_data["timestamp"] = time.time()
                session.search_history.append(search_data)
                
                # Keep only last 50 searches
                if len(session.search_history) > 50:
                    session.search_history = session.search_history[-50:]
                
                session.last_activity = time.time()
                
                if self.redis_client:
                    self._store_session_redis(session)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding search history: {e}")
            return False
    
    def set_location_data(self, session_id: str, location_data: Dict[str, Any]) -> bool:
        """Set location data for session"""
        try:
            session = self.get_session(session_id)
            if not session:
                return False
            
            with self.lock:
                session.location_data = location_data
                session.last_activity = time.time()
                
                if self.redis_client:
                    self._store_session_redis(session)
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting location data: {e}")
            return False
    
    def update_rate_limit(self, session_id: str, requests: int, window_start: float) -> bool:
        """Update rate limit data"""
        try:
            session = self.get_session(session_id)
            if not session:
                return False
            
            with self.lock:
                session.rate_limit_data = {
                    "requests": requests,
                    "window_start": window_start
                }
                session.last_activity = time.time()
                
                if self.redis_client:
                    self._store_session_redis(session)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating rate limit: {e}")
            return False
    
    def get_user_sessions(self, user_id: str) -> List[UserSession]:
        """Get all sessions for a user"""
        try:
            with self.lock:
                session_ids = self.user_sessions.get(user_id, [])
                sessions = []
                
                for session_id in session_ids:
                    session = self.get_session(session_id)
                    if session and session.is_active:
                        sessions.append(session)
                
                return sessions
                
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return []
    
    def deactivate_session(self, session_id: str) -> bool:
        """Deactivate a session"""
        try:
            session = self.get_session(session_id)
            if not session:
                return False
            
            with self.lock:
                session.is_active = False
                session.last_activity = time.time()
                
                if self.redis_client:
                    self._store_session_redis(session)
            
            logger.info(f"Deactivated session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deactivating session: {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session completely"""
        try:
            with self.lock:
                if session_id in self.sessions:
                    session = self.sessions[session_id]
                    
                    # Remove from user sessions
                    if session.user_id and session.user_id in self.user_sessions:
                        self.user_sessions[session.user_id] = [
                            sid for sid in self.user_sessions[session.user_id] 
                            if sid != session_id
                        ]
                    
                    del self.sessions[session_id]
                
                # Remove from Redis
                if self.redis_client:
                    self.redis_client.delete(f"session:{session_id}")
            
            logger.info(f"Deleted session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False
    
    def _cleanup_worker(self):
        """Background worker to clean up expired sessions"""
        while True:
            try:
                current_time = time.time()
                expired_sessions = []
                
                with self.lock:
                    for session_id, session in self.sessions.items():
                        if (current_time - session.last_activity) > self.session_timeout:
                            expired_sessions.append(session_id)
                    
                    for session_id in expired_sessions:
                        self.delete_session(session_id)
                
                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
                time.sleep(300)
    
    def _store_session_redis(self, session: UserSession):
        """Store session in Redis"""
        try:
            if self.redis_client:
                session_data = asdict(session)
                self.redis_client.setex(
                    f"session:{session.session_id}",
                    self.session_timeout,
                    pickle.dumps(session_data)
                )
        except Exception as e:
            logger.error(f"Error storing session in Redis: {e}")
    
    def _get_session_redis(self, session_id: str) -> Optional[UserSession]:
        """Get session from Redis"""
        try:
            if self.redis_client:
                session_data = self.redis_client.get(f"session:{session_id}")
                if session_data:
                    data = pickle.loads(session_data)
                    return UserSession(**data)
        except Exception as e:
            logger.error(f"Error getting session from Redis: {e}")
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics"""
        try:
            with self.lock:
                active_sessions = sum(1 for s in self.sessions.values() if s.is_active)
                total_sessions = len(self.sessions)
                unique_users = len(self.user_sessions)
                
                return {
                    "active_sessions": active_sessions,
                    "total_sessions": total_sessions,
                    "unique_users": unique_users,
                    "session_timeout": self.session_timeout,
                    "redis_connected": self.redis_client is not None
                }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

# Global session manager instance
session_manager = SessionManager() 