import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid
import os

logger = logging.getLogger(__name__)

@dataclass
class ConversationMessage:
    message_id: str
    session_id: str
    user_id: Optional[str]
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    message_type: str = "text"
    metadata: Dict[str, Any] = None

@dataclass
class ConversationSession:
    session_id: str
    user_id: Optional[str]
    created_at: datetime
    last_activity: datetime
    messages: List[ConversationMessage]
    context: Dict[str, Any]
    preferences: Dict[str, Any]
    is_active: bool = True

class ConversationMemory:
    """
    Conversation memory and persistence system
    """
    
    def __init__(self, storage_path: str = "conversation_data"):
        self.storage_path = storage_path
        self.sessions: Dict[str, ConversationSession] = {}
        self.user_sessions: Dict[str, List[str]] = {}  # user_id -> list of session_ids
        
        # Performance tracking
        self.stats = {
            'total_sessions': 0,
            'active_sessions': 0,
            'total_messages': 0,
            'avg_session_length': 0.0
        }
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
        
        logger.info(f"Conversation Memory initialized with storage: {storage_path}")
    
    def create_session(self, session_id: str, user_id: Optional[str] = None, 
                      initial_context: Dict[str, Any] = None) -> ConversationSession:
        """Create a new conversation session"""
        try:
            # Create session
            session = ConversationSession(
                session_id=session_id,
                user_id=user_id,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                messages=[],
                context=initial_context or {},
                preferences={}
            )
            
            # Store session
            self.sessions[session_id] = session
            
            # Track user sessions
            if user_id:
                if user_id not in self.user_sessions:
                    self.user_sessions[user_id] = []
                self.user_sessions[user_id].append(session_id)
            
            # Update stats
            self.stats['total_sessions'] += 1
            self.stats['active_sessions'] = len(self.sessions)
            
            # Save to disk
            self._save_session(session)
            
            logger.info(f"Created conversation session: {session_id} (user: {user_id})")
            return session
            
        except Exception as e:
            logger.error(f"Error creating conversation session: {e}")
            raise
    
    def add_message(self, session_id: str, role: str, content: str, 
                   user_id: Optional[str] = None, message_type: str = "text",
                   metadata: Dict[str, Any] = None) -> ConversationMessage:
        """Add a message to a conversation session"""
        try:
            session = self.sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            # Create message
            message = ConversationMessage(
                message_id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=user_id,
                role=role,
                content=content,
                timestamp=datetime.now(),
                message_type=message_type,
                metadata=metadata or {}
            )
            
            # Add to session
            session.messages.append(message)
            session.last_activity = datetime.now()
            
            # Update stats
            self.stats['total_messages'] += 1
            
            # Save to disk
            self._save_session(session)
            
            logger.debug(f"Added message to session {session_id}: {role} - {content[:50]}...")
            return message
            
        except Exception as e:
            logger.error(f"Error adding message to session {session_id}: {e}")
            raise
    
    def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        try:
            session = self.sessions.get(session_id)
            if not session:
                return []
            
            # Get recent messages
            recent_messages = session.messages[-limit:] if limit > 0 else session.messages
            
            # Convert to dict format
            history = []
            for message in recent_messages:
                history.append({
                    'role': message.role,
                    'content': message.content,
                    'timestamp': message.timestamp.isoformat(),
                    'message_type': message.message_type,
                    'metadata': message.metadata
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting conversation history for {session_id}: {e}")
            return []
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get session context"""
        try:
            session = self.sessions.get(session_id)
            if not session:
                return {}
            
            return session.context.copy()
            
        except Exception as e:
            logger.error(f"Error getting session context for {session_id}: {e}")
            return {}
    
    def update_session_context(self, session_id: str, context_updates: Dict[str, Any]):
        """Update session context"""
        try:
            session = self.sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            # Update context
            session.context.update(context_updates)
            session.last_activity = datetime.now()
            
            # Save to disk
            self._save_session(session)
            
            logger.debug(f"Updated context for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error updating session context for {session_id}: {e}")
            raise
    
    def update_user_preferences(self, session_id: str, preferences: Dict[str, Any]):
        """Update user preferences for a session"""
        try:
            session = self.sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            # Update preferences
            session.preferences.update(preferences)
            session.last_activity = datetime.now()
            
            # Save to disk
            self._save_session(session)
            
            logger.debug(f"Updated preferences for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error updating preferences for session {session_id}: {e}")
            raise
    
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all sessions for a user"""
        try:
            session_ids = self.user_sessions.get(user_id, [])
            sessions = []
            
            for session_id in session_ids:
                session = self.sessions.get(session_id)
                if session:
                    sessions.append({
                        'session_id': session.session_id,
                        'created_at': session.created_at.isoformat(),
                        'last_activity': session.last_activity.isoformat(),
                        'message_count': len(session.messages),
                        'is_active': session.is_active
                    })
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting user sessions for {user_id}: {e}")
            return []
    
    def end_session(self, session_id: str):
        """End a conversation session"""
        try:
            session = self.sessions.get(session_id)
            if not session:
                return
            
            # Mark as inactive
            session.is_active = False
            session.last_activity = datetime.now()
            
            # Update stats
            self.stats['active_sessions'] = len([s for s in self.sessions.values() if s.is_active])
            
            # Save to disk
            self._save_session(session)
            
            logger.info(f"Ended conversation session: {session_id}")
            
        except Exception as e:
            logger.error(f"Error ending session {session_id}: {e}")
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up old inactive sessions"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cleaned_count = 0
            
            for session_id, session in list(self.sessions.items()):
                if not session.is_active and session.last_activity < cutoff_date:
                    # Remove session
                    del self.sessions[session_id]
                    
                    # Remove from user sessions
                    if session.user_id and session.user_id in self.user_sessions:
                        self.user_sessions[session.user_id] = [
                            sid for sid in self.user_sessions[session.user_id] 
                            if sid != session_id
                        ]
                    
                    # Delete file
                    self._delete_session_file(session_id)
                    cleaned_count += 1
            
            # Update stats
            self.stats['total_sessions'] = len(self.sessions)
            self.stats['active_sessions'] = len([s for s in self.sessions.values() if s.is_active])
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old sessions")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}")
            return 0
    
    def search_conversations(self, user_id: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search conversations for a user"""
        try:
            session_ids = self.user_sessions.get(user_id, [])
            results = []
            
            for session_id in session_ids:
                session = self.sessions.get(session_id)
                if not session:
                    continue
                
                # Search in messages
                matching_messages = []
                for message in session.messages:
                    if query.lower() in message.content.lower():
                        matching_messages.append({
                            'role': message.role,
                            'content': message.content,
                            'timestamp': message.timestamp.isoformat()
                        })
                
                if matching_messages:
                    results.append({
                        'session_id': session_id,
                        'created_at': session.created_at.isoformat(),
                        'matching_messages': matching_messages[:5]  # Limit to 5 matches per session
                    })
                
                if len(results) >= limit:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching conversations for {user_id}: {e}")
            return []
    
    def get_conversation_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get conversation analytics for a user"""
        try:
            session_ids = self.user_sessions.get(user_id, [])
            total_messages = 0
            total_sessions = len(session_ids)
            avg_session_length = 0
            
            for session_id in session_ids:
                session = self.sessions.get(session_id)
                if session:
                    total_messages += len(session.messages)
            
            if total_sessions > 0:
                avg_session_length = total_messages / total_sessions
            
            return {
                'total_sessions': total_sessions,
                'total_messages': total_messages,
                'avg_session_length': avg_session_length,
                'active_sessions': len([sid for sid in session_ids if self.sessions.get(sid, {}).is_active])
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics for {user_id}: {e}")
            return {}
    
    def _save_session(self, session: ConversationSession):
        """Save session to disk"""
        try:
            filepath = os.path.join(self.storage_path, f"{session.session_id}.json")
            
            # Convert session to dict
            session_data = {
                'session_id': session.session_id,
                'user_id': session.user_id,
                'created_at': session.created_at.isoformat(),
                'last_activity': session.last_activity.isoformat(),
                'is_active': session.is_active,
                'context': session.context,
                'preferences': session.preferences,
                'messages': [asdict(msg) for msg in session.messages]
            }
            
            # Convert datetime objects in messages
            for msg in session_data['messages']:
                msg['timestamp'] = msg['timestamp']
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving session {session.session_id}: {e}")
    
    def _load_session(self, session_id: str) -> Optional[ConversationSession]:
        """Load session from disk"""
        try:
            filepath = os.path.join(self.storage_path, f"{session_id}.json")
            
            if not os.path.exists(filepath):
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert messages back to objects
            messages = []
            for msg_data in data['messages']:
                message = ConversationMessage(
                    message_id=msg_data['message_id'],
                    session_id=msg_data['session_id'],
                    user_id=msg_data['user_id'],
                    role=msg_data['role'],
                    content=msg_data['content'],
                    timestamp=datetime.fromisoformat(msg_data['timestamp']),
                    message_type=msg_data.get('message_type', 'text'),
                    metadata=msg_data.get('metadata', {})
                )
                messages.append(message)
            
            # Create session
            session = ConversationSession(
                session_id=data['session_id'],
                user_id=data['user_id'],
                created_at=datetime.fromisoformat(data['created_at']),
                last_activity=datetime.fromisoformat(data['last_activity']),
                messages=messages,
                context=data.get('context', {}),
                preferences=data.get('preferences', {}),
                is_active=data.get('is_active', True)
            )
            
            return session
            
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return None
    
    def _delete_session_file(self, session_id: str):
        """Delete session file from disk"""
        try:
            filepath = os.path.join(self.storage_path, f"{session_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            logger.error(f"Error deleting session file {session_id}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation memory statistics"""
        total_messages = sum(len(session.messages) for session in self.sessions.values())
        active_sessions = len([s for s in self.sessions.values() if s.is_active])
        
        avg_session_length = 0
        if len(self.sessions) > 0:
            avg_session_length = total_messages / len(self.sessions)
        
        return {
            **self.stats,
            'total_messages': total_messages,
            'active_sessions': active_sessions,
            'avg_session_length': avg_session_length,
            'storage_path': self.storage_path
        }

# Global conversation memory instance
_conversation_memory = None

def get_conversation_memory() -> ConversationMemory:
    """Get global conversation memory instance"""
    global _conversation_memory
    if _conversation_memory is None:
        _conversation_memory = ConversationMemory()
    return _conversation_memory
