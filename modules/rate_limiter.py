import time
import threading
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import redis
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Rate limit configuration for different user plans"""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int
    window_size: int = 60  # seconds

class RateLimiter:
    """Enhanced rate limiter for multiple users with different plans"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.rate_limits: Dict[str, RateLimitConfig] = {
            "basic": RateLimitConfig(30, 1000, 10000, 5),
            "plus": RateLimitConfig(60, 2000, 20000, 10),
            "pro": RateLimitConfig(120, 5000, 50000, 20)
        }
        
        # In-memory storage for rate limit data
        self.rate_limit_data: Dict[str, Dict[str, any]] = defaultdict(dict)
        self.lock = threading.RLock()
        
        # Redis for distributed rate limiting (optional)
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                logger.info("Connected to Redis for distributed rate limiting")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
    
    def is_allowed(self, identifier: str, plan: str = "basic") -> Tuple[bool, Dict[str, any]]:
        """Check if request is allowed based on rate limits"""
        try:
            config = self.rate_limits.get(plan, self.rate_limits["basic"])
            current_time = time.time()
            
            # Get rate limit data
            rate_data = self._get_rate_limit_data(identifier)
            
            # Check minute limit
            minute_key = f"minute_{int(current_time / 60)}"
            minute_requests = rate_data.get(minute_key, 0)
            
            if minute_requests >= config.requests_per_minute:
                return False, {
                    "allowed": False,
                    "limit_type": "minute",
                    "limit": config.requests_per_minute,
                    "current": minute_requests,
                    "reset_time": (int(current_time / 60) + 1) * 60
                }
            
            # Check hour limit
            hour_key = f"hour_{int(current_time / 3600)}"
            hour_requests = rate_data.get(hour_key, 0)
            
            if hour_requests >= config.requests_per_hour:
                return False, {
                    "allowed": False,
                    "limit_type": "hour",
                    "limit": config.requests_per_hour,
                    "current": hour_requests,
                    "reset_time": (int(current_time / 3600) + 1) * 3600
                }
            
            # Check day limit
            day_key = f"day_{int(current_time / 86400)}"
            day_requests = rate_data.get(day_key, 0)
            
            if day_requests >= config.requests_per_day:
                return False, {
                    "allowed": False,
                    "limit_type": "day",
                    "limit": config.requests_per_day,
                    "current": day_requests,
                    "reset_time": (int(current_time / 86400) + 1) * 86400
                }
            
            # Update counters
            self._increment_counters(identifier, minute_key, hour_key, day_key)
            
            return True, {
                "allowed": True,
                "minute_remaining": config.requests_per_minute - minute_requests - 1,
                "hour_remaining": config.requests_per_hour - hour_requests - 1,
                "day_remaining": config.requests_per_day - day_requests - 1
            }
            
        except Exception as e:
            logger.error(f"Error checking rate limit for {identifier}: {e}")
            return True, {"allowed": True, "error": str(e)}
    
    def _get_rate_limit_data(self, identifier: str) -> Dict[str, any]:
        """Get rate limit data from memory or Redis"""
        try:
            # Try Redis first
            if self.redis_client:
                data = self.redis_client.get(f"rate_limit:{identifier}")
                if data:
                    return json.loads(data)
            
            # Fallback to memory
            with self.lock:
                return self.rate_limit_data[identifier]
                
        except Exception as e:
            logger.error(f"Error getting rate limit data: {e}")
            return {}
    
    def _increment_counters(self, identifier: str, minute_key: str, hour_key: str, day_key: str):
        """Increment rate limit counters"""
        try:
            # Update memory
            with self.lock:
                if identifier not in self.rate_limit_data:
                    self.rate_limit_data[identifier] = {}
                
                self.rate_limit_data[identifier][minute_key] = \
                    self.rate_limit_data[identifier].get(minute_key, 0) + 1
                self.rate_limit_data[identifier][hour_key] = \
                    self.rate_limit_data[identifier].get(hour_key, 0) + 1
                self.rate_limit_data[identifier][day_key] = \
                    self.rate_limit_data[identifier].get(day_key, 0) + 1
            
            # Update Redis
            if self.redis_client:
                data = self._get_rate_limit_data(identifier)
                data[minute_key] = data.get(minute_key, 0) + 1
                data[hour_key] = data.get(hour_key, 0) + 1
                data[day_key] = data.get(day_key, 0) + 1
                
                self.redis_client.setex(
                    f"rate_limit:{identifier}",
                    86400,  # 24 hours
                    json.dumps(data)
                )
                
        except Exception as e:
            logger.error(f"Error incrementing counters: {e}")
    
    def get_remaining_requests(self, identifier: str, plan: str = "basic") -> Dict[str, int]:
        """Get remaining requests for different time windows"""
        try:
            config = self.rate_limits.get(plan, self.rate_limits["basic"])
            current_time = time.time()
            rate_data = self._get_rate_limit_data(identifier)
            
            minute_key = f"minute_{int(current_time / 60)}"
            hour_key = f"hour_{int(current_time / 3600)}"
            day_key = f"day_{int(current_time / 86400)}"
            
            return {
                "minute_remaining": max(0, config.requests_per_minute - rate_data.get(minute_key, 0)),
                "hour_remaining": max(0, config.requests_per_hour - rate_data.get(hour_key, 0)),
                "day_remaining": max(0, config.requests_per_day - rate_data.get(day_key, 0))
            }
            
        except Exception as e:
            logger.error(f"Error getting remaining requests: {e}")
            return {"minute_remaining": 0, "hour_remaining": 0, "day_remaining": 0}
    
    def reset_limits(self, identifier: str):
        """Reset rate limits for an identifier"""
        try:
            with self.lock:
                if identifier in self.rate_limit_data:
                    del self.rate_limit_data[identifier]
            
            if self.redis_client:
                self.redis_client.delete(f"rate_limit:{identifier}")
                
            logger.info(f"Reset rate limits for {identifier}")
            
        except Exception as e:
            logger.error(f"Error resetting rate limits: {e}")
    
    def get_stats(self) -> Dict[str, any]:
        """Get rate limiter statistics"""
        try:
            with self.lock:
                total_identifiers = len(self.rate_limit_data)
                total_requests = sum(
                    sum(data.values()) for data in self.rate_limit_data.values()
                )
                
                return {
                    "total_identifiers": total_identifiers,
                    "total_requests": total_requests,
                    "redis_connected": self.redis_client is not None,
                    "rate_limits": {
                        plan: {
                            "requests_per_minute": config.requests_per_minute,
                            "requests_per_hour": config.requests_per_hour,
                            "requests_per_day": config.requests_per_day
                        }
                        for plan, config in self.rate_limits.items()
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting rate limiter stats: {e}")
            return {}

class TokenBucketRateLimiter:
    """Token bucket rate limiter for burst handling"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def is_allowed(self, tokens_required: int = 1) -> bool:
        """Check if request is allowed based on available tokens"""
        with self.lock:
            self._refill_tokens()
            
            if self.tokens >= tokens_required:
                self.tokens -= tokens_required
                return True
            return False
    
    def _refill_tokens(self):
        """Refill tokens based on time elapsed"""
        current_time = time.time()
        time_elapsed = current_time - self.last_refill
        tokens_to_add = time_elapsed * self.refill_rate
        
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = current_time

class SlidingWindowRateLimiter:
    """Sliding window rate limiter for precise rate limiting"""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = []
        self.lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed in sliding window"""
        current_time = time.time()
        
        with self.lock:
            # Remove expired requests
            self.requests = [req_time for req_time in self.requests 
                           if current_time - req_time < self.window_size]
            
            # Check if we can add new request
            if len(self.requests) < self.max_requests:
                self.requests.append(current_time)
                return True
            
            return False

# Global rate limiter instances
rate_limiter = RateLimiter()
token_bucket_limiter = TokenBucketRateLimiter(capacity=100, refill_rate=10)
sliding_window_limiter = SlidingWindowRateLimiter(window_size=60, max_requests=30) 