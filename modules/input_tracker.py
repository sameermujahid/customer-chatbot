import time
from collections import defaultdict
import logging
import json
import os
from datetime import datetime
from modules.config import PLAN_INPUT_LIMITS, UserPlan

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserInputTracker:
    def __init__(self):
        self.input_counts = defaultdict(lambda: {'count': 0, 'last_reset': time.time()})
        self.session_data_file = 'session_data.json'
        self.load_session_data()

    def load_session_data(self):
        """Load session data from file if it exists"""
        try:
            if os.path.exists(self.session_data_file):
                with open(self.session_data_file, 'r') as f:
                    data = json.load(f)
                    for session_id, session_info in data.items():
                        self.input_counts[session_id] = session_info
                print(f"Loaded {len(data)} sessions from file")
        except Exception as e:
            logger.error(f"Error loading session data: {str(e)}")

    def save_session_data(self):
        """Save session data to file"""
        try:
            with open(self.session_data_file, 'w') as f:
                json.dump(dict(self.input_counts), f)
            print("Session data saved successfully")
        except Exception as e:
            logger.error(f"Error saving session data: {str(e)}")

    def can_accept_input(self, session_id, plan):
        """Check if the user can make another query based on their plan"""
        self._check_reset(session_id)
        max_inputs = self._get_max_inputs(plan)
        current_count = self.input_counts[session_id]['count']
        
        print(f"Session {session_id} - Plan: {plan}, Current count: {current_count}, Max inputs: {max_inputs}")
        return current_count < max_inputs

    def add_input(self, session_id, plan):
        """Add an input to the user's count"""
        self._check_reset(session_id)
        if self.can_accept_input(session_id, plan):
            self.input_counts[session_id]['count'] += 1
            self.save_session_data()
            print(f"Added input for session {session_id}. New count: {self.input_counts[session_id]['count']}")
            return True
        return False

    def get_remaining_inputs(self, session_id, plan):
        """Get the number of remaining inputs for the user"""
        self._check_reset(session_id)
        max_inputs = self._get_max_inputs(plan)
        current_count = self.input_counts[session_id]['count']
        remaining = max(0, max_inputs - current_count)
        print(f"Session {session_id} - Remaining inputs: {remaining}")
        return remaining

    def get_usage_stats(self, session_id):
        """Get usage statistics for a session"""
        try:
            user_data = self.input_counts[session_id]
            current_time = time.time()
            remaining_time = 24 - ((current_time - user_data['last_reset']) / 3600)
            
            return {
                'total_used': user_data['count'],
                'last_reset': datetime.fromtimestamp(user_data['last_reset']).isoformat(),
                'remaining_time': remaining_time
            }
        except Exception as e:
            logger.error(f"Error in get_usage_stats: {str(e)}")
            return {
                'total_used': 0,
                'last_reset': datetime.fromtimestamp(time.time()).isoformat(),
                'remaining_time': 24
            }

    def _check_reset(self, session_id):
        """Check if the 24-hour period has passed and reset if necessary"""
        current_time = time.time()
        last_reset = self.input_counts[session_id]['last_reset']
        
        if current_time - last_reset >= 24 * 3600:  # 24 hours in seconds
            self.input_counts[session_id] = {'count': 0, 'last_reset': current_time}
            self.save_session_data()
            print(f"Reset count for session {session_id}")

    def _get_max_inputs(self, plan):
        """Get the maximum number of inputs allowed for a plan"""
        try:
            # If plan is a UserPlan enum, get its value
            if isinstance(plan, UserPlan):
                plan = plan.value
            
            # Convert plan to lowercase string for comparison
            plan = str(plan).lower()
            
            plan_limits = {
                'basic': 5,
                'plus': 20,
                'pro': 50
            }
            return plan_limits.get(plan, 5)  # Default to basic plan if unknown
        except Exception as e:
            logger.error(f"Error getting max inputs for plan {plan}: {str(e)}")
            return 5  # Default to basic plan on error

    def get_usage_stats(self, session_id):
        """Get usage statistics for a session"""
        try:
            user_data = self.input_counts[session_id]
            current_time = time.time()
            remaining_time = 24 - ((current_time - user_data['last_reset']) / 3600)
            
            return {
                'total_used': user_data['count'],
                'last_reset': datetime.fromtimestamp(user_data['last_reset']).isoformat(),
                'remaining_time': remaining_time
            }
        except Exception as e:
            logger.error(f"Error in get_usage_stats: {str(e)}")
            return {
                'total_used': 0,
                'last_reset': datetime.fromtimestamp(time.time()).isoformat(),
                'remaining_time': 24
            } 