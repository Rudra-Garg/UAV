# demand_generator.py
"""
Encapsulates the advanced, stateful logic for generating realistic
sequences of task requests, including sessions, user types, and time patterns.
"""

import numpy as np

from config import (
    POPULARITY_ZIPF_ALPHA,
    NUM_SERVICE_TYPES,
    NUM_CONTENT_TYPES
)

# --- Configuration ---
NUM_REQUESTS_TO_GENERATE = 10000000
OUTPUT_FILENAME = "task_request_data.csv"

# Pattern parameters
TEMPORAL_LOCALITY_PROB = 0.75
SESSION_LENGTH_MEAN = 8
SESSION_LENGTH_STD = 3
BURST_PROBABILITY = 0.15

# New advanced features
USE_SERVICE_CHAINS = True
USE_TIME_PATTERNS = True
USE_USER_TYPES = True
USE_CONTENT_CORRELATION = True


# --- Model Classes (Helper classes for generating patterns) ---

class ServiceTransitionModel:
    """Models which services are likely to follow each other"""

    def __init__(self, num_services):
        self.num_services = num_services
        self.transitions = self._create_transition_matrix()

    def _create_transition_matrix(self):
        transitions = {}
        for s in range(self.num_services):
            probs = np.array([1.0 / (i + 1) ** POPULARITY_ZIPF_ALPHA for i in range(self.num_services)])
            for target in range(self.num_services):
                distance = abs(s - target)
                if distance <= 2:
                    probs[target] *= (3 - distance)
            probs /= probs.sum()
            transitions[s] = probs
        return transitions

    def next_service(self, current_service):
        return np.random.choice(self.num_services, p=self.transitions[current_service])


class TimeOfDayModel:
    """Models how service popularity changes throughout the day"""

    def __init__(self, num_services):
        self.num_services = num_services
        self.current_hour = 0
        self.service_peaks = {s: ((18, 23) if s % 5 == 0 else (6, 10) if s % 5 == 1 else (9, 17) if s % 5 == 2 else (20,
                                                                                                                     2) if s % 5 == 3 else (
            12, 20)) for s in range(num_services)}

    def get_service_boost(self, service):
        start, end = self.service_peaks[service]
        in_peak = (self.current_hour >= start or self.current_hour <= end) if end < start else (
                start <= self.current_hour <= end)
        return 3.0 if in_peak else 1.0

    def advance_time(self, num_requests):
        self.current_hour = int((self.current_hour + num_requests / 6000) % 24)


class UserTypeModel:
    """Models different user behavior patterns"""

    def __init__(self):
        self.user_types = ['power', 'casual', 'bursty', 'explorer']
        self.current_user_type = 'casual'
        self.requests_until_switch = 100
        self.type_info = {
            'power': {'locality': 0.85, 'session': (15, 5)},
            'casual': {'locality': 0.70, 'session': (8, 3)},
            'bursty': {'locality': 0.95, 'session': (20, 8)},
            'explorer': {'locality': 0.50, 'session': (3, 1)}
        }

    def get_locality_prob(self):
        return self.type_info[self.current_user_type]['locality']

    def get_session_length(self):
        mean, std = self.type_info[self.current_user_type]['session']
        return max(1, int(np.random.normal(mean, std)))

    def maybe_switch_user(self):
        self.requests_until_switch -= 1
        if self.requests_until_switch <= 0:
            self.current_user_type = np.random.choice(self.user_types, p=[0.2, 0.5, 0.15, 0.15])
            self.requests_until_switch = np.random.randint(50, 200)


class ContentServiceCorrelation:
    """Models which content types go with which services"""

    def __init__(self, num_services, num_contents):
        self.num_services = num_services
        self.num_contents = num_contents
        self.service_content_prefs = {}
        for s in range(num_services):
            preferred = [(s + i) % self.num_contents for i in range(3)]
            probs = np.ones(num_contents) * 0.5
            for p in preferred:
                probs[p] = 5.0
            self.service_content_prefs[s] = probs / probs.sum()

    def sample_content_for_service(self, service):
        return np.random.choice(self.num_contents, p=self.service_content_prefs[service])


def sample_zipf_service():
    return (np.random.zipf(POPULARITY_ZIPF_ALPHA, 1)[0] - 1) % NUM_SERVICE_TYPES


def sample_zipf_content():
    return (np.random.zipf(POPULARITY_ZIPF_ALPHA, 1)[0] - 1) % NUM_CONTENT_TYPES


# --- Main Orchestrator Class ---
class DemandGenerator:
    def __init__(self):
        # Initialize all the sub-models
        self.transition_model = ServiceTransitionModel(NUM_SERVICE_TYPES)
        self.time_model = TimeOfDayModel(NUM_SERVICE_TYPES)
        self.user_model = UserTypeModel()
        self.content_model = ContentServiceCorrelation(NUM_SERVICE_TYPES, NUM_CONTENT_TYPES)

        # Session state
        self.current_service = sample_zipf_service()
        self.session_remaining = self.user_model.get_session_length()

        # Burst state
        self.in_burst = False
        self.burst_remaining = 0

        self.requests_generated = 0

    def generate_next_request(self):
        """Generates the next (service, content) pair based on the internal state."""
        self.requests_generated += 1

        # Update stateful models
        self.user_model.maybe_switch_user()
        if self.requests_generated % 1000 == 0:
            self.time_model.advance_time(1000)

        # Determine the next service type
        locality_prob = self.user_model.get_locality_prob()

        if self.in_burst and self.burst_remaining > 0:
            service_type = self.current_service
            self.burst_remaining -= 1
        elif not self.in_burst and np.random.rand() < (BURST_PROBABILITY / 100):
            self.in_burst = True
            self.burst_remaining = np.random.randint(15, 30)
            service_type = self.current_service
        else:
            self.in_burst = False
            self.session_remaining -= 1
            if self.session_remaining > 0 and np.random.rand() < locality_prob:
                service_type = self.current_service
            else:
                new_service = self.transition_model.next_service(self.current_service)
                boost = self.time_model.get_service_boost(new_service)
                if boost < 2.0 and np.random.rand() > 0.5:
                    new_service = sample_zipf_service()

                self.current_service = new_service
                service_type = self.current_service
                self.session_remaining = self.user_model.get_session_length()

        # Determine the content type
        content_type = None
        if np.random.rand() < 0.5:
            content_type = self.content_model.sample_content_for_service(service_type)

        return service_type, content_type

    def reset(self):
        """Resets the generator's state for a new episode."""
        self.current_service = sample_zipf_service()
        self.session_remaining = self.user_model.get_session_length()
        self.in_burst = False
        self.burst_remaining = 0
        self.requests_generated = 0
        self.time_model.current_hour = np.random.randint(0, 24)  # Start at a random time of day
