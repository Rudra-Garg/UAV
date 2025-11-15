# create_task_dataset.py
"""
Advanced task dataset generator with realistic caching patterns.
This script's only job is to generate the data and save it.
Analysis is handled by a separate script: analyze_dataset.py
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

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


def generate_advanced_dataset():
    """Generates and saves the dataset with advanced realistic patterns."""
    print("--- Advanced Data Generation with Realistic Caching Patterns ---")
    # ... (print enabled features as before) ...

    # Initialize models
    transition_model = ServiceTransitionModel(NUM_SERVICE_TYPES) if USE_SERVICE_CHAINS else None
    time_model = TimeOfDayModel(NUM_SERVICE_TYPES) if USE_TIME_PATTERNS else None
    user_model = UserTypeModel() if USE_USER_TYPES else None
    content_model = ContentServiceCorrelation(NUM_SERVICE_TYPES, NUM_CONTENT_TYPES) if USE_CONTENT_CORRELATION else None

    requests_data = []
    current_service = sample_zipf_service()
    locality_prob = TEMPORAL_LOCALITY_PROB
    session_remaining = max(1, int(np.random.normal(SESSION_LENGTH_MEAN, SESSION_LENGTH_STD)))

    in_burst = False
    burst_remaining = 0

    for i in tqdm(range(NUM_REQUESTS_TO_GENERATE), desc="Generating Requests"):
        if user_model:
            user_model.maybe_switch_user()
            locality_prob = user_model.get_locality_prob()

        if time_model and i % 1000 == 0:
            time_model.advance_time(1000)

        # Generation logic for service_type
        if in_burst and burst_remaining > 0:
            service_type = current_service
            burst_remaining -= 1
        elif not in_burst and np.random.rand() < BURST_PROBABILITY / 100:
            in_burst = True
            burst_remaining = np.random.randint(15, 30)
            service_type = current_service
        else:
            in_burst = False
            session_remaining -= 1
            if session_remaining > 0 and np.random.rand() < locality_prob:
                service_type = current_service
            else:
                if transition_model and np.random.rand() < 0.6:
                    new_service = transition_model.next_service(current_service)
                else:
                    new_service = sample_zipf_service()

                if time_model:
                    boost = time_model.get_service_boost(new_service)
                    if boost < 2.0 and np.random.rand() > 0.5:
                        new_service = sample_zipf_service()

                current_service = new_service
                service_type = current_service

                session_remaining = user_model.get_session_length() if user_model else max(1, int(np.random.normal(
                    SESSION_LENGTH_MEAN, SESSION_LENGTH_STD)))

        # Generation logic for content_type
        content_type = None
        if np.random.rand() < 0.5:
            content_type = content_model.sample_content_for_service(
                service_type) if content_model else sample_zipf_content()

        requests_data.append({'service': service_type, 'content': content_type})

    print("\nGeneration complete. Saving to CSV...")
    df = pd.DataFrame(requests_data)
    df.to_csv(OUTPUT_FILENAME, index=False)
    print(f"\n✅ Dataset with {len(df):,} records saved to '{OUTPUT_FILENAME}'")


if __name__ == "__main__":
    generate_advanced_dataset()
