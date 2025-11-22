import numpy as np
import pandas as pd

from config import NUM_SERVICE_TYPES, NUM_CONTENT_TYPES, AREA_WIDTH, AREA_HEIGHT

# Load data
AZURE_MATRIX_FILE = 'azure_workload_matrix.npy'
AZURE_META_FILE = 'azure_service_meta.csv'


class AzureTraceModel:
    def __init__(self):
        try:
            # Load shape: (20, 1440) -> (Num_Services, Minutes_in_Day)
            self.workload = np.load(AZURE_MATRIX_FILE)
            # Add a small epsilon to avoid divide by zero
            self.workload = self.workload + 1e-5
            print("--- Loaded Azure Public Dataset Trace ---")
        except FileNotFoundError:
            print("ERROR: Run process_azure_data.py first! Falling back to random.")
            self.workload = np.random.rand(NUM_SERVICE_TYPES, 1440)

        self.num_services = self.workload.shape[0]
        self.total_minutes = self.workload.shape[1]
        self.current_minute_idx = 0

        # Pre-calculate probabilities per minute
        # Shape: (1440, 20) - Probability of each service at each minute
        self.service_probs_per_minute = self.workload.T / self.workload.T.sum(axis=1, keepdims=True)

        # --- NEW: Load Triggers ---
        meta_df = pd.read_csv(AZURE_META_FILE)
        self.triggers = meta_df['Trigger'].values  # Array of strings, e.g., ['http', 'timer'...]
        # Pre-assign Content Preferences based on Triggers
        self.service_content_map = {}
        for service_id in range(NUM_SERVICE_TYPES):
            trigger = self.triggers[service_id]

            if trigger == 'http':
                # HTTP services access content via Zipf (User behavior)
                # They might access ANY content, but some are popular
                self.service_content_map[service_id] = 'zipf'

            elif trigger in ['timer', 'queue', 'event']:
                # Automated services usually access specific, predictable data
                # Assign a specific subset of 5 content items to this service
                start_content = (service_id * 5) % NUM_CONTENT_TYPES
                self.service_content_map[service_id] = list(range(start_content, start_content + 5))

            else:
                self.service_content_map[service_id] = 'random'

    def get_content_for_service(self, service_id):
        """
        Returns a Content ID based on the Service's Trigger Type.
        """
        strategy = self.service_content_map[service_id]

        if strategy == 'zipf':
            # Standard Zipf distribution (viral content)
            # Using -1 because numpy zipf returns 1-based index
            return (np.random.zipf(1.5) - 1) % NUM_CONTENT_TYPES

        elif isinstance(strategy, list):
            # This service only ever accesses these specific 5 files
            return np.random.choice(strategy)

        else:
            # Random
            return np.random.randint(0, NUM_CONTENT_TYPES)

    def get_service_prob(self, time_step, vehicle_position):
        """
        Returns the probability distribution of services for a specific time and location.
        This injects SPATIAL BIAS into the temporal Azure data.
        """
        # 1. Map simulation step to Azure Minute (assuming 1 sim step = 1 real second? Or just loop it)
        # Let's say 60 simulation steps = 1 dataset minute
        minute_idx = (time_step // 60) % self.total_minutes

        base_probs = self.service_probs_per_minute[minute_idx].copy()

        # 2. SPATIAL BIAS INJECTION
        # This is CRITICAL. Without this, all UAVs see the same global patterns (homogenization).
        # We boost specific services based on vehicle position (X, Y).

        x, y = vehicle_position[0], vehicle_position[1]

        # Define 4 zones
        if x < AREA_WIDTH / 2 and y < AREA_HEIGHT / 2:
            # Zone 1 (Bottom-Left): Boost Services 0-4
            base_probs[0:5] *= 5.0
        elif x >= AREA_WIDTH / 2 and y < AREA_HEIGHT / 2:
            # Zone 2 (Bottom-Right): Boost Services 5-9
            base_probs[5:10] *= 5.0
        elif x < AREA_WIDTH / 2 and y >= AREA_HEIGHT / 2:
            # Zone 3 (Top-Left): Boost Services 10-14
            base_probs[10:15] *= 5.0
        else:
            # Zone 4 (Top-Right): Boost Services 15-19
            base_probs[15:20] *= 5.0

        # Re-normalize after boosting
        return base_probs / base_probs.sum()


class DemandGenerator:
    def __init__(self):
        self.azure_model = AzureTraceModel()

    def generate_next_request(self, time_step, vehicle_position):
        # 1. Determine Service (based on Time + Location)
        probs = self.azure_model.get_service_prob(time_step, vehicle_position)
        service_type = np.random.choice(NUM_SERVICE_TYPES, p=probs)

        # 2. Determine Content (based on Service Trigger)
        content_type = self.azure_model.get_content_for_service(service_type)

        return service_type, content_type

    def reset(self):
        pass
