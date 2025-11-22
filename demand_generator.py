# demand_generator.py
import numpy as np
import pandas as pd
import os
from config import NUM_SERVICE_TYPES, NUM_CONTENT_TYPES, NUM_ZONES

# Load data
AZURE_MATRIX_FILE = 'azure_workload_matrix.npy'
AZURE_META_FILE = 'azure_service_meta.csv'

# MATCHING WORKFLOW LOGIC FROM OFFLINE GENERATOR
WORKFLOWS = {
    0: [1, 2],  # Login -> Dashboard / Data
    1: [3, 4],  # Dashboard -> Analytics / Settings
    5: [6, 7, 8],  # Search -> Result -> Image -> Details
    10: [11, 12],  # Auth -> Token -> Session
    15: [16]  # Upload -> Process
}
WORKFLOW_PROBABILITY = 0.85


class AzureTraceModel:
    def __init__(self):
        if not os.path.exists(AZURE_MATRIX_FILE) or not os.path.exists(AZURE_META_FILE):
            print("⚠️ WARNING: Azure data files not found. Falling back to random generation.")
            self.workload = np.random.rand(NUM_SERVICE_TYPES, 1440)
            self.triggers = ['random'] * NUM_SERVICE_TYPES
        else:
            self.workload = np.load(AZURE_MATRIX_FILE).astype(np.float64) + 1e-5
            meta_df = pd.read_csv(AZURE_META_FILE)
            self.triggers = meta_df['Trigger'].values

        self.probs_per_minute = self.workload.T / self.workload.T.sum(axis=1, keepdims=True)
        self.total_minutes = self.probs_per_minute.shape[0]

        self.service_content_map = {}
        for sid in range(NUM_SERVICE_TYPES):
            trigger = self.triggers[sid] if sid < len(self.triggers) else 'others'
            if trigger == 'http':
                self.service_content_map[sid] = 'zipf'
            elif trigger in ['timer', 'queue', 'event']:
                start_id = (sid * 5) % NUM_CONTENT_TYPES
                self.service_content_map[sid] = list(range(start_id, start_id + 5))
            else:
                self.service_content_map[sid] = 'random'

    def get_service_prob(self, time_step, zone_id):
        minute_idx = (time_step // 60) % self.total_minutes
        base_probs = self.probs_per_minute[minute_idx].copy()

        # --- SPATIAL BIAS (Must match Offline Generator) ---
        boost_factor = 20.0
        if zone_id == 0:
            base_probs[0:5] *= boost_factor
        elif zone_id == 1:
            base_probs[5:10] *= boost_factor
        elif zone_id == 2:
            base_probs[10:15] *= boost_factor
        elif zone_id == 3:
            base_probs[15:20] *= boost_factor

        return base_probs / base_probs.sum()

    def get_content(self, service_id):
        strategy = self.service_content_map.get(service_id, 'random')
        if strategy == 'zipf':
            c = np.random.zipf(1.5) - 1
            return c % NUM_CONTENT_TYPES
        elif isinstance(strategy, list):
            return np.random.choice(strategy)
        else:
            return np.random.randint(0, NUM_CONTENT_TYPES)


class DemandGenerator:
    def __init__(self):
        self.azure_model = AzureTraceModel()
        # Track active workflows per vehicle: { vehicle_id: [next_service_id, ...] }
        self.vehicle_queues = {}

    def generate_next_request(self, time_step, vehicle_id, zone_id):
        """
        Generates the next request for a specific vehicle, respecting workflows.
        """
        service_type = -1

        # 1. Check if vehicle is in a workflow
        if vehicle_id in self.vehicle_queues and self.vehicle_queues[vehicle_id]:
            # high chance to continue chain
            if np.random.rand() < WORKFLOW_PROBABILITY:
                service_type = self.vehicle_queues[vehicle_id].pop(0)
                # Clean up if empty
                if not self.vehicle_queues[vehicle_id]:
                    del self.vehicle_queues[vehicle_id]
            else:
                # Broke the chain
                del self.vehicle_queues[vehicle_id]

        # 2. If no service selected yet, pick new one based on Zone/Time
        if service_type == -1:
            probs = self.azure_model.get_service_prob(time_step, zone_id)
            service_type = np.random.choice(NUM_SERVICE_TYPES, p=probs)

            # 3. Does this start a new workflow?
            if service_type in WORKFLOWS:
                # Copy the workflow steps to the vehicle's queue
                self.vehicle_queues[vehicle_id] = list(WORKFLOWS[service_type])

        # 4. Get Content
        content_type = self.azure_model.get_content(service_type)

        return service_type, content_type

    def reset(self):
        self.vehicle_queues.clear()
