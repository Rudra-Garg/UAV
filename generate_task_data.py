# generate_task_data.py
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import NUM_SERVICE_TYPES, NUM_CONTENT_TYPES, NUM_ZONES

# --- Configuration ---
NUM_REQUESTS_TO_GENERATE = 2000000
OUTPUT_FILENAME = "task_request_data.csv"
AZURE_MATRIX_FILE = "azure_workload_matrix.npy"
AZURE_META_FILE = "azure_service_meta.csv"

# --- NEW: Define Workflows (Sequential dependencies) ---
# Format: { Trigger_Service_ID: [Next_Possible_Service_A, Next_Possible_Service_B] }
# This simulates a user "logging in" -> "dashboard" -> "profile"
WORKFLOWS = {
    0: [1, 2],  # Service 0 triggers 1 or 2
    1: [3, 4],  # Service 1 triggers 3 or 4
    5: [6, 7, 8],  # Service 5 triggers 6, 7, or 8
    10: [11, 12],
    15: [16]
}
WORKFLOW_PROBABILITY = 0.85  # 85% chance to follow workflow if active


class AzureGeneratorModel:
    def __init__(self):
        if not os.path.exists(AZURE_MATRIX_FILE):
            raise FileNotFoundError(f"Missing {AZURE_MATRIX_FILE}. Run process_azure_data.py first!")

        self.workload = np.load(AZURE_MATRIX_FILE).astype(np.float64) + 1e-5
        self.probs_per_minute = self.workload.T / self.workload.T.sum(axis=1, keepdims=True)
        self.total_minutes = self.probs_per_minute.shape[0]

        if not os.path.exists(AZURE_META_FILE):
            raise FileNotFoundError(f"Missing {AZURE_META_FILE}. Run process_azure_data.py first!")

        meta_df = pd.read_csv(AZURE_META_FILE)
        self.triggers = meta_df['Trigger'].values

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

    def get_service_prob(self, minute_idx, zone_id):
        base_probs = self.probs_per_minute[minute_idx].copy()

        # Spatial Bias: Boost specific services based on Zone
        # This gives the LSTM 'spatial context' to learn
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
        strategy = self.service_content_map[service_id]
        if strategy == 'zipf':
            c = np.random.zipf(1.5) - 1
            return c % NUM_CONTENT_TYPES
        elif isinstance(strategy, list):
            return np.random.choice(strategy)
        else:
            return np.random.randint(0, NUM_CONTENT_TYPES)


def generate_dataset():
    print(f"--- Generating {NUM_REQUESTS_TO_GENERATE:,} requests (With Sequential Workflows) ---")

    try:
        model = AzureGeneratorModel()
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        return

    data = []

    # Simulation pointers
    requests_per_minute = 60
    current_minute = 0
    current_zone = 0
    requests_until_zone_switch = 5000

    # Workflow queue for sequential patterns
    active_workflow_queue = []

    for i in tqdm(range(NUM_REQUESTS_TO_GENERATE)):
        # Time progression
        if i % requests_per_minute == 0:
            current_minute = (current_minute + 1) % model.total_minutes

        # Space progression (Simulating a user/group moving between zones)
        if i % requests_until_zone_switch == 0:
            current_zone = (current_zone + 1) % NUM_ZONES

        service_id = -1

        # --- LOGIC: Workflow vs Random ---
        # 1. If inside a workflow, high chance to follow it
        if active_workflow_queue and np.random.rand() < WORKFLOW_PROBABILITY:
            service_id = active_workflow_queue.pop(0)
        else:
            # 2. Otherwise, pick based on Azure prob + Zone bias
            active_workflow_queue = []  # Reset queue
            probs = model.get_service_prob(current_minute, current_zone)
            service_id = np.random.choice(NUM_SERVICE_TYPES, p=probs)

            # 3. Does this service start a new workflow?
            if service_id in WORKFLOWS:
                active_workflow_queue.extend(WORKFLOWS[service_id])

        content_id = model.get_content(service_id)

        data.append({
            'service': service_id,
            'content': content_id,
            'zone': current_zone  # Crucial for context-aware LSTM
        })

    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_FILENAME, index=False)
    print(f"\n✅ Saved dataset to {OUTPUT_FILENAME}")
    print("   Sample Data:")
    print(df.head())


if __name__ == "__main__":
    generate_dataset()
