# create_task_dataset.py
"""
A lightweight, standalone script to generate a synthetic dataset of task requests.

This script directly uses the task generation logic and parameters from the
project's config to create a large CSV file for training the LSTM
cache predictor, without running the full reinforcement learning simulation.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import the configuration parameters that define task generation
from config import (
    POPULARITY_ZIPF_ALPHA,
    NUM_SERVICE_TYPES,
    NUM_CONTENT_TYPES
)

# --- Configuration ---
NUM_REQUESTS_TO_GENERATE = 10000000  # Generate half a million data points
OUTPUT_FILENAME = "task_request_data.csv"


# ---

def generate_dataset():
    """Generates and saves the task request dataset."""
    print("--- Starting Lightweight Data Generation ---")
    print(f"Generating {NUM_REQUESTS_TO_GENERATE:,} task requests...")

    requests_data = []

    # Use tqdm for a nice progress bar
    for _ in tqdm(range(NUM_REQUESTS_TO_GENERATE)):

        # --- This logic is copied directly from Task.__init__ in entities.py ---
        # This ensures our generated data has the exact same statistical properties
        # as the data seen by the agents during simulation.

        # 1. Generate a service type based on the Zipf distribution
        # The `[0] - 1` part extracts the number and makes it 0-indexed.
        # The modulo ensures it's a valid service ID.
        service_type = (np.random.zipf(POPULARITY_ZIPF_ALPHA, 1)[0] - 1) % NUM_SERVICE_TYPES

        # 2. Generate a content type with a 50% probability
        content_type = None
        if np.random.rand() < 0.5:
            content_type = (np.random.zipf(POPULARITY_ZIPF_ALPHA, 1)[0] - 1) % NUM_CONTENT_TYPES

        # 3. Store the result
        requests_data.append({
            'service': service_type,
            'content': content_type
        })

    print("\nGeneration complete.")

    # Convert the list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(requests_data)

    # Save the DataFrame to a CSV file
    df.to_csv(OUTPUT_FILENAME, index=False)

    print(f"✅ Dataset with {len(df):,} records saved to '{OUTPUT_FILENAME}'")


if __name__ == "__main__":
    generate_dataset()
