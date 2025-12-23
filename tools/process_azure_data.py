import os

import numpy as np
import pandas as pd

from config import AZURE_MATRIX_FILE, AZURE_META_FILE

# You must provide the path to your raw Azure CSV here
# If you don't have it, the DemandGenerator will fallback to random noise.
INPUT_FILE = "D:/azurefunctions-dataset2019.tar/invocations_per_function_md.anon.d01.csv"
NUM_SERVICES_TO_KEEP = 20


def process_trace():
    print(f"--- Processing Azure Trace ---")

    if not os.path.exists(INPUT_FILE):
        print(f"❌ ERROR: Raw Azure input file not found at {INPUT_FILE}")
        print("Please download the Azure Functions Trace 2019 dataset or update the path.")
        return

    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)
    # Rename columns (cols 4 to end are the minutes)
    minute_cols = [f'min_{i}' for i in range(1440)]
    df.columns = ['Owner', 'App', 'Function', 'Trigger'] + minute_cols

    # 2. Sum invocations to find the Top K popular functions
    df['total_invocations'] = df[minute_cols].sum(axis=1)

    # Sort by popularity and take Top 20
    top_services = df.nlargest(NUM_SERVICES_TO_KEEP, 'total_invocations').reset_index(drop=True)

    # 3. Save Workload Matrix (The Time-Series Data)
    workload_matrix = top_services[minute_cols].values

    # Ensure directory exists
    os.makedirs(os.path.dirname(AZURE_MATRIX_FILE), exist_ok=True)

    np.save(AZURE_MATRIX_FILE, workload_matrix)

    # 4. Save Metadata (The Trigger Data)
    top_services['Service_ID'] = top_services.index
    top_services[['Service_ID', 'Trigger']].to_csv(AZURE_META_FILE, index=False)

    print(f"✅ Saved Matrix: {workload_matrix.shape}")
    print(f"✅ Saved Meta: {AZURE_META_FILE}")


if __name__ == "__main__":
    process_trace()
