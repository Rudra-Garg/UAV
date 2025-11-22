import numpy as np
import pandas as pd

INPUT_FILE = "D:/azurefunctions-dataset2019.tar/invocations_per_function_md.anon.d01.csv"
OUTPUT_MATRIX = 'azure_workload_matrix.npy'
OUTPUT_META = 'azure_service_meta.csv'
NUM_SERVICES_TO_KEEP = 20


def process_trace():
    print(f"--- Processing Azure Trace: {INPUT_FILE} ---")

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
    np.save(OUTPUT_MATRIX, workload_matrix)

    # 4. Save Metadata (The Trigger Data)
    # We create a simple CSV mapping: Service_ID (0-19), Trigger_Type
    top_services['Service_ID'] = top_services.index
    top_services[['Service_ID', 'Trigger']].to_csv(OUTPUT_META, index=False)

    print(f"✅ Saved Matrix: {workload_matrix.shape}")
    print(f"✅ Saved Meta: {OUTPUT_META}")
    print("\nTop 5 Selected Services:")
    print(top_services[['Service_ID', 'Trigger', 'total_invocations']].head())


if __name__ == "__main__":
    process_trace()
