"""
Phase 2: Offline Training
Loads the generated task request data (based on Azure Traces) and trains
the LSTM Cache Predictor model.
"""
import multiprocessing as mp
import os

import pandas as pd
import torch

from cache_predictor import LSTMCachePredictor, train_predictor_from_df
from config import PREDICTION_SEQUENCE_LENGTH, DEVICE

# --- Configuration ---
INPUT_DATA_FILE = "task_request_data.csv"
OUTPUT_MODEL_FILE = "lstm_cache_predictor.pth"

# Azure data is complex (bursty + sparse).
# It takes more epochs to converge than synthetic data.
TRAINING_EPOCHS = 15
BATCH_SIZE = 1024  # Increased batch size for faster processing of large trace data


def validate_dataset(df):
    """Sanity checks to ensure the Azure data generation worked correctly."""
    required_cols = ['service', 'content']

    # 1. Check Columns
    if not all(col in df.columns for col in required_cols):
        print(f"❌ ERROR: CSV missing columns. Found {df.columns}, expected {required_cols}")
        return False

    # 2. Check for Empty Data
    if len(df) < PREDICTION_SEQUENCE_LENGTH * 2:
        print(f"❌ ERROR: Dataset too small ({len(df)} rows). Run generate_task_data.py with more requests.")
        return False

    # 3. Check for Diversity (If Variance is 0, something broke in the generator)
    if df['service'].nunique() < 2:
        print("❌ ERROR: 'service' column has no diversity (only 1 type found). Generator logic failed.")
        return False

    return True


def main():
    print(f"--- Phase 2: Starting Offline Model Training on {DEVICE} ---")
    print(f"    Dataset: {INPUT_DATA_FILE}")
    print(f"    Epochs:  {TRAINING_EPOCHS}")

    # 1. Load the dataset
    if not os.path.exists(INPUT_DATA_FILE):
        print(f"❌ ERROR: Data file '{INPUT_DATA_FILE}' not found.")
        print("   Please run 'generate_task_data.py' (with the new Azure logic) first.")
        return

    df = pd.read_csv(INPUT_DATA_FILE)
    print(f"-> Loaded {len(df):,} records.")

    # 2. Validate Data Quality
    if not validate_dataset(df):
        return

    # 3. Initialize the model
    print("-> Initializing LSTM Model...")
    predictor_model = LSTMCachePredictor().to(DEVICE)

    # 4. Train the model
    # Note: The cache_predictor.py logic will handle the service prediction loss.
    # Since Azure data has high temporal correlation, we expect the loss to drop significantly.
    train_predictor_from_df(
        predictor_model,
        df,
        sequence_length=PREDICTION_SEQUENCE_LENGTH,
        epochs=TRAINING_EPOCHS,
        batch_size=BATCH_SIZE
    )

    # 5. Save the trained model
    torch.save(predictor_model.state_dict(), OUTPUT_MODEL_FILE)
    print(f"\n✅ Model training complete.")
    print(f"✅ Weights saved to '{OUTPUT_MODEL_FILE}'.")


if __name__ == "__main__":
    # Essential for PyTorch DataLoader num_workers > 0 on Windows/Linux
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    main()
