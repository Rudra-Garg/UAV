"""
Phase 2: Offline Training
Loads the generated task request data and trains the LSTM Cache Predictor model.
The final trained model weights are saved to a file.
"""
import pandas as pd
import torch
import multiprocessing as mp
from cache_predictor import LSTMCachePredictor, train_predictor_from_df
from config import PREDICTION_SEQUENCE_LENGTH, DEVICE

# --- Configuration ---
INPUT_DATA_FILE = "task_request_data.csv"
OUTPUT_MODEL_FILE = "lstm_cache_predictor.pth"
TRAINING_EPOCHS = 5  # Start with  5 epochs. You can increase this if the loss is still decreasing.


def main():
    print(f"--- Phase 2: Starting Offline Model Training on {DEVICE} ---")

    # 1. Load the dataset
    try:
        df = pd.read_csv(INPUT_DATA_FILE)
    except FileNotFoundError:
        print(f"ERROR: Data file not found at '{INPUT_DATA_FILE}'.")
        print("Please run 'create_task_dataset.py' first.")
        return

    print(f"Loaded {len(df):,} records from '{INPUT_DATA_FILE}'.")

    # 2. Initialize the model
    predictor_model = LSTMCachePredictor().to(DEVICE)

    # 3. Train the model using the function from cache_predictor.py
    train_predictor_from_df(predictor_model, df,
                            sequence_length=PREDICTION_SEQUENCE_LENGTH,
                            epochs=TRAINING_EPOCHS)

    # 4. Save the trained model's state dictionary
    torch.save(predictor_model.state_dict(), OUTPUT_MODEL_FILE)
    print(f"\n✅ Model training complete. Weights saved to '{OUTPUT_MODEL_FILE}'.")


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        pass

    main()
