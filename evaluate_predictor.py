# evaluate_predictor.py
"""
Tests the performance of the trained LSTM cache predictor by measuring
its Top-K accuracy on a held-out test set.
"""
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np

from cache_predictor import LSTMCachePredictor
from config import PREDICTION_SEQUENCE_LENGTH, DEVICE, NUM_SERVICE_TYPES, NUM_ZONES

# --- Configuration ---
MODEL_FILE = "lstm_cache_predictor.pth"
DATA_FILE = "task_request_data.csv"
TEST_SET_SIZE = 0.2  # Use 20% of the data for testing


def calculate_accuracy(model, df, sequence_length):
    """Calculates Top-1, Top-3, and Top-5 accuracy."""
    print("\n--- Evaluating Predictor Performance ---")
    model.eval()

    # --- Prepare Test Data ---
    # Take the last 20% of the data as our test set
    split_index = int(len(df) * (1 - TEST_SET_SIZE))
    test_df = df.iloc[split_index:]

    sequences_s, sequences_z, labels = [], [], []
    service_col = test_df['service'].values

    # Check if zone exists (backward compatibility)
    if 'zone' in test_df.columns:
        zone_col = test_df['zone'].values
    else:
        print("⚠️ Warning: 'zone' column missing. Filling with zeros.")
        zone_col = np.zeros_like(service_col)

    print(f"Generating test sequences from {len(test_df):,} records...")

    # Create sequences (non-vectorized for clarity/safety on test set)
    # Note: For very large test sets, this might be slow, but safe for 400k records.
    for i in tqdm(range(len(test_df) - sequence_length), desc="Building Test Set"):
        sequences_s.append(service_col[i:i + sequence_length])
        sequences_z.append(zone_col[i:i + sequence_length])
        labels.append(service_col[i + sequence_length])

    X_test_s = torch.LongTensor(np.array(sequences_s))
    X_test_z = torch.LongTensor(np.array(sequences_z))
    y_test = torch.LongTensor(np.array(labels))

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total = 0

    batch_size = 1024

    with torch.no_grad():
        # Evaluate in batches to avoid running out of memory
        num_batches = (len(X_test_s) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(X_test_s), batch_size), total=num_batches, desc="Evaluating"):
            seq_batch_s = X_test_s[i:i + batch_size].to(DEVICE)
            seq_batch_z = X_test_z[i:i + batch_size].to(DEVICE)
            label_batch = y_test[i:i + batch_size].to(DEVICE)

            # Pass BOTH inputs to the model
            service_preds, _ = model(seq_batch_s, seq_batch_z)

            # --- Top-1 Accuracy ---
            _, predicted_top1 = torch.max(service_preds, 1)
            top1_correct += (predicted_top1 == label_batch).sum().item()

            # --- Top-K Accuracy ---
            # Get top 5 indices: [batch_size, 5]
            _, predicted_topk = torch.topk(service_preds, k=5, dim=1)

            # Expand labels to [batch_size, 1] for broadcasting
            label_batch_expanded = label_batch.view(-1, 1)

            # Check matches
            matches = (predicted_topk == label_batch_expanded)  # [batch, 5] boolean

            top3_correct += matches[:, :3].any(dim=1).sum().item()
            top5_correct += matches[:, :5].any(dim=1).sum().item()

            total += len(label_batch)

    print("\n--- Evaluation Results ---")
    print(f"Total test samples: {total:,}")
    print(f"Top-1 Accuracy: {100 * top1_correct / total:.2f}%")
    print(f"Top-3 Accuracy: {100 * top3_correct / total:.2f}%")
    print(f"Top-5 Accuracy: {100 * top5_correct / total:.2f}%")


def main():
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"ERROR: Data file not found at '{DATA_FILE}'.")
        return

    # Initialize model with new architecture parameters
    model = LSTMCachePredictor(
        embedding_dim=64,
        zone_emb_dim=16,
        lstm_hidden_dim=128
    ).to(DEVICE)

    try:
        model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
        print(f"Loaded model from {MODEL_FILE}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{MODEL_FILE}'.")
        print("Please run 'train_cache_predictor.py' first.")
        return
    except RuntimeError as e:
        print(f"ERROR: Model architecture mismatch. {e}")
        return

    calculate_accuracy(model, df, PREDICTION_SEQUENCE_LENGTH)


if __name__ == "__main__":
    main()
