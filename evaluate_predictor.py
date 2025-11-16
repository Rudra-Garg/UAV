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
from config import PREDICTION_SEQUENCE_LENGTH, DEVICE, NUM_SERVICE_TYPES

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

    sequences, labels = [], []
    service_col = test_df['service'].values
    for i in range(len(test_df) - sequence_length):
        sequences.append(service_col[i:i + sequence_length])
        labels.append(service_col[i + sequence_length])

    X_test = torch.LongTensor(np.array(sequences))
    y_test = torch.LongTensor(labels)

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        # Evaluate in batches to avoid running out of memory
        for i in tqdm(range(0, len(X_test), 512), desc="Evaluating Accuracy"):
            seq_batch = X_test[i:i + 512].to(DEVICE)
            label_batch = y_test[i:i + 512]

            service_preds, _ = model(seq_batch)

            # --- Top-1 Accuracy ---
            _, predicted_top1 = torch.max(service_preds.cpu(), 1)
            top1_correct += (predicted_top1 == label_batch).sum().item()

            # --- Top-K Accuracy ---
            _, predicted_topk = torch.topk(service_preds.cpu(), k=5, dim=1)
            # Check if the true label is within the top 3 or top 5 predictions
            for j in range(len(label_batch)):
                true_label = label_batch[j]
                if true_label in predicted_topk[j, :3]:
                    top3_correct += 1
                if true_label in predicted_topk[j, :5]:
                    top5_correct += 1

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

    model = LSTMCachePredictor().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{MODEL_FILE}'.")
        print("Please run 'train_cache_predictor.py' first.")
        return

    calculate_accuracy(model, df, PREDICTION_SEQUENCE_LENGTH)


if __name__ == "__main__":
    main()
