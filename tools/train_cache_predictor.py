import multiprocessing as mp

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
from config import *
from prediction import LSTMCachePredictor


os.makedirs("models", exist_ok=True)
# --- Configuration ---
INPUT_DATA_FILE = TASK_REQUEST_DATA_FILE
OUTPUT_MODEL_FILE = os.path.join("models", "lstm_cache_predictor.pth")
TRAINING_EPOCHS = 20  # Increased epochs
BATCH_SIZE = 1024


def validate_dataset(df):
    required_cols = ['service', 'content', 'zone']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ ERROR: CSV missing columns. Found {df.columns}, expected {required_cols}")
        return False
    if len(df) < 500:
        print("❌ ERROR: Dataset too small.")
        return False
    return True


def _create_sequences_chunk(args):
    """Helper: Processes Service AND Zone columns."""
    service_col, zone_col, sequence_length, start_idx, end_idx = args
    seq_s, seq_z, labels = [], [], []

    # Safety check for index bounds
    end_loop = min(end_idx - sequence_length, len(service_col) - sequence_length)
    if start_idx >= end_loop:
        return [], [], []

    for i in range(start_idx, end_loop):
        seq_s.append(service_col[i: i + sequence_length])
        seq_z.append(zone_col[i: i + sequence_length])
        labels.append(service_col[i + sequence_length])

    return seq_s, seq_z, labels


def train_predictor_from_df(model, df, sequence_length=10, epochs=10, batch_size=512):
    print(f"Preparing data sequences (Service + Zone) from {len(df):,} records...")

    service_col = df['service'].values
    zone_col = df['zone'].values

    # Use fewer processes to avoid overhead on smaller systems, but at least 2
    num_processes = max(2, min(mp.cpu_count() - 1, 8))
    chunk_size = len(service_col) // num_processes

    tasks = []
    for i in range(num_processes):
        start = i * chunk_size
        # For the last chunk, go to the end
        end = len(service_col) if i == num_processes - 1 else (i + 1) * chunk_size
        # IMPORTANT: Ensure overlap logic if strict sequencing matters, 
        # but for training data generation, missing boundary seqs is fine.
        tasks.append((service_col, zone_col, sequence_length, start, end))

    all_seq_s, all_seq_z, all_labels = [], [], []

    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(_create_sequences_chunk, tasks), total=len(tasks), desc="Processing Chunks"))

    for s, z, l in results:
        all_seq_s.extend(s)
        all_seq_z.extend(z)
        all_labels.extend(l)

    print(f"Generated {len(all_labels):,} training sequences.")

    # Create Tensors
    X_s = torch.LongTensor(np.array(all_seq_s))
    X_z = torch.LongTensor(np.array(all_seq_z))
    y = torch.LongTensor(np.array(all_labels))

    dataset = TensorDataset(X_s, X_z, y)

    num_workers = min(os.cpu_count(), 4)
    # pinned memory helps transfer to GPU faster
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    # --- Optimizer & Scheduler ---
    optimizer = optim.Adam(model.parameters(), lr=0.002)  # Slightly higher start
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    print(f"\nStarting training on {DEVICE}...")

    for epoch in range(epochs):
        epoch_loss = 0.0

        # Progress bar
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_s, batch_z, batch_y in pbar:
            batch_s = batch_s.to(DEVICE)
            batch_z = batch_z.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            service_preds, _ = model(batch_s, batch_z)

            loss = criterion(service_preds, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1} Avg Loss: {avg_loss:.4f}")

        # Step the scheduler
        scheduler.step(avg_loss)

    model.eval()


def main():
    print(f"--- Phase 2: Starting Offline Model Training on {DEVICE} ---")

    if not os.path.exists(INPUT_DATA_FILE):
        print(f"❌ ERROR: {INPUT_DATA_FILE} not found.")
        return

    df = pd.read_csv(INPUT_DATA_FILE)
    if not validate_dataset(df):
        return

    # Re-initialize model to ensure clean slate
    predictor_model = LSTMCachePredictor(
        embedding_dim=64,
        zone_emb_dim=16,
        lstm_hidden_dim=128
    ).to(DEVICE)

    train_predictor_from_df(
        predictor_model,
        df,
        sequence_length=10,  # Keep short for efficiency
        epochs=TRAINING_EPOCHS,
        batch_size=BATCH_SIZE
    )

    torch.save(predictor_model.state_dict(), OUTPUT_MODEL_FILE)
    print(f"\n✅ Weights saved to '{OUTPUT_MODEL_FILE}'.")


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
