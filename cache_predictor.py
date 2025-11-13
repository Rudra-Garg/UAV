# cache_predictor.py
import multiprocessing as mp
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from config import NUM_SERVICE_TYPES, NUM_CONTENT_TYPES, DEVICE


# --- 1. The LSTM Model Architecture (Unchanged) ---
class LSTMCachePredictor(nn.Module):
    def __init__(self, embedding_dim=32, lstm_hidden_dim=64, num_lstm_layers=2):
        super(LSTMCachePredictor, self).__init__()
        self.service_embedding = nn.Embedding(NUM_SERVICE_TYPES + 1, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_lstm_layers, batch_first=True, dropout=0.2)
        self.service_output = nn.Linear(lstm_hidden_dim, NUM_SERVICE_TYPES)
        self.content_output = nn.Linear(lstm_hidden_dim, NUM_CONTENT_TYPES)

    def forward(self, service_seq):
        embedded_seq = self.service_embedding(service_seq)
        lstm_out, _ = self.lstm(embedded_seq)
        last_time_step_out = lstm_out[:, -1, :]
        service_preds = self.service_output(last_time_step_out)
        content_preds = self.content_output(last_time_step_out)
        return service_preds, content_preds


# --- 2. NEW: Parallel Function for Data Preparation ---
def _create_sequences_chunk(args):
    """Helper function to process a chunk of data in a separate process."""
    service_col, sequence_length, start_idx, end_idx = args
    sequences, s_labels = [], []
    # The loop now only runs on a small chunk of the data
    for i in range(start_idx, end_idx - sequence_length):
        sequences.append(service_col[i:i + sequence_length])
        s_labels.append(service_col[i + sequence_length])
    return sequences, s_labels


# --- 3. UPDATED: The Training Function for Offline Use ---
def train_predictor_from_df(model, df, sequence_length=10, epochs=10):
    """
    Trains the predictor model from a DataFrame, now with parallel data preparation.
    """
    if len(df) < sequence_length * 10:
        print("Warning: Dataset is too small for effective training.")
        return

    # --- OPTIMIZATION 1: Parallel Data Preparation ---
    print("Preparing data sequences in parallel...")
    service_col = df['service'].values
    num_processes = max(1, mp.cpu_count() - 2)  # Leave a couple of cores free
    chunk_size = len(service_col) // num_processes

    # Create a list of arguments for each process
    tasks = [(service_col, sequence_length, i * chunk_size, (i + 1) * chunk_size) for i in range(num_processes)]
    # Ensure the last chunk goes to the end of the array
    tasks[-1] = (service_col, sequence_length, (num_processes - 1) * chunk_size, len(service_col))

    sequences, s_labels = [], []
    with mp.Pool(processes=num_processes) as pool:
        # Use tqdm to show progress for the parallel processing
        results = list(tqdm(pool.imap(_create_sequences_chunk, tasks), total=len(tasks), desc="Processing Chunks"))

    # Combine results from all processes
    for res_seq, res_labels in results:
        sequences.extend(res_seq)
        s_labels.extend(res_labels)

    X = torch.LongTensor(np.array(sequences)).to(DEVICE)
    y_service = torch.LongTensor(s_labels).to(DEVICE)

    dataset = TensorDataset(X, y_service)

    # --- OPTIMIZATION 2: Parallel Data Loading ---
    # Set num_workers > 0 to use subprocesses for data loading.
    # This feeds data to the GPU without blocking the main training loop.
    num_workers = min(os.cpu_count(), 8)  # Use up to 8 workers or all available CPUs
    print(f"Using {num_workers} workers for data loading.")
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=num_workers, pin_memory=True)

    # --- Training ---
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("\nStarting model training...")
    # --- OPTIMIZATION 3: Check GPU Availability ---
    # The DEVICE variable from config.py should automatically handle this.
    # We just confirm it's being used.
    print(f"Training on device: {DEVICE}")

    for epoch in range(epochs):
        epoch_loss = 0.0
        # The tqdm progress bar now wraps the DataLoader
        for seq_batch, service_label_batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            # Data is already on the correct device thanks to the loader and .to(DEVICE) on the Tensors
            optimizer.zero_grad()
            service_preds, _ = model(seq_batch)
            loss = criterion(service_preds, service_label_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1} complete. Average Loss: {avg_loss:.4f}")

    model.eval()
