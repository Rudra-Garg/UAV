# cache_predictor.py
import multiprocessing as mp
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from config import NUM_SERVICE_TYPES, NUM_CONTENT_TYPES, NUM_ZONES, DEVICE


# --- UPDATED MODEL ARCHITECTURE ---
class LSTMCachePredictor(nn.Module):
    def __init__(self, embedding_dim=64, zone_emb_dim=16, lstm_hidden_dim=128, num_lstm_layers=2):
        super(LSTMCachePredictor, self).__init__()

        # 1. Service Embedding
        self.service_embedding = nn.Embedding(NUM_SERVICE_TYPES + 1, embedding_dim)

        # 2. Zone Embedding (NEW) - Gives context to the request
        self.zone_embedding = nn.Embedding(NUM_ZONES + 1, zone_emb_dim)

        # Input to LSTM is concatenated embeddings
        input_dim = embedding_dim + zone_emb_dim

        self.lstm = nn.LSTM(
            input_dim,
            lstm_hidden_dim,
            num_lstm_layers,
            batch_first=True,
            dropout=0.2
        )

        self.service_output = nn.Linear(lstm_hidden_dim, NUM_SERVICE_TYPES)
        self.content_output = nn.Linear(lstm_hidden_dim, NUM_CONTENT_TYPES)

    def forward(self, service_seq, zone_seq):
        # service_seq: [batch, seq_len]
        # zone_seq:    [batch, seq_len]

        serv_emb = self.service_embedding(service_seq)  # [batch, seq, 64]
        zone_emb = self.zone_embedding(zone_seq)  # [batch, seq, 16]

        # Concatenate: [batch, seq, 80]
        combined_input = torch.cat((serv_emb, zone_emb), dim=2)

        lstm_out, _ = self.lstm(combined_input)

        # We only care about the last time step for prediction
        last_time_step_out = lstm_out[:, -1, :]

        service_preds = self.service_output(last_time_step_out)
        content_preds = self.content_output(last_time_step_out)

        return service_preds, content_preds


def _create_sequences_chunk(args):
    """Helper: Processes Service AND Zone columns."""
    service_col, zone_col, sequence_length, start_idx, end_idx = args
    seq_s, seq_z, labels = [], [], []

    for i in range(start_idx, end_idx - sequence_length):
        seq_s.append(service_col[i: i + sequence_length])
        seq_z.append(zone_col[i: i + sequence_length])
        labels.append(service_col[i + sequence_length])

    return seq_s, seq_z, labels


def train_predictor_from_df(model, df, sequence_length=10, epochs=10):
    """Trains the predictor using both Service and Zone data."""

    if len(df) < sequence_length * 10:
        print("Warning: Dataset too small.")
        return

    print("Preparing data sequences (Service + Zone)...")
    service_col = df['service'].values
    zone_col = df['zone'].values

    num_processes = max(1, mp.cpu_count() - 2)
    chunk_size = len(service_col) // num_processes

    # Pack arguments (Now including zone_col)
    tasks = [(service_col, zone_col, sequence_length, i * chunk_size, (i + 1) * chunk_size)
             for i in range(num_processes)]
    # Fix last chunk
    tasks[-1] = (service_col, zone_col, sequence_length, (num_processes - 1) * chunk_size, len(service_col))

    all_seq_s, all_seq_z, all_labels = [], [], []

    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(_create_sequences_chunk, tasks), total=len(tasks), desc="Processing Chunks"))

    for s, z, l in results:
        all_seq_s.extend(s)
        all_seq_z.extend(z)
        all_labels.extend(l)

    # Convert to Tensors
    X_s = torch.LongTensor(np.array(all_seq_s))
    X_z = torch.LongTensor(np.array(all_seq_z))
    y = torch.LongTensor(np.array(all_labels))

    dataset = TensorDataset(X_s, X_z, y)

    num_workers = min(os.cpu_count(), 4)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=num_workers, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    print(f"\nStarting training on {DEVICE}...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        # Updated loop to unpack 3 values from loader
        for batch_s, batch_z, batch_y in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            batch_s = batch_s.to(DEVICE)
            batch_z = batch_z.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass with TWO inputs
            service_preds, _ = model(batch_s, batch_z)

            loss = criterion(service_preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1} Avg Loss: {epoch_loss / len(loader):.4f}")

    model.eval()
