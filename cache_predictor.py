import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from collections import Counter

from config import NUM_SERVICE_TYPES, NUM_CONTENT_TYPES, DEVICE
from typing import Tuple


class LSTMCachePredictor(nn.Module):
    """Optimized LSTM Cache Predictor"""

    def __init__(self, embedding_dim: int = 64, lstm_hidden_dim: int = 128, num_lstm_layers: int = 2):
        super(LSTMCachePredictor, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers

        self.service_embedding = nn.Embedding(NUM_SERVICE_TYPES + 1, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim,
            lstm_hidden_dim,
            num_lstm_layers,
            batch_first=True,
            dropout=0.3 if num_lstm_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(0.3)

        self.service_output = nn.Linear(lstm_hidden_dim, NUM_SERVICE_TYPES)
        self.content_output = nn.Linear(lstm_hidden_dim, NUM_CONTENT_TYPES)

    def forward(self, service_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded_seq = self.service_embedding(service_seq)
        lstm_out, (h_n, c_n) = self.lstm(embedded_seq)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)

        service_preds = self.service_output(last_hidden)
        content_preds = self.content_output(last_hidden)

        return service_preds, content_preds


def create_sequences_vectorized(service_col, sequence_length):
    """Vectorized sequence creation"""
    n = len(service_col)
    num_sequences = n - sequence_length

    if num_sequences <= 0:
        return np.array([]), np.array([])

    indices = np.arange(sequence_length)[None, :] + np.arange(num_sequences)[:, None]
    sequences = service_col[indices]
    labels = service_col[sequence_length:]

    return sequences, labels


def analyze_data_distribution(df, service_col, labels):
    """Analyze the dataset to understand why model isn't learning"""
    print("\n" + "=" * 60)
    print("DATA DISTRIBUTION ANALYSIS")
    print("=" * 60)

    # 1. Check label distribution
    label_counts = Counter(labels)
    print(f"\n1. Label Distribution (top 10 most frequent):")
    for label, count in label_counts.most_common(10):
        pct = (count / len(labels)) * 100
        print(f"   Service {label}: {count:,} ({pct:.2f}%)")

    # 2. Calculate baseline accuracy (always predict most common)
    most_common_label = label_counts.most_common(1)[0][0]
    baseline_acc = label_counts[most_common_label] / len(labels)
    print(f"\n2. Baseline Accuracy (always predict most common): {baseline_acc:.4f} ({baseline_acc * 100:.2f}%)")

    # 3. Calculate entropy of distribution
    probs = np.array([count / len(labels) for count in label_counts.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(len(label_counts))
    print(f"\n3. Label Entropy: {entropy:.4f} / {max_entropy:.4f} (max)")
    print(f"   Uniformity: {entropy / max_entropy:.2%} (100% = perfectly uniform)")

    # 4. Check for sequential patterns
    print(f"\n4. Sequential Pattern Check:")
    # Check if next service depends on previous
    transitions = {}
    for i in range(len(service_col) - 1):
        curr = service_col[i]
        next_svc = service_col[i + 1]
        if curr not in transitions:
            transitions[curr] = Counter()
        transitions[curr][next_svc] += 1

    # Calculate average transition entropy
    transition_entropies = []
    for curr, next_counts in transitions.items():
        total = sum(next_counts.values())
        probs = np.array([c / total for c in next_counts.values()])
        ent = -np.sum(probs * np.log2(probs + 1e-10))
        transition_entropies.append(ent)

    avg_transition_entropy = np.mean(transition_entropies)
    print(f"   Average transition entropy: {avg_transition_entropy:.4f}")
    print(f"   (Lower = more predictable, ~{max_entropy:.2f} = random)")

    # 5. Check actual Zipf distribution
    if 'service' in df.columns:
        service_counts = df['service'].value_counts()
        print(f"\n5. Zipf Distribution Check:")
        print(f"   Most common service appears: {service_counts.iloc[0]:,} times")
        print(f"   Least common service appears: {service_counts.iloc[-1]:,} times")
        print(f"   Ratio (should be high for Zipf): {service_counts.iloc[0] / service_counts.iloc[-1]:.2f}x")

    print("=" * 60 + "\n")

    return baseline_acc, entropy, avg_transition_entropy


def train_predictor_from_df(model, df, sequence_length=10, epochs=10, batch_size=512):
    """
    Enhanced training with diagnostics and adaptive learning
    """
    if len(df) < sequence_length * 10:
        print("Warning: Dataset is too small for effective training.")
        return

    print("Preparing data sequences (vectorized)...")
    service_col = df['service'].values

    sequences, s_labels = create_sequences_vectorized(service_col, sequence_length)

    if len(sequences) == 0:
        print("Error: No sequences created.")
        return

    print(f"Created {len(sequences):,} training sequences")

    # DIAGNOSTIC: Analyze data distribution
    baseline_acc, entropy, trans_entropy = analyze_data_distribution(df, service_col, s_labels)

    # Create tensors
    X = torch.from_numpy(sequences).long()
    y_service = torch.from_numpy(s_labels).long()

    # Split into train and validation
    val_size = int(0.1 * len(X))
    train_size = len(X) - val_size

    train_dataset = TensorDataset(X[:train_size], y_service[:train_size])
    val_dataset = TensorDataset(X[train_size:], y_service[train_size:])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Use lower learning rate and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    criterion = nn.CrossEntropyLoss()

    model.train()
    print(f"\nStarting model training on {DEVICE}...")
    print(f"Baseline accuracy to beat: {baseline_acc:.4f} ({baseline_acc * 100:.2f}%)\n")

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for seq_batch, service_label_batch in progress_bar:
            seq_batch = seq_batch.to(DEVICE, non_blocking=True)
            service_label_batch = service_label_batch.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            service_preds, _ = model(seq_batch)
            loss = criterion(service_preds, service_label_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(service_preds, 1)
            total += service_label_batch.size(0)
            correct += (predicted == service_label_batch).sum().item()

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        avg_train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for seq_batch, service_label_batch in val_loader:
                seq_batch = seq_batch.to(DEVICE, non_blocking=True)
                service_label_batch = service_label_batch.to(DEVICE, non_blocking=True)

                service_preds, _ = model(seq_batch)
                loss = criterion(service_preds, service_label_batch)

                val_loss += loss.item()
                _, predicted = torch.max(service_preds, 1)
                val_total += service_label_batch.size(0)
                val_correct += (predicted == service_label_batch).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} ({train_acc * 100:.2f}%)")
        print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} ({val_acc * 100:.2f}%)")
        print(f"  Baseline: {baseline_acc:.4f} ({baseline_acc * 100:.2f}%)")

        if val_acc > baseline_acc:
            print(f"  ✓ Model beats baseline by {(val_acc - baseline_acc) * 100:.2f}%")
        else:
            print(f"  ✗ Model underperforms baseline by {(baseline_acc - val_acc) * 100:.2f}%")

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  ★ New best validation loss!")

    model.eval()
    print("\nTraining complete!")
