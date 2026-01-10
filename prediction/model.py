# cache_predictor.py
import torch
import torch.nn as nn

from config import NUM_SERVICE_TYPES, NUM_CONTENT_TYPES, NUM_ZONES


# --- MODEL ARCHITECTURE ---
class LSTMCachePredictor(nn.Module):
    def __init__(self, embedding_dim=64, zone_emb_dim=16, lstm_hidden_dim=128, num_lstm_layers=2):
        super(LSTMCachePredictor, self).__init__()

        # 1. Service Embedding
        self.service_embedding = nn.Embedding(NUM_SERVICE_TYPES + 1, embedding_dim)

        # 2. Zone Embedding (Context)
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

    def predict_top_k(self, recent_requests, zone_id, k_services=5, k_content=5):
        """
        Predict top-k services and content types based on recent requests.
        
        Args:
            recent_requests: List of (service_id, zone_id) tuples
            zone_id: Current zone ID for context
            k_services: Number of top services to return
            k_content: Number of top content types to return
            
        Returns:
            top_services: List of top-k service IDs
            top_content: List of top-k content IDs
        """
        self.eval()
        with torch.no_grad():
            # Extract service IDs and zone IDs from recent requests
            service_ids = [req[0] for req in recent_requests]
            zone_ids = [req[1] for req in recent_requests]
            
            # Convert to tensors (batch size = 1)
            service_seq = torch.tensor([service_ids], dtype=torch.long)
            zone_seq = torch.tensor([zone_ids], dtype=torch.long)
            
            # Get predictions
            service_preds, content_preds = self.forward(service_seq, zone_seq)
            
            # Get top-k services
            _, top_service_indices = torch.topk(service_preds[0], min(k_services, service_preds.size(1)))
            top_services = top_service_indices.tolist()
            
            # Get top-k content
            _, top_content_indices = torch.topk(content_preds[0], min(k_content, content_preds.size(1)))
            top_content = top_content_indices.tolist()
            
        return top_services, top_content


# --- Helper functions for data processing ---
def _create_sequences_chunk(args):
    """Helper: Processes Service AND Zone columns."""
    service_col, zone_col, sequence_length, start_idx, end_idx = args
    seq_s, seq_z, labels = [], [], []

    for i in range(start_idx, end_idx - sequence_length):
        seq_s.append(service_col[i: i + sequence_length])
        seq_z.append(zone_col[i: i + sequence_length])
        labels.append(service_col[i + sequence_length])

    return seq_s, seq_z, labels
