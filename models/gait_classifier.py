"""
gait_classifier.py — CNN-LSTM gait phase classifier and LSTM autoencoder.

Phase 4: train on pose sequences with labels (contact / swing / push-off).
Until labels exist, these modules are imported by train scripts only.

Input convention: (batch, seq_len, 66) = 33 joints × 2 (x, y) normalized coords.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaitPhaseClassifier(nn.Module):
    """
    Temporal CNN + LSTM for gait phase classification.

    Default 3 classes: stance / swing / push (adjust num_phases to match labels).
    """

    def __init__(
        self,
        input_size: int = 66,
        hidden_size: int = 128,
        num_phases: int = 3,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(128, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_phases)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq, features) -> conv expects (batch, features, seq)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)
        lstm_out, (h_n, _c_n) = self.lstm(x)
        x = h_n[-1]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits, lstm_out


class GaitAnomalyDetector(nn.Module):
    """
    Encoder–decoder LSTM: reconstruction error = anomaly score for unusual gait.
    """

    def __init__(self, input_size: int = 66, hidden_size: int = 128):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_size, input_size, batch_first=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, c_n) = self.encoder_lstm(x)
        # Repeat hidden across time for a simple decode (expand to seq length)
        dec_in = h_n[-1].unsqueeze(1).expand(-1, x.size(1), -1)
        # Do not pass encoder hidden state directly into decoder state:
        # decoder hidden size is `input_size`, while encoder hidden is `hidden_size`.
        decoded, _ = self.decoder_lstm(dec_in)
        return decoded, h_n[-1]


class GaitAnomalyDetectorV2(nn.Module):
    """
    Stronger anomaly model variant:
    - Bidirectional multi-layer LSTM encoder
    - Temporal attention over encoded sequence
    - Decoder with residual connection to input
    """

    def __init__(
        self,
        input_size: int = 66,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.enc_out_dim = hidden_size * self.num_directions

        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.attn_score = nn.Linear(self.enc_out_dim, 1)
        self.context_proj = nn.Sequential(
            nn.Linear(self.enc_out_dim, self.enc_out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.decoder_lstm = nn.LSTM(
            input_size=self.enc_out_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.output_proj = nn.Linear(self.enc_out_dim, input_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enc_seq, _ = self.encoder_lstm(x)  # [B, T, enc_out_dim]

        # Attention over encoder time axis to build one global context vector.
        attn_logits = self.attn_score(enc_seq).squeeze(-1)  # [B, T]
        attn_w = torch.softmax(attn_logits, dim=1)
        context = torch.sum(enc_seq * attn_w.unsqueeze(-1), dim=1)  # [B, enc_out_dim]
        context = self.context_proj(context)

        # Repeat context through time and decode sequence.
        dec_in = context.unsqueeze(1).expand(-1, x.size(1), -1)
        dec_seq, _ = self.decoder_lstm(dec_in)  # [B, T, enc_out_dim]
        decoded = self.output_proj(dec_seq)  # [B, T, input_size]

        # Residual skip keeps baseline identity mapping path available.
        decoded = decoded + x
        return decoded, context


def build_anomaly_model(
    variant: str,
    input_size: int,
    hidden_size: int,
    num_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = True,
) -> nn.Module:
    """Factory for anomaly model variants used by train/eval scripts."""
    v = (variant or "baseline").lower()
    if v in {"baseline", "v1"}:
        return GaitAnomalyDetector(input_size=input_size, hidden_size=hidden_size)
    if v in {"v2", "advanced", "bilstm_attn"}:
        return GaitAnomalyDetectorV2(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
    raise ValueError(f"Unknown anomaly model variant: {variant}")
