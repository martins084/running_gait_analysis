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
        decoded, _ = self.decoder_lstm(dec_in, (h_n, c_n))
        return decoded, h_n[-1]
