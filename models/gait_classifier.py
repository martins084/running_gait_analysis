"""
gait_classifier.py — Gait phase classifier, LSTM/TCN autoencoders, supervised injury classifier.

Model inventory
---------------
GaitPhaseClassifier       CNN-LSTM, per-frame stance/swing/push labels  (seq-to-seq)
GaitAnomalyDetector       Encoder-decoder LSTM autoencoder              (V1 baseline)
GaitAnomalyDetectorV2     BiLSTM + temporal attention autoencoder       (V2 — improved)
GaitAnomalyDetectorV3     Dilated TCN autoencoder                       (V3 — strongest unsupervised)
GaitInjuryClassifier      BiGRU + attention, supervised binary output   (best when labels available)

Input convention (all models): (batch, seq_len, feature_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gait Phase Classifier (seq-to-seq)
# ---------------------------------------------------------------------------

class GaitPhaseClassifier(nn.Module):
    """
    1D-CNN feature extractor + LSTM temporal model → per-frame phase logits.

    Returns per-frame logits: (batch, seq_len, num_phases).
    Phases default: 0=stance, 1=swing, 2=push.
    """

    def __init__(
        self,
        input_size: int = 66,
        hidden_size: int = 128,
        num_phases: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(64)
        self.norm2 = nn.BatchNorm1d(128)
        self.lstm = nn.LSTM(128, hidden_size, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, num_phases)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        h = x.transpose(1, 2)                    # (B, F, T)
        h = F.relu(self.norm1(self.conv1(h)))
        h = F.relu(self.norm2(self.conv2(h)))
        h = h.transpose(1, 2)                    # (B, T, 128)
        lstm_out, _ = self.lstm(h)               # (B, T, hidden*2) — bidirectional
        h = F.relu(self.fc1(lstm_out))           # (B, T, 64)
        h = self.dropout(h)
        return self.fc2(h)                       # (B, T, num_phases) — per-frame logits


# ---------------------------------------------------------------------------
# V1: Baseline encoder-decoder LSTM autoencoder
# ---------------------------------------------------------------------------

class GaitAnomalyDetector(nn.Module):
    """
    Encoder–decoder LSTM: reconstruction error = anomaly score.
    V1 baseline — kept for reproducibility of earlier runs.
    """

    def __init__(self, input_size: int = 66, hidden_size: int = 128):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_size, input_size, batch_first=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, _c_n) = self.encoder_lstm(x)
        dec_in = h_n[-1].unsqueeze(1).expand(-1, x.size(1), -1)
        decoded, _ = self.decoder_lstm(dec_in)
        return decoded, h_n[-1]


# ---------------------------------------------------------------------------
# V2: BiLSTM + temporal attention autoencoder (improved)
# ---------------------------------------------------------------------------

class GaitAnomalyDetectorV2(nn.Module):
    """
    Bidirectional multi-layer LSTM encoder → temporal attention bottleneck →
    bidirectional LSTM decoder initialised from projected context.

    Improvements over V1:
    - Bidirectional encoder captures past + future context.
    - Temporal attention learns which frames carry the most diagnostic signal.
    - Decoder hidden state initialised from bottleneck (not zero).
    - Input LayerNorm for stable training.
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

        self.input_norm = nn.LayerNorm(input_size)

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
        # Project context to initialise decoder (h0, c0).
        self._dec_h_dim = num_layers * self.num_directions * hidden_size
        self.context_to_h0 = nn.Linear(self.enc_out_dim, self._dec_h_dim)
        self.context_to_c0 = nn.Linear(self.enc_out_dim, self._dec_h_dim)

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
        B, T, _ = x.shape
        x = self.input_norm(x)

        enc_seq, _ = self.encoder_lstm(x)        # [B, T, enc_out_dim]

        # Temporal attention → context vector.
        attn_w = torch.softmax(self.attn_score(enc_seq).squeeze(-1), dim=1)   # [B, T]
        context = (enc_seq * attn_w.unsqueeze(-1)).sum(dim=1)                 # [B, enc_out_dim]
        context = self.context_proj(context)

        # Initialise decoder hidden/cell from context.
        h0 = self.context_to_h0(context)                                        # [B, h_dim]
        c0 = self.context_to_c0(context)
        h0 = h0.view(B, self.num_layers * self.num_directions, self.hidden_size).transpose(0, 1).contiguous()
        c0 = c0.view(B, self.num_layers * self.num_directions, self.hidden_size).transpose(0, 1).contiguous()

        dec_in = context.unsqueeze(1).expand(-1, T, -1)                         # [B, T, enc_out_dim]
        dec_seq, _ = self.decoder_lstm(dec_in, (h0, c0))                        # [B, T, enc_out_dim]
        decoded = self.output_proj(dec_seq)                                      # [B, T, input_size]
        return decoded, context


# ---------------------------------------------------------------------------
# V3: Dilated TCN autoencoder
# ---------------------------------------------------------------------------

class _TCNBlock(nn.Module):
    """Residual dilated 1-D convolution block with LayerNorm + GELU."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if causal:
            self._pad = (kernel_size - 1) * dilation
        else:
            self._pad = (kernel_size - 1) * dilation // 2
        self.causal = causal
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=self._pad, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        h = self.conv(x)
        if self.causal:
            h = h[:, :, : x.size(2)]
        h = h.transpose(1, 2)       # [B, T, C]
        h = self.norm(h)
        h = h.transpose(1, 2)       # [B, C, T]
        h = F.gelu(h)
        h = self.drop(h)
        return x + h                # residual


class GaitAnomalyDetectorV3(nn.Module):
    """
    Dilated causal TCN encoder → global-average-pool bottleneck →
    non-causal TCN decoder → reconstruction.

    Advantages over LSTM:
    - Parallelisable (no sequential dependency).
    - Exponentially growing receptive field covers long strides efficiently.
    - No vanishing gradient through time.
    """

    def __init__(
        self,
        input_size: int = 66,
        channels: int = 128,
        num_layers: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.2,
        bottleneck_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Linear(input_size, channels)

        self.encoder = nn.ModuleList([
            _TCNBlock(channels, kernel_size, dilation=2 ** i, dropout=dropout, causal=True)
            for i in range(num_layers)
        ])
        self.bottleneck = nn.Sequential(
            nn.Linear(channels, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, channels),
        )
        self.decoder = nn.ModuleList([
            _TCNBlock(channels, kernel_size, dilation=2 ** i, dropout=dropout, causal=False)
            for i in range(num_layers)
        ])
        self.output_proj = nn.Linear(channels, input_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        x = self.input_norm(x)
        h = self.input_proj(x).transpose(1, 2)    # [B, channels, T]

        for block in self.encoder:
            h = block(h)

        ctx = h.mean(dim=2)                        # [B, channels]  global-avg-pool
        ctx = self.bottleneck(ctx)                 # [B, channels]

        h = ctx.unsqueeze(2).expand(-1, -1, T)    # [B, channels, T]
        for block in self.decoder:
            h = block(h)

        decoded = self.output_proj(h.transpose(1, 2))   # [B, T, input_size]
        return decoded, ctx


# ---------------------------------------------------------------------------
# Supervised binary injury classifier
# ---------------------------------------------------------------------------

class GaitInjuryClassifier(nn.Module):
    """
    Bidirectional GRU encoder + temporal attention → binary injury logit.

    Use this when you have injury labels — it directly optimises AUROC
    and will outperform unsupervised reconstruction-based anomaly models.

    Input : (batch, seq_len, input_size)
    Output: (batch,)  scalar logit (pre-sigmoid); use BCEWithLogitsLoss.
    """

    def __init__(
        self,
        input_size: int = 66,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        enc_dim = hidden_size * 2          # bidirectional
        self.attn = nn.Linear(enc_dim, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(enc_dim),
            nn.Linear(enc_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        seq, _ = self.gru(x)                                          # [B, T, enc_dim]
        w = torch.softmax(self.attn(seq).squeeze(-1), dim=1)          # [B, T]
        ctx = (seq * w.unsqueeze(-1)).sum(dim=1)                      # [B, enc_dim]
        return self.head(ctx).squeeze(-1)                             # [B]


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_anomaly_model(
    variant: str,
    input_size: int,
    hidden_size: int,
    num_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = True,
) -> nn.Module:
    """Return an anomaly autoencoder by variant name."""
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
    if v in {"v3", "tcn"}:
        return GaitAnomalyDetectorV3(
            input_size=input_size,
            channels=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unknown anomaly model variant: {variant!r}. Choose v1, v2, or v3.")


def build_classifier_model(
    input_size: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
) -> GaitInjuryClassifier:
    """Return a supervised binary injury classifier."""
    return GaitInjuryClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
