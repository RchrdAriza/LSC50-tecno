"""Temporal sign classifier for landmark tracks.

A GRU reads the per-frame landmark vector across time and predicts the sign
class from the final hidden state. Supports concatenating multiple landmark
sets (body, hands, face) by passing a multi-column input.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LandmarkGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        rnn_out = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_out, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x: (N, T, D) padded. lengths: (N,) valid timesteps per sample."""
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        if self.rnn.bidirectional:
            # hidden: (num_layers * 2, N, H) -> take last layer, both directions
            last = hidden[-2:]  # (2, N, H)
            out = torch.cat([last[0], last[1]], dim=-1)  # (N, 2H)
        else:
            out = hidden[-1]  # (N, H)
        return self.head(out)


def build_model(input_dim: int, num_classes: int, **kwargs) -> LandmarkGRU:
    return LandmarkGRU(input_dim=input_dim, num_classes=num_classes, **kwargs)