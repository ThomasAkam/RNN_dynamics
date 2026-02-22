from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class TinyGRUPredictor(nn.Module):
    """
    Optional pre-MLP -> GRU(hidden_units) -> Linear(obs)
    Predicts y_{t+1} from y_{1:t} with one-step-ahead supervision.
    """

    def __init__(self, obs_dim: int, hidden_units: int, pre_mlp_dim: int | None = None):
        super().__init__()
        self.pre_mlp_dim = pre_mlp_dim
        if pre_mlp_dim is None:
            self.input_map = nn.Linear(obs_dim, hidden_units)
            self.pre_mlp = None
        else:
            self.pre_mlp = nn.Sequential(
                nn.Linear(obs_dim, pre_mlp_dim),
                nn.ReLU(),
                nn.Linear(pre_mlp_dim, hidden_units),
            )
            self.input_map = None

        self.gru = nn.GRU(
            input_size=hidden_units,
            hidden_size=hidden_units,
            num_layers=1,
            batch_first=True,
        )
        self.output_map = nn.Linear(hidden_units, obs_dim)

    def forward(self, obs_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pre_mlp is None:
            x = self.input_map(obs_seq)
        else:
            x = self.pre_mlp(obs_seq)
        h, _ = self.gru(x)
        y_hat = self.output_map(h)
        return y_hat, h
