"""Local forecasting models. Each model exposes a shared backbone and a per-zone head,
so the federated code can aggregate the backbone and keep heads local (FedPer-style)."""
from __future__ import annotations

import torch
import torch.nn as nn


class GRUForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 pred_len: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F] -> [B, P]
        out, _ = self.encoder(x)
        return self.head(out[:, -1])

    def backbone_state_dict(self) -> dict:
        return self.encoder.state_dict()

    def head_state_dict(self) -> dict:
        return self.head.state_dict()


def build_model(cfg: dict) -> nn.Module:
    d, m = cfg["data"], cfg["model"]
    input_size = 5 if d["use_time_features"] else 1   # traffic + 4 time features
    if m["name"] == "gru":
        return GRUForecaster(
            input_size=input_size,
            hidden_size=m["hidden_size"],
            num_layers=m["num_layers"],
            pred_len=d["pred_len"],
            dropout=m["dropout"],
        )
    raise ValueError(f"Unknown model: {m['name']}")