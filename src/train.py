"""Local training and evaluation. These are the building blocks the federated rounds reuse.

Run the local-only baseline (each zone trains its own model, no federation):
    python -m src.train
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from src.data import build_zone_data, load_config
from src.models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_local(model, loader, epochs, lr, device, global_state=None, mu=0.0) -> float:
    """Train one model on one zone's training loader.

    If global_state is given and mu > 0, a FedProx proximal term keeps the local weights
    close to the global ones. Returns the mean loss of the last epoch.
    """
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.HuberLoss()
    ref = None
    if mu > 0 and global_state is not None:
        ref = {k: v.detach().clone().to(device) for k, v in global_state.items()}

    last = 0.0
    for _ in range(epochs):
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            if ref is not None:
                prox = sum(((p - ref[name]) ** 2).sum()
                           for name, p in model.named_parameters() if name in ref)
                loss = loss + (mu / 2) * prox
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
            n += len(x)
        last = total / max(n, 1)
    return last


@torch.no_grad()
def evaluate(model, loader, scaler, device) -> dict:
    """MAE, RMSE and WMAPE in the original traffic units."""
    model.to(device).eval()
    preds, trues = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).cpu().numpy())
        trues.append(y.numpy())
    p = scaler.inverse(np.concatenate(preds))
    t = scaler.inverse(np.concatenate(trues))
    err = p - t
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "wmape": float(np.abs(err).sum() / (np.abs(t).sum() + 1e-8)),
    }


def run_local_only(cfg: dict, zones: dict | None = None, device=None) -> dict:
    """Baseline: every zone trains its own model and is tested on its own test split."""
    device = device or get_device(cfg["train"]["device"])
    epochs = cfg["train"].get("local_only_epochs", 5)
    zones = zones if zones is not None else build_zone_data(cfg)

    results = {}
    for zone, zd in zones.items():
        model = build_model(cfg)
        loss = train_local(model, zd.loaders["train"], epochs, cfg["train"]["lr"], device)
        results[zone] = evaluate(model, zd.loaders["test"], zd.scaler, device)
        print(f"{zone}  train_loss={loss:.4f}  "
              f"MAE={results[zone]['mae']:.1f}  RMSE={results[zone]['rmse']:.1f}  "
              f"WMAPE={results[zone]['wmape']:.3f}")

    wm = [r["wmape"] for r in results.values()]
    print(f"\nLocal-only: mean WMAPE={np.mean(wm):.3f}  worst zone WMAPE={np.max(wm):.3f}")
    return results


if __name__ == "__main__":
    cfg = load_config()
    run_local_only(cfg)