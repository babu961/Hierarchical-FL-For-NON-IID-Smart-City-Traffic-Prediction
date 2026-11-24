"""Zone-level traffic data for training: one zone = one federated client.

Pandas-only helpers (reading, cleaning, scaling options) live in src/data_utils.py.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.data_utils import (
    ZoneScaler, data_opts, load_config, load_zone_series,
    make_scaler, split_sizes, time_features,
)

__all__ = ["ZoneData", "WindowDataset", "build_zone_data", "load_config", "load_zone_series"]


@dataclass
class ZoneData:
    loaders: dict          # {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    scaler: ZoneScaler
    n_train: int           # training windows actually used (aggregation weight)


class WindowDataset(Dataset):
    """Sliding windows. features: [T, F], column 0 is the normalized traffic target."""

    def __init__(self, features, input_len: int, pred_len: int):
        self.x = torch.tensor(features, dtype=torch.float32)
        self.input_len = input_len
        self.pred_len = pred_len
        self.n = max(0, len(features) - input_len - pred_len + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = self.x[i : i + self.input_len]                                     # [L, F]
        y = self.x[i + self.input_len : i + self.input_len + self.pred_len, 0]  # [P]
        return x, y


def build_zone_data(cfg: dict, series: dict | None = None) -> dict[str, ZoneData]:
    """Chronological split, per-zone scaling, windowing and loaders for every usable zone.

    Pass `series` (from load_zone_series) to avoid reading the file again.
    """
    d, t = cfg["data"], cfg["train"]
    o = data_opts(d)
    L, P = d["input_len"], d["pred_len"]
    if series is None:
        series = load_zone_series(cfg)

    out = {}
    for zone, s in series.items():
        n_train, n_val, _ = split_sizes(len(s), d["train_ratio"], d["val_ratio"])
        if n_train <= L + P:
            warnings.warn(f"Skipping zone {zone}: training part too short")
            continue

        # the scaler is fitted on the training part only
        scaler = make_scaler(o["normalization"], s.values[:n_train])
        target = scaler.transform(s.values.astype(np.float64))[:, None]
        feats = (np.concatenate([target, time_features(s.index)], axis=1)
                 if d["use_time_features"] else target)

        # val/test start L steps early so their first window has full history;
        # no target overlaps the training period
        splits = {
            "train": feats[:n_train],
            "val": feats[n_train - L : n_train + n_val],
            "test": feats[n_train + n_val - L :],
        }
        loaders, counts, ok = {}, {}, True
        for name, arr in splits.items():
            ds = WindowDataset(arr, L, P)
            if len(ds) == 0:
                ok = False
                break
            if name == "train" and o["train_stride"] > 1:
                ds = Subset(ds, range(0, len(ds), o["train_stride"]))
            counts[name] = len(ds)
            loaders[name] = DataLoader(ds, batch_size=t["batch_size"],
                                       shuffle=(name == "train"))
        if not ok:
            warnings.warn(f"Skipping zone {zone}: a split has no full windows")
            continue
        out[zone] = ZoneData(loaders=loaders, scaler=scaler, n_train=counts["train"])
    return out


if __name__ == "__main__":
    cfg = load_config()
    zones = build_zone_data(cfg)
    print(f"{len(zones)} zones loaded")
    for z, zd in list(zones.items())[:5]:
        x, y = next(iter(zd.loaders["train"]))
        print(z, "train windows:", zd.n_train, "x:", tuple(x.shape), "y:", tuple(y.shape))