"""Torch-free data helpers shared by the training code and the web app.

Everything that only needs pandas/numpy lives here, so the web app can analyse and
preview datasets without importing PyTorch.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

NORMALIZATIONS = ["standard", "minmax", "none"]
FILL_METHODS = ["interpolate", "ffill", "zero"]
RESAMPLE_AGGS = ["mean", "sum", "max"]
OPTION_DEFAULTS = {
    "normalization": "standard",   # per-zone, fitted on the training split only
    "fill_method": "interpolate",  # how gaps and missing values are filled
    "resample_agg": "mean",        # how several readings inside one interval are combined
    "train_stride": 1,             # use every n-th training window (faster on CPU)
}


def load_config(path: str | Path = "configs/base.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_opts(d: dict) -> dict:
    """Preprocessing options with defaults filled in (older configs lack them)."""
    opts = dict(OPTION_DEFAULTS)
    opts.update({k: d[k] for k in OPTION_DEFAULTS if k in d})
    return opts


@dataclass
class ZoneScaler:
    """Per-zone scaling fitted on the training split only."""
    kind: str
    shift: float
    scale: float

    def transform(self, x):
        return (x - self.shift) / self.scale

    def inverse(self, x):
        return x * self.scale + self.shift


def make_scaler(kind: str, values) -> ZoneScaler:
    v = np.asarray(values, dtype=np.float64)
    if kind == "standard":
        return ZoneScaler(kind, float(v.mean()), float(v.std() + 1e-8))
    if kind == "minmax":
        return ZoneScaler(kind, float(v.min()), float(v.max() - v.min() + 1e-8))
    if kind == "none":
        return ZoneScaler(kind, 0.0, 1.0)
    raise ValueError(f"Unknown normalization: {kind}")


def time_features(index: pd.DatetimeIndex) -> np.ndarray:
    hour = index.hour + index.minute / 60.0
    dow = index.dayofweek
    return np.stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
        ],
        axis=1,
    )


def read_dataset(d: dict, nrows: int | None = None) -> pd.DataFrame:
    """Read the CSV, parse the time column and coerce the target to numbers.

    Rows whose timestamp cannot be parsed are dropped; their count is kept in df.attrs["bad_time"].
    """
    df = pd.read_csv(d["path"], nrows=nrows)
    for col in (d["zone_col"], d["time_col"], d["target_col"]):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found. Available columns: {', '.join(map(str, df.columns))}")
    df[d["time_col"]] = pd.to_datetime(df[d["time_col"]], errors="coerce")
    tcol = d["target_col"]
    df[tcol] = pd.to_numeric(df[tcol], errors="coerce").astype(float)
    neg = int((df[tcol] < 0).sum())
    df[tcol] = df[tcol].mask(df[tcol] < 0)      # negative counts are invalid: treat as missing
    bad = int(df[d["time_col"]].isna().sum())
    out = df.dropna(subset=[d["time_col"]])
    out.attrs["bad_time"] = bad
    out.attrs["negative"] = neg
    return out


def regularize(s: pd.Series, freq_minutes: int, agg: str, fill: str) -> pd.Series:
    """Put a series on a regular time grid and fill gaps."""
    resampler = s.resample(f"{int(freq_minutes)}min")
    rs = resampler.sum(min_count=1) if agg == "sum" else getattr(resampler, agg)()
    if fill == "interpolate":
        rs = rs.interpolate(limit_direction="both")
    elif fill == "ffill":
        rs = rs.ffill().bfill()
    elif fill == "zero":
        rs = rs.fillna(0.0)
    else:
        raise ValueError(f"Unknown fill method: {fill}")
    return rs.fillna(0.0)  # a zone that is entirely missing ends up as zeros


def prepare_zone(g: pd.DataFrame, d: dict):
    """One zone's rows -> (regular series, quality stats)."""
    o = data_opts(d)
    s = g.set_index(d["time_col"])[d["target_col"]].sort_index()
    raw_rows = int(len(s))
    s = s[~s.index.duplicated(keep="first")]
    unique_rows = int(len(s))
    missing_values = int(s.isna().sum())
    out = regularize(s, d["freq_minutes"], o["resample_agg"], o["fill_method"])
    stats = {
        "raw_rows": raw_rows,
        "duplicates": raw_rows - unique_rows,
        "missing_values": missing_values,
        "grid_rows": int(len(out)),
        "inserted_rows": max(0, int(len(out)) - unique_rows),
    }
    return out, stats


def load_zone_series(cfg: dict) -> dict[str, pd.Series]:
    """Read the dataset and return {zone_id: regular time series of traffic}."""
    d = cfg["data"]
    df = read_dataset(d)
    series = {}
    for zone, g in df.groupby(d["zone_col"]):
        s, _ = prepare_zone(g, d)
        if len(s) < d["min_zone_length"]:
            warnings.warn(f"Skipping zone {zone}: only {len(s)} rows")
            continue
        series[str(zone)] = s
    return series


def split_sizes(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return n_train, n_val, n - n_train - n_val


def count_windows(n_rows: int, input_len: int, pred_len: int, stride: int = 1) -> int:
    total = n_rows - input_len - pred_len + 1
    return 0 if total <= 0 else (total - 1) // max(1, stride) + 1