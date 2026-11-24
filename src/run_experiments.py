"""Run every method with the same compute budget and save the results JSON.

    python -m src.run_experiments

The environment variable HFL_RESULTS overrides the output path (the dashboard uses this).

Seeds: set  fed.seeds: [0, 1, 2]  in configs/base.yaml to repeat every method with several
seeds. The saved numbers are then the mean over seeds, and each method also gets
std_wmape (spread of the mean error across seeds) and per_seed_wmape. With no
fed.seeds key, one run uses cfg["seed"], exactly as before.
"""
from __future__ import annotations

import copy
import json
import os
import statistics
import time
from pathlib import Path

from src.data import build_zone_data
from src.data_utils import load_config, load_zone_series
from src.federated import (
    fed_cfg, run_centralized, run_flat, run_hierarchical, summarize,
)
from src.train import get_device, run_local_only, set_seed

ALL_METHODS = ["centralized", "local_only", "fedavg", "fedprox", "hierarchical"]
DEFAULT_METHODS = ["centralized", "local_only", "fedprox", "hierarchical"]   # unchanged default
LABELS = {
    "centralized": "Centralized (upper bound)",
    "local_only": "Local-only",
    "fedavg": "Flat FedAvg",
    "fedprox": "Flat FedProx",
    "hierarchical": "Hierarchical (clustered)",
}


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _avg(items):
    """Average a list of result structures: numbers are averaged, dicts recursed, the rest taken from the first."""
    first = items[0]
    if _is_num(first):
        return sum(items) / len(items)
    if isinstance(first, dict):
        return {k: _avg([it[k] for it in items]) if all(k in it for it in items) else first[k]
                for k in first}
    return first


def _run_one(name, cfg, zones, device, f, series, seed):
    """One method, one seed. Returns (result, history_or_None, clusters_or_None)."""
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = seed
    set_seed(seed)
    hist = assignment = None
    if name == "centralized":
        res = run_centralized(cfg, zones, device)
    elif name == "local_only":
        cfg["train"]["local_only_epochs"] = f["rounds"] * f["local_epochs"]   # same epochs as federated
        res = summarize(run_local_only(cfg, zones, device))
    elif name == "fedavg":
        res, hist = run_flat(cfg, zones, device, mu=0.0)
    elif name == "fedprox":
        res, hist = run_flat(cfg, zones, device, mu=f["mu"])
    elif name == "hierarchical":
        res, hist, assignment = run_hierarchical(cfg, zones, device, series=series)
    return res, hist, assignment


def main(cfg_path: str = "configs/base.yaml", out_path: str | None = None):
    out_path = out_path or os.environ.get("HFL_RESULTS", "results/results.json")
    cfg = load_config(cfg_path)
    f = fed_cfg(cfg)
    methods = cfg.get("fed", {}).get("methods", DEFAULT_METHODS)
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown:
        raise ValueError(f"Unknown methods in config: {unknown}")
    seeds = cfg.get("fed", {}).get("seeds") or [cfg["seed"]]

    device = get_device(cfg["train"]["device"])
    series = load_zone_series(cfg)       # read the file once
    zones = build_zone_data(cfg, series)
    if not zones:
        raise ValueError("No zone has enough data. Check min_zone_length and the train/val ratios.")
    print(f"{len(zones)} zones, device={device}, rounds={f['rounds']}, "
          f"local_epochs={f['local_epochs']}, mu={f['mu']}, seeds={seeds}")

    results = {
        "settings": {
            "zones": len(zones), "rounds": f["rounds"], "local_epochs": f["local_epochs"],
            "mu": f["mu"], "n_clusters": f["n_clusters"], "global_every": f["global_every"],
            "input_len": cfg["data"]["input_len"], "pred_len": cfg["data"]["pred_len"],
            "model": cfg["model"]["name"],
            "finetune_epochs": f["finetune_epochs"],
            "normalization": cfg["data"].get("normalization", "standard"),
            "train_stride": cfg["data"].get("train_stride", 1),
            "seeds": list(seeds),
        },
        "labels": LABELS,
        "methods": {},
        "history": {},
        "clusters": {},
        "seconds": {},
    }

    for name in methods:
        print(f"\n== {LABELS[name]} ==")
        t0 = time.time()
        runs = []
        for s in seeds:
            res, hist, assignment = _run_one(name, cfg, zones, device, f, series, s)
            runs.append(res)
            if s == seeds[0]:                       # curves and clusters come from the first seed
                if hist is not None:
                    results["history"][name] = hist
                if assignment is not None:
                    results["clusters"] = assignment
            print(f"   seed {s}: mean WMAPE={res['mean_wmape']:.3f}")
        res = _avg(runs)
        if len(runs) > 1:
            w = [r["mean_wmape"] for r in runs]
            res["std_wmape"] = statistics.stdev(w)
            res["per_seed_wmape"] = w
        results["methods"][name] = res
        results["seconds"][name] = round(time.time() - t0, 1)
        print(f"-> mean WMAPE={res['mean_wmape']:.3f}  worst zone={res['worst_wmape']:.3f}  "
              f"({results['seconds'][name]}s)")

        # save after every method so a long run is never lost
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

    print("\nSummary (test split)")
    for name, res in results["methods"].items():
        extra = f" ± {res['std_wmape']:.3f}" if "std_wmape" in res else ""
        print(f"  {LABELS[name]:<28} mean WMAPE={res['mean_wmape']:.3f}{extra}  "
              f"worst={res['worst_wmape']:.3f}  MAE={res['mean_mae']:.1f}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()