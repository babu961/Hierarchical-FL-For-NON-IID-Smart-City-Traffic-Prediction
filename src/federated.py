"""Federated training methods.

- run_flat:         all zones -> one server (FedAvg when mu=0, FedProx when mu>0)
- run_hierarchical: zones -> clusters -> global (clustered, shared encoder, per-cluster heads)
- run_centralized:  pooled-data upper bound
"""
from __future__ import annotations

import copy

import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import ConcatDataset, DataLoader

from src.data_utils import load_zone_series, split_sizes
from src.models import build_model
from src.train import evaluate, train_local


def fed_cfg(cfg: dict) -> dict:
    f = dict(rounds=5, local_epochs=1, mu=0.01, n_clusters=4, global_every=2, finetune_epochs=2)
    f.update(cfg.get("fed", {}))
    return f


def clone_state(state: dict) -> dict:
    return {k: v.detach().clone().cpu() for k, v in state.items()}


def weighted_average(states: list[dict], weights: list[float]) -> dict:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    return {
        k: sum(s[k].float() * (w / total) for s, w in zip(states, weights))
        for k in states[0]
    }


def eval_all(zones: dict, model_of: dict, split: str, device) -> dict:
    return {
        z: evaluate(model_of[z], zd.loaders[split], zd.scaler, device)
        for z, zd in zones.items()
    }


def summarize(per_zone: dict) -> dict:
    return {
        "per_zone": per_zone,
        "mean_wmape": float(np.mean([m["wmape"] for m in per_zone.values()])),
        "worst_wmape": float(np.max([m["wmape"] for m in per_zone.values()])),
        "mean_mae": float(np.mean([m["mae"] for m in per_zone.values()])),
        "mean_rmse": float(np.mean([m["rmse"] for m in per_zone.values()])),
    }


def _mean_wmape(per_zone: dict) -> float:
    return float(np.mean([m["wmape"] for m in per_zone.values()]))


def _local_update(work, state, loader, f, lr, device, mu):
    """Start the working model from `state`, train on one zone, return its new weights."""
    work.load_state_dict(state)
    train_local(work, loader, f["local_epochs"], lr, device,
                global_state=state if mu > 0 else None, mu=mu)
    return clone_state(work.state_dict())


# ----------------------------------------------------------------------------- clustering
def zone_profiles(cfg: dict, zone_names: list[str], series: dict | None = None) -> np.ndarray:
    """Weekly shape of each zone: 168 hourly bins, standardized per zone, training part only.
    Clustering on shape rather than volume groups zones that behave alike."""
    if series is None:
        series = load_zone_series(cfg)
    d = cfg["data"]
    profs = []
    for z in zone_names:
        s = series[z]
        n_train, _, _ = split_sizes(len(s), d["train_ratio"], d["val_ratio"])
        tr = s.iloc[:n_train]
        zs = (tr - tr.mean()) / (tr.std() + 1e-8)
        key = np.asarray(tr.index.dayofweek * 24 + tr.index.hour)
        profs.append(zs.groupby(key).mean().reindex(range(168)).fillna(0.0).values)
    return np.stack(profs)


def cluster_zones(cfg: dict, zones: dict, series: dict | None = None) -> dict:
    names = list(zones)
    k = min(fed_cfg(cfg)["n_clusters"], len(names))
    labels = KMeans(n_clusters=k, n_init=10, random_state=cfg["seed"]).fit_predict(
        zone_profiles(cfg, names, series)
    )
    return {z: int(l) for z, l in zip(names, labels)}


# ----------------------------------------------------------------------------- methods
def run_flat(cfg: dict, zones: dict, device, mu: float = 0.0, return_model: bool = False):
    """FedAvg (mu=0) or FedProx (mu>0): every zone talks directly to one global server."""
    f, lr = fed_cfg(cfg), cfg["train"]["lr"]
    glob = build_model(cfg)
    work = build_model(cfg)
    weights = [zd.n_train for zd in zones.values()]
    history = []
    for r in range(f["rounds"]):
        g_state = clone_state(glob.state_dict())
        states = [_local_update(work, g_state, zd.loaders["train"], f, lr, device, mu)
                  for zd in zones.values()]
        glob.load_state_dict(weighted_average(states, weights))
        val = eval_all(zones, {z: glob for z in zones}, "val", device)
        history.append(_mean_wmape(val))
        print(f"  round {r + 1}/{f['rounds']}  val WMAPE={history[-1]:.3f}")
    test = eval_all(zones, {z: glob for z in zones}, "test", device)
    if return_model:
        return summarize(test), history, glob
    return summarize(test), history


def run_hierarchical(cfg: dict, zones: dict, device, mu: float | None = None,
                     assignment: dict | None = None, series: dict | None = None):
    """Three tiers: zones -> cluster aggregators -> global server.

    - Tier 2: inside each cluster, zone models are averaged (FedProx toward the cluster model).
    - Tier 3: every `global_every` rounds the encoder is averaged across clusters;
      each cluster keeps its own prediction head.
    `mu` defaults to the config value, the same value run_flat uses for FedProx.
    """
    f, lr = fed_cfg(cfg), cfg["train"]["lr"]
    mu = f["mu"] if mu is None else mu
    assignment = assignment or cluster_zones(cfg, zones, series)
    cids = sorted(set(assignment.values()))
    members = {c: [z for z in zones if assignment[z] == c] for c in cids}
    sizes = {c: sum(zones[z].n_train for z in members[c]) for c in cids}

    init = clone_state(build_model(cfg).state_dict())
    enc_keys = [k for k in init if k.startswith("encoder.")]
    c_state = {c: copy.deepcopy(init) for c in cids}
    work = build_model(cfg)
    models = {c: build_model(cfg) for c in cids}

    history = []
    for r in range(f["rounds"]):
        for c in cids:
            states = [_local_update(work, c_state[c], zones[z].loaders["train"], f, lr, device, mu)
                      for z in members[c]]
            c_state[c] = weighted_average(states, [zones[z].n_train for z in members[c]])

        if (r + 1) % f["global_every"] == 0 or r == f["rounds"] - 1:
            backbone = weighted_average(
                [{k: c_state[c][k] for k in enc_keys} for c in cids],
                [sizes[c] for c in cids],
            )
            for c in cids:
                c_state[c].update(backbone)

        for c in cids:
            models[c].load_state_dict(c_state[c])
        val = eval_all(zones, {z: models[assignment[z]] for z in zones}, "val", device)
        history.append(_mean_wmape(val))
        print(f"  round {r + 1}/{f['rounds']}  val WMAPE={history[-1]:.3f}")

    test = eval_all(zones, {z: models[assignment[z]] for z in zones}, "test", device)
    return summarize(test), history, assignment


def run_centralized(cfg: dict, zones: dict, device):
    """Upper bound: one model trained on all zones' pooled training windows."""
    f, lr = fed_cfg(cfg), cfg["train"]["lr"]
    pooled = ConcatDataset([zd.loaders["train"].dataset for zd in zones.values()])
    loader = DataLoader(pooled, batch_size=cfg["train"]["batch_size"], shuffle=True)
    m = build_model(cfg)
    train_local(m, loader, f["rounds"] * f["local_epochs"], lr, device)
    return summarize(eval_all(zones, {z: m for z in zones}, "test", device))


# ----------------------------------------------------------------------------- personal heads