"""Preprocessing: raw CSV -> one regular, cleaned series per zone.

Steps, in order:
  1. parse timestamps and numbers; rows with unreadable timestamps are dropped
  2. negative traffic values are treated as missing
  3. duplicate timestamps within a zone are dropped (first one kept)
  4. each zone is put on a regular grid (freq_minutes) and gaps are filled
  5. zones with fewer than min_zone_length rows are skipped

Usage (from the project root):
    python -m src.preprocess                          # print the quality report
    python -m src.preprocess --out data/cleaned.csv   # also save the cleaned long table
"""
from __future__ import annotations

import argparse
import warnings

import pandas as pd

from src.data_utils import load_config, prepare_zone, read_dataset


def clean_zones(cfg: dict):
    """Return (series per kept zone, quality stats per zone, file-level stats)."""
    d = cfg["data"]
    df = read_dataset(d)
    series, stats = {}, {}
    for zone, g in df.groupby(d["zone_col"]):
        s, st = prepare_zone(g, d)
        st["kept"] = len(s) >= d["min_zone_length"]
        if st["kept"]:
            series[str(zone)] = s
        else:
            warnings.warn(f"Skipping zone {zone}: only {len(s)} rows")
        stats[str(zone)] = st
    file_stats = {"bad_time": df.attrs.get("bad_time", 0),
                  "negative": df.attrs.get("negative", 0)}
    return series, stats, file_stats


def quality_report(stats: dict, file_stats: dict) -> str:
    lines = [
        f"Unreadable timestamps dropped : {file_stats['bad_time']}",
        f"Negative values set to missing: {file_stats['negative']}",
        "",
        f"{'zone':<12}{'raw':>9}{'dups':>8}{'missing':>9}{'grid':>9}{'filled':>9}  status",
    ]
    for z, st in stats.items():
        status = "kept" if st["kept"] else "skipped (too short)"
        lines.append(
            f"{z:<12}{st['raw_rows']:>9}{st['duplicates']:>8}{st['missing_values']:>9}"
            f"{st['grid_rows']:>9}{st['inserted_rows']:>9}  {status}"
        )
    kept = sum(st["kept"] for st in stats.values())
    lines.append(f"\n{kept} of {len(stats)} zones kept")
    return "\n".join(lines)


def to_long_table(series: dict) -> pd.DataFrame:
    parts = []
    for z, s in series.items():
        part = s.rename("traffic").reset_index()
        part.columns = ["timestamp", "traffic"]
        part.insert(0, "zone_id", z)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Clean the traffic dataset and report its quality.")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default=None, help="write the cleaned long table to this CSV")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    series, stats, file_stats = clean_zones(cfg)
    print(quality_report(stats, file_stats))

    if args.out:
        table = to_long_table(series)
        table.to_csv(args.out, index=False)
        print(f"\nWrote {len(table)} rows to {args.out}")


if __name__ == "__main__":
    main()