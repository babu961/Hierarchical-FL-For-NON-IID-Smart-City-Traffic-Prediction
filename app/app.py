"""Backend for the results dashboard.

Run from the project root:
    python app/app.py
then open http://127.0.0.1:5001
"""
import io
import json
import os
import re
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

import pandas as pd
import yaml
from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STATIC = HERE / "static"
CONFIG = ROOT / "configs" / "base.yaml"
UPLOAD = ROOT / "data" / "uploaded.csv"          # cleaned upload, in the config's column names
RESULTS = ROOT / "results" / "results.json"      # built-in dataset results (never touched by uploads)
RESULTS_UP = ROOT / "results" / "results_uploaded.json"

app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

# idle | running | done | error. "log" holds the last lines the runner printed.
_run = {"state": "idle", "error": None, "log": []}
_lock = threading.Lock()


def load_cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def data_target() -> Path:
    """The built-in CSV (from configs/base.yaml)."""
    p = Path(load_cfg()["data"]["path"])
    return p if p.is_absolute() else ROOT / p


def is_running() -> bool:
    return _run["state"] == "running"


@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


# ---------------------------------------------------------------- results
def _read_json(path: Path, missing_msg: str):
    if not path.exists():
        return jsonify({"error": missing_msg}), 404
    try:
        with open(path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    except json.JSONDecodeError:
        return jsonify({"error": "Results are being written, retrying..."}), 503


@app.get("/api/results")
def results():
    # Each dataset has its own file, so a run on an upload never disturbs the built-in results.
    # The runner saves after every method, so partial results are served while a run is going.
    if request.args.get("dataset") == "uploaded":
        return _read_json(RESULTS_UP, "No results for the uploaded dataset yet. Run the experiments on it.")
    return _read_json(RESULTS, "No results yet. Run: python -m src.run_experiments")


# ---------------------------------------------------------------- upload and run
@app.post("/api/upload")
def upload():
    """Body: raw CSV text. Query: ts, val (required) and zone (optional) = the column names."""
    if is_running():
        return jsonify({"error": "A run is in progress. Wait for it to finish."}), 409
    body = request.get_data()
    if not body.strip():
        return jsonify({"error": "Empty file."}), 400
    ts = request.args.get("ts", "")
    val = request.args.get("val", "")
    zone = request.args.get("zone", "") or None
    if not ts or not val:
        return jsonify({"error": "Column names for the timestamp and traffic are required."}), 400
    try:
        df = pd.read_csv(io.BytesIO(body))
    except Exception as e:
        return jsonify({"error": f"Could not parse the CSV: {e}"}), 400
    needed = [ts, val] + ([zone] if zone else [])
    absent = [c for c in needed if c not in df.columns]
    if absent:
        return jsonify({"error": f"Columns not found: {', '.join(absent)}"}), 400

    d = load_cfg()["data"]
    out = pd.DataFrame({
        d["time_col"]: df[ts],
        d["target_col"]: pd.to_numeric(df[val], errors="coerce"),
        d["zone_col"]: df[zone].astype(str) if zone else "uploaded",
    })
    UPLOAD.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(UPLOAD, index=False)
    RESULTS_UP.unlink(missing_ok=True)           # old results belong to the previous upload
    _run.update(state="idle", error=None, log=[])
    return jsonify({"ok": True, "rows": len(out)})


def _do_run():
    """Run the experiments on the upload. The runner is pointed at the upload and at its own
    results file through environment variables, so the built-in data and results are never moved."""
    tail = deque(maxlen=14)
    outcome = ("error", "unknown error")
    try:
        RESULTS_UP.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_UP.unlink(missing_ok=True)
        env = dict(os.environ, HFL_DATA=str(UPLOAD), HFL_RESULTS=str(RESULTS_UP), PYTHONUNBUFFERED="1")
        p = subprocess.Popen([sys.executable, "-m", "src.run_experiments"], cwd=ROOT, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            line = line.rstrip()
            if line:
                tail.append(line)
                _run["log"] = list(tail)         # the page shows this while the run is going
        p.wait()
        if p.returncode == 0 and RESULTS_UP.exists():
            outcome = ("done", None)
        else:
            outcome = ("error", "\n".join(tail)[-1500:] or "run failed")
    except Exception as e:
        outcome = ("error", str(e))
    finally:
        _run.update(state=outcome[0], error=outcome[1])


@app.post("/api/run")
def run():
    with _lock:
        if is_running():
            return jsonify({"error": "A run is already in progress."}), 409
        if not UPLOAD.exists():
            return jsonify({"error": "Upload a CSV first."}), 400
        _run.update(state="running", error=None, log=[])
    threading.Thread(target=_do_run, daemon=True).start()
    return jsonify({"ok": True})


@app.get("/api/run/status")
def run_status():
    return jsonify(_run)


# ---------------------------------------------------------------- zones (built-in dataset)
_cache = {"key": None, "data": None}


def _find(cols, pattern):
    for c in cols:
        if re.search(pattern, str(c), re.I):
            return c
    return None


def _profile(ts, vals):
    df = pd.DataFrame({"ts": ts, "v": vals}).dropna()
    df["h"] = df["ts"].dt.hour
    df["we"] = df["ts"].dt.dayofweek >= 5
    wd, we = df[~df["we"]], df[df["we"]]
    weekday = wd.groupby("h")["v"].mean().reindex(range(24)).fillna(0.0)
    weekend = we.groupby("h")["v"].mean().reindex(range(24)).fillna(0.0)
    wd_mean = float(wd["v"].mean()) if len(wd) else None
    we_mean = float(we["v"].mean()) if len(we) else None
    return {
        "weekday": [float(x) for x in weekday],
        "weekend": [float(x) for x in weekend],
        "mean": float(df["v"].mean()),
        "peak_hour": int(weekday.values.argmax()),
        "weekend_ratio": (we_mean / wd_mean) if wd_mean and we_mean else None,
        "rows": int(len(df)),
        "start": df["ts"].min().strftime("%Y-%m-%d %H:%M"),
        "end": df["ts"].max().strftime("%Y-%m-%d %H:%M"),
    }


def build_zones(path: Path):
    df = pd.read_csv(path)
    cols = list(df.columns)
    ts_col = _find(cols, r"time|date|stamp") or cols[0]
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])

    zone_col = _find([c for c in cols if c != ts_col], r"zone|sensor|node|location|area|^id$")
    zones = {}
    if zone_col is not None:
        val_col = _find([c for c in cols if c not in (ts_col, zone_col)],
                        r"flow|count|traffic|volume|vehicle|value|speed")
        if val_col is None:
            num = [c for c in df.select_dtypes("number").columns if c != zone_col]
            if not num:
                raise ValueError("no numeric traffic column found")
            val_col = num[0]
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        for name, g in df.groupby(zone_col):
            zones[str(name)] = _profile(g[ts_col], g[val_col])
    else:
        for c in df.select_dtypes("number").columns:
            zones[str(c)] = _profile(df[ts_col], df[c])

    if not zones:
        raise ValueError("no zones found in the CSV")
    return {"source": path.name, "zones": zones}


@app.get("/api/zones")
def zones():
    path = data_target()
    if not path.exists():
        return jsonify({"error": "No built-in dataset found. Upload a CSV on the previous page."}), 404
    try:
        key = (str(path), path.stat().st_mtime)
        if _cache["key"] != key:
            _cache["data"] = build_zones(path)
            _cache["key"] = key
        return jsonify(_cache["data"])
    except Exception as e:
        return jsonify({"error": f"Could not read {path.name}: {e}"}), 500


if __name__ == "__main__":
    # use_reloader=False so a file change cannot restart the server mid-run.
    # Set HFL_DEBUG=1 to turn debug mode on while developing.
    app.run(host="127.0.0.1", port=5001, debug=os.environ.get("HFL_DEBUG") == "1", use_reloader=False)