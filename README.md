# Hierarchical Federated Learning for Non-IID Smart-City Traffic Prediction
a live demo link:
      https://hierarchical-fl-for-non-iid-smart-city.onrender.com/

Smart-city traffic differs from road to road: each zone peaks at a different hour, so the data is **non-IID**. This project forecasts traffic for every zone **without pooling the raw records**. Each zone trains locally and shares only model weights. Zones with similar daily rhythms are grouped into clusters, and each cluster learns its own model (hierarchical federated learning).

A web dashboard, "The Roundabout", walks through the idea, lets you upload your own traffic CSV, runs the experiments, and shows the results.

## Methods compared

| Method | What it does |
|---|---|
| Centralized | Pools all raw data. An upper bound, not a privacy-preserving baseline. |
| Local-only | Each zone trains alone, with no sharing. |
| Flat FedProx | One global model, averaged across all zones, with a proximal term to limit drift. |
| Hierarchical | Clusters zones by daily rhythm (k-means), then trains a model per cluster with shared learning on top. |

FedAvg is also supported. Add `fedavg` to `fed.methods` in `configs/base.yaml` to include it.

Forecasting model: GRU. Metric: WMAPE on held-out data (lower is better).

## Project structure

```
app/            Flask backend and the dashboard page (app/app.py, app/static/)
src/            Data loading, models, federated training, experiment runner
configs/        base.yaml (all settings)
data/           Traffic CSV files
results/        results.json written by the experiment runner
```

## Setup

Python 3.10 or newer is recommended.

```
pip install torch flask pandas pyyaml scikit-learn
```

(If the project has a `requirements.txt`, use `pip install -r requirements.txt` instead.)

## Run

1. Run the experiments (writes `results/results.json`):
   ```
   python -m src.run_experiments
   ```
2. Start the dashboard:
   ```
   python app/app.py
   ```
3. Open http://127.0.0.1:5000

Each finished method appears on the Results page as soon as it completes.

### Several seeds (error bars)

For more reliable comparisons, add seeds to `configs/base.yaml`:

```yaml
fed:
  seeds: [0, 1, 2]
```

Results are then averaged over seeds, and the dashboard shows the standard deviation as error bars.

## Using your own data

On the **Your data** page, upload a CSV with:

- a **timestamp** column,
- a numeric **traffic count** column,
- optionally a **zone** column (sensor, road or area).

Column names are detected automatically. The file is cleaned in the browser (duplicates removed, missing values filled, outliers clipped), and only the cleaned file is sent to the local backend. Then click **Run experiments**.

## Privacy

Raw traffic records stay inside their zone. Only model weights are shared, and in the hierarchical method the global server sees only cluster-level aggregates. This reduces exposure, but it is **not a formal privacy guarantee**: membership inference and gradient inversion are still possible. Differential privacy and secure aggregation are listed as future work.

## Limitations

- Short CPU training and small data. Treat small differences between methods as possibly noise unless you run several seeds.
- The built-in dataset may be synthetic, which validates the pipeline rather than real-world accuracy.
- Forecast accuracy can be lower than centralized training. That is the cost of keeping data local.

## Future work

Stronger baselines (Ditto, IFCA, PatchTST), real city sensor data, graph models for neighbouring zones, secure aggregation and differential privacy, asynchronous clients, and dynamic re-clustering.
