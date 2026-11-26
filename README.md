# Canadian housing price forecaster (MLOps with Airflow)

Forecasts the MLS® benchmark home price **3 months ahead** for Canada, BC and Alberta
(Greater Vancouver, Fraser Valley, Victoria, Calgary, Edmonton, and more), using free public data.

```
config.yaml ── regions, sources, forecasts, monitoring thresholds
     │
ingest (weekly) ── fetch_<source> ×N ──▶ Asset housing://raw/<source>
     │                                         │ triggers when all its sources update
forecast_<name> ── has_new_month ▶ train_evaluate_promote ▶ publish_forecasts ──▶ Asset housing://forecasts/<name>
     │                                                                                  │ triggers
monitor_<name> ── score past forecasts against actual prices; fails (= alert) on drift
     │
FastAPI :8000 ── GET /forecast/{region}?target=<name>
```

- **Data:** [CREA MLS® HPI](https://www.crea.ca/housing-market-stats/mls-home-price-index/hpi-tool/) (monthly,
  seasonally adjusted composite benchmark price) and the [Bank of Canada Valet API](https://www.bankofcanada.ca/valet/docs)
  (5-year mortgage rate, policy rate). Source: CREA MLS® HPI. Downloaded data stays in `data/` and is not committed.
- **Ingest** keeps an immutable snapshot of every download and validates it. Bad data never replaces `latest`, so
  nothing downstream runs on it.
- **Forecast** fits every candidate model, keeps the one with the lowest error over the last 24 months, and registers
  it in the MLflow model registry with the `production` alias only if it beats both "no change" and the current
  production model. Every month's forecasts are kept.
- **Monitor** compares old forecasts with what actually happened once their month arrives, and fails if live error
  exceeds `max_error_ratio` × the model's holdout error.

## Run it (Docker, about 4 GB RAM)
```bash
docker compose up -d --build          # Airflow UI: http://localhost:8080  (airflow / airflow)
```
In the UI, unpause `forecast_home_price` and `monitor_home_price`, then `ingest`. Unpausing `ingest` starts this
week's run; the other two follow automatically. Then:
```bash
curl localhost:8000/regions
curl localhost:8000/forecast/calgary
```
Stop: `docker compose down` (add `-v` to also delete Airflow's database).

## How to extend
| To add | Do this |
|---|---|
| A region or province | Add `SHEET_NAME: Display name` under `regions:` in `config.yaml` (CREA sheet names, e.g. `GREATER_TORONTO`, `OTTAWA`, `MONTREAL_CMA`). |
| A data source | Create `housing/sources/<name>.py` following the contract in `housing/sources/__init__.py`, add it to `SOURCES`, and list it under a forecast's `sources:`. Its numeric columns become features (level + 3-month change) automatically. |
| A candidate model | Add it to `CANDIDATES` in `housing/model.py` (anything with `fit`/`predict`), and list it under `candidates:`. |
| A new thing to forecast (e.g. rents) | Add an entry under `forecasts:` with its `target: <source>.<column>`. `forecast_<name>` and `monitor_<name>` DAGs appear automatically, and the API serves it via `?target=<name>`. |
| Cloud storage | Set `DATA_URI=s3://bucket/path` (or `gs://…`), install `s3fs` (or `gcsfs`), and point `MLFLOW_TRACKING_URI` at an MLflow server. No code changes. |

Old production models keep working after these changes: each model stores the regions and features it was trained on.

## Storage layout (`DATA_URI`)
```
raw/<source>/fetched=YYYY-MM-DD.parquet   every download, never overwritten
raw/<source>/latest.parquet               last download that passed validation
forecasts/<name>/as_of=YYYY-MM.parquet    every month's published forecasts (with model version)
monitoring/<name>.parquet                 live error per forecast month
state/<name>.json                         last month trained on
mlflow.db, mlartifacts/                   MLflow runs and model registry
```

## Develop / test
```bash
uv venv -p 3.12 .venv
uv pip install -p .venv -r requirements.txt \
  -c https://raw.githubusercontent.com/apache/airflow/constraints-3.4.0/constraints-3.12.txt
.venv/bin/pytest
```
Browse MLflow runs and registered models:
`uv run --with mlflow mlflow ui --backend-store-uri sqlite:///data/mlflow.db`

## Layout
- `config.yaml`: what the pipelines cover
- `housing/`: plain Python, tested without Airflow
  - `sources/`: one module per data source; `features.py`, `model.py` (candidates, gate, registry),
    `monitoring.py`, `pipeline.py` (the steps the DAGs call), `storage.py`, `config.py`, `assets.py`
- `dags/`: `ingest.py`, `forecast.py`, `monitor.py` (thin; forecast and monitor DAGs are generated from config)
- `src/app.py`: forecast API
- `docker-compose.yaml`: Airflow 3.4 (LocalExecutor) + Postgres + API
- `.github/workflows/ci.yml`: tests on every push and PR; builds and pushes both images to ghcr.io from `main`
