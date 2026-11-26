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
  (5-year mortgage rate, policy rate). Source: CREA MLS® HPI. Downloaded data lives in the datastore, never in git.
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

## Datastore
Local and cloud speak the same protocols (S3 for files, Postgres for MLflow), so switching is settings only.

| Holds | Local (default) | Cloud |
|---|---|---|
| Files: raw snapshots, forecasts, monitoring, state, model files | `seaweedfs` container, S3 API at `localhost:8333`, bucket `housing` | Cloudflare R2 bucket |
| MLflow runs and model registry | `mlflow-db` Postgres container at `localhost:5433` | Neon Postgres |

Local data persists in Docker volumes (`seaweedfs-volume`, `mlflow-db-volume`); `docker compose down -v` deletes it.

**Switch to cloud**
1. Cloudflare: create an R2 bucket and an API token with Object Read & Write on it.
2. Neon: create a project and copy its connection string.
3. `cp .env.cloud.example .env.cloud`, fill it in (`.env.cloud` is git-ignored).
4. `docker compose --env-file .env.cloud up -d`, then trigger `ingest`.

**Look at the data**
```bash
# SQL over the Parquet files with DuckDB
uv run --with duckdb python -c "
import duckdb; c = duckdb.connect()
c.sql(\"CREATE SECRET (TYPE s3, KEY_ID 'housing', SECRET 'housing-secret', ENDPOINT 'localhost:8333', URL_STYLE 'path', USE_SSL false)\")
c.sql(\"SELECT * FROM 's3://housing/forecasts/home_price/*.parquet'\").show()"

# MLflow UI (runs and registered models)
uv run --with mlflow --with 'psycopg[binary]' mlflow ui \
  --backend-store-uri postgresql+psycopg://mlflow:mlflow@localhost:5433/mlflow
```

## How to extend
| To add | Do this |
|---|---|
| A region or province | Add `SHEET_NAME: Display name` under `regions:` in `config.yaml` (CREA sheet names, e.g. `GREATER_TORONTO`, `OTTAWA`, `MONTREAL_CMA`). |
| A data source | Create `housing/sources/<name>.py` following the contract in `housing/sources/__init__.py`, add it to `SOURCES`, and list it under a forecast's `sources:`. Its numeric columns become features (level + 3-month change) automatically. |
| A candidate model | Add it to `CANDIDATES` in `housing/model.py` (anything with `fit`/`predict`), and list it under `candidates:`. |
| A new thing to forecast (e.g. rents) | Add an entry under `forecasts:` with its `target: <source>.<column>`. `forecast_<name>` and `monitor_<name>` DAGs appear automatically, and the API serves it via `?target=<name>`. |
| Cloud storage | Set `DATA_URI=s3://bucket/path` (or `gs://…`), install `s3fs` (or `gcsfs`), and point `MLFLOW_TRACKING_URI` at an MLflow server. No code changes. |

Old production models keep working after these changes: each model stores the regions and features it was trained on.

## Storage layout (inside `DATA_URI`, e.g. `s3://housing`)
```
raw/<source>/fetched=YYYY-MM-DD.parquet   every download, never overwritten
raw/<source>/latest.parquet               last download that passed validation
forecasts/<name>/as_of=YYYY-MM.parquet    every month's published forecasts (with model version)
monitoring/<name>.parquet                 live error per forecast month
state/<name>.json                         last month trained on
mlartifacts/<name>/                       MLflow model files (run records live in Postgres)
```

## Develop / test
```bash
uv venv -p 3.12 .venv
uv pip install -p .venv -r requirements.txt \
  -c https://raw.githubusercontent.com/apache/airflow/constraints-3.4.0/constraints-3.12.txt
.venv/bin/pytest
```
Tests and the venv default to a local `data/` folder and SQLite, so they need no Docker.
To point the venv at the Docker datastore instead:
`DATA_URI=s3://housing AWS_ENDPOINT_URL=http://localhost:8333 AWS_ACCESS_KEY_ID=housing AWS_SECRET_ACCESS_KEY=housing-secret MLFLOW_S3_ENDPOINT_URL=http://localhost:8333 MLFLOW_TRACKING_URI=postgresql+psycopg://mlflow:mlflow@localhost:5433/mlflow`

## Layout
- `config.yaml`: what the pipelines cover
- `housing/`: plain Python, tested without Airflow
  - `sources/`: one module per data source; `features.py`, `model.py` (candidates, gate, registry),
    `monitoring.py`, `pipeline.py` (the steps the DAGs call), `storage.py`, `config.py`, `assets.py`
- `dags/`: `ingest.py`, `forecast.py`, `monitor.py` (thin; forecast and monitor DAGs are generated from config)
- `src/app.py`: forecast API
- `docker-compose.yaml`: Airflow 3.4 (LocalExecutor) + datastore (SeaweedFS, MLflow Postgres) + API
- `.env.cloud.example`: settings template for the cloud datastore (R2 + Neon)
- `.github/workflows/ci.yml`: tests on every push and PR; builds and pushes both images to ghcr.io from `main`
