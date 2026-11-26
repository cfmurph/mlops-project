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

## Run dev (Docker, about 4 GB RAM)
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

## Environments
| | Dev | Prod |
|---|---|---|
| Where | Your laptop: `docker compose up -d` | Oracle Cloud Always Free ARM VM, deployed by GitHub Actions |
| Code | Bind-mounted `dags/`, `housing/`, `config.yaml`: edits apply live | Baked into images tagged with the git SHA (exactly what CI tested) |
| Datastore | Local SeaweedFS + Postgres containers | Cloudflare R2 + Neon Postgres |
| Secrets | Dev defaults in `docker-compose.yaml` | GitHub Environment secret `PROD_ENV_FILE`, written to the VM at deploy time |
| Airflow UI | http://localhost:8080 | VM's localhost only: `ssh -L 8080:localhost:8080 <user>@<vm>`, then http://localhost:8080 |
| Forecast API | http://localhost:8000 | `http://<vm>:8000` |

`/health`, the Airflow UI title ("Housing (dev)" / "Housing (prod)") and MLflow run tags show which environment you're in.
Prod = `docker-compose.yaml` + `docker-compose.prod.yaml` + `envs/prod.env`.

**Release flow**
1. Work on a branch and test in dev. Open a PR: CI runs the tests and checks both environments' compose configs.
2. Merge to `main`: CI builds arm64 images tagged with the commit SHA and pushes them to ghcr.io.
3. The deploy job waits for approval (Actions → the run → **Review deployments** → Approve). It then copies the compose
   files to the VM, writes `envs/prod.env`, pulls the images, restarts, and fails unless `/health` reports `prod`.
4. **Roll back:** Actions → CI/CD → **Run workflow**, with `image_tag` set to an earlier commit SHA (same approval).

**One-time prod setup**
1. **VM:** Oracle Cloud → create an Always Free *Ampere A1* instance (Ubuntu, e.g. 2 OCPU / 12 GB). Free ARM capacity is
   sometimes "out of capacity" in a region; retry later or pick another availability domain. In the VCN security list,
   allow TCP 8000. Copy `deploy/bootstrap-vm.sh` to the VM and run it once.
2. **Deploy key:** `ssh-keygen -t ed25519 -f prod_deploy -N ""`, append `prod_deploy.pub` to the VM's `~/.ssh/authorized_keys`.
3. **Datastore:** create an R2 bucket (e.g. `housing-prod`) with an API token (Object Read & Write), and a Neon project.
4. **Settings:** copy `envs/prod.env.example`, fill it in (generate the Airflow secrets as the comments show).
5. **GitHub** → Settings → Environments → `production` → add secrets: `PROD_HOST` (VM IP), `PROD_USER` (e.g. `ubuntu`),
   `PROD_SSH_KEY` (contents of `prod_deploy`), `PROD_ENV_FILE` (the whole filled-in file). Then delete the local copies.
6. Run a deploy (merge to `main`, or **Run workflow**), approve it, then in the prod Airflow UI unpause
   `forecast_home_price`, `monitor_home_price`, then `ingest`.

## Datastore
Local and cloud speak the same protocols (S3 for files, Postgres for MLflow), so they differ only in settings.

| Holds | Dev (local containers) | Prod (cloud) |
|---|---|---|
| Files: raw snapshots, forecasts, monitoring, state, model files | `seaweedfs` container, S3 API at `localhost:8333`, bucket `housing` | Cloudflare R2 bucket |
| MLflow runs and model registry | `mlflow-db` Postgres container at `localhost:5433` | Neon Postgres |

Local data persists in Docker volumes (`seaweedfs-volume`, `mlflow-db-volume`); `docker compose down -v` deletes it.

**Look at the dev data**
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
- `docker-compose.prod.yaml`, `envs/prod.env.example`: prod overlay and its settings template
- `deploy/`: `bootstrap-vm.sh` (one-time VM setup), `deploy.sh` (run on the VM by CI)
- `.github/workflows/ci.yml`: tests + config checks on every PR; on `main` builds images and deploys to prod after approval
