# Canadian housing price forecaster (MLOps with Airflow)

Forecasts the MLS® benchmark home price **3 months ahead** for Canada, BC and Alberta
(Greater Vancouver, Fraser Valley, Victoria, Calgary, Edmonton, and more), using free public data.

```
Airflow, weekly:
fetch_crea ─┐
            ├─▶ validate ─▶ has_new_month? ─▶ build_features ─▶ train_evaluate_promote ─▶ publish_forecasts
fetch_boc ──┘                (skips if no                        (best candidate must beat      (data/production/
                              new CREA month)                     "no change" + production)      forecasts.csv)
                                                                                                       │
                                                                       FastAPI :8000 /forecast/{region} ◀┘
```

**Data:** [CREA MLS® Home Price Index](https://www.crea.ca/housing-market-stats/mls-home-price-index/hpi-tool/)
(monthly, seasonally adjusted composite benchmark price) and the
[Bank of Canada Valet API](https://www.bankofcanada.ca/valet/docs) (5-year mortgage rate, policy rate).
Source: CREA MLS® HPI. Downloaded data stays in `data/` and is not committed.

**Model selection:** each run fits a gradient-boosting model and a "damped momentum" rule
(half of the last 3 months' change continues) and keeps whichever has the lower error on the last
24 months. It is promoted only if it beats both "prices stay the same" and the current production
model. Runs are tracked in MLflow (`data/mlflow.db`).

## Run the pipeline (Docker, about 4 GB RAM)
```bash
docker compose up -d --build          # Airflow UI: http://localhost:8080  (airflow / airflow)
```
In the UI, unpause **housing_pipeline** and trigger it. Then:
```bash
curl localhost:8000/regions
curl localhost:8000/forecast/calgary
```
Stop: `docker compose down` (add `-v` to also delete Airflow's database).

## Develop / test
```bash
uv venv -p 3.12 .venv
uv pip install -p .venv -r requirements.txt \
  -c https://raw.githubusercontent.com/apache/airflow/constraints-3.4.0/constraints-3.12.txt
.venv/bin/pytest
```
Browse MLflow runs: `uv run --with mlflow mlflow ui --backend-store-uri sqlite:///data/mlflow.db`

## Layout
- `housing/`: plain Python (fetch, validate, features, train/gate/publish), tested without Airflow
- `dags/housing_pipeline.py`: thin Airflow wrappers
- `src/app.py`: forecast API
- `docker-compose.yaml`: Airflow 3.4 (LocalExecutor) + Postgres + API
- `.github/workflows/ci.yml`: tests on every push; builds and pushes both images to ghcr.io from `main`
