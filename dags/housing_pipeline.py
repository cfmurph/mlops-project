"""Weekly: fetch housing data, validate, and when a new month has arrived retrain, gate, promote and publish."""
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from airflow.sdk import dag, task

DATA = Path(os.environ.get("HOUSING_DATA_DIR", "/opt/airflow/data"))
RAW, PROD = DATA / "raw", DATA / "production"
LAST_TRAINED = DATA / "last_trained_month.txt"


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW / f"{name}.csv", parse_dates=["date"])


# max_active_runs=1: runs share files in DATA, so they must not overlap.
@dag(schedule="@weekly", start_date=datetime(2026, 1, 1), catchup=False, max_active_runs=1, tags=["housing"])
def housing_pipeline():
    @task
    def fetch_crea():
        from housing.data import fetch_crea
        RAW.mkdir(parents=True, exist_ok=True)
        fetch_crea().to_csv(RAW / "crea.csv", index=False)

    @task
    def fetch_boc():
        from housing.data import fetch_boc
        RAW.mkdir(parents=True, exist_ok=True)
        fetch_boc().to_csv(RAW / "boc.csv", index=False)

    @task
    def validate():
        from housing.data import validate
        validate(_read("crea"), _read("boc"))

    @task.short_circuit
    def has_new_month() -> bool:
        latest = f"{_read('crea')['date'].max():%Y-%m}"
        return not LAST_TRAINED.exists() or LAST_TRAINED.read_text().strip() != latest

    @task
    def build_features():
        from housing.features import build_features
        build_features(_read("crea"), _read("boc")).to_csv(RAW / "features.csv", index=False)

    @task
    def train_evaluate_promote() -> dict:
        from housing.model import promote, train_evaluate
        model, metrics = train_evaluate(_read("features"), PROD, DATA / "mlflow.db")
        metrics["promoted"] = promote(model, metrics, PROD)
        LAST_TRAINED.write_text(metrics["data_month"])
        return metrics

    @task
    def publish_forecasts():
        from housing.model import publish_forecasts
        publish_forecasts(_read("features"), PROD)

    [fetch_crea(), fetch_boc()] >> validate() >> has_new_month() >> build_features() >> train_evaluate_promote() >> publish_forecasts()


housing_pipeline()
