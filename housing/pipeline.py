"""The steps the DAGs run. Plain functions, so they can be tested and run without Airflow."""
import mlflow
import numpy as np
import pandas as pd

from housing import monitoring, storage
from housing.config import Config, Forecast
from housing.features import build_features
from housing.model import load_production, train_evaluate_promote, use_experiment
from housing.sources import SOURCES


def ingest(source: str, cfg: Config, today: pd.Timestamp | None = None) -> int:
    """Fetch a source, keep an immutable snapshot, validate, then make it the latest version.

    A source that fails validation raises before `latest` changes, so downstream DAGs aren't triggered.
    """
    regions = list(cfg.regions)
    df = SOURCES[source].fetch(regions)
    storage.write(df, "raw", source, f"fetched={pd.Timestamp.today():%Y-%m-%d}.parquet")
    SOURCES[source].validate(df, regions, today)
    storage.write(df, "raw", source, "latest.parquet")
    return len(df)


def _tables(fc: Forecast) -> dict[str, pd.DataFrame]:
    return {s: storage.read("raw", s, "latest.parquet") for s in fc.sources}


def _latest_month(fc: Forecast) -> str:
    return f"{storage.read('raw', fc.target_source, 'latest.parquet')['date'].max():%Y-%m}"


def has_new_month(fc: Forecast) -> bool:
    state = storage.read_json("state", f"{fc.name}.json", default={})
    return state.get("last_trained_month") != _latest_month(fc)


def train(fc: Forecast, cfg: Config) -> dict:
    metrics = train_evaluate_promote(fc, build_features(fc, _tables(fc)), list(cfg.regions))
    storage.write_json({"last_trained_month": metrics["data_month"]}, "state", f"{fc.name}.json")
    return metrics


def publish(fc: Forecast) -> pd.DataFrame | None:
    """Forecast each region from its latest month with the production model; keep every month's forecasts."""
    production = load_production(fc.name)
    if production is None:
        return None  # nothing has passed the gate yet
    model, version = production
    df = build_features(fc, _tables(fc))
    latest = df[df["date"] == df.groupby("region")["date"].transform("max")]
    out = pd.DataFrame({
        "region": latest["region"],
        "as_of": latest["date"].dt.strftime("%Y-%m"),
        "horizon_month": (latest["date"] + pd.DateOffset(months=fc.horizon_months)).dt.strftime("%Y-%m"),
        "current_value": latest["value"].round(),
        "forecast_value": (latest["value"] * np.exp(model.predict(latest))).round(),
        **version,
    })
    storage.write(out, "forecasts", fc.name, f"as_of={out['as_of'].max()}.parquet")
    return out


def monitor(fc: Forecast) -> pd.DataFrame:
    files = storage.list_files("forecasts", fc.name)
    history = pd.concat([storage.read("forecasts", fc.name, f) for f in files]) if files else pd.DataFrame()
    actuals = _tables(fc)[fc.target_source].rename(columns={fc.target_column: "value"})
    report = monitoring.score(history, actuals)
    storage.write(report, "monitoring", f"{fc.name}.parquet")
    if not report.empty:
        use_experiment(fc.name)
        with mlflow.start_run(run_name=f"monitor {report['as_of'].max()}"):
            mlflow.set_tag("type", "monitoring")
            mlflow.log_metric("live_mae_latest", float(report.iloc[-1]["live_mae"]))
    monitoring.check(report, fc.max_error_ratio)
    return report
