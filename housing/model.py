"""Train, evaluate against baselines, promote, and publish forecasts."""
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from housing.features import HORIZON, model_input

HOLDOUT_MONTHS = 24


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split. Training targets must end before the holdout starts, so leave a HORIZON-month gap."""
    labeled = df.dropna(subset=["target"])
    holdout_start = labeled["date"].max() - pd.DateOffset(months=HOLDOUT_MONTHS - 1)
    train = labeled[labeled["date"] < holdout_start - pd.DateOffset(months=HORIZON)]
    return train, labeled[labeled["date"] >= holdout_start]


def _mae(model, test: pd.DataFrame) -> float:
    return float(mean_absolute_error(test["target"], model.predict(model_input(test))))


def train_evaluate(df: pd.DataFrame, prod_dir: Path, mlruns_dir: Path) -> tuple[HistGradientBoostingRegressor, dict]:
    """Fit a candidate and score it, the naive 'no change' baseline, and the production model on one holdout.

    MAE is on 3-month log price change, so 0.02 is roughly 2% of price.
    """
    train, test = split(df)
    model = HistGradientBoostingRegressor(random_state=0).fit(model_input(train), train["target"])
    prod = prod_dir / "model.joblib"
    metrics = {
        "mae": _mae(model, test),
        "baseline_mae": float(test["target"].abs().mean()),  # predicting zero change
        "production_mae": _mae(joblib.load(prod), test) if prod.exists() else None,
        "data_month": f"{df['date'].max():%Y-%m}",
        "train_rows": len(train),
        "holdout_rows": len(test),
    }

    mlflow.set_tracking_uri(mlruns_dir.resolve().as_uri())
    mlflow.set_experiment("housing")
    with mlflow.start_run(run_name=metrics["data_month"]):
        mlflow.log_params({"horizon_months": HORIZON, "holdout_months": HOLDOUT_MONTHS, "data_month": metrics["data_month"]})
        mlflow.log_metrics({k: v for k, v in metrics.items() if k.endswith("mae") and v is not None})
    return model, metrics


def should_promote(metrics: dict) -> bool:
    beats_baseline = metrics["mae"] < metrics["baseline_mae"]
    beats_production = metrics["production_mae"] is None or metrics["mae"] <= metrics["production_mae"]
    return beats_baseline and beats_production


def promote(model, metrics: dict, prod_dir: Path) -> bool:
    if not should_promote(metrics):
        return False
    prod_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, prod_dir / "model.joblib")
    (prod_dir / "metadata.json").write_text(json.dumps({**metrics, "promoted_at": pd.Timestamp.now().isoformat()}, indent=2))
    return True


def publish_forecasts(df: pd.DataFrame, prod_dir: Path) -> pd.DataFrame:
    """Forecast each region's price HORIZON months after its latest month, using the production model."""
    model = joblib.load(prod_dir / "model.joblib")
    latest = df[df["date"] == df.groupby("region")["date"].transform("max")]
    out = pd.DataFrame({
        "region": latest["region"],
        "as_of": latest["date"].dt.strftime("%Y-%m"),
        "horizon_month": (latest["date"] + pd.DateOffset(months=HORIZON)).dt.strftime("%Y-%m"),
        "current_price": latest["price"].round(),
        "forecast_price": (latest["price"] * np.exp(model.predict(model_input(latest)))).round(),
    })
    out.to_csv(prod_dir / "forecasts.csv", index=False)
    return out
