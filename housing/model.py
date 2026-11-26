"""Candidates, training with a promotion gate, and the MLflow model registry."""
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from housing import storage
from housing.config import Forecast
from housing.features import feature_columns

ALIAS = "production"


class DampedMomentum:
    """Rule-based candidate: half of the last 3 months' change continues."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return 0.5 * X["chg_3m"].fillna(0).to_numpy()


# To add a candidate: add it here and list its name under `candidates` in config.yaml.
CANDIDATES = {
    "gradient_boosting": lambda: HistGradientBoostingRegressor(
        max_depth=2, learning_rate=0.05, max_iter=100, min_samples_leaf=50, random_state=0),
    "damped_momentum": DampedMomentum,
}


class ForecastModel:
    """An estimator plus the inputs it was trained on, so it keeps working when regions or sources change.

    Missing feature columns become NaN; regions it never saw become a missing category.
    """

    def __init__(self, estimator, features: list[str], regions: list[str]):
        self.estimator, self.features, self.regions = estimator, features, regions

    def _input(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.reindex(columns=self.features)
        return X.assign(region=pd.Categorical(df["region"], categories=self.regions))

    def fit(self, df: pd.DataFrame) -> "ForecastModel":
        self.estimator.fit(self._input(df), df["target"])
        return self

    def predict(self, df: pd.DataFrame):
        return self.estimator.predict(self._input(df))


def split(df: pd.DataFrame, fc: Forecast) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split. Training targets must end before the holdout starts, so leave a horizon-sized gap."""
    labeled = df.dropna(subset=["target"])
    holdout_start = labeled["date"].max() - pd.DateOffset(months=fc.holdout_months - 1)
    train = labeled[labeled["date"] < holdout_start - pd.DateOffset(months=fc.horizon_months)]
    return train[train["date"] >= fc.train_start], labeled[labeled["date"] >= holdout_start]


def mae(model, df: pd.DataFrame) -> float:
    """Mean absolute error on log change (0.01 is roughly 1% of the value)."""
    return float(mean_absolute_error(df["target"], model.predict(df)))


def should_promote(metrics: dict) -> bool:
    beats_baseline = metrics["mae"] < metrics["baseline_mae"]
    beats_production = metrics["production_mae"] is None or metrics["mae"] <= metrics["production_mae"]
    return beats_baseline and beats_production


# --- MLflow ---------------------------------------------------------------------------------

def use_experiment(name: str) -> None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{storage.uri('mlflow.db')}"))
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=storage.absolute_uri("mlartifacts", name))
    mlflow.set_experiment(name)


def load_production(name: str) -> tuple[ForecastModel, dict] | None:
    """The model behind the `production` alias and its version info, or None if nothing is promoted yet."""
    use_experiment(name)
    try:
        version = MlflowClient().get_model_version_by_alias(name, ALIAS)
    except MlflowException:
        return None
    model = mlflow.sklearn.load_model(f"models:/{name}@{ALIAS}")
    return model, {"model_version": int(version.version), "holdout_mae": float(version.tags["holdout_mae"])}


def train_evaluate_promote(fc: Forecast, df: pd.DataFrame, regions: list[str]) -> dict:
    """Fit every candidate, keep the best, and register it as production if it passes the gate."""
    use_experiment(fc.name)
    train, test = split(df, fc)
    features = feature_columns(df)
    fitted = {n: ForecastModel(CANDIDATES[n](), features, regions).fit(train) for n in fc.candidates}
    scores = {n: mae(m, test) for n, m in fitted.items()}
    best = min(scores, key=scores.get)
    production = load_production(fc.name)
    metrics = {
        "model": best,
        "mae": scores[best],
        **{f"{n}_mae": s for n, s in scores.items()},
        "baseline_mae": float(test["target"].abs().mean()),  # predicting no change
        "production_mae": mae(production[0], test) if production else None,
        "data_month": f"{df['date'].max():%Y-%m}",
    }
    metrics["promoted"] = should_promote(metrics)

    with mlflow.start_run(run_name=metrics["data_month"]):
        mlflow.log_params({"model": best, "data_month": metrics["data_month"], "horizon_months": fc.horizon_months,
                           "holdout_months": fc.holdout_months, "train_start": fc.train_start,
                           "n_features": len(features), "n_regions": len(regions)})
        mlflow.log_metrics({k: v for k, v in metrics.items() if k.endswith("mae") and v is not None})
        mlflow.set_tags({"promoted": metrics["promoted"], "env": os.environ.get("HOUSING_ENV", "dev")})
        if metrics["promoted"]:
            info = mlflow.sklearn.log_model(fitted[best], name="model", registered_model_name=fc.name,
                                            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
            client = MlflowClient()
            version = info.registered_model_version
            client.set_model_version_tag(fc.name, version, "holdout_mae", str(metrics["mae"]))
            client.set_registered_model_alias(fc.name, ALIAS, version)
            metrics["model_version"] = int(version)
    return metrics
