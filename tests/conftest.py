import os
import tempfile

# Keep Airflow from writing config/db into ~/airflow during tests.
os.environ.setdefault("AIRFLOW_HOME", tempfile.mkdtemp())

import numpy as np
import pandas as pd
import pytest

from housing.config import load


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Every test gets its own data folder and MLflow database."""
    monkeypatch.setenv("DATA_URI", str(tmp_path / "data"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")


@pytest.fixture
def cfg():
    return load()


@pytest.fixture
def fc(cfg):
    return cfg.forecasts["home_price"]


@pytest.fixture
def crea(cfg) -> pd.DataFrame:
    """Synthetic monthly prices: each region grows at its own steady rate, plus noise."""
    dates = pd.date_range("2010-01-01", periods=180, freq="MS")
    rng = np.random.default_rng(0)
    rows = [
        {"date": d, "region": r, "price": 400_000 * np.exp(0.003 * (i + 1) * t / 12 + rng.normal(0, 0.002))}
        for i, r in enumerate(cfg.regions) for t, d in enumerate(dates)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def boc(crea) -> pd.DataFrame:
    dates = sorted(crea["date"].unique())
    return pd.DataFrame({"date": dates, "mortgage_5y": np.linspace(5, 6, len(dates)),
                         "policy_rate": np.linspace(1, 3, len(dates))})


@pytest.fixture
def tables(crea, boc) -> dict[str, pd.DataFrame]:
    return {"crea_hpi": crea, "boc_rates": boc}
