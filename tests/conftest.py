import os
import tempfile

# Keep Airflow from writing config/db into ~/airflow during tests.
os.environ.setdefault("AIRFLOW_HOME", tempfile.mkdtemp())

import numpy as np
import pandas as pd
import pytest

from housing.data import REGIONS


@pytest.fixture
def crea() -> pd.DataFrame:
    """Synthetic monthly prices: each region grows at its own steady rate, plus noise."""
    dates = pd.date_range("2015-01-01", periods=120, freq="MS")
    rng = np.random.default_rng(0)
    rows = [
        {"date": d, "region": r, "price": 400_000 * np.exp(0.003 * (i + 1) * t / 12 + rng.normal(0, 0.002))}
        for i, r in enumerate(REGIONS) for t, d in enumerate(dates)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def boc(crea) -> pd.DataFrame:
    dates = sorted(crea["date"].unique())
    return pd.DataFrame({"date": dates, "mortgage_5y": np.linspace(5, 6, len(dates)), "policy_rate": np.linspace(1, 3, len(dates))})
