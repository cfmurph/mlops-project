import numpy as np
import pandas as pd
import pytest

from housing.monitoring import check, score

ACTUALS = pd.DataFrame({"date": pd.to_datetime(["2026-04-01", "2026-04-01"]), "region": ["CALGARY", "EDMONTON"],
                        "value": [100.0, 200.0]})


def _history(calgary_forecast, edmonton_forecast, horizon="2026-04"):
    return pd.DataFrame({"region": ["CALGARY", "EDMONTON"], "as_of": "2026-01", "horizon_month": horizon,
                         "forecast_value": [calgary_forecast, edmonton_forecast], "model_version": 1,
                         "holdout_mae": 0.01})


def test_score_matches_forecasts_to_actuals():
    report = score(_history(101.0, 198.0), ACTUALS)
    assert len(report) == 1 and report.loc[0, "regions"] == 2
    expected = (abs(np.log(101 / 100)) + abs(np.log(198 / 200))) / 2
    assert np.isclose(report.loc[0, "live_mae"], expected)


def test_forecasts_not_yet_due_are_ignored():
    assert score(_history(101.0, 198.0, horizon="2026-07"), ACTUALS).empty
    assert score(pd.DataFrame(), ACTUALS).empty


def test_check_raises_only_on_drift():
    check(score(_history(101.0, 198.0), ACTUALS), max_error_ratio=1.5)  # ~1% error vs 1% holdout: fine
    with pytest.raises(ValueError, match="drifted"):
        check(score(_history(110.0, 180.0), ACTUALS), max_error_ratio=1.5)  # ~10% error
