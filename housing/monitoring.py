"""Score past forecasts against what actually happened."""
import numpy as np
import pandas as pd


def score(history: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Live error per forecast month and model version, for forecasts whose horizon month has arrived.

    history: published forecasts (region, as_of, horizon_month, forecast_value, model_version, holdout_mae)
    actuals: target source rows (date, region, value)
    Error is |log forecast - log actual|, the same scale as the holdout MAE.
    """
    cols = ["as_of", "model_version", "holdout_mae", "live_mae", "regions"]
    if history.empty:
        return pd.DataFrame(columns=cols)
    actual = actuals.assign(horizon_month=actuals["date"].dt.strftime("%Y-%m"))[["region", "horizon_month", "value"]]
    m = history.merge(actual.rename(columns={"value": "actual"}), on=["region", "horizon_month"])
    m["abs_error"] = (np.log(m["forecast_value"]) - np.log(m["actual"])).abs()
    return (m.groupby(["as_of", "model_version", "holdout_mae"], as_index=False)
             .agg(live_mae=("abs_error", "mean"), regions=("region", "count")))[cols]


def check(report: pd.DataFrame, max_error_ratio: float) -> None:
    """Raise when any forecast month's live error is much worse than its model's holdout error."""
    bad = report[report["live_mae"] > max_error_ratio * report["holdout_mae"]]
    if not bad.empty:
        raise ValueError(f"forecast error drifted above {max_error_ratio}x holdout error:\n{bad.to_string(index=False)}")
