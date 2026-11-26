"""Build model features from any set of sources, plus the forecast target."""
import numpy as np
import pandas as pd

from housing.config import Forecast

NOT_FEATURES = {"date", "value", "target"}


def build_features(fc: Forecast, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per region and month.

    - value: the forecast target's current level (e.g. benchmark price)
    - chg_{k}m: log change of value over the past k months
    - target: log change over the next fc.horizon_months (NaN for the most recent months)
    - <source>__<column> and <source>__<column>__chg_3m for every other numeric source column
    Features look only backwards; only `target` looks forwards.
    """
    df = (tables[fc.target_source][["date", "region", fc.target_column]]
          .rename(columns={fc.target_column: "value"})
          .sort_values(["region", "date"]).reset_index(drop=True))
    log_value = np.log(df["value"])
    by_region = log_value.groupby(df["region"])
    for k in (1, 3, 6, 12):
        df[f"chg_{k}m"] = log_value - by_region.shift(k)
    df["month"] = df["date"].dt.month
    df["target"] = by_region.shift(-fc.horizon_months) - log_value

    for name in fc.sources:
        df = df.merge(_source_features(name, tables[name], fc), on=_keys(tables[name]), how="left")
    return df


def _keys(table: pd.DataFrame) -> list[str]:
    return ["date", "region"] if "region" in table else ["date"]


def _source_features(name: str, table: pd.DataFrame, fc: Forecast) -> pd.DataFrame:
    keys = _keys(table)
    skip = fc.target_column if name == fc.target_source else None  # already used as `value`
    cols = [c for c in table.select_dtypes("number").columns if c not in keys and c != skip]
    t = table.sort_values(keys).reset_index(drop=True)
    past = t.groupby("region")[cols].shift(3) if "region" in keys else t[cols].shift(3)
    out = t[keys].copy()
    for c in cols:
        out[f"{name}__{c}"] = t[c]
        out[f"{name}__{c}__chg_3m"] = t[c] - past[c]
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NOT_FEATURES]
