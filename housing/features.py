"""Turn prices + rates into model features and a 3-month-ahead target."""
import numpy as np
import pandas as pd

from housing.data import REGIONS

HORIZON = 3  # months ahead to forecast
FEATURES = [
    "region", "month", "chg_1m", "chg_3m", "chg_6m", "chg_12m",
    "mortgage_5y", "policy_rate", "mortgage_5y_chg_3m", "policy_rate_chg_3m",
]


def build_features(crea: pd.DataFrame, boc: pd.DataFrame) -> pd.DataFrame:
    df = crea.merge(boc, on="date", how="left").sort_values(["region", "date"]).reset_index(drop=True)
    by_region = df.groupby("region")
    log_price = np.log(df["price"])
    log_by_region = log_price.groupby(df["region"])

    # Features look only backwards (shift(k) with k > 0).
    for k in (1, 3, 6, 12):
        df[f"chg_{k}m"] = log_price - log_by_region.shift(k)
    for c in ("mortgage_5y", "policy_rate"):
        df[f"{c}_chg_3m"] = df[c] - by_region[c].shift(3)
    df["month"] = df["date"].dt.month

    # Target looks forwards; it is NaN for the last HORIZON months of each region.
    df["target"] = log_by_region.shift(-HORIZON) - log_price
    return df


def model_input(df: pd.DataFrame) -> pd.DataFrame:
    """Feature columns with region as a fixed categorical, so train and predict agree."""
    return df[FEATURES].assign(region=pd.Categorical(df["region"], categories=REGIONS))
