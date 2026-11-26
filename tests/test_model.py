import numpy as np
import pandas as pd

from housing.features import HORIZON, build_features
from housing.model import promote, publish_forecasts, should_promote, split, train_evaluate


def test_target_is_future_change(crea, boc):
    df = build_features(crea, boc)
    cal = df[df["region"] == "CALGARY"].reset_index(drop=True)
    assert np.isclose(cal.loc[0, "target"], np.log(cal.loc[HORIZON, "price"] / cal.loc[0, "price"]))
    assert cal["target"].tail(HORIZON).isna().all()


def test_features_do_not_use_future_prices(crea, boc):
    before = build_features(crea, boc)
    cutoff = crea["date"].sort_values().unique()[60]
    changed = crea.assign(price=crea["price"].where(crea["date"] <= cutoff, 1.0))
    after = build_features(changed, boc)
    cols = [c for c in before if c.startswith("chg_")]
    mask = before["date"] <= cutoff
    pd.testing.assert_frame_equal(before.loc[mask, cols], after.loc[mask, cols])


def test_split_has_gap_so_training_targets_never_reach_holdout(crea, boc):
    train, test = split(build_features(crea, boc))
    assert train["date"].max() + pd.DateOffset(months=HORIZON) < test["date"].min()


def test_gate():
    assert should_promote({"mae": 0.01, "baseline_mae": 0.02, "production_mae": None})
    assert should_promote({"mae": 0.01, "baseline_mae": 0.02, "production_mae": 0.015})
    assert not should_promote({"mae": 0.03, "baseline_mae": 0.02, "production_mae": None})
    assert not should_promote({"mae": 0.015, "baseline_mae": 0.02, "production_mae": 0.01})


def test_end_to_end_train_promote_publish(crea, boc, tmp_path):
    df = build_features(crea, boc)
    prod = tmp_path / "production"
    model, metrics = train_evaluate(df, prod, tmp_path / "mlruns")
    assert metrics["mae"] < metrics["baseline_mae"]  # synthetic data has a learnable trend
    assert promote(model, metrics, prod)
    out = publish_forecasts(df, prod)
    assert len(out) == df["region"].nunique()
    assert (prod / "forecasts.csv").exists() and (prod / "metadata.json").exists()

    # Retraining on the same data scores the same as production, so it still promotes (ties allowed).
    _, metrics2 = train_evaluate(df, prod, tmp_path / "mlruns")
    assert metrics2["production_mae"] is not None and should_promote(metrics2)
