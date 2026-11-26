import dataclasses

import numpy as np
import pandas as pd

from housing import pipeline, storage
from housing.features import build_features, feature_columns
from housing.model import CANDIDATES, ForecastModel, load_production, should_promote, split


def test_target_is_future_change(fc, tables):
    df = build_features(fc, tables)
    cal = df[df["region"] == "CALGARY"].reset_index(drop=True)
    h = fc.horizon_months
    assert np.isclose(cal.loc[0, "target"], np.log(cal.loc[h, "value"] / cal.loc[0, "value"]))
    assert cal["target"].tail(h).isna().all()


def test_features_do_not_use_future_values(fc, tables, crea):
    before = build_features(fc, tables)
    cutoff = crea["date"].sort_values().unique()[60]
    changed = {**tables, "crea_hpi": crea.assign(price=crea["price"].where(crea["date"] <= cutoff, 1.0))}
    after = build_features(fc, changed)
    cols = [c for c in feature_columns(before) if c != "region"]
    mask = before["date"] <= cutoff
    pd.testing.assert_frame_equal(before.loc[mask, cols], after.loc[mask, cols])


def test_new_source_columns_become_features_automatically(fc, tables, boc):
    unemployment = boc[["date"]].assign(rate=np.linspace(7, 5, len(boc)))
    fc2 = dataclasses.replace(fc, sources=fc.sources + ("statcan_jobs",))
    df = build_features(fc2, {**tables, "statcan_jobs": unemployment})
    assert {"statcan_jobs__rate", "statcan_jobs__rate__chg_3m", "boc_rates__mortgage_5y"} <= set(feature_columns(df))


def test_split_has_gap_so_training_targets_never_reach_holdout(fc, tables):
    train, test = split(build_features(fc, tables), fc)
    assert train["date"].max() + pd.DateOffset(months=fc.horizon_months) < test["date"].min()
    assert train["date"].min() >= pd.Timestamp(fc.train_start)


def test_model_survives_new_regions_and_missing_features(fc, tables, cfg):
    df = build_features(fc, tables).dropna(subset=["target"])
    model = ForecastModel(CANDIDATES["gradient_boosting"](), feature_columns(df), list(cfg.regions)).fit(df)
    new = df.tail(5).assign(region="GREATER_TORONTO").drop(columns=["boc_rates__policy_rate"])
    assert np.isfinite(model.predict(new)).all()


def test_gate():
    assert should_promote({"mae": 0.01, "baseline_mae": 0.02, "production_mae": None})
    assert should_promote({"mae": 0.01, "baseline_mae": 0.02, "production_mae": 0.015})
    assert not should_promote({"mae": 0.03, "baseline_mae": 0.02, "production_mae": None})
    assert not should_promote({"mae": 0.015, "baseline_mae": 0.02, "production_mae": 0.01})


def test_end_to_end_train_register_publish(fc, cfg, tables):
    for name, df in tables.items():
        storage.write(df, "raw", name, "latest.parquet")
    assert pipeline.publish(fc) is None  # nothing promoted yet
    assert pipeline.has_new_month(fc)

    metrics = pipeline.train(fc, cfg)
    assert metrics["promoted"] and metrics["model_version"] == 1
    assert metrics["mae"] == min(metrics[f"{c}_mae"] for c in fc.candidates)
    assert not pipeline.has_new_month(fc)

    model, version = load_production(fc.name)
    assert version["model_version"] == 1 and version["holdout_mae"] == metrics["mae"]

    out = pipeline.publish(fc)
    assert len(out) == len(cfg.regions) and (out["model_version"] == 1).all()
    assert storage.list_files("forecasts", fc.name) == [f"as_of={out['as_of'].max()}.parquet"]

    # Same data again ties with production, so it promotes as version 2.
    assert pipeline.train(fc, cfg)["model_version"] == 2
