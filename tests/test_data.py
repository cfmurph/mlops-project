import io

import pandas as pd
import pytest

from housing.data import REGIONS, parse_crea, validate


def test_parse_crea_reads_one_sheet_per_region():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        for r in REGIONS + ["ONTARIO"]:
            pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=2, freq="MS"), "Composite_HPI_SA": 100,
                          "Composite_Benchmark_SA": [500_000, 510_000]}).to_excel(w, sheet_name=r, index=False)
    df = parse_crea(io.BytesIO(buf.getvalue()))
    assert list(df.columns) == ["date", "region", "price"]
    assert set(df["region"]) == set(REGIONS)  # other provinces ignored
    assert len(df) == 2 * len(REGIONS)


def _today(crea):
    return crea["date"].max() + pd.DateOffset(months=1)


def test_validate_accepts_good_data(crea, boc):
    validate(crea, boc, today=_today(crea))


def test_validate_rejects_missing_region(crea, boc):
    with pytest.raises(ValueError, match="missing regions"):
        validate(crea[crea["region"] != "CALGARY"], boc, today=_today(crea))


def test_validate_rejects_bad_prices(crea, boc):
    crea.loc[0, "price"] = -1
    with pytest.raises(ValueError, match="non-positive"):
        validate(crea, boc, today=_today(crea))


def test_validate_rejects_stale_data(crea, boc):
    with pytest.raises(ValueError, match="stale"):
        validate(crea, boc, today=crea["date"].max() + pd.DateOffset(months=6))
