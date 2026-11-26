import io

import pandas as pd
import pytest

from housing.sources import SOURCES, boc_rates, crea_hpi


def _today(df):
    return df["date"].max() + pd.DateOffset(months=1)


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_every_source_follows_the_contract(name):
    src = SOURCES[name]
    assert src.NAME == name
    assert callable(src.fetch) and callable(src.validate)


def test_crea_parse_reads_configured_regions_only(cfg):
    regions = list(cfg.regions)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        for r in regions + ["ONTARIO"]:
            pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=2, freq="MS"), "Composite_HPI_SA": 100,
                          "Composite_Benchmark_SA": [500_000, 510_000]}).to_excel(w, sheet_name=r, index=False)
    df = crea_hpi.parse(io.BytesIO(buf.getvalue()), regions)
    assert list(df.columns) == ["date", "region", "price"]
    assert set(df["region"]) == set(regions)
    assert len(df) == 2 * len(regions)


def test_crea_validate(cfg, crea):
    regions = list(cfg.regions)
    crea_hpi.validate(crea, regions, today=_today(crea))
    with pytest.raises(ValueError, match="missing regions"):
        crea_hpi.validate(crea[crea["region"] != "CALGARY"], regions, today=_today(crea))
    with pytest.raises(ValueError, match="stale"):
        crea_hpi.validate(crea, regions, today=crea["date"].max() + pd.DateOffset(months=6))
    bad = crea.copy()
    bad.loc[0, "price"] = -1
    with pytest.raises(ValueError, match="non-positive"):
        crea_hpi.validate(bad, regions, today=_today(crea))


def test_boc_validate(boc):
    boc_rates.validate(boc, [], today=_today(boc))
    bad = boc.copy()
    bad.loc[bad.index[-1], "policy_rate"] = None
    with pytest.raises(ValueError, match="policy_rate"):
        boc_rates.validate(bad, [], today=_today(boc))
