"""Download and validate CREA MLS® HPI prices and Bank of Canada interest rates."""
import io
import re
import zipfile

import pandas as pd
import requests

CREA_PAGE = "https://www.crea.ca/housing-market-stats/mls-home-price-index/hpi-tool/"
CREA_ZIP_RE = r"https://www\.crea\.ca/files/mls-hpi-data/[^\"']+\.zip"
CREA_FILE = "Seasonally Adjusted (M).xlsx"
BOC_URL = "https://www.bankofcanada.ca/valet/observations/{}/json"
BOC_SERIES = {"V80691335": "mortgage_5y", "V39079": "policy_rate"}
HEADERS = {"User-Agent": "Mozilla/5.0 (housing-forecast pipeline)"}

REGIONS = [
    "AGGREGATE",  # all of Canada
    "BRITISH_COLUMBIA", "GREATER_VANCOUVER", "FRASER_VALLEY", "LOWER_MAINLAND", "VICTORIA",
    "VANCOUVER_ISLAND", "INTERIOR_BC", "CHILLIWACK_AND_DISTRICT",
    "ALBERTA", "CALGARY", "EDMONTON",
]


def _get(url: str) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r


def parse_crea(xlsx) -> pd.DataFrame:
    """One sheet per region -> long table: date, region, price (benchmark $, seasonally adjusted)."""
    sheets = pd.read_excel(xlsx, sheet_name=REGIONS, usecols=["Date", "Composite_Benchmark_SA"])
    df = pd.concat(s.assign(region=r) for r, s in sheets.items())
    return df.rename(columns={"Date": "date", "Composite_Benchmark_SA": "price"})[["date", "region", "price"]]


def fetch_crea() -> pd.DataFrame:
    # The zip's filename changes every month, so find the current link on the page.
    url = re.search(CREA_ZIP_RE, _get(CREA_PAGE).text).group(0)
    with zipfile.ZipFile(io.BytesIO(_get(url).content)) as zf:
        return parse_crea(zf.open(CREA_FILE))


def fetch_boc() -> pd.DataFrame:
    """Monthly averages of each Bank of Canada series: date, mortgage_5y, policy_rate."""
    cols = []
    for series, name in BOC_SERIES.items():
        obs = _get(BOC_URL.format(series)).json()["observations"]
        s = pd.Series({o["d"]: float(o[series]["v"]) for o in obs if o.get(series, {}).get("v")}, name=name)
        cols.append(s.set_axis(pd.to_datetime(s.index)).resample("MS").mean())
    return pd.concat(cols, axis=1).rename_axis("date").reset_index()


def validate(crea: pd.DataFrame, boc: pd.DataFrame, today: pd.Timestamp | None = None, max_age_months: int = 3) -> None:
    """Raise ValueError listing every problem, so a bad download stops the pipeline."""
    today = today or pd.Timestamp.today()
    problems = []
    if missing := set(REGIONS) - set(crea["region"]):
        problems.append(f"missing regions: {sorted(missing)}")
    if (crea["price"] <= 0).any():
        problems.append("non-positive prices")
    latest = crea["date"].max()
    if latest < today - pd.DateOffset(months=max_age_months):
        problems.append(f"CREA data is stale: latest month {latest:%Y-%m}")
    if crea.loc[crea["date"] == latest, "price"].isna().any():
        problems.append("missing prices in latest month")
    if boc.tail(3)[list(BOC_SERIES.values())].isna().any().any():
        problems.append("missing Bank of Canada rates in recent months")
    if problems:
        raise ValueError("; ".join(problems))
