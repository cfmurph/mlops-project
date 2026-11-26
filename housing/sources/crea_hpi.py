"""CREA MLS® Home Price Index: monthly seasonally adjusted composite benchmark price per region."""
import io
import re
import zipfile

import pandas as pd
import requests

from housing.sources import checks

NAME = "crea_hpi"
PAGE = "https://www.crea.ca/housing-market-stats/mls-home-price-index/hpi-tool/"
ZIP_RE = r"https://www\.crea\.ca/files/mls-hpi-data/[^\"']+\.zip"
XLSX = "Seasonally Adjusted (M).xlsx"
HEADERS = {"User-Agent": "Mozilla/5.0 (housing-forecast pipeline)"}


def _get(url: str) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r


def parse(xlsx, regions: list[str]) -> pd.DataFrame:
    """One sheet per region -> date, region, price."""
    sheets = pd.read_excel(xlsx, sheet_name=list(regions), usecols=["Date", "Composite_Benchmark_SA"])
    df = pd.concat(s.assign(region=r) for r, s in sheets.items())
    return df.rename(columns={"Date": "date", "Composite_Benchmark_SA": "price"})[["date", "region", "price"]]


def fetch(regions: list[str]) -> pd.DataFrame:
    # The zip's filename changes every month, so find the current link on the page.
    url = re.search(ZIP_RE, _get(PAGE).text).group(0)
    with zipfile.ZipFile(io.BytesIO(_get(url).content)) as zf:
        return parse(zf.open(XLSX), regions)


def validate(df: pd.DataFrame, regions: list[str], today: pd.Timestamp | None = None) -> None:
    problems = checks.missing_regions(df, regions) + checks.stale(df, today)
    if (df["price"] <= 0).any():
        problems.append("non-positive prices")
    latest = df[df["date"] == df["date"].max()]
    if latest["price"].isna().any():
        problems.append("missing prices in latest month")
    checks.raise_if(problems, NAME)
