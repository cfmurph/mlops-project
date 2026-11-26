"""Bank of Canada Valet API: 5-year conventional mortgage rate and policy rate, as monthly averages."""
import pandas as pd
import requests

from housing.sources import checks

NAME = "boc_rates"
URL = "https://www.bankofcanada.ca/valet/observations/{}/json"
SERIES = {"V80691335": "mortgage_5y", "V39079": "policy_rate"}


def fetch(regions: list[str]) -> pd.DataFrame:
    cols = []
    for series, name in SERIES.items():
        r = requests.get(URL.format(series), timeout=120)
        r.raise_for_status()
        obs = r.json()["observations"]
        s = pd.Series({o["d"]: float(o[series]["v"]) for o in obs if o.get(series, {}).get("v")}, name=name)
        cols.append(s.set_axis(pd.to_datetime(s.index)).resample("MS").mean())
    return pd.concat(cols, axis=1).rename_axis("date").reset_index()


def validate(df: pd.DataFrame, regions: list[str], today: pd.Timestamp | None = None) -> None:
    checks.raise_if(checks.stale(df, today) + checks.missing_recent(df, list(SERIES.values())), NAME)
