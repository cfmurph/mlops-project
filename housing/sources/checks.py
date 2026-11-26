"""Checks shared by sources. Each returns a list of problems (empty when fine)."""
import pandas as pd


def stale(df: pd.DataFrame, today: pd.Timestamp | None, max_age_months: int = 3) -> list[str]:
    latest = df["date"].max()
    if latest < (today or pd.Timestamp.today()) - pd.DateOffset(months=max_age_months):
        return [f"stale: latest month {latest:%Y-%m}"]
    return []


def missing_recent(df: pd.DataFrame, cols: list[str], months: int = 3) -> list[str]:
    recent = df[df["date"] > df["date"].max() - pd.DateOffset(months=months)]
    return [f"missing recent values in {c}" for c in cols if recent[c].isna().any()]


def missing_regions(df: pd.DataFrame, regions: list[str]) -> list[str]:
    missing = set(regions) - set(df["region"])
    return [f"missing regions: {sorted(missing)}"] if missing else []


def raise_if(problems: list[str], source: str) -> None:
    if problems:
        raise ValueError(f"{source}: " + "; ".join(problems))
