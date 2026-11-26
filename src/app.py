"""Serve the latest published housing forecasts. Run: uvicorn src.app:app"""
import csv
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

FORECASTS = Path(os.environ.get("FORECASTS_PATH", "data/production/forecasts.csv"))
SOURCE = "Source: CREA MLS® HPI (seasonally adjusted composite benchmark price); Bank of Canada"
app = FastAPI(title="Canadian housing forecast")


def _forecasts() -> dict[str, dict]:
    # Read on every request so newly published forecasts appear without a restart.
    if not FORECASTS.exists():
        raise HTTPException(503, "No forecasts yet: run the housing_pipeline DAG")
    with FORECASTS.open() as f:
        return {row["region"]: row for row in csv.DictReader(f)}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/regions")
def regions():
    return sorted(_forecasts())


@app.get("/forecast/{region}")
def forecast(region: str):
    row = _forecasts().get(region.upper())
    if row is None:
        raise HTTPException(404, f"Unknown region {region!r}; see /regions")
    return {**row, "current_price": float(row["current_price"]), "forecast_price": float(row["forecast_price"]), "source": SOURCE}
