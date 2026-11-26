"""Serve the latest published forecasts. Run: uvicorn src.app:app"""
from fastapi import FastAPI, HTTPException

from housing import storage
from housing.config import load

CFG = load()
SOURCE = "Source: CREA MLS® HPI (seasonally adjusted composite benchmark price); Bank of Canada"
app = FastAPI(title="Canadian housing forecast")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/regions")
def regions():
    return CFG.regions


@app.get("/forecast/{region}")
def forecast(region: str, target: str = "home_price"):
    if target not in CFG.forecasts:
        raise HTTPException(404, f"Unknown target {target!r}; options: {sorted(CFG.forecasts)}")
    files = storage.list_files("forecasts", target)  # read per request, so new forecasts appear without a restart
    if not files:
        raise HTTPException(503, f"No forecasts yet: run the forecast_{target} DAG")
    df = storage.read("forecasts", target, files[-1])
    rows = df[df["region"] == region.upper()]
    if rows.empty:
        raise HTTPException(404, f"Unknown region {region!r}; see /regions")
    row = rows.iloc[0]
    return {
        "region": row["region"], "name": CFG.regions.get(row["region"], row["region"]), "target": target,
        "as_of": row["as_of"], "horizon_month": row["horizon_month"],
        "current_value": float(row["current_value"]), "forecast_value": float(row["forecast_value"]),
        "model_version": int(row["model_version"]), "source": SOURCE,
    }
