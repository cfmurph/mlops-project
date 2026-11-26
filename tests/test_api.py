import pandas as pd
from fastapi.testclient import TestClient

from housing import storage
from src.app import app

client = TestClient(app)


def test_no_forecasts_yet():
    assert client.get("/forecast/calgary").status_code == 503


def test_forecast_serves_latest_month(cfg):
    assert client.get("/regions").json() == cfg.regions
    row = {"region": "CALGARY", "horizon_month": "2027-01", "current_value": 600000.0, "model_version": 1}
    storage.write(pd.DataFrame([{**row, "as_of": "2026-09", "forecast_value": 1.0}]), "forecasts", "home_price", "as_of=2026-09.parquet")
    storage.write(pd.DataFrame([{**row, "as_of": "2026-10", "forecast_value": 606000.0}]), "forecasts", "home_price", "as_of=2026-10.parquet")
    body = client.get("/forecast/calgary").json()
    assert body["forecast_value"] == 606000.0 and body["name"] == "Calgary" and "CREA" in body["source"]
    assert client.get("/forecast/TORONTO").status_code == 404
    assert client.get("/forecast/calgary?target=rent").status_code == 404
