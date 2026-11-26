from fastapi.testclient import TestClient

import src.app as app_module

client = TestClient(app_module.app)


def test_no_forecasts_yet(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "FORECASTS", tmp_path / "missing.csv")
    assert client.get("/regions").status_code == 503


def test_forecast(tmp_path, monkeypatch):
    f = tmp_path / "forecasts.csv"
    f.write_text("region,as_of,horizon_month,current_price,forecast_price\nCALGARY,2026-10,2027-01,600000,606000\n")
    monkeypatch.setattr(app_module, "FORECASTS", f)
    assert client.get("/regions").json() == ["CALGARY"]
    body = client.get("/forecast/calgary").json()
    assert body["forecast_price"] == 606000.0 and "CREA" in body["source"]
    assert client.get("/forecast/TORONTO").status_code == 404
