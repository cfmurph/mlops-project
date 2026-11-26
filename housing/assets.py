"""Airflow Assets that connect the DAGs: ingest -> forecast_<name> -> monitor_<name>."""
from airflow.sdk import Asset


def raw(source: str) -> Asset:
    return Asset(f"housing://raw/{source}")


def forecasts(name: str) -> Asset:
    return Asset(f"housing://forecasts/{name}")
