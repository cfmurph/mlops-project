"""One monitor_<name> DAG per forecast: after new forecasts are published, score past forecasts against actuals.

The task fails (Airflow's alert) when live error drifts above max_error_ratio x the model's holdout error.
"""
from datetime import datetime

from airflow.sdk import dag, task

from housing import assets
from housing.config import Forecast, load

CFG = load()


def monitor_dag(fc: Forecast):
    @dag(dag_id=f"monitor_{fc.name}", schedule=[assets.forecasts(fc.name)], start_date=datetime(2026, 1, 1),
         catchup=False, max_active_runs=1, tags=["housing"])
    def _dag():
        @task
        def score_past_forecasts() -> int:
            from housing import pipeline
            return len(pipeline.monitor(fc))

        score_past_forecasts()

    return _dag()


for fc in CFG.forecasts.values():
    monitor_dag(fc)
