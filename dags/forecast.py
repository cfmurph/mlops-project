"""One forecast_<name> DAG per entry under `forecasts:` in config.yaml, triggered when all its sources update."""
from datetime import datetime

from airflow.sdk import dag, task

from housing import assets
from housing.config import Forecast, load

CFG = load()


def forecast_dag(fc: Forecast):
    @dag(dag_id=f"forecast_{fc.name}", schedule=[assets.raw(s) for s in fc.sources], start_date=datetime(2026, 1, 1),
         catchup=False, max_active_runs=1, tags=["housing"])
    def _dag():
        @task.short_circuit
        def has_new_month() -> bool:
            from housing import pipeline
            return pipeline.has_new_month(fc)

        @task
        def train_evaluate_promote() -> dict:
            from housing import pipeline
            return pipeline.train(fc, CFG)

        @task(outlets=[assets.forecasts(fc.name)])
        def publish_forecasts():
            from housing import pipeline
            pipeline.publish(fc)

        has_new_month() >> train_evaluate_promote() >> publish_forecasts()

    return _dag()


for fc in CFG.forecasts.values():
    forecast_dag(fc)
