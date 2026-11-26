"""Weekly: fetch every source in config.yaml. Each success updates that source's Asset, which triggers forecasts."""
from datetime import datetime

from airflow.sdk import dag, task

from housing import assets
from housing.config import load

CFG = load()


@dag(dag_id="ingest", schedule="@weekly", start_date=datetime(2026, 1, 1), catchup=False, max_active_runs=1,
     tags=["housing"])
def ingest():
    for source in CFG.sources:
        @task(task_id=f"fetch_{source}", outlets=[assets.raw(source)])
        def fetch(source: str = source) -> int:
            from housing import pipeline
            return pipeline.ingest(source, CFG)

        fetch()


ingest()
