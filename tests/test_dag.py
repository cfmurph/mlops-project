from pathlib import Path

from airflow.dag_processing.dagbag import DagBag


def test_dag_loads():
    bag = DagBag(dag_folder=str(Path(__file__).parent.parent / "dags"), include_examples=False)
    assert bag.import_errors == {}
    dag = bag.get_dag("housing_pipeline")
    assert dag.task_ids[0] in {"fetch_crea", "fetch_boc"} and "publish_forecasts" in dag.task_ids
