from pathlib import Path

from airflow.dag_processing.dagbag import DagBag


def test_dags_load_and_are_wired_by_assets(cfg):
    bag = DagBag(dag_folder=str(Path(__file__).parent.parent / "dags"))
    assert bag.import_errors == {}

    ingest = bag.dags["ingest"]
    assert set(ingest.task_ids) == {f"fetch_{s}" for s in cfg.sources}

    for name, fc in cfg.forecasts.items():
        forecast, monitor = bag.dags[f"forecast_{name}"], bag.dags[f"monitor_{name}"]
        assert forecast.task_ids == ["has_new_month", "train_evaluate_promote", "publish_forecasts"]
        assert {a.uri for a in forecast.timetable.asset_condition.objects} == {f"housing://raw/{s}" for s in fc.sources}
        assert [a.uri for a in monitor.timetable.asset_condition.objects] == [f"housing://forecasts/{name}"]
