import pandas as pd

from housing import storage
from housing.model import CANDIDATES
from housing.sources import SOURCES


def test_config_references_only_known_parts(cfg):
    assert cfg.regions
    for fc in cfg.forecasts.values():
        assert set(fc.sources) <= set(SOURCES), "add the source to housing/sources/__init__.py"
        assert set(fc.candidates) <= set(CANDIDATES), "add the candidate to CANDIDATES in housing/model.py"
        assert fc.target_source in fc.sources


def test_storage_round_trip():
    df = pd.DataFrame({"a": [1, 2]})
    storage.write(df, "x", "y.parquet")
    storage.write_json({"k": 1}, "x", "z.json")
    pd.testing.assert_frame_equal(storage.read("x", "y.parquet"), df)
    assert storage.read_json("x", "z.json") == {"k": 1}
    assert storage.read_json("x", "missing.json", default={}) == {}
    assert storage.list_files("x") == ["y.parquet", "z.json"]
    assert storage.list_files("nope") == []
