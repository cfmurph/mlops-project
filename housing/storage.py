"""All file IO goes through here. DATA_URI is a local folder or a cloud URI (s3://bucket/x, gs://bucket/x)."""
import json
import os
from pathlib import Path

import fsspec
import pandas as pd


def uri(*parts: str) -> str:
    return "/".join([os.environ.get("DATA_URI", "data").rstrip("/"), *parts])


def absolute_uri(*parts: str) -> str:
    """Like uri() but absolute, for tools that need a full URI (MLflow artifact locations)."""
    u = uri(*parts)
    return u if "://" in u else Path(u).resolve().as_uri()


def _fs(path: str):
    return fsspec.core.url_to_fs(path)


def exists(*parts: str) -> bool:
    fs, path = _fs(uri(*parts))
    return fs.exists(path)


def list_files(*parts: str) -> list[str]:
    """File names (not paths) directly under a folder, sorted; empty if the folder doesn't exist."""
    fs, path = _fs(uri(*parts))
    return sorted(Path(p).name for p in fs.ls(path, detail=False)) if fs.exists(path) else []


def write(df: pd.DataFrame, *parts: str) -> None:
    fs, path = _fs(uri(*parts))
    fs.makedirs(str(Path(path).parent), exist_ok=True)
    df.to_parquet(uri(*parts), index=False)


def read(*parts: str) -> pd.DataFrame:
    return pd.read_parquet(uri(*parts))


def write_json(obj, *parts: str) -> None:
    fs, path = _fs(uri(*parts))
    fs.makedirs(str(Path(path).parent), exist_ok=True)
    with fs.open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(*parts: str, default=None):
    if not exists(*parts):
        return default
    fs, path = _fs(uri(*parts))
    with fs.open(path) as f:
        return json.load(f)
