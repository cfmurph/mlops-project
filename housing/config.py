"""Load config.yaml into typed objects."""
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass(frozen=True)
class Forecast:
    name: str
    target: str  # "<source>.<column>"
    sources: tuple[str, ...]
    horizon_months: int
    holdout_months: int
    train_start: str
    candidates: tuple[str, ...]
    max_error_ratio: float

    @property
    def target_source(self) -> str:
        return self.target.split(".")[0]

    @property
    def target_column(self) -> str:
        return self.target.split(".")[1]


@dataclass(frozen=True)
class Config:
    regions: dict[str, str]  # CREA sheet name -> display name
    forecasts: dict[str, Forecast]

    @property
    def sources(self) -> list[str]:
        return sorted({s for f in self.forecasts.values() for s in f.sources})


def load(path: str | Path | None = None) -> Config:
    raw = yaml.safe_load(Path(path or os.environ.get("HOUSING_CONFIG", DEFAULT_PATH)).read_text())
    forecasts = {
        name: Forecast(
            name=name, target=f["target"], sources=tuple(f["sources"]), horizon_months=f["horizon_months"],
            holdout_months=f["holdout_months"], train_start=str(f["train_start"]),
            candidates=tuple(f["candidates"]), max_error_ratio=f["monitor"]["max_error_ratio"],
        )
        for name, f in raw["forecasts"].items()
    }
    return Config(regions=raw["regions"], forecasts=forecasts)
