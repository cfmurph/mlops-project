"""Data sources. To add one, create a module with this contract and register it below:

    NAME: str                                   # used in config.yaml
    fetch(regions: list[str]) -> DataFrame      # monthly rows dated the 1st of the month:
                                                #   date, [region,] numeric columns...
                                                #   include `region` only for regional data
    validate(df, regions, today=None) -> None   # raise ValueError listing every problem

Every numeric column then becomes a model feature (level + 3-month change) automatically.
"""
from housing.sources import boc_rates, crea_hpi

SOURCES = {m.NAME: m for m in (crea_hpi, boc_rates)}
