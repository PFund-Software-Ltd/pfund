from typing import Any, Literal

import os

from pydantic import BaseModel, Field, ConfigDict, field_validator

from pfeed.enums import IOFormat
from pfund.enums.database import Database


class BaseEngineSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid', frozen=True)

    auto_download_data: bool = Field(
        default=False,
        description="""
            if True, the engine will automatically download data using pfeed if no data is found in the storage specified in add_data()'s storage_config.
            if False, the engine will not download data automatically.
        """,
    )
    cache_materialized_data: bool | Literal['auto'] = Field(
        default='auto',
        description="""
            Controls whether materialized data is cached to the CURATED data layer for faster future retrieval.
            Materialized data is the output of MarketDataStore.materialize(), where raw/stored data is
            processed (e.g. cleaned from RAW, resampled from tick to bar) into a format pfund can use.
            - 'auto': cache only when the stored resolution differs from the requested resolution (e.g. tick data resampled to second bars).
            - True: always cache retrieved data to the CURATED layer.
            - False: never cache, always process on the fly.
        """,
    )

    num_batch_workers: int | None = Field(
        default=None,
        description="""
            number of workers to fetch data using pfeed, equivalent to the parameter 'num_batch_workers' in pfeed.
            if None, Ray will NOT be used for fetching data and it will be done sequentially.
        """,
    )
    storage_options: dict[Database, dict[str, Any]] = Field(default_factory=dict)
    io_options: dict[IOFormat, dict[str, Any]] = Field(default_factory=dict)
 
    @field_validator('cache_materialized_data', mode='before')
    @classmethod
    def _normalize_cache_resampled_data(cls, v: Any) -> bool | str:
        if isinstance(v, str):
            return v.lower()
        return v