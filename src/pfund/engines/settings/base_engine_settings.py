from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pfund.enums import DataLake


class BaseEngineSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    datalake: DataLake | str = Field(
        default=DataLake.DELTALAKE,
        description="data lake for writing and appending component data",
    )
    persist: bool = Field(
        default=False,
        description="""
            if True, the settings will be persisted to settings.toml.
            if False, the settings will not be persisted to settings.toml.
        """,
    )
    auto_download_data: bool = Field(
        default=True,
        description="""
            if True, the engine will automatically download data using pfeed if no data is found in the storage specified in add_data()'s storage_config.
            if False, the engine will not download data automatically.
        """,
    )
    cache_materialized_data: bool | Literal["auto"] = Field(
        default="auto",
        description="""
            Controls whether materialized data is cached to the CURATED data layer for faster future retrieval.
            Materialized data is the output of MarketDataStore.materialize(), where raw/stored data is
            processed (e.g. cleaned from RAW, resampled from tick to bar) into a format pfund can use.
            - 'auto': cache only when the stored resolution differs from the requested resolution (e.g. tick data resampled to second bars).
            - True: always cache retrieved data to the CURATED layer.
            - False: never cache, always process on the fly.
        """,
    )

    @field_validator("cache_materialized_data", mode="before")
    @classmethod
    def _normalize_cache_materialized_data(cls, v: Any) -> bool | str:
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("datalake", mode="before")
    @classmethod
    def _validate_datalake(cls, v: DataLake | str) -> DataLake:
        if not isinstance(v, DataLake):
            return DataLake[v.upper()]
        return v
