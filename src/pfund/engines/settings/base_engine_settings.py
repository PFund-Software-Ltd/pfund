from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self

if TYPE_CHECKING:
    from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pfund.enums import DataLake, Environment


class BaseEngineSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    datalake: DataLake | str = Field(
        default=DataLake.DELTALAKE,
        description="data lake for writing and appending component data",
    )
    datalake_path: str | None = Field(
        default=None,
        description=(
            "data lake path, such as /data/pfund or s3://bucket/prefix. "
            + "Unset means the engine's data_path, resolved on every run so it "
            + "follows a later pf.configure(data_path=...)."
        ),
    )
    # datalake_storage_options: dict[str, Any] = Field(
    #     default_factory=dict,
    #     description="non-secret options passed to the data lake storage backend",
    # )
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
    warn_overwrite: bool = Field(
        default=True,
        description="""
        Ask for confirmation before engine.run(overwrite=True) clears the reused run folder
        (e.g. default_run/) and all artifacts inside it, including component outputs,
        models, and checkpoints.

        This setting is ignored when an mtflow run is active because every mtflow run
        receives its own unique run folder instead of reusing and clearing an existing one.
        """,
    )

    @staticmethod
    def file_path(engine_name: str) -> Path:
        from pfund.config import get_config

        path = get_config().get_settings_file_path(engine_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def load(cls, engine_name: str, env: Environment) -> Self:
        from pfund_kit.utils import toml

        data = toml.load(cls.file_path(engine_name)) or {}
        env_data = data.get(env, {})
        settings = cls(**{k: v for k, v in env_data.items() if k in cls.model_fields})
        # Always write back — this adds new fields with defaults and drops removed fields automatically
        settings.save(engine_name, env)
        return settings

    def save(self, engine_name: str, env: Environment) -> None:
        from pfund_kit.utils import toml

        # Drop unset values: toml has no null, and the dumper would write them
        # as the string "None", which reloads as a literal path named "None".
        data = {env: {k: v for k, v in self.model_dump().items() if v is not None}}
        toml.dump(data, self.file_path(engine_name), mode="update", auto_inline=True)

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
