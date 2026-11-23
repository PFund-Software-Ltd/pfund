from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

if TYPE_CHECKING:
    from pathlib import Path

from pfeed.io.table_io import TableIO
from pfund_blocks.enums.env import Environment
from pfund_kit.settings import Settings
from pydantic import ConfigDict, Field, field_validator


class BaseEngineSettings(Settings):
    __version__: ClassVar[str] = "0.1.0"

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # excluded: an IO instance can't be written to settings.toml
    datalake: TableIO | None = Field(
        default=None,
        exclude=True,
        description=(
            "data lake for writing and appending component data, e.g. pe.DuckLakeIO(base_path=...). "
            + "Unset means pe.DuckLakeIO under the engine's data_path. Not persisted to settings.toml."
        ),
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
            if True, the engine will automatically download data using pfeed if no data is found in the io of add_data()'s config (DataConfig.io),
            and store it there.
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

    @classmethod
    def _validate_file(cls, data: dict[str, Any]) -> None:
        # One version covers every environment. Validate all sections before
        # advancing it, including environments not being loaded by this call.
        from pfund.engines.settings import ENGINE_SETTINGS

        for env, fields in data.items():
            if env not in ENGINE_SETTINGS:
                raise ValueError(f"Unknown engine settings section {env!r}")
            try:
                ENGINE_SETTINGS[env].model_validate(fields)
            except ValueError as error:
                raise ValueError(f"Invalid settings for {env}: {error}") from error

    @staticmethod
    def file_path(engine_name: str) -> Path:
        from pfund.config import get_config

        path = get_config().get_settings_file_path(engine_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def load(cls, engine_name: str, env: Environment) -> Self:
        return cls.load_file(cls.file_path(engine_name), section=env)

    def save(self, engine_name: str, env: Environment) -> None:
        self.save_file(self.file_path(engine_name), section=env)

    @field_validator("cache_materialized_data", mode="before")
    @classmethod
    def _normalize_cache_materialized_data(cls, v: Any) -> bool | str:
        if isinstance(v, str):
            return v.lower()
        return v
