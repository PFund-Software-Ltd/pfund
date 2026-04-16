from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator

from pfund.enums import Environment


class BaseEngineSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid', frozen=True)

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
    max_rows: int | None = Field(
        default=None,
        description="""
            Maximum number of data rows kept in memory.
            Once exceeded, oldest rows are dropped (sliding window).
            if None, all rows of data will be kept (unlimited).
        """,
    )

    def save(self, env: Environment):
        '''saves current settings to settings.toml'''
        from pfund_kit.utils import toml
        from pfund import get_config

        # write settings to settings.toml
        config = get_config()
        env_section = env.value
        data = {env_section: self.model_dump()}
        toml.dump(data, config.settings_file_path, mode='update', auto_inline=True)
 
    @field_validator('cache_materialized_data', mode='before')
    @classmethod
    def _normalize_cache_resampled_data(cls, v: Any) -> bool | str:
        if isinstance(v, str):
            return v.lower()
        return v
    
    @field_validator('max_rows', mode='after')
    @classmethod
    def _warn_if_max_rows_is_not_set(cls, v: int | None) -> int | None:
        from pfund_kit.style import cprint, RichColor, TextStyle
        if v is None:
            cprint(
                "WARNING: max_rows is not set in engine settings, data will be unbounded",
                style=TextStyle.BOLD + RichColor.YELLOW,
            )
        return v
