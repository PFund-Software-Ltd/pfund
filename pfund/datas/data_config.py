from __future__ import annotations
from typing import Annotated, ClassVar, cast

from pydantic import BaseModel, Field, ConfigDict, field_validator, PrivateAttr

from pfeed.enums import DataSource
from pfund.datas.timeframe import Timeframe
from pfund.datas.resolution import Resolution


class DataConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    data_source: DataSource | str | None = Field(default=None)
    data_origin: str = ''
    # data_resolutions = primary_resolution + extra_resolutions defined in add_data()
    _data_resolutions: list[Resolution] = PrivateAttr(init=False)
    num_batch_workers: int | None = Field(
        default=None,
        description="""
        number of workers for batch processing (e.g. retrieve, download data) using pfeed, equivalent to the parameter 'num_workers' in pfeed.
        if None, Ray will NOT be used and it will be done sequentially.
        """,
    )
    num_stream_workers: int | None = Field(
        default=None,
        description="""
        number of workers for streaming data using pfeed, equivalent to the parameter 'num_workers' in pfeed.
        if None, Ray will NOT be used and it will be done sequentially.
        """,
    )
    
    # TODO: handle quote_L1 resampled by quote_L2?
    resample: dict[Annotated[Resolution | str, "ResampleeResolution"], Annotated[Resolution | str, "ResamplerResolution"]] = Field(
        default_factory=dict, 
        description='key is the resolution to resample to (resamplee), value is the resolution to resample from (resampler), e.g. {"1h": "1m"} means 1 hour bar is resampled by 1 minute bar.'
    )
    shift: dict[Resolution | str, Annotated[int, Field(strict=True, gt=0, lt=60)]] = Field(
        default_factory=dict,
        description='shifts the start_ts of the bar by a number, only supports "minute", "hour", "day" timeframe. e.g. {"1h": 30} means the hour bar starts at 00:30-01:30.'
    )
    push_incomplete_bar: bool = Field(default=False)
    skip_first_bar: dict[Resolution | str, bool] = Field(
        default_factory=dict,
        description='''
            skip the first bar due to incomplete data.
            In live trading, the first bar is very likely incomplete due to resampling
            In backtesting, the first bar might be incomplete due to shifting 
            e.g. hourly bar shifts 30 minutes, first bar is 00:00 to 00:30, which is incomplete
        '''
    )
    stale_bar_timeout: dict[Resolution | str, Annotated[float, Field(strict=True, gt=0)]] = Field(
        default_factory=dict,
        description="time (in seconds) after a bar's expected completion (bar.end_ts) to wait for any delayed updates before flushing the bar."
    )
    
    @field_validator('data_source', mode='before')
    @classmethod
    def validate_data_source(cls, v: DataSource | str | None) -> DataSource | None:
        if v is not None:
            return DataSource[v.upper()]
        return None
    
    @field_validator('resample', mode='before')
    @classmethod
    def validate_resample(cls, v: dict[Resolution | str, Resolution | str]) -> dict[Resolution, Resolution]:
        return {
            Resolution(resamplee_resolution): Resolution(resampler_resolution) 
            for resamplee_resolution, resampler_resolution in v.items()
        }
    
    @field_validator('resample', mode='after')
    @classmethod
    def validate_resample_after(cls, v: dict[Resolution, Resolution]) -> dict[Resolution, Resolution]:
        for resamplee_resolution, resampler_resolution in v.items():
            if not resamplee_resolution.is_bar():
                raise ValueError(f'{resamplee_resolution=} is not a bar resolution (e.g. "1s", "1m", "1h", "1d")')
            if resampler_resolution.is_quote():
                raise ValueError(f'{resampler_resolution=} in "resample" cannot be a quote resolution')
            if not resampler_resolution > resamplee_resolution:
                raise ValueError(f'Cannot use lower/equal resolution "{resampler_resolution}" to resample "{resamplee_resolution}"')
        return v

    @field_validator('shift', mode='before')
    @classmethod
    def validate_shift(cls, v: dict[Resolution | str, int]) -> dict[Resolution, int]:
        return { Resolution(resolution): shift for resolution, shift in v.items() }
    
    @field_validator('shift', mode='after')
    @classmethod
    def validate_shift_after(cls, v: dict[Resolution, int]) -> dict[Resolution, int]:
        for resolution, shift in v.items():
            if not resolution.is_bar():
                raise ValueError(f'{resolution=} in "shift" must be a bar resolution (e.g. "1m", "1h", "1d")')
            if resolution.is_second():
                raise ValueError(f'{resolution=} in "shift" must not be a second resolution (e.g. "1s"), there is no shifting in second bars')
            if resolution.is_day() and shift >= 24:
                raise ValueError(f'{shift=} must be less than 24 for {resolution=}')
        return v

    @field_validator('skip_first_bar', mode='before')
    @classmethod
    def validate_skip_first_bar(cls, v: dict[Resolution | str, bool]) -> dict[Resolution, bool]:
        return { Resolution(resolution): is_skip for resolution, is_skip in v.items() }

    @field_validator('stale_bar_timeout', mode='before')
    @classmethod
    def validate_stale_bar_timeout(cls, v: dict[Resolution | str, float]) -> dict[Resolution, float]:
        return { Resolution(resolution): timeout for resolution, timeout in v.items() }
    
    @field_validator('stale_bar_timeout', mode='after')
    @classmethod
    def validate_stale_bar_timeout_after(cls, v: dict[Resolution, float]) -> dict[Resolution, float]:
        for resolution, timeout in v.items():
            if not resolution.is_bar():
                raise ValueError(f'{resolution=} in "stale_bar_timeout" must be a bar resolution (e.g. "1s", "1m", "1h", "1d")')
            resolution_in_seconds = resolution.to_seconds()
            if timeout >= resolution_in_seconds:
                raise ValueError(f'{resolution=} {timeout=} in "stale_bar_timeout" must be less than {resolution_in_seconds} seconds')
        return v
    
    @property
    def data_resolutions(self) -> list[Resolution]:
        return self._data_resolutions
    
    @data_resolutions.setter
    def data_resolutions(self, resolutions: list[Resolution]):
        data_resolutions = cast(
            list[Resolution], 
            list(set(
                # NOTE: automatically include resamplee, resampler, shift resolutions
                resolutions + 
                list(self.resample.keys()) + 
                list(self.resample.values()) +
                list(self.shift.keys())
            ))
        )
        # validate the data resolutions
        quote_resolution_exists = False
        tick_resolution_exists = False
        for resolution in data_resolutions:
            # assert at most one quote and tick resolution exists
            if resolution.is_quote():
                if quote_resolution_exists:
                    raise ValueError('only one quote resolution is supported')
                quote_resolution_exists = True
            elif resolution.is_tick():
                if tick_resolution_exists:
                    raise ValueError('only one tick resolution is supported')
                tick_resolution_exists = True
            # REVIEW: support week, month, year?
            elif resolution.is_week() or resolution.is_month() or resolution.is_year():
                raise ValueError(f'{resolution=} is not supported, resolution is at least daily')
        self._data_resolutions = data_resolutions
    
    def resolve_resolution(
        self,
        resolution: Resolution,
        supported_resolutions: dict[Timeframe, list[int]]
    ) -> Resolution | None:
        """Converts the resolution into an officially supported one
        Returns None if the resolution is not officially supported.
        """
        period: int = resolution.period
        unit: Timeframe = resolution.timeframe
        if unit in supported_resolutions:
            supported_periods: list[int] = supported_resolutions[unit]
            if period in supported_periods:
                return resolution
            else:
                # find supported periods that evenly divide the requested period,
                # then pick the smallest for highest granularity resampling
                # e.g. 6m with supported [1, 3, 5, 15, 30] -> divisors=[1, 3] -> use 1m
                if divisors := [p for p in supported_periods if period % p == 0]:
                    smallest_period = min(divisors)
                    return Resolution(str(smallest_period) + unit.canonical)
        # if resolution is already at tick level, no more higher resolution to try to convert to
        if resolution.is_tick():
            return None
        else:
            # switch to a higher (more granular) resolution unit and retry,
            # e.g. minute -> second, hour -> minute, day -> hour
            # if the resolution has a shift, use the shift value as the period, e.g. {'1h': 30}, use the shift value 30
            # otherwise use the default: 60 for minute/hour, 24 for day
            is_in_shift = resolution in self.shift
            if resolution.is_minute():
                shift_unit: int = self.shift[resolution] if is_in_shift else 60
                higher_resolution = Resolution(f'{shift_unit}s')
            elif resolution.is_hour():
                shift_unit: int = self.shift[resolution] if is_in_shift else 60
                higher_resolution = Resolution(f'{shift_unit}m')
            elif resolution.is_day():
                shift_unit: int = self.shift[resolution] if is_in_shift else 24
                higher_resolution = Resolution(f'{shift_unit}h')
            else:
                higher_resolution = resolution.higher()
            return self.resolve_resolution(higher_resolution, supported_resolutions)

    # TODO: detect bar shift based on the returned data by e.g. Yahoo Finance, its hourly data starts from 9:30 to 10:30 etc.
    def auto_shift(self) -> DataConfig:
        return self
        
    def auto_resample(self, supported_resolutions: dict[Timeframe, list[int]]) -> DataConfig:
        resample_dict: dict[
            Annotated[Resolution | str, "ResampleeResolution"], 
            Annotated[Resolution | str, "ResamplerResolution"]
        ] = {}
        # if a resolution is not officially supported, 
        # make it as resamplee where the supported resolution is the resampler
        for resolution in self.data_resolutions:
            if resolution in self.resample:
                continue
            resolved_resolution: Resolution | None = self.resolve_resolution(resolution, supported_resolutions)
            if resolved_resolution is None:
                raise Exception(f'{resolution=} is not supported')
            elif resolution.is_strict_equal(resolved_resolution):
                continue
            else:
                resample_dict[resolution] = resolved_resolution
        resolved_data_config = DataConfig(
            **self.model_dump(exclude={'resample'}),
            resample={**self.resample, **resample_dict},
        )
        resolved_data_config.data_resolutions = self.data_resolutions
        return resolved_data_config
    
    def auto_skip_first_bar(self) -> DataConfig:
        skip_first_bar_dict: dict[Resolution, bool] = {}
        for resolution in self.data_resolutions:
            if resolution in self.skip_first_bar:
                continue
            # the first bar is very likely incomplete due to resampling, excluding cases like resample={'1h': '60m'}
            if resolution in self.resample and resolution != self.resample[resolution]:
                skip_first_bar = True
            else:
                skip_first_bar = False
            skip_first_bar_dict[resolution] = skip_first_bar
        resolved_data_config = DataConfig(
            **self.model_dump(exclude={'skip_first_bar'}),
            skip_first_bar={**self.skip_first_bar, **skip_first_bar_dict},
        )
        resolved_data_config.data_resolutions = self.data_resolutions
        return resolved_data_config
    
    def auto_set_stale_bar_timeout(self) -> DataConfig:
        stale_bar_timeout_dict: dict[Resolution, float] = {}
        for resolution in self.data_resolutions:
            if resolution in self.stale_bar_timeout:
                continue
            stale_bar_timeout = resolution.to_seconds() * 0.1
            stale_bar_timeout_dict[resolution] = stale_bar_timeout
        resolved_data_config = DataConfig(
            **self.model_dump(exclude={'stale_bar_timeout'}),
            stale_bar_timeout={**self.stale_bar_timeout, **stale_bar_timeout_dict},
        )
        resolved_data_config.data_resolutions = self.data_resolutions
        return resolved_data_config
