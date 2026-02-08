from typing import Annotated, ClassVar

from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator, PrivateAttr

from pfund.datas.resolution import Resolution


class DataConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    _primary_resolution: Resolution = PrivateAttr(init=False)
    extra_resolutions: list[Resolution] = Field(
        default_factory=list, 
        description='extra resolutions, e.g. "1t" for tick data, "1q" for quote data'
    )
    # TODO: handle quote_L1 resampled by quote_L2
    resample: dict[Annotated[Resolution, "ResampleeResolution"], Annotated[Resolution, "ResamplerResolution"]] = Field(
        default_factory=dict, 
        description='key is the resolution to resample to (resamplee), value is the resolution to resample from (resampler), e.g. {"1h": "1m"} means 1 hour bar is resampled by 1 minute bar.'
    )
    shift: dict[Resolution, Annotated[int, Field(strict=True, gt=0, lt=60)]] = Field(
        default_factory=dict,
        description='shifts the start_ts of the bar by a number, only supports "minute", "hour", "day" timeframe. e.g. {"1h": 30} means the hour bar starts at 00:30-01:30.'
    )
    skip_first_bar: dict[Resolution, bool] = Field(
        default_factory=dict,
        description='''
            skip the first bar due to incomplete data.
            In live trading, the first bar is very likely incomplete due to resampling
            In backtesting, the first bar might be incomplete due to shifting 
            e.g. hourly bar shifts 30 minutes, first bar is 00:00 to 00:30, which is incomplete
        '''
    )
    stale_bar_timeout: dict[Resolution, Annotated[float, Field(strict=True, gt=0)]] = Field(
        default_factory=dict,
        description="time (in seconds) after a bar's expected completion (bar.end_ts) to wait for any delayed updates before flushing the bar."
    )
    
    @property
    def primary_resolution(self) -> Resolution:
        return self._primary_resolution
    
    @primary_resolution.setter
    def primary_resolution(self, value: Resolution | str):  # pyright: ignore[reportPropertyTypeMismatch]
        if isinstance(value, str):
            value = Resolution(value)
        if not value.is_bar():
            raise ValueError(f'primary_resolution={repr(value)} must be a bar resolution (e.g. "1s", "1m", "1h", "1d")')
        self._primary_resolution = value
        
    @property
    def resolutions(self) -> list[Resolution]:
        if hasattr(self, '_primary_resolution') and self._primary_resolution not in self.extra_resolutions:
            return [self._primary_resolution] + self.extra_resolutions
        else:
            return self.extra_resolutions
    
    # REVIEW: support week, month, year?
    @staticmethod
    def _assert_at_least_daily_resolution(resolution: Resolution):
        if resolution.is_week() or resolution.is_month() or resolution.is_year():
            raise ValueError(f'{resolution=} is not supported')
    
    def _set_default_skip_first_bar(self, resolution: Resolution):
        if resolution in self.skip_first_bar or not resolution.is_bar():
            return
        # the first bar is very likely incomplete due to resampling, excluding cases like resample={'1h': '60m'}
        if resolution in self.resample and resolution != self.resample[resolution]:
            default_skip_first_bar = True
        else:
            default_skip_first_bar = False
        self.skip_first_bar[resolution] = default_skip_first_bar
    
    def _set_default_stale_bar_timeout(self, resolution: Resolution):
        if resolution in self.stale_bar_timeout or not resolution.is_bar():
            return
        default_timeout = resolution.to_seconds() * 0.1
        self.stale_bar_timeout[resolution] = default_timeout
    
    @model_validator(mode='after')
    def validate_after(self):
        for resolution in self.shift:
            self.add_extra_resolution(resolution)
        for resamplee_resolution, resampler_resolution in self.resample.items():
            self.add_extra_resolution(resamplee_resolution)
            self.add_extra_resolution(resampler_resolution)
        for resolution in self.resolutions:
            self._assert_at_least_daily_resolution(resolution)
            if not resolution.is_bar():
                continue
            self._set_default_skip_first_bar(resolution)
            self._set_default_stale_bar_timeout(resolution)
        return self
    
    @field_validator('extra_resolutions', mode='before')
    @classmethod
    def validate_extra_resolutions(cls, v: list[Resolution | str]) -> list[Resolution]:
        return list(set(Resolution(resolution) for resolution in v))

    @field_validator('extra_resolutions', mode='after')
    @classmethod
    def validate_extra_resolutions_after(cls, v: list[Resolution]) -> list[Resolution]:
        quote_resolutions = [r for r in v if r.is_quote()]
        if len(quote_resolutions) > 1:
            raise ValueError(f'only one quote resolution is supported, got {len(quote_resolutions)}')
        tick_resolutions = [r for r in v if r.is_tick()]
        if len(tick_resolutions) > 1:
            raise ValueError(f'only one tick resolution is supported, got {len(tick_resolutions)}')
        return v

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
    
    def add_extra_resolution(self, resolution: Resolution):
        if resolution not in self.extra_resolutions:
            self.extra_resolutions.append(resolution)
            
    # TODO: detect bar shift based on the returned data by e.g. Yahoo Finance, its hourly data starts from 9:30 to 10:30 etc.
    def auto_shift(self):
        pass
    
    def auto_resample(self, supported_resolutions: dict) -> bool:
        '''Resamples the resolutions automatically if not supported officially.
        Returns True if auto_resampling is needed.
        '''
        def _convert_to_supported_resolution(_resolution: Resolution) -> Resolution | None:
            """Converts the resolution into an officially supported one
            Returns None if the resolution is not officially supported.
            """
            period, timeframe = _resolution.period, _resolution.timeframe
            if timeframe.unit in supported_resolutions:
                supported_periods = supported_resolutions.get(timeframe.unit, [])
                if period in supported_periods:
                    return _resolution
                else:
                    if divisors := [p for p in supported_periods if period % p == 0]:
                        smallest_period = min(divisors)
                        return Resolution(str(smallest_period) + str(timeframe))
            # if resolution is already at tick level, no more higher resolution to try to convert to
            if _resolution.is_tick():
                return None
            else:
                is_in_shift = _resolution in self.shift
                # if resolution is in shift, e.g. {'1h': 30}, then use the shift value, otherwise use 60 for minute, 24 for hour
                if _resolution.is_minute():
                    unit = self.shift[_resolution] if is_in_shift else 60
                    higher_resolution = Resolution(f'{unit}s')
                elif _resolution.is_hour():
                    unit = self.shift[_resolution] if is_in_shift else 60
                    higher_resolution = Resolution(f'{unit}m')
                elif _resolution.is_day():
                    unit = self.shift[_resolution] if is_in_shift else 24
                    higher_resolution = Resolution(f'{unit}h')
                else:
                    higher_resolution = _resolution.higher()
                return _convert_to_supported_resolution(higher_resolution)
        
        output_resample = {}
        input_resample = self.resample.copy()

        for resolution in self.resolutions:
            # quote and tick data cannot be resamplee
            if resolution.is_quote() or resolution.is_tick() or resolution in input_resample:
                continue
            supported_resolution: Resolution | None = _convert_to_supported_resolution(resolution)
            if supported_resolution is None:
                raise Exception(f'{resolution=} is not officially supported')
            else:
                if not resolution.is_strict_equal(supported_resolution):
                    output_resample[resolution] = supported_resolution
        
        temp_resample = {**input_resample, **output_resample}
        for resamplee_resolution, resampler_resolution in temp_resample.items():
            # e.g. resample = {'1m': '1s', '1s': '1t'}, '1s' is both a resamplee and a resampler
            # skip if resampler is also a resamplee, only deal with the root resampler
            is_resampler_also_a_resamplee = resampler_resolution in temp_resample
            if is_resampler_also_a_resamplee:
                continue
            supported_resolution: Resolution | None = _convert_to_supported_resolution(resampler_resolution)
            if supported_resolution is None:
                raise Exception(f'{resampler_resolution=} is not officially supported')
            else:
                if not resampler_resolution.is_strict_equal(supported_resolution):
                    output_resample[resamplee_resolution] = supported_resolution
                
        if is_auto_resampled := (output_resample != input_resample):
            self.resample = {**self.resample, **output_resample}  # Triggers validation automatically
        return is_auto_resampled
