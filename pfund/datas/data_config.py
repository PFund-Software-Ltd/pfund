from typing_extensions import Annotated

from pydantic import BaseModel, Field, ConfigDict, model_validator

from pfeed.enums import DataSource
from pfund.datas.resolution import Resolution


# TODO: use field_validator?
class DataConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_source: DataSource
    data_origin: str=''
    primary_resolution: Resolution = Field(description='primary resolution used for trading, must be a bar resolution (e.g. "1s", "1m", "1h", "1d")')
    extra_resolutions: list[Resolution] = Field(default_factory=list, description='extra resolutions, e.g. "1t" for tick data, "1q" for quote data')
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
    def resolutions(self) -> list[Resolution]:
        return [self.primary_resolution] + self.extra_resolutions

    @model_validator(mode='after')
    def validate_after(self):
        assert self.primary_resolution.is_bar(), f'resolution={repr(self.primary_resolution)} must be a bar resolution (e.g. "1s", "1m", "1h", "1d")'
        assert self.primary_resolution not in self.extra_resolutions, f'resolution={repr(self.primary_resolution)} should not be included in "extra_resolutions"'
        quote_resolutions = [r for r in self.resolutions if r.is_quote()]
        if quote_resolutions:
            assert len(quote_resolutions) == 1, f'only one quote resolution is supported, got {len(quote_resolutions)}'
        tick_resolutions = [r for r in self.resolutions if r.is_tick()]
        if tick_resolutions:
            assert len(tick_resolutions) == 1, f'only one tick resolution is supported, got {len(tick_resolutions)}'
        self._validate_resample()
        self._validate_shift()
        self._validate_stale_bar_timeout()
        # set default values
        for resolution in self.resolutions:
            self._set_default_skip_first_bar(resolution)
            self._set_default_stale_bar_timeout(resolution)
        return self
        
    @model_validator(mode='before')
    @classmethod
    def validate_before(cls, data: dict) -> dict:
        if isinstance(data['primary_resolution'], str):
            data['primary_resolution'] = Resolution(data['primary_resolution'])
        data['extra_resolutions'] = list(set(Resolution(resolution) for resolution in data.get('extra_resolutions', [])))
        data['resample']: dict[Annotated[Resolution, "ResampleeResolution"], Annotated[Resolution, "ResamplerResolution"]] = {
            Resolution(resamplee_resolution): Resolution(resampler_resolution) for resamplee_resolution, resampler_resolution in data.get('resample', {}).items()
        }
        data['shift'] = {Resolution(resolution): shift for resolution, shift in data.get('shift', {}).items()}
        data['skip_first_bar'] = {Resolution(resolution): is_skip for resolution, is_skip in data.get('skip_first_bar', {}).items()}
        data['stale_bar_timeout'] = {Resolution(resolution): timeout for resolution, timeout in data.get('stale_bar_timeout', {}).items()}
        return data
    
    def _validate_resample(self):
        for resamplee_resolution, resampler_resolution in self.resample.items():
            assert resamplee_resolution.is_bar(), f'{resamplee_resolution=} is not a bar resolution (e.g. "1s", "1m", "1h", "1d")'
            assert not resampler_resolution.is_quote(), f'{resampler_resolution=} in "resample" cannot be a quote resolution'
            assert resampler_resolution >= resamplee_resolution, f'Cannot use lower/equal resolution "{resampler_resolution}" to resample "{resamplee_resolution}"'
            self.add_extra_resolution(resamplee_resolution)
            self.add_extra_resolution(resampler_resolution)
    
    def _validate_shift(self):
        for resolution, shift in self.shift.items():
            assert resolution.is_bar() and not resolution.is_second(), f'{resolution=} in "shift" is not a supported bar resolution (e.g. "1m", "1h", "1d"), there is no shifting in second bars'
            if resolution.is_day():
                assert shift < 24, f'{shift=} must be less than 24 for {resolution=}'
            self.add_extra_resolution(resolution)
            
    def _validate_stale_bar_timeout(self):
        for resolution, timeout in self.stale_bar_timeout.items():
            assert resolution.is_bar(), f'{resolution=} in "stale_bar_timeout" must be a bar resolution (e.g. "1s", "1m", "1h", "1d")'
            resolution_in_seconds = resolution.to_seconds()
            assert timeout < resolution_in_seconds, f'{resolution=} {timeout=} in "stale_bar_timeout" must be less than {resolution_in_seconds} seconds'
        
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
            self.update_resample(output_resample)
        return is_auto_resampled
    
    def add_extra_resolution(self, resolution: Resolution):
        if resolution not in self.extra_resolutions:
            self.extra_resolutions.append(resolution)
            self._set_default_skip_first_bar(resolution)
            self._set_default_stale_bar_timeout(resolution)
            
    def update_resample(self, resample: dict):
        self.resample.update(resample)
        for resamplee_resolution, resampler_resolution in resample.items():
            self.add_extra_resolution(resamplee_resolution)
            self.add_extra_resolution(resampler_resolution)
        self._validate_resample()
    
    # TODO: detect bar shift based on the returned data by e.g. Yahoo Finance, its hourly data starts from 9:30 to 10:30 etc.
    def update_shift(self):
        pass
    
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