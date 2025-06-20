from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import datetime
    from pfeed.enums import DataSource
    from pfund.datas.resolution import Resolution
    from pfund.datas.timeframe import Timeframe
    from pfund.products.product_base import BaseProduct

import sys
import time

from pfund.enums import PublicDataChannel
from pfund.datas.resolution import Resolution
from pfund.datas.data_time_based import TimeBasedData
from pfund.utils.utils import convert_ts_to_dt


class Bar:
    def __init__(self, resolution: Resolution, shift: int=0):
        self._resolution: Resolution = resolution
        self._period: int = resolution.period
        self._timeframe: Timeframe = resolution.timeframe
        self._resolution_in_seconds: int = resolution.to_seconds()
        self._shift_in_seconds: int = self._calculate_shift_in_seconds(shift)
        self.clear()

    def clear(self):
        self._open = 0.0
        self._high = 0.0
        self._low = sys.float_info.max
        self._close = 0.0
        self._volume = 0.0
        self._start_ts = self._end_ts = self._ts = 0.0
        self._is_ready = False
    
    @property
    def resolution(self) -> Resolution:
        return self._resolution
    
    @property
    def open(self) -> float:
        return self._open
    
    @property
    def high(self) -> float:
        return self._high
    
    @property
    def low(self) -> float:
        return self._low
    
    @property
    def close(self) -> float:
        return self._close
    
    @property
    def volume(self) -> float:
        return self._volume
    
    @property
    def start_ts(self) -> float:
        return self._start_ts
    
    @property
    def start_dt(self) -> datetime | None:
        return convert_ts_to_dt(self._start_ts) if self._start_ts else None
    
    @property
    def end_ts(self) -> float:
        return self._end_ts
    
    @property
    def end_dt(self) -> datetime | None:
        return convert_ts_to_dt(self._end_ts) if self._end_ts else None
    
    @property
    def ts(self) -> float:
        return self._ts
    
    @property
    def dt(self) -> datetime | None:
        return convert_ts_to_dt(self._ts) if self._ts else None

    def _set_ready(self):
        self._is_ready = True
    
    def is_empty(self):
        return not self.open

    def is_ready(self, now: float | None=None):
        # if not ready, check if the bar is ready
        if not self._is_ready:
            now = now or time.time()
            is_ready = now >= self._end_ts > 0
            if is_ready:
                self._set_ready()
        return self._is_ready

    def _calculate_shift_in_seconds(self, shift: int) -> int:
        '''
        Calculate the shift in seconds that will be used to shift the start_ts of the bar.
        e.g. 
        for a 1H bar, shift=30 means the bar will shift 30 minutes forward,
        making the bar start from 9:30 to 10:30 instead of from 9:00 to 10:00.
        '''
        if self._timeframe.is_minute():
            max_shift = 60  # in seconds
        elif self._timeframe.is_hour():
            max_shift = 60  # in minutes
        elif self._timeframe.is_day():
            max_shift = 24  # in hours
        else:
            raise ValueError(f'shifting is not supported for resolution={self._resolution}')
        assert 0 <= shift < max_shift, f'{shift=} should be between 0 and {max_shift}'
        shift_unit = shift / max_shift
        seconds_per_unit = self._resolution_in_seconds / self._period
        shift_in_seconds = int(shift_unit * seconds_per_unit)
        return shift_in_seconds
    
    def update(self, o: float, h: float, l: float, c: float, v: float, ts: float, is_incremental: bool):
        self._update_ts(ts)
        if is_incremental:
            if not self._open:
                self._open = o
            if h > self._high:
                self._high = h
            if l < self._low:
                self._low = l
            self._close = c
            self._volume += v
        else:
            self._open, self._high, self._low, self._close, self._volume = o, h, l, c, v
            self._set_ready()
    
    def _update_ts(self, ts: float):
        self._ts = ts
        if not self._start_ts:
            self._start_ts = self._ts // self._resolution_in_seconds * self._resolution_in_seconds + self._shift_in_seconds
            self._end_ts = self._start_ts + self._resolution_in_seconds - 1  # exclusively


class BarData(TimeBasedData):
    def __init__(
        self,
        data_source: DataSource,
        data_origin: str,
        product: BaseProduct, 
        resolution: Resolution, 
        shift: int=0, 
        skip_first_bar: bool=True
    ):
        super().__init__(data_source, data_origin, product, resolution)
        self._bar = Bar(resolution, shift=shift)
        self._skip_first_bar = skip_first_bar

    def __getattr__(self, attr):
        if '_bar' in self.__dict__:
            return getattr(self._bar, attr)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")
    
    @property
    def channel(self) -> PublicDataChannel:
        return PublicDataChannel.candlestick
    
    @property
    def bar(self):
        return self._bar
    
    @property
    def kline(self):
        return self._bar
    
    @property
    def candlestick(self):
        return self._bar
    
    def skip_first_bar(self) -> bool:
        if self._skip_first_bar:
            self._skip_first_bar = False
        return self._skip_first_bar

    # TODO: handle backfilling
    def on_bar(self, o, h, l, c, v, ts: float, is_incremental=True, is_backfill=False, **extra_data):
        '''
        Args:
            is_incremental: if True, the bar update is incremental, otherwise it is a full bar update
                some exchanges may push incremental bar updates, some may only push when the bar is complete
        '''
        self._bar.update(o, h, l, c, v, ts, is_incremental)
        self.update_ts(ts)
        self.update_extra_data(extra_data)
        for resamplee in self._resamplees:
            resamplee.on_bar(o, h, l, c, v, ts, is_incremental=True)

    # use tick updates to update bar
    def on_tick(self, price, quantity, ts, is_backfill=False):
        self.on_bar(price, price, price, price, quantity, ts, is_incremental=True, is_backfill=is_backfill)

    def is_second(self):
        return self.bar._timeframe.is_second()

    def is_minute(self):
        return self.bar._timeframe.is_minute()

    def is_hour(self):
        return self.bar._timeframe.is_hour()

    def is_day(self):
        return self.bar._timeframe.is_day()

    def is_ready(self, now: float | None=None):
        return self._bar.is_ready(now=now)
    
    def clear(self):
        self._bar.clear()

    def __str__(self):
        bar_type = 'Bar'
        if not self.bar.start_ts:
            bar_info = ['None']
        else:
            bar_info = list(map(str, [self.bar.open, self.bar.high, self.bar.low, self.bar.close, self.bar.volume]))
        return '_'.join(
                [
                    'START-' + str(self.bar.start_dt),
                    str(self.product),
                    str(self.resolution),
                    bar_type
                ] + bar_info + [
                    'END-' + str(self.bar.end_dt),
                    'LAST-' + str(self.bar.ts)
                ]
            )