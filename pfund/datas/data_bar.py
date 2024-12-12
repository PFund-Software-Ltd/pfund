from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.datas.resolution import Resolution
    from pfund.products.product_base import BaseProduct

import sys
import logging

from pfund.datas.resolution import Resolution
from pfund.datas.data_time_based import TimeBasedData
from pfund.utils.utils import convert_ts_to_dt


logger = logging.getLogger('data_manager')


class Bar:
    def __init__(self, product: BaseProduct, resolution: Resolution, shift: int=0):
        self.bkr = product.bkr
        self.exch = product.exch
        self.pdt = product.name
        self.product = product
        self.resolution = resolution
        self.period = resolution.period
        self.timeframe = resolution.timeframe
        self.unit = self.get_unit()
        self.shift_unit = self.get_shift_unit(shift)
        self.clear()

    def clear(self):
        self.o = self.open = 0.0
        self.h = self.high = 0.0
        self.l = self.low = sys.float_info.max
        self.c = self.close = 0.0
        self.v = self.volume = 0.0
        self._start_ts = self._end_ts = self.ts = 0.0
        
    def __str__(self):
        bar_type = 'Bar'
        if not self._start_ts:
            bar_info = ['None']
        else:
            bar_info = list(map(str, [self.o, self.h, self.l, self.c, self.v]))
        return '_'.join(
                [
                    'START-' + str(self.start_dt),
                    str(self.product),
                    str(self.resolution),
                    bar_type
                ] + bar_info + [
                    'END-' + str(self.end_dt),
                    'LAST-' + str(self.ts)
                ]
            )
    
    @property
    def start_ts(self):
        return self._start_ts
    
    @property
    def start_dt(self):
        return convert_ts_to_dt(self._start_ts) if self._start_ts else None
    
    @property
    def end_ts(self):
        return self._end_ts
    
    @property
    def end_dt(self):
        return convert_ts_to_dt(self._end_ts) if self._end_ts else None
    
    @property
    def dt(self):
        return convert_ts_to_dt(self.ts) if self.ts else None
    
    def is_empty(self):
        return not self.open

    def is_ready(self, now: int):
        return now >= self._end_ts > 0

    def get_shift_unit(self, shift: int):
        '''
        Calculate the shift unit in seconds that will be used to shift the start_ts of the bar.
        e.g. 
        for a 1H bar, shift=30 means the bar will shift 30 minutes forward,
        making the bar start from 9:30 to 10:30 instead of from 9:00 to 10:00.
        '''
        # shift is not supported for resolutions higher than HOUR
        if self.timeframe.is_second() or self.timeframe.is_minute():
            if shift:
                raise Exception(f'{shift=} is not supported for {self.timeframe}')
            else:
                shift_unit = shift
        else:
            if self.timeframe.is_hour():
                max_shift = 60
            elif self.timeframe.is_day():
                max_shift = 24
            # REVIEW: 7 is not accurate
            elif self.timeframe.is_week():
                max_shift = 7
            # REVIEW: 30 is not accurate
            elif self.timeframe.is_month():
                max_shift = 30
            assert 0 <= shift < max_shift, f'{shift=} should be between 0 and {max_shift}'
            shift_unit = (shift / max_shift) * (self.unit / self.period)  # in seconds
        return shift_unit
    
    def get_unit(self):
        if self.timeframe.is_second():
            unit = 1 * self.period
        elif self.timeframe.is_minute():
            unit = 60 * self.period
        elif self.timeframe.is_hour():
            unit = 60 * 60 * self.period
        elif self.timeframe.is_day():
            unit = 60 * 60 * 24 * self.period
        elif self.timeframe.is_week():
            unit = 60 * 60 * 24 * 7 * self.period
        elif self.timeframe.is_month():
            unit = 60 * 60 * 24 * 7 * 4 * self.period
        return unit

    def update(self, o, h, l, c, v, ts, is_volume_aggregated):
        if not self.o:
            self.o = self.open = o
        if h > self.h:
            self.h = self.high = h
        if l < self.l:
            self.l = self.low = l
        self.c = self.close = c
        self.v = self.volume = self.v * is_volume_aggregated + v
        self.ts = ts
        if not self._start_ts:
            self._start_ts = self.ts // self.unit * self.unit + self.shift_unit
            self._end_ts = self._start_ts + self.unit - 1  # exclusively


class BarData(TimeBasedData):
    def __init__(self, product, resolution: Resolution, shifts: dict[str, int] | None=None, skip_first_bar: bool=True):
        super().__init__(product, resolution)
        if shifts and repr(resolution) in shifts:
            shift = shifts[repr(resolution)]
        else:
            shift = 0
        self._bar = Bar(product, resolution, shift=shift)
        self._timeframe = self._bar.timeframe
        self._skip_first_bar = skip_first_bar

    def __getattr__(self, attr):
        if '_bar' in self.__dict__:
            return getattr(self._bar, attr)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")
    
    # for resampled data, the first bar is very likely incomplete
    # so users can choose to skip it
    def skip_first_bar(self) -> bool:
        skip_first_bar = self._skip_first_bar
        self._skip_first_bar = False
        return skip_first_bar

    def on_bar(self, o, h, l, c, v, ts, is_volume_aggregated=False, is_backfill=False, **kwargs):
        self._bar.update(o, h, l, c, v, ts, is_volume_aggregated)

    # use tick updates to update bar
    def on_tick(self, px, qty, ts, is_backfill=False):
        self.on_bar(px, px, px, px, qty, ts, is_volume_aggregated=True, is_backfill=is_backfill)

    def is_second(self):
        return self._timeframe.is_second()

    def is_minute(self):
        return self._timeframe.is_minute()

    def is_hour(self):
        return self._timeframe.is_hour()

    def is_day(self):
        return self._timeframe.is_day()

    def is_week(self):
        return self._timeframe.is_week()

    def is_month(self):
        return self._timeframe.is_month()

    def is_ready(self, now: int):
        return self._bar.is_ready(now)
    
    def clear(self):
        self._bar.clear()

    @property
    def bar(self):
        return self._bar

Kline = BarData