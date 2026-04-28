from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from datetime import datetime
    from pfeed.storages.storage_config import StorageConfig
    from pfund.datas.data_config import DataConfig
    from pfund.datas.resolution import Resolution
    from pfund.datas.timeframe import Timeframe
    from pfund.entities.products.product_base import BaseProduct

import sys

from pfund.datas.resolution import Resolution
from pfund.datas.data_market import MarketData
from pfund_kit.utils.temporal import convert_ts_to_dt


# OPTIMIZE: too slow for second data
class Bar:
    def __init__(self, resolution: Resolution, shift: int=0):
        self._resolution: Resolution = resolution
        self._period: int = resolution.period
        self._timeframe: Timeframe = resolution.timeframe
        self._resolution_in_seconds: int = resolution.to_seconds()
        self._shift_in_seconds: int = self._calculate_shift_in_seconds(shift) if shift != 0 else 0
        self.clear()

    def clear(self):
        self._open = 0.0
        self._high = 0.0
        self._low = sys.float_info.max
        self._close = 0.0
        self._volume = 0.0
        self._start_ts = self._end_ts = self._ts = 0.0
        self._is_closed = False
    
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

    def _set_closed(self):
        self._is_closed = True
    
    def is_empty(self):
        return not self.open

    def is_closed(self, now: float | None=None) -> bool:
        # check if the bar is closed
        if not self._is_closed and now is not None:
            is_closed = now >= self._end_ts > 0
            if is_closed:
                self._set_closed()
        return self._is_closed

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
    
    def update(
        self, 
        o: float, h: float, l: float, c: float, v: float,  # noqa: E741
        is_incremental: bool, is_snapshot: bool,
        ts: float | None=None, start_ts: float | None=None, end_ts: float | None=None,
    ):  # noqa: E741
        if is_incremental:
            if self.is_closed(now=ts):
                self.clear()
            if not self._open:
                self._open = o
            if h > self._high:
                self._high = h
            if l < self._low:
                self._low = l
            self._close = c
            if is_snapshot:
                self._volume = v
            else:
                self._volume += v
        else:
            self._start_ts, self._end_ts, self._ts = start_ts, end_ts, ts
            self._open, self._high, self._low, self._close, self._volume = o, h, l, c, v
            self._set_closed()
        self._update_ts(ts=ts, start_ts=start_ts, end_ts=end_ts, is_incremental=is_incremental)

    def _update_ts(self, ts: float | None=None, start_ts: float | None=None, end_ts: float | None=None, is_incremental: bool=True):
        '''Determine the bar's time window (start_ts, end_ts) with the following priority:
        1. start_ts and end_ts provided (e.g. Bybit) → use them directly
        2. Only start_ts provided (e.g. OKX) → derive end_ts from resolution
        3. Only end_ts provided → derive start_ts from resolution
        4. Only ts provided + is_incremental=True → ts is within the bar, floor to derive start_ts
        5. Only ts provided + is_incremental=False → ts is past the bar, shift back one period
        '''
        def _compute_end_ts_from_start_ts(_start_ts: float) -> float:
            return _start_ts + self._resolution_in_seconds - 0.001
        def _compute_start_ts_from_end_ts(_end_ts: float) -> float:
            from math import ceil
            return ceil(_end_ts + 0.001) - self._resolution_in_seconds
        if ts:
            self._ts = ts
        if self._start_ts and self._end_ts:
            return
        if start_ts:
            self._start_ts = start_ts
            self._end_ts = end_ts if end_ts else _compute_end_ts_from_start_ts(start_ts)
        elif end_ts:
            self._end_ts = end_ts
            self._start_ts = _compute_start_ts_from_end_ts(end_ts)
        elif ts:
            if is_incremental:
                # ts is within the current bar, safe to derive both
                self._start_ts = ts // self._resolution_in_seconds * self._resolution_in_seconds + self._shift_in_seconds
            else:
                # ts is past the bar (confirmation timestamp), shift back one period
                self._start_ts = ts // self._resolution_in_seconds * self._resolution_in_seconds - self._resolution_in_seconds + self._shift_in_seconds
            self._end_ts = _compute_end_ts_from_start_ts(self._start_ts)
        else:
            raise ValueError('ts, start_ts, and end_ts are all None')


# OPTIMIZE: too slow for second data, especially using tick data to resample to bar data
class BarData(MarketData):
    def __init__(
        self,
        product: BaseProduct,
        resolution: Resolution,
        data_config: DataConfig | None=None,
        storage_config: StorageConfig | None=None,
    ):
        super().__init__(
            product=product, 
            resolution=resolution, 
            data_config=data_config, 
            storage_config=storage_config,
        )
        self._bar = Bar(resolution, shift=self.config.shift.get(resolution, 0))
        self._stale_bar_timeout = self.config.stale_bar_timeout[resolution]
        self._skip_first_bar = self.config.skip_first_bar[resolution]

    def __getattr__(self, attr):
        if '_bar' in self.__dict__:
            return getattr(self._bar, attr)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")
    
    @property
    def bar(self):
        return self._bar
    kline = bar
    candlestick = bar
    
    def on_bar(
        self, 
        o: float, h: float, l: float, c: float, v: float,  # noqa: E741
        is_incremental: bool,
        is_snapshot: bool=True,
        # TODO: handle backfilling
        is_backfill: bool=False,
        ts: float | None=None, start_ts: float | None=None, end_ts: float | None=None,
        msg_ts: float | None=None,
        extra_data: dict[str, Any] | None=None,
    ):
        '''
        Args:
            ts: the timestamp of when the bar data was generated/confirmed.
                This is NOT the bar's time range — it's the moment the exchange produced the update.
                When is_incremental=True, ts falls within (start_ts, end_ts) since the bar is still open.
                When is_incremental=False (complete bar), ts > end_ts because the exchange can only
                confirm a bar is complete after its time window has passed.
            start_ts: the start timestamp of the bar's time window, if provided by the exchange.
            end_ts: the end timestamp of the bar's time window, if provided by the exchange.
                If start_ts/end_ts are not provided, they are derived from ts (see _update_ts).
            msg_ts: the timestamp of the message sent
            is_incremental: if True, the bar is not yet complete (still forming).
                If False, the bar is complete/closed.
                Some exchanges push updates while the bar is still open, others only push when the bar is closed.
            is_snapshot: if True, the OHLCV values represent the full state of the bar so far
                (e.g. volume is the bar's total volume up to now), so values are overwritten on each update.
                If False, the values are deltas (e.g. volume is per-trade), so values are accumulated.
                Typically True for exchange bar/kline pushes, False when building bars from tick data.
        '''
        # NOTE: use msg_ts if ts is not provided
        ts = ts or msg_ts
        self._bar.update(
            o, h, l, c, v, 
            ts=ts, start_ts=start_ts, end_ts=end_ts,
            is_incremental=is_incremental, is_snapshot=is_snapshot
        )
        self.update_timestamps(ts=ts, msg_ts=msg_ts)
        if extra_data is not None:
            self.update_extra_data(extra_data)

    def is_second(self):
        return self.bar._timeframe.is_second()

    def is_minute(self):
        return self.bar._timeframe.is_minute()

    def is_hour(self):
        return self.bar._timeframe.is_hour()

    def is_day(self):
        return self.bar._timeframe.is_day()

    def is_closed(self, now: float | None=None) -> bool:
        is_closed = self._bar.is_closed(now=now)
        if is_closed and self._skip_first_bar:
            self._skip_first_bar = False
            return False
        return is_closed
    
    def __str__(self):
        is_closed = self.bar._is_closed
        header = f"[BarData {'●' if is_closed else '○'}] {self.product.source} | {self.product.name} | {self.resolution}"
        if not self.bar.start_ts or self.bar.is_empty():
            return header
        ohlcv = f"  {self.bar.open} / {self.bar.high} / {self.bar.low} / {self.bar.close} | volume={self.bar.volume}"
        if self.bar.start_dt and self.bar.end_dt and self.bar.dt:
            time_info = f"  {self.bar.start_dt.strftime('%H:%M:%S')} → {self.bar.end_dt.strftime('%H:%M:%S')} | update@{self.bar.dt.strftime('%H:%M:%S.%f')[:-3]}"
            return f"{header}\n{ohlcv}\n{time_info}"
        return f"{header}\n{ohlcv}"
