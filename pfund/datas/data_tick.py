import time
from collections import deque

from pfund.datas.data_time_based import TimeBasedData
from pfund.utils.utils import convert_ts_to_dt


class TickData(TimeBasedData):
    def __init__(self, product, resolution, **kwargs):
        super().__init__(product, resolution)
        self.px = self.price = 0.0
        self.qty = self.quantity = self.volume = 0.0
        self.ts = 0.0
        
        assert 0 < self.period <= 1, f'period {self.period} is not supported for TickData'
        # TODO: support period > 1?
        # self._is_appended = (self.period > 1)
        # self.ticks = deque(maxlen=self.period)
        
        # used to collect extra info e.g. IB's `tickAttribLast` when calling reqTickByTickData()
        self.info = {}

    def on_tick(self, px, qty, ts, is_backfill=False, **kwargs):
        self.px = self.price = px
        self.qty = self.quantity = self.volume = qty
        self.info = kwargs
        now = time.time()
        if ts:
            self.latency = self.lat = ts - now
            self.ts = ts
        else:
            self.ts = now
            
        # if self._is_appended:
        #     self.ticks.append((px, qty, ts))

    @property
    def dt(self):
        return convert_ts_to_dt(self.ts) if self.ts else None

TradeBook = TickData