import time
from collections import deque

from pfund.datas.data_time_based import TimeBasedData
from pfund.utils.utils import convert_ts_to_dt


class QuoteData(TimeBasedData):
    def __init__(self, product, resolution, **kwargs):
        super().__init__(product, resolution)
        self.bids = ()
        self.asks = ()
        self.ts = 0.0
        
        assert 0 < self.period <= 1, f'period {self.period} is not supported for QuoteData'
        # TODO: support period > 1?
        # self._is_appended = (self.period > 1)
        # self.last_bids = deque(maxlen=self.period)
        # self.last_asks = deque(maxlen=self.period)
        
        # used to collect extra info e.g. IB's `tickAttribLast` when calling reqTickByTickData()
        self.info = {}

    def on_quote(self, bids, asks, ts, is_backfill=False, **kwargs):
        # e.g. IB's reqMktData may send only asks with empty bids
        if bids:
            self.bids = bids
        if asks:
            self.asks = asks
        self.info = kwargs
        now = time.time()
        if ts:
            self.latency = self.lat = ts - now
            self.ts = ts
        else:
            self.ts = now
            
        # if self._is_appended:
        #     self.last_bids.append(bids or self.bids)
        #     self.last_asks.append(asks or self.asks)
    
    @property
    def dt(self):
        return convert_ts_to_dt(self.ts) if self.ts else None
    
OrderBook = QuoteData