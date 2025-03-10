from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.datas.resolution import Resolution
    from pfund.products.product_base import BaseProduct

import time
from collections import deque

from pfund.datas.data_time_based import TimeBasedData
from pfund.utils.utils import convert_ts_to_dt


class QuoteData(TimeBasedData):
    def __init__(self, product: BaseProduct, resolution: Resolution, **kwargs):
        from pfund.typing.data_kwargs import QuoteDataKwargs
        kwargs = QuoteDataKwargs(**kwargs).model_dump()
        super().__init__(product, resolution)
        self._orderbook_depth = kwargs['orderbook_depth']
        self._orderbook_level = resolution.orderbook_level
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
    
    @property
    def orderbook_level(self):
        return self._orderbook_level
    
    @property
    def orderbook_depth(self):
        return self._orderbook_depth
    
    
OrderBook = QuoteData