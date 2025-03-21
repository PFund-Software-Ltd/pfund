from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.datas.resolution import Resolution
    from pfund.products.product_base import BaseProduct

from collections import deque

from pfund.datas.data_time_based import TimeBasedData


class QuoteData(TimeBasedData):
    def __init__(self, product: BaseProduct, resolution: Resolution, orderbook_depth: int=1):
        super().__init__(product, resolution)
        self._orderbook_depth = orderbook_depth
        self._orderbook_level = resolution.orderbook_level
        self.bids = ()
        self.asks = ()
        
        assert 0 < self.period <= 1, f'period {self.period} is not supported for QuoteData'
        # TODO: support period > 1?
        # self._is_appended = (self.period > 1)
        # self.last_bids = deque(maxlen=self.period)
        # self.last_asks = deque(maxlen=self.period)

    def on_quote(self, bids, asks, ts: float | None, is_backfill=False, **extra_data):
        # e.g. IB's reqMktData may send only asks with empty bids
        if bids:
            self.bids = bids
        if asks:
            self.asks = asks
        self.update_ts(ts)
        self.update_extra_data(extra_data)
        
        for resamplee in self._resamplees:
            resamplee.on_quote(bids, asks, ts)
            
        # if self._is_appended:
        #     self.last_bids.append(bids or self.bids)
        #     self.last_asks.append(asks or self.asks)
    
    @property
    def orderbook_level(self):
        return self._orderbook_level
    
    @property
    def orderbook_depth(self):
        return self._orderbook_depth
