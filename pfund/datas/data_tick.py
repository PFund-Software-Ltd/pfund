from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfeed.enums import DataSource
    from pfund.datas.resolution import Resolution
    from pfund.products.product_base import BaseProduct

import time
from collections import deque

from pfund.datas.data_market import MarketData


class TickData(MarketData):
    def __init__(
        self,
        data_source: DataSource,
        data_origin: str,
        product: BaseProduct,
        resolution: Resolution
    ):
        super().__init__(data_source, data_origin, product, resolution)
        self._price = self._quantity = 0.0
        assert 0 < self.period <= 1, f'period {self.period} is not supported for TickData'
        # TODO: support period > 1?
        # self._is_appended = (self.period > 1)
        # self.ticks = deque(maxlen=self.period)
    
    @property
    def price(self):
        return self._price
    
    @property
    def quantity(self):
        return self._quantity
    
    @property
    def volume(self):
        return self._quantity
        
    def on_tick(self, price, quantity, ts: float | None, is_backfill=False, **extra_data):
        self.price = price
        self.quantity = quantity
        self.update_ts(ts)
        self.update_extra_data(extra_data)
        
        for resamplee in self._resamplees:
            resamplee.on_tick(price, quantity, ts)
            
        # if self._is_appended:
        #     self.ticks.append((px, qty, ts))
