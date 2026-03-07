from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from pfeed.enums import DataSource
    from pfund.datas.resolution import Resolution
    from pfund.entities.products.product_base import BaseProduct

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
        self._price = self._volume = 0.0
        assert 0 < self.period <= 1, f'period {self.period} is not supported for TickData'
        # TODO: support period > 1?
        # self._is_appended = (self.period > 1)
        # self.ticks = deque(maxlen=self.period)
    
    @property
    def price(self):
        return self._price
    
    @property
    def quantity(self):
        return self._volume
    
    @property
    def volume(self):
        return self._volume
        
    def on_tick(
        self, 
        price: float, volume: float, ts: float, 
        # TODO: handle backfilling
        is_backfill=False, 
        msg_ts: float | None=None,
        extra_data: dict[str, Any] | None=None,
        custom_data: dict[str, Any] | None=None,
    ):
        self._price = price
        self._volume = volume
        self.update_timestamps(ts=ts, msg_ts=msg_ts)
        if extra_data is not None:
            self.update_extra_data(extra_data)
        if custom_data is not None:
            self.update_custom_data(custom_data)
        for resamplee in self._resamplees:
            resamplee.on_tick(
                price=price, volume=volume, ts=ts, 
                is_backfill=is_backfill,
                msg_ts=msg_ts, 
                extra_data=extra_data,
                custom_data=custom_data,
            )
            
        # if self._is_appended:
        #     self.ticks.append((px, qty, ts))
