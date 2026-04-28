from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from pfeed.storages.storage_config import StorageConfig
    from pfund.datas.resolution import Resolution
    from pfund.datas.data_config import DataConfig
    from pfund.entities.products.product_base import BaseProduct

import time
from collections import deque

from pfund.datas.data_market import MarketData


class TickData(MarketData):
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
    ):
        self._price = price
        self._volume = volume
        self.update_timestamps(ts=ts, msg_ts=msg_ts)
        if extra_data is not None:
            self.update_extra_data(extra_data)
        # if self._is_appended:
        #     self.ticks.append((px, qty, ts))
