from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import datetime
    from pfund.datas.timeframe import Timeframe
    from pfeed.enums import DataSource
    from pfund.products.product_base import BaseProduct
    from pfund.datas import QuoteData, TickData, BarData
    from pfund.datas.resolution import Resolution

import time

from pfund.datas.data_base import BaseData


class TimeBasedData(BaseData):
    def __init__(
        self,
        data_source: DataSource,
        data_origin: str,
        product: BaseProduct,
        resolution: Resolution
    ):
        super().__init__(data_source, data_origin, product)
        self._ts = 0.0
        self._latency = None
        self.resolution: Resolution = resolution
        self.period: int = resolution.period
        self.timeframe: Timeframe = resolution.timeframe
        self._resamplers = set()  # data used to be resampled into another data
        self._resamplees = set()  # opposite of resampler
        self._extra_data = {}

    def __repr__(self):
        return f'{self.product.tv}:{self.product.name}:{repr(self.resolution)}'

    def __str__(self):
        return f'{self.product}|Data={self.resolution}'

    def __eq__(self, other):
        if not isinstance(other, TimeBasedData):
            return NotImplemented
        return (self.product, self.resolution) == (other.product, other.resolution)
    
    def __hash__(self):
        return hash((self.product, self.resolution))
    
    @property
    def zmq_channel(self: QuoteData | TickData | BarData) -> str:
        return f'{self.product.tv}:{self.channel}:{repr(self.resolution)}:{self.product.name}:'
    
    @property
    def ts(self):
        return self._ts
    
    @property
    def latency(self):
        return self._latency
    
    @property
    def extra_data(self):
        return self._extra_data
    
    @property
    def dt(self) -> datetime.datetime | None:
        from pfund.utils.utils import convert_ts_to_dt
        return convert_ts_to_dt(self._ts) if self._ts else None
    
    def update_extra_data(self, extra_data):
        self._extra_data = extra_data
    
    def update_ts(self, ts: float | None):
        if ts:
            self._ts = ts
            self.update_latency(ts)
    
    def update_latency(self, ts: float):
        self._latency = time.time() - ts
    
    def is_time_based(self):
        return True
    
    def is_quote_l1(self):
        return self.is_quote() and self.resolution.orderbook_level == 1

    def is_quote_l2(self):
        return self.is_quote() and self.resolution.orderbook_level == 2

    def is_quote_l3(self):
        return self.is_quote() and self.resolution.orderbook_level == 3

    def is_quote(self):
        return self.timeframe.is_quote()

    def is_tick(self):
        return self.timeframe.is_tick()

    def is_bar(self):
        return (
            self.is_second() or
            self.is_minute() or
            self.is_hour() or
            self.is_day()
        )
    
    def is_second(self):
        return self.timeframe.is_second()

    def is_minute(self):
        return self.timeframe.is_minute()

    def is_hour(self):
        return self.timeframe.is_hour()

    def is_day(self):
        return self.timeframe.is_day()

    def is_resamplee(self):
        return bool(self._resamplers)

    def is_resampler(self):
        return bool(self._resamplees)

    def get_resamplees(self) -> set[TimeBasedData]:
        return self._resamplees
    
    def get_resamplers(self) -> set[TimeBasedData]:
        return self._resamplers
    
    def _add_resampler(self, data_resampler: TimeBasedData):
        self._resamplers.add(data_resampler)
    
    def _remove_resampler(self, data_resampler: TimeBasedData):
        self._resamplers.remove(data_resampler)
    
    def _add_resamplee(self, data_resamplee: TimeBasedData):
        self._resamplees.add(data_resamplee)
    
    def _remove_resamplee(self, data_resamplee: TimeBasedData):
        self._resamplees.remove(data_resamplee)

    def bind_resampler(self, data_resampler: TimeBasedData):
        self._add_resampler(data_resampler)
        data_resampler._add_resamplee(self)
        
    def unbind_resampler(self, data_resampler: TimeBasedData):
        self._remove_resampler(data_resampler)
        data_resampler._remove_resamplee(self)
