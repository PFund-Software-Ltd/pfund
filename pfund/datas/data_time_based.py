from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import datetime
    from pfund.datas.timeframe import Timeframe
    from pfeed.enums import DataSource
    from pfund.products.product_base import BaseProduct
    from pfund.datas.resolution import Resolution

from pfund.datas.data_base import BaseData


class TimeBasedData(BaseData):
    def __init__(
        self,
        data_source: DataSource,
        data_origin: str,
        product: BaseProduct,
        resolution: Resolution
    ):
        '''
        Args:
            ts: is the timestamp of the last updated data, e.g. timestamp of a candlestick
            msg_ts: is the timestamp of the data sent by the trading venue
        '''
        super().__init__(data_source, data_origin, product=product)
        self._ts = 0.0
        self._msg_ts = 0.0
        self.resolution: Resolution = resolution
        self.period: int = resolution.period
        self.timeframe: Timeframe = resolution.timeframe
        self._resamplers = set()  # data used to be resampled into another data
        self._resamplees = set()  # opposite of resampler
    
    def __eq__(self, other):
        if not isinstance(other, TimeBasedData):
            return NotImplemented
        return (self.product, self.resolution) == (other.product, other.resolution)
    
    def __hash__(self):
        return hash((self.product, self.resolution))
    
    @property
    def ts(self):
        return self._ts
    
    @property
    def msg_ts(self):
        return self._msg_ts
    
    @property
    def dt(self) -> datetime.datetime | None:
        from pfund.utils.utils import convert_ts_to_dt
        return convert_ts_to_dt(self._ts) if self._ts else None
    
    def update_timestamps(self, ts: float, msg_ts: float | None=None):
        self._ts = ts
        if msg_ts:
            self._msg_ts = msg_ts
    
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
