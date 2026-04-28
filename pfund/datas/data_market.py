from __future__ import annotations
from typing import TYPE_CHECKING, Any, ClassVar
if TYPE_CHECKING:
    from pfund.datas.data_config import DataConfig
    from pfeed.storages.storage_config import StorageConfig
    from pfund.entities.products.product_base import BaseProduct
    from pfund.datas.resolution import Resolution
    from pfund.datas.timeframe import Timeframe
    from pfund.enums import Broker, CryptoExchange
    
from pfeed.enums import DataCategory
from pfund.datas.data_time_based import TimeBasedData


class MarketData(TimeBasedData):
    category: ClassVar[DataCategory] = DataCategory.MARKET_DATA
    
    def __init__(
        self,
        product: BaseProduct,
        resolution: Resolution,
        data_config: DataConfig | None=None,
        storage_config: StorageConfig | None=None,
    ):
        super().__init__(data_config=data_config, storage_config=storage_config)
        if self.config.data_source is None:
            self.config.data_source = product.source
        self.product: BaseProduct = product
        self.resolution: Resolution = resolution
        self.period: int = resolution.period
        self.timeframe: Timeframe = resolution.timeframe
        self._resamplers = []  # data used to be resampled into another data
        self._resamplees = []  # opposite of resampler
    
    @property
    def broker(self) -> Broker | None:
        return self.product.broker
    
    @property
    def exchange(self) -> CryptoExchange | str | None:
        return self.product.exchange

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

    def get_resamplees(self) -> list[TimeBasedData]:
        return self._resamplees
    
    def get_resamplers(self) -> list[TimeBasedData]:
        return self._resamplers
    
    def _add_resampler(self, data_resampler: TimeBasedData):
        if data_resampler not in self._resamplers:
            self._resamplers.append(data_resampler)
    
    def _remove_resampler(self, data_resampler: TimeBasedData):
        if data_resampler in self._resamplers:
            self._resamplers.remove(data_resampler)
    
    def _add_resamplee(self, data_resamplee: TimeBasedData):
        if data_resamplee not in self._resamplees:
            self._resamplees.append(data_resamplee)
    
    def _remove_resamplee(self, data_resamplee: TimeBasedData):
        if data_resamplee in self._resamplees:
            self._resamplees.remove(data_resamplee)

    def bind_resampler(self, data_resampler: TimeBasedData):
        self._add_resampler(data_resampler)
        data_resampler._add_resamplee(self)
        
    def unbind_resampler(self, data_resampler: TimeBasedData):
        self._remove_resampler(data_resampler)
        data_resampler._remove_resamplee(self)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            'product': self.product.to_dict(),
            'resolution': repr(self.resolution),
        }
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MarketData):
            return NotImplemented
        return (self.product, self.resolution) == (other.product, other.resolution)
    
    def __hash__(self):
        return hash((self.product, self.resolution))

    def __str__(self):
        return f'Data={self.product}|{self.resolution}'
