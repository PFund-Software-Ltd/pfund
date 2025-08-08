from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from pfeed.enums import DataSource, DataCategory
    from pfund.enums import TradingVenue, Broker, CryptoExchange

from abc import ABC, abstractmethod
from pfund.products.product_base import BaseProduct


class BaseData(ABC):
    def __init__(self, data_source: DataSource, data_origin: str, product: BaseProduct | None=None):
        self.source: DataSource = data_source
        self.origin: str = data_origin or data_source.value
        self.product: BaseProduct | None = product
        
    @abstractmethod
    def to_dict(self) -> dict:
        pass
    
    @property
    @abstractmethod
    def category(self) -> DataCategory:
        pass
    
    def is_time_based(self):
        return False
    
    @property
    def trading_venue(self) -> TradingVenue | None:
        return self.product.trading_venue if self.product else None
    tv = trading_venue
    
    @property
    def broker(self) -> Broker | None:
        return self.product.broker if self.product else None
    bkr = broker
    
    @property
    def exchange(self) -> CryptoExchange | str | None:
        return self.product.exchange if self.product else None
    exch = exchange

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseData):
            return NotImplemented
        return (
            self.source == other.source
            and self.origin == other.origin
            and self.product == other.product
        )
    
    def __hash__(self):
        return hash((self.source, self.origin, self.product))