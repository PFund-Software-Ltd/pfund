from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from pfeed.enums import DataSource
    from pfund.enums import TradingVenue, Broker, CryptoExchange

from pfund.products.product_base import BaseProduct


class BaseData:
    def __init__(self, data_source: DataSource, data_origin: str, product: BaseProduct):
        self.data_source: DataSource = data_source
        self.data_origin: str = data_origin
        self.product: BaseProduct = product
    
    def is_time_based(self):
        return False
    
    @property
    def trading_venue(self) -> TradingVenue:
        return self.product.trading_venue
    tv = trading_venue
    
    @property
    def broker(self) -> Broker:
        return self.product.broker
    bkr = broker
    
    @property
    def exchange(self) -> CryptoExchange | str:
        return self.product.exchange
    exch = exchange

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseData):
            return NotImplemented
        return (
            self.data_source == other.data_source
            and self.data_origin == other.data_origin
            and self.product == other.product
        )
    
    def __hash__(self):
        return hash((self.data_source, self.data_origin, self.product))