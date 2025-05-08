from typing import Any

from pfeed.enums import DataSource
from pfund.products.product_base import BaseProduct


class BaseData:
    def __init__(self, data_source: DataSource, data_origin: str, product: BaseProduct):
        self.data_source: DataSource = data_source
        self.data_origin: str = data_origin
        self.broker: str = product.broker
        self.exchange: str = product.exchange
        self.product: BaseProduct = product
    
    def is_crypto(self):
        return self.product.is_crypto()

    def is_time_based(self):
        return False

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseData):
            return NotImplemented
        return self.product == other.product
    
    def __hash__(self):
        return hash(self.product)