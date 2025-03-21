from typing import Any

from pfund.products.product_base import BaseProduct


class BaseData:
    def __init__(self, product: BaseProduct):
        self.bkr: str = product.bkr
        self.exch: str = product.exch
        self.pdt: str = product.name
        self.product = product
    
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