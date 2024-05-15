from __future__ import annotations

from abc import ABC

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.products.product_base import BaseProduct


class BaseUniverse(ABC):
    def get_asset_class(self, product_type: str):
        return self._assets[product_type]
    
    def add(self, product: BaseProduct):
        asset_class: dict = self.get_asset_class(product.ptype)
        asset_class[product.exch][product.name] = product
    
    def update(self, product: BaseProduct):
        asset_class: dict = self.get_asset_class(product.ptype)
        if repr(product) in asset_class:
            asset_class[product.exch][product.name] = product
        else:
            raise ValueError(f'{product} not in {asset_class}')
    
    def has(self, product: BaseProduct) -> bool:
        asset_class: dict = self.get_asset_class(product.ptype)
        return product.exch in asset_class and product.name in asset_class[product.exch]
    
    def remove(self, product: BaseProduct):
        asset_class: dict = self.get_asset_class(product.ptype)
        if self.has(product):
            del asset_class[product.exch][product.name]
        else:
            raise ValueError(f'{product} not in {asset_class}')
    
    # TODO: add more functionalities, e.g. 
    # - get_products_by_risk_level(asset_class)
    # - get_stocks_by_sector()
    # - ...