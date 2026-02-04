from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.products.product_base import BaseProduct

from collections import defaultdict


class BaseUniverse:
    @classmethod
    def from_products(cls, products: list[BaseProduct]) -> BaseUniverse:
        universe = cls()
        for product in products:
            universe.add(product)
        return universe
    
    def get(
        self, 
        ptype: str, 
        exch: str='', 
        pdt: str=''
    ) -> defaultdict[str, dict[str, BaseProduct]] | dict[str, BaseProduct] | BaseProduct | None:
        assets = self._get_assets(ptype)
        exch, pdt = exch.upper(), pdt.upper()
        if not exch and not pdt:
            return assets
        elif exch and not pdt:
            return assets.get(exch, None)
        elif exch and pdt:
            return assets[exch].get(pdt, None)
        else:  # not exch and pdt
            return {exch: product for exch, pdt_to_product in assets.items() for product in pdt_to_product.values() if str(product) == pdt}
            
    def add(self, product: BaseProduct):
        assets: dict = self._get_assets(product.ptype)
        assets[product.exch][str(product)] = product
    
    def update(self, product: BaseProduct):
        assets: dict = self._get_assets(product.ptype)
        if self.has(product):
            assets[product.exch][str(product)] = product
        else:
            raise ValueError(f'{product} not in {assets}')
     
    def has(self, product: BaseProduct) -> bool:
        assets: dict = self._get_assets(product.ptype)
        return product.exch in assets and str(product) in assets[product.exch]
    
    def remove(self, product: BaseProduct):
        assets: dict = self._get_assets(product.ptype)
        if self.has(product):
            del assets[product.exch][str(product)]
        else:
            raise ValueError(f'{product} not in {assets}')
    
    # TODO: add more functionalities, e.g. 
    # - get_products_by_risk_level(assets)
    # - get_stocks_by_sector()
    # - ...