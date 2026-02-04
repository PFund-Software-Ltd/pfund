from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.products.product_base import BaseProduct

from collections import defaultdict

from rich.console import Console

from pfund.universes.universe_base import BaseUniverse
from pfund.universes.universe_cefi import CeFiUniverse
from pfund.universes.universe_tradfi import TradFiUniverse
from pfund.universes.universe_defi import DeFiUniverse
from pfund.mixins.assets.all_assets_mixin import AllAssetsMixin


class Universe(AllAssetsMixin, BaseUniverse):
    '''A (unified) universe that combines multiple sub-universes from different brokers.'''
    def __init__(self):
        BaseUniverse.__init__(self)
        self._sub_universes = {}  # {bkr: universe}
        self.setup_assets()
        
    def initialize(self, products: list[BaseProduct]):
        for bkr in {product.bkr for product in products}:
            products_per_bkr = [product for product in products if product.bkr == bkr]
            universe: BaseUniverse = self._add_sub_universe(bkr, products_per_bkr)
            # e.g. allows using 'universe.crypto' to access CeFiUniverse
            setattr(self, bkr.lower(), universe)

        # TODO: use global() to dynamically create attributes?
        for attr in (
            'stocks', 
            'futures', 
            'options', 
            'cashes', 
            'cryptos', 
            'bonds', 
            'funds', 
            'cmdties', 
            'perps', 
            'iperps', 
            'ifutures'
        ):
            # combine assets from sub-universes, e.g. self.futures = futures in crypto universe + futures in tradfi universe
            setattr(self, attr, self.combine_dicts(*(getattr(uni, attr) for uni in self._sub_universes.values() if hasattr(uni, attr))))
            if attr == 'cryptos':
                self.spots = self.cryptos
    
    @staticmethod
    # Function to combine nested dictionaries without copying
    def combine_dicts(*dicts: defaultdict[str, dict]) -> defaultdict[str, dict]:
        combined = defaultdict(dict)
        for d in dicts:
            for key, sub_dict in d.items():
                if key not in combined:
                    combined[key] = sub_dict
                else:
                    Console().print(
                        f'WARNING: Duplicate {key=} found in Universe.combine_dicts(), '
                        f"please check if there's any conflict between sub-universes."
                        f"e.g. exch=SMART in IB could also be used in another broker.",
                        style='bold'
                    )
                    combined[key].update(sub_dict)
        return combined
    
    def _add_sub_universe(self, bkr: str, products: list[BaseProduct]) -> BaseUniverse:
        if bkr not in self._sub_universes:
            if bkr == 'CRYPTO':
                universe = CeFiUniverse.from_products(products)
            elif bkr == 'DEFI':
                universe = DeFiUniverse.from_products(products)
            else:
                universe = TradFiUniverse.from_products(products)
            self._sub_universes[bkr] = universe
        return self._sub_universes[bkr]