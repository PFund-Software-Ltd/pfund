from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.products.product_base import BaseProduct

from collections import defaultdict

from rich.console import Console

from pfund.universes import BaseUniverse, CryptoUniverse, TradfiUniverse, DefiUniverse
from pfund.const.common import SUPPORTED_TRADFI_PRODUCT_TYPES, SUPPORTED_CRYPTO_PRODUCT_TYPES
from pfund.mixins.assets import TradfiAssetsMixin, CryptoAssetsMixin, DefiAssetsMixin


class Universe(TradfiAssetsMixin, CryptoAssetsMixin, DefiAssetsMixin, BaseUniverse):
    '''A (unified) universe that combines multiple sub-universes from different brokers.'''
    def __init__(self):
        BaseUniverse.__init__(self)
        self._sub_universes = {}  # {bkr: universe}
        all_assets = {}
        TradfiAssetsMixin.setup_assets(self)
        all_assets.update(self._all_assets)
        CryptoAssetsMixin.setup_assets(self)
        all_assets.update(self._all_assets)
        DefiAssetsMixin.setup_assets(self)
        all_assets.update(self._all_assets)
        self._all_assets = all_assets
        
    def initialize(self, products: list[BaseProduct]):
        for bkr in {product.bkr for product in products}:
            products_per_bkr = [product for product in products if product.bkr == bkr]
            universe: BaseUniverse = self._add_sub_universe(bkr, products_per_bkr)
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
    
    def _get_assets(self, ptype: str) -> defaultdict[str, dict[str, BaseProduct]]:
        ptype = ptype.upper()
        # TODO: add SUPPORTED_DEFI_PRODUCT_TYPES
        if ptype not in SUPPORTED_TRADFI_PRODUCT_TYPES + SUPPORTED_CRYPTO_PRODUCT_TYPES:
            raise KeyError(f'Invalid {ptype=}, supported choices: {SUPPORTED_TRADFI_PRODUCT_TYPES+SUPPORTED_CRYPTO_PRODUCT_TYPES}')
        else:
            return self._all_assets[ptype]
    
    def _add_sub_universe(self, bkr: str, products: list[BaseProduct]) -> BaseUniverse:
        if bkr not in self._sub_universes:
            if bkr == 'CRYPTO':
                universe = CryptoUniverse.from_products(products)
            elif bkr == 'DEFI':
                universe = DefiUniverse.from_products(products)
            else:
                universe = TradfiUniverse.from_products(products)
            self._sub_universes[bkr] = universe
        return self._sub_universes[bkr]