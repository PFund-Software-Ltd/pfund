from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.products.product_base import BaseProduct

from collections import defaultdict

from rich.console import Console

from pfund.universes import BaseUniverse, CryptoUniverse, TradfiUniverse, DefiUniverse
from pfund.const.common import SUPPORTED_PRODUCT_TYPES, SUPPORTED_CRYPTO_PRODUCT_TYPES


class Universe(BaseUniverse):
    '''A (unified) universe that combines multiple sub-universes from different brokers.'''
    def __init__(self):
        super().__init__()
        self._sub_universes = {}  # {bkr: universe}
            
    def initialize(self, products: list[BaseProduct]):
        products_per_bkr = {bkr: [product for product in products if product.bkr == bkr] for bkr in {product.bkr for product in products}}
        for bkr, products in products_per_bkr.items():
            self._add_sub_universe(bkr, products)
        
        self.stocks = self.combine_dicts(*(u.stocks for u in self._sub_universes.values() if hasattr(u, 'stocks')))
        self.futures = self.combine_dicts(*(u.futures for u in self._sub_universes.values() if hasattr(u, 'futures')))
        self.options = self.combine_dicts(*(u.options for u in self._sub_universes.values() if hasattr(u, 'options')))
        self.cashes = self.combine_dicts(*(u.cashes for u in self._sub_universes.values() if hasattr(u, 'cashes')))
        self.spots = self.cryptos = self.combine_dicts(*(u.cryptos for u in self._sub_universes.values() if hasattr(u, 'cryptos')))
        self.bonds = self.combine_dicts(*(u.bonds for u in self._sub_universes.values() if hasattr(u, 'bonds')))
        self.funds = self.combine_dicts(*(u.funds for u in self._sub_universes.values() if hasattr(u, 'funds')))
        self.cmdties = self.combine_dicts(*(u.cmdties for u in self._sub_universes.values() if hasattr(u, 'cmdties')))
        
        self.perps = self.combine_dicts(*(u.perps for u in self._sub_universes.values() if hasattr(u, 'perps')))
        self.iperps = self.combine_dicts(*(u.iperps for u in self._sub_universes.values() if hasattr(u, 'iperps')))
        self.ifutures = self.combine_dicts(*(u.ifutures for u in self._sub_universes.values() if hasattr(u, 'ifutures')))
        
        # TODO: DeFi
        # self.liquidity_pools = ... (no need to combine dicts)
        
        self._all_assets = {
            # ptype: asset_class
            'STK': self.stocks,
            'FUT': self.futures,
            'OPT': self.options,
            'CASH': self.cashes,
            'CRYPTO': self.cryptos,
            'BOND': self.bonds,
            'FUND': self.funds,
            'CMDTY': self.cmdties,
            'SPOT': self.spots,
            'PERP': self.perps,
            'IPERP': self.iperps,
            'IFUT': self.ifutures,
        }
    
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
    
    def _get_assets(self, product_type: str):
        try:
            return super()._get_assets(product_type)
        except KeyError:
            raise KeyError(f'Invalid {product_type=}, supported asset classes: {SUPPORTED_PRODUCT_TYPES+SUPPORTED_CRYPTO_PRODUCT_TYPES}')
    
    def _add_sub_universe(self, bkr: str, products: list[BaseProduct]):
        if bkr == 'CRYPTO':
            universe = CryptoUniverse.from_products(products)
        elif bkr == 'DEFI':
            universe = DefiUniverse.from_products(products)
        else:
            universe = TradfiUniverse.from_products(products)
        self._sub_universes[bkr] = universe
        setattr(self, bkr.lower(), universe)