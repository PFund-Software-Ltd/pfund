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

        self.stocks = defaultdict(dict)  # {exch: {pdt: product}}
        self.futures = defaultdict(dict)
        self.options = defaultdict(dict)
        self.cashes = defaultdict(dict)
        self.spots = self.cryptos = defaultdict(dict)
        self.bonds = defaultdict(dict)
        self.funds = defaultdict(dict)
        self.cmdties = defaultdict(dict)
        self.perps = defaultdict(dict)
        self.iperps = defaultdict(dict)
        self.ifutures = defaultdict(dict)
        
        # TODO: DeFi
        # self.liquidity_pools = ... (no need to combine dicts)
        
        self._all_assets = {
            # ptype: assets
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
            
    def initialize(self, products: list[BaseProduct]):
        for bkr in {product.bkr for product in products}:
            products_per_bkr = [product for product in products if product.bkr == bkr]
            universe = self._add_sub_universe(bkr, products_per_bkr)
            setattr(self, bkr.lower(), universe)

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
    
    def _get_assets(self, product_type: str):
        try:
            return super()._get_assets(product_type)
        except KeyError:
            raise KeyError(f'Invalid {product_type=}, supported asset classes: {SUPPORTED_PRODUCT_TYPES+SUPPORTED_CRYPTO_PRODUCT_TYPES}')
    
    def _add_sub_universe(self, bkr: str, products: list[BaseProduct]) -> BaseUniverse:
        if bkr == 'CRYPTO':
            universe = CryptoUniverse.from_products(products)
        elif bkr == 'DEFI':
            universe = DefiUniverse.from_products(products)
        else:
            universe = TradfiUniverse.from_products(products)
        self._sub_universes[bkr] = universe
        return universe