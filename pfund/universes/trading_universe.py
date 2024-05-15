from collections import defaultdict

from pfund.universes.base_universe import BaseUniverse
from pfund.const.common import SUPPORTED_PRODUCT_TYPES


class TradingUniverse(BaseUniverse):
    def __init__(self):
        super().__init__()
        self.stocks = defaultdict(dict)  # {exch: {pdt: product}}
        self.futures = defaultdict(dict)
        self.options = defaultdict(dict)
        self.cashes = defaultdict(dict)
        self.cryptos = defaultdict(dict)
        self.bonds = defaultdict(dict)
        self.funds = defaultdict(dict)
        self.cmdties = defaultdict(dict)
        self._assets = {
            # ptype: asset_class
            'STK': self.stocks,
            'FUT': self.futures,
            'OPT': self.options,
            'CASH': self.cashes,
            'CRYPTO': self.cryptos,
            'BOND': self.bonds,
            'FUND': self.funds,
            'CMDTY': self.cmdties,
        }
    
    def get_asset_class(self, product_type: str):
        try:
            return super().get_asset_class(product_type)
        except KeyError:
            raise KeyError(f'Invalid {product_type=}, {SUPPORTED_PRODUCT_TYPES=}')