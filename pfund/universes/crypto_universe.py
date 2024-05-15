from collections import defaultdict

from pfund.universes.base_universe import BaseUniverse
from pfund.const.common import SUPPORTED_CRYPTO_PRODUCT_TYPES

    
class CryptoUniverse(BaseUniverse):
    def __init__(self):
        super().__init__()
        self.spots = defaultdict(dict)  # {exch: {pdt: product}}
        self.perps = defaultdict(dict)
        self.iperps = defaultdict(dict)
        self.futures = defaultdict(dict)
        self.ifutures = defaultdict(dict)
        self.options = defaultdict(dict)
        self._assets = {
            # ptype: asset_class
            'SPOT': self.spots,
            'PERP': self.perps,
            'IPERP': self.iperps,
            'FUT': self.futures,
            'IFUT': self.ifutures,
            'OPT': self.options,
        }

    def get_asset_class(self, product_type: str):
        try:
            return super().get_asset_class(product_type)
        except KeyError:
            raise KeyError(f'Invalid {product_type=}, {SUPPORTED_CRYPTO_PRODUCT_TYPES=}')
    