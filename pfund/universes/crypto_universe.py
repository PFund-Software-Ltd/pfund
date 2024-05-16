from collections import defaultdict

from pfund.universes.base_universe import BaseUniverse
from pfund.const.common import SUPPORTED_CRYPTO_PRODUCT_TYPES

    
class CryptoUniverse(BaseUniverse):
    def __init__(self):
        super().__init__()
        self.spots = self.cryptos = defaultdict(dict)  # {exch: {pdt: product}}
        self.perps = defaultdict(dict)
        self.iperps = defaultdict(dict)
        self.futures = defaultdict(dict)
        self.ifutures = defaultdict(dict)
        self.options = defaultdict(dict)
        self._all_assets = {
            # ptype: assets
            'SPOT': self.spots,
            'PERP': self.perps,
            'IPERP': self.iperps,
            'FUT': self.futures,
            'IFUT': self.ifutures,
            'OPT': self.options,
        }

    def _get_assets(self, ptype: str):
        try:
            return super()._get_assets(ptype)
        except KeyError:
            raise KeyError(f'Invalid {ptype=}, {SUPPORTED_CRYPTO_PRODUCT_TYPES=}')