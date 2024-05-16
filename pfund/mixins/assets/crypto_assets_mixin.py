from collections import defaultdict

from pfund.const.common import SUPPORTED_CRYPTO_PRODUCT_TYPES


class CryptoAssetsMixin:
    def setup_assets(self):
        self.spots = self.cryptos = defaultdict(dict)  # {exch: {pdt: e.g. position/product}}
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
        
    def _get_assets(self, ptype: str) -> defaultdict[str, dict]:
        ptype = ptype.upper()
        if ptype not in SUPPORTED_CRYPTO_PRODUCT_TYPES:
            raise KeyError(f'Invalid {ptype=}, {SUPPORTED_CRYPTO_PRODUCT_TYPES=}')
        else:
            return self._all_assets[ptype]