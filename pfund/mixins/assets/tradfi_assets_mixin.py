from collections import defaultdict

from pfund.const.common import SUPPORTED_TRADFI_PRODUCT_TYPES


class TradfiAssetsMixin:
    def setup_assets(self):
        self.stocks = defaultdict(dict)  # {exch: {pdt: e.g. position/product}}
        self.futures = defaultdict(dict)
        self.options = defaultdict(dict)
        self.cashes = defaultdict(dict)
        self.cryptos = self.spots = defaultdict(dict)
        self.bonds = defaultdict(dict)
        self.funds = defaultdict(dict)
        self.cmdties = defaultdict(dict)
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
        }
    
    def _get_assets(self, ptype: str) -> defaultdict[str, dict]:
        ptype = ptype.upper()
        if ptype not in SUPPORTED_TRADFI_PRODUCT_TYPES:
            raise KeyError(f'Invalid {ptype=}, {SUPPORTED_TRADFI_PRODUCT_TYPES=}')
        else:
            return self._all_assets[ptype]