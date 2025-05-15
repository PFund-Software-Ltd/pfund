from collections import defaultdict

from pfund.enums import TradFiProductType


class TradFiAssetsMixin:
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
        if ptype not in TradFiProductType.__members__:
            raise KeyError(f'Invalid {ptype=}, supported product types: {list(TradFiProductType.__members__)}')
        else:
            return self._all_assets[ptype]