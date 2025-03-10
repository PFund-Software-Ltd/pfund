from collections import defaultdict

from pfund.enums import CeFiProductType


class CeFiAssetsMixin:
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
        if ptype not in CeFiProductType.__members__:
            raise KeyError(f'Invalid {ptype=}, supported product type: {list(CeFiProductType.__members__)}')
        else:
            return self._all_assets[ptype]