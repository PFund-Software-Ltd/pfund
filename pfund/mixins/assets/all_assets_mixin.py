from collections import defaultdict

from pfund.mixins.assets import TradFiAssetsMixin, CeFiAssetsMixin, DeFiAssetsMixin
from pfund.const.enums import TradFiProductType, CeFiProductType


class AllAssetsMixin(TradFiAssetsMixin, CeFiAssetsMixin, DeFiAssetsMixin):
    def setup_assets(self):
        all_assets = {}
        TradFiAssetsMixin.setup_assets(self)
        all_assets.update(self._all_assets)
        CeFiAssetsMixin.setup_assets(self)
        all_assets.update(self._all_assets)
        DeFiAssetsMixin.setup_assets(self)
        all_assets.update(self._all_assets)
        self._all_assets = all_assets
    
    def _get_assets(self, ptype: str) -> defaultdict[str, dict]:
        ptype = ptype.upper()
        # TODO: add SUPPORTED_DEFI_PRODUCT_TYPES
        SUPPORTED_PRODUCT_TYPES = list(TradFiProductType.__members__) + list(CeFiProductType.__members__)
        if ptype not in SUPPORTED_PRODUCT_TYPES:
            raise KeyError(f'Invalid {ptype=}, supported choices: {SUPPORTED_PRODUCT_TYPES}')
        else:
            return self._all_assets[ptype]
    