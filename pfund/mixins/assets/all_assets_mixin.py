from collections import defaultdict

from pfund.mixins.assets import TradfiAssetsMixin, CryptoAssetsMixin, DefiAssetsMixin
from pfund.const.common import SUPPORTED_TRADFI_PRODUCT_TYPES, SUPPORTED_CRYPTO_PRODUCT_TYPES


class AllAssetsMixin(TradfiAssetsMixin, CryptoAssetsMixin, DefiAssetsMixin):
    def setup_assets(self):
        all_assets = {}
        TradfiAssetsMixin.setup_assets(self)
        all_assets.update(self._all_assets)
        CryptoAssetsMixin.setup_assets(self)
        all_assets.update(self._all_assets)
        DefiAssetsMixin.setup_assets(self)
        all_assets.update(self._all_assets)
        self._all_assets = all_assets
    
    def _get_assets(self, ptype: str) -> defaultdict[str, dict]:
        ptype = ptype.upper()
        # TODO: add SUPPORTED_DEFI_PRODUCT_TYPES
        if ptype not in SUPPORTED_TRADFI_PRODUCT_TYPES + SUPPORTED_CRYPTO_PRODUCT_TYPES:
            raise KeyError(f'Invalid {ptype=}, supported choices: {SUPPORTED_TRADFI_PRODUCT_TYPES+SUPPORTED_CRYPTO_PRODUCT_TYPES}')
        else:
            return self._all_assets[ptype]
    