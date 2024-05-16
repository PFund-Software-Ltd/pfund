from pfund.universes.base_universe import BaseUniverse
from pfund.mixins.assets.crypto_assets_mixin import CryptoAssetsMixin

    
class CryptoUniverse(CryptoAssetsMixin, BaseUniverse):
    def __init__(self):
        super().__init__()
        self.setup_assets()
