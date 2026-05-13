from pfund.mixins.assets.cefi_assets_mixin import CeFiAssetsMixin
from pfund.universes.universe_base import BaseUniverse


class CeFiUniverse(CeFiAssetsMixin, BaseUniverse):
    def __init__(self):
        super().__init__()
        self.setup_assets()


CryptoUniverse = CeFiUniverse
