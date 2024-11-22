from pfund.universes.universe_base import BaseUniverse
from pfund.mixins.assets.defi_assets_mixin import DeFiAssetsMixin


class DeFiUniverse(DeFiAssetsMixin, BaseUniverse):
    def __init__(self):
        super().__init__()
        self.setup_assets()