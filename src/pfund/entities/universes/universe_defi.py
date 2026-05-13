from pfund.mixins.assets.defi_assets_mixin import DeFiAssetsMixin
from pfund.universes.universe_base import BaseUniverse


class DeFiUniverse(DeFiAssetsMixin, BaseUniverse):
    def __init__(self):
        super().__init__()
        self.setup_assets()
