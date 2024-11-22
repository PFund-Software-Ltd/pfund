from pfund.universes.universe_base import BaseUniverse
from pfund.mixins.assets.tradfi_assets_mixin import TradFiAssetsMixin


class TradFiUniverse(TradFiAssetsMixin, BaseUniverse):
    def __init__(self):
        super().__init__()
        self.setup_assets()