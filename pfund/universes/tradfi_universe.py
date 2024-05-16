from pfund.universes.base_universe import BaseUniverse
from pfund.mixins.assets.tradfi_assets_mixin import TradfiAssetsMixin


class TradfiUniverse(TradfiAssetsMixin, BaseUniverse):
    def __init__(self):
        super().__init__()
        self.setup_assets()