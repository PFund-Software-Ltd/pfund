from pfund.universes.base_universe import BaseUniverse
from pfund.mixins.assets.defi_assets_mixin import DefiAssetsMixin


class DefiUniverse(DefiAssetsMixin, BaseUniverse):
    def __init__(self):
        super().__init__()
        self.setup_assets()