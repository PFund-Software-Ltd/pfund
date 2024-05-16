from pfund.portfolios.base_portfolio import BasePortfolio
from pfund.mixins.assets.defi_assets_mixin import DefiAssetsMixin


class DefiPortfolio(DefiAssetsMixin, BasePortfolio):
    def __init__(self):
        super().__init__()
        self.setup_assets()