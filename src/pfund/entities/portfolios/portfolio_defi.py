from pfund.mixins.assets.defi_assets_mixin import DeFiAssetsMixin
from pfund.portfolios.portfolio_base import BasePortfolio


class DeFiPortfolio(DeFiAssetsMixin, BasePortfolio):
    def __init__(self):
        super().__init__()
        self.setup_assets()
