from pfund.portfolios.portfolio_base import BasePortfolio
from pfund.mixins.assets.defi_assets_mixin import DeFiAssetsMixin


class DeFiPortfolio(DeFiAssetsMixin, BasePortfolio):
    def __init__(self):
        super().__init__()
        self.setup_assets()