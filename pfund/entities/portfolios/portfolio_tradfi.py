from pfund.portfolios.portfolio_base import BasePortfolio
from pfund.mixins.assets.tradfi_assets_mixin import TradFiAssetsMixin


class TradFiPortfolio(TradFiAssetsMixin, BasePortfolio):
    def __init__(self):
        super().__init__()
        self.setup_assets()