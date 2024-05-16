from pfund.portfolios.base_portfolio import BasePortfolio
from pfund.mixins.assets.tradfi_assets_mixin import TradfiAssetsMixin


class TradfiPortfolio(TradfiAssetsMixin, BasePortfolio):
    def __init__(self):
        super().__init__()
        self.setup_assets()