from pfund.portfolios.portfolio_base import BasePortfolio
from pfund.mixins.assets.cefi_assets_mixin import CeFiAssetsMixin


class CeFiPortfolio(CeFiAssetsMixin, BasePortfolio):
    def __init__(self):
        super().__init__()
        self.setup_assets()

CryptoPortfolio = CeFiPortfolio