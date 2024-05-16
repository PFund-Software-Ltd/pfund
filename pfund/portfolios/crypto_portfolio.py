from pfund.portfolios.base_portfolio import BasePortfolio
from pfund.mixins.assets.crypto_assets_mixin import CryptoAssetsMixin


class CryptoPortfolio(CryptoAssetsMixin, BasePortfolio):
    def __init__(self):
        super().__init__()
        self.setup_assets()