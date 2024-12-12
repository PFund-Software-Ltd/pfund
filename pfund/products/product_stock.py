from typing import Any

from pfund.products.product_base import BaseProduct


class StockProduct(BaseProduct):
    def model_post_init(self, __context: Any):
        super().model_post_init(__context)
        self.symbol = self.base_asset
    