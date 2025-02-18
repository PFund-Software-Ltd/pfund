from pfund.products.product_base import BaseProduct


class FXProduct(BaseProduct):
    def _create_symbol(self) -> str:
        return self.base_asset + self.quote_asset

