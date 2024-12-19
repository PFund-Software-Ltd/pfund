from pfund.products.product_base import BaseProduct


class StockProduct(BaseProduct):
    def _create_symbol(self) -> str:
        return self.base_asset
    