from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.products.product_base import BaseProduct


class StockMixin:
    def _create_symbol(self: BaseProduct) -> str:
        return self.base_asset
    