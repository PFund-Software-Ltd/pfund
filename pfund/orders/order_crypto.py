from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pfund.products.product_crypto import CryptoProduct
    from pfund.accounts.account_crypto import CryptoAccount

from pfund.orders.order_base import BaseOrder


class CryptoOrder(BaseOrder):
    account: CryptoAccount
    product: CryptoProduct
