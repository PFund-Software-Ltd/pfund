from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.products.product_base import BaseProduct

from pfund.products.mixins.forex import ForexMixin


class CryptoMixin(ForexMixin):
    pass