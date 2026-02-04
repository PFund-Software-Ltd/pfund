from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.entities.products.product_base import BaseProduct

from pfund.entities.products.mixins.forex import ForexMixin


class CryptoMixin(ForexMixin):
    pass