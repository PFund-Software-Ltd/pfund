from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from pfund.entities.products.mixins.forex import ForexMixin


class CryptoMixin(ForexMixin):
    pass
