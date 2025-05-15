from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from pfund.products.product_base import BaseProduct

from pfund.products.mixins.future import FutureMixin


class PerpetualMixin(FutureMixin):
    expiration: None = None
    contract_code: None = None
    
    # override FutureMixin._derive_contract_code
    def _derive_contract_code(self) -> None:
        return None