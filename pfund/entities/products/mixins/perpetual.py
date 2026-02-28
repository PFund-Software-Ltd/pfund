from __future__ import annotations
from typing import TYPE_CHECKING
from typing_extensions import override
if TYPE_CHECKING:
    from pfund.entities.products.product_base import BaseProduct

from pfund.entities.products.mixins.future import FutureMixin


class PerpetualMixin(FutureMixin):
    expiration: None = None
    contract_code: None = None
    
    @override  # override FutureMixin._derive_contract_code
    def _derive_contract_code(self) -> None:
        return None
    
    @override  # override FutureMixin._create_symbol
    def _create_symbol(self: FutureMixin | BaseProduct) -> str:
        '''
        Creates default symbol e.g. ESZ23
        '''
        return self.base_asset + '_' + self.quote_asset
