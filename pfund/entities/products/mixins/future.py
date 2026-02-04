from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.products.product_base import BaseProduct

from datetime import date

from pfund.products.mixins.derivative import DerivativeMixin


class FutureMixin(DerivativeMixin):
    expiration: date
    contract_code: str=''
    
    def __mixin_post_init__(self):
        super().__mixin_post_init__()
        self.contract_code = self._derive_contract_code()
    
    def _derive_contract_code(self):
        from pfund.enums import FutureMonthCode
        expiration_year, expiration_month, _ = str(self.expiration).split('-')
        expiration_year = expiration_year[-2:]
        month_code = [code.value for code in FutureMonthCode][int(expiration_month) - 1]
        contract_code = month_code + expiration_year
        return contract_code
    
    def _create_symbol(self: FutureMixin | BaseProduct) -> str:
        '''
        Creates default symbol e.g. ESZ23
        '''
        return self.base_asset + self.contract_code
