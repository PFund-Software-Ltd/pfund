from typing import Any

from datetime import date

from pfund.products.product_derivative import DerivativeProduct
from pfund.const.enums import FutureMonthCode


class FutureProduct(DerivativeProduct):
    expiration: date
    contract_code: str=''
    
    def model_post_init(self, __context: Any):
        super().model_post_init(__context)
        self.contract_code = self._derive_contract_code(self.expiration)
    
    @staticmethod
    def _derive_contract_code(expiration: date) -> str:
        expiration_year, expiration_month, _ = str(expiration).split('-')
        expiration_year = expiration_year[-2:]
        month_code = [code.value for code in FutureMonthCode][int(expiration_month) - 1]
        return month_code + expiration_year
    
    def _create_specs(self) -> dict:
        '''Create specifications that make a product unique'''
        return {
            'expiration': self.expiration,
        }
        
    def _create_product_name(self) -> str:
        return '_'.join([self.basis, str(self.expiration)])
    
    def _create_symbol(self) -> str:
        '''
        Creates default symbol e.g. ESZ23
        '''
        symbol = self.base_asset + self.contract_code
        return symbol