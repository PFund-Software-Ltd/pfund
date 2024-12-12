from decimal import Decimal
from datetime import date

from pydantic import Field

from pfund.products.product_derivative import DerivativeProduct
from pfund.const.enums.option_type import OptionType


class OptionProduct(DerivativeProduct):
    expiration: date
    strike_price: Decimal = Field(ge=0.0)
    option_type: OptionType

    def _create_specs(self) -> dict:
        '''Create specifications that make a product unique'''
        return {
            'strike_price': self.strike_price,
            'expiration': self.expiration,
            'option_type': self.option_type,
        }
    
    def _create_product_name(self) -> str:
        return '_'.join([self.basis, str(self.expiration), self.option_type.value, str(self.strike_price)])
 