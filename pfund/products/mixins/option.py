from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.products.product_base import BaseProduct
    
from decimal import Decimal
from datetime import date

from pydantic import Field, field_validator

from pfund.products.mixins.derivative import DerivativeMixin
from pfund.enums.option_type import OptionType


class OptionMixin(DerivativeMixin):
    expiration: date
    strike_price: Decimal = Field(ge=0.0)
    option_type: OptionType
    
    @field_validator('option_type', mode='before')
    @classmethod
    def _uppercase_option_type(cls, option_type: str) -> str:
        return option_type.upper()
    
    @field_validator('strike_price', mode='after')
    @classmethod
    def _remove_redundant_zeros(cls, strike_price: Decimal) -> Decimal:
        return strike_price.quantize(Decimal('1')) if strike_price == strike_price.to_integral_value() else strike_price

    def _create_symbol(self: OptionMixin | BaseProduct) -> str:
        '''
        Creates default symbol e.g. TSLA241213C00075000 following OSI format (Option Symbology Initiative)
        '''
        expiration = self.expiration.strftime('%y%m%d')  # convert expiration to yymmdd
        num_of_digits = 8
        strike_price = str(int(self.strike_price * 1000)).zfill(num_of_digits)
        symbol = self.base_asset + expiration + self.option_type.value[0] + strike_price
        return symbol
