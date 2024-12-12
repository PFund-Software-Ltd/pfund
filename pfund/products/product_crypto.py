from __future__ import annotations
from typing import Any, Literal

import os
from decimal import Decimal

from pfund.const.enums import CeFiProductType, CryptoExchange
from pfund.utils.utils import load_yaml_file
from pfund.products.product_base import get_product_class
from pfund.const.paths import CACHE_PATH
from pfund.products.product_base import BaseProduct


def get_CryptoProduct(product_basis: str) -> type[BaseProduct]:
    Product = get_product_class(product_basis)
    
    class CryptoProduct(Product):
        exch: CryptoExchange | str
        type: CeFiProductType | str
        category: str
        tick_size: Decimal | None = None
        lot_size: Decimal | None = None
        
        # EXTEND: may add taker_fee, maker_fee, multiplier
        # but the current problem is these values can't be obtained from apis consistently across exchanges,
        # and for fees, they can be different for different accounts,
        # so for now let users set them manually in e.g. strategy's config
        
        # NOTE: take in exchange-specific arguments, not used for now
        # kwargs: dict = Field(default_factory=dict)

        @classmethod
        def get_required_specs(cls) -> set[str]:
            '''
            Get specifications such as 'expiration', 'strike_price'
            '''
            non_specs_required_fields = {
                field_name for field_name, field in BaseProduct.model_fields.items()
                if field.is_required()
            }
            return {
                field_name for field_name, field in Product.model_fields.items()
                if field.is_required() and field_name not in non_specs_required_fields
            }
            
        def model_post_init(self, __context: Any):
            super().model_post_init(__context)
            if isinstance(self.exch, str):
                self.exch = CryptoExchange[self.exch.upper()]
            if isinstance(self.type, str):
                self.type = CeFiProductType[self.type.upper()]
            self.category = self.category.upper()
            self._load_config()
        
        def __getattr__(self, name: str) -> Any:
            if name in self.kwargs:
                return self.kwargs[name]
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            
        def _load_config(self):
            file_path = f'{CACHE_PATH}/{self.exch.value.lower()}/market_configs.yml'
            if not os.path.exists(file_path):
                return
            config = load_yaml_file(file_path)[self.category]
            if self.name not in config:
                raise ValueError(
                    f'Product {self.name} not found in {self.exch} market configs:\n'
                    f'This could mean either:\n'
                    f'  - The product {self.name} does not exist on {self.exch}\n'
                    f'  - Your market configs are outdated\n'
                    f'Solutions:\n'
                    f'  - Check if the product exists on {self.exch}\n'
                    f'  - Check if there is a typo in USD, USDC or USDT\n'
                    f'  - To update configs, reinitialize the exchange with:\n'
                    f'    exchange(refetch_market_configs=True) \n'
                )
            self.tick_size = Decimal(config[self.name]['tick_size'])
            self.lot_size = Decimal(config[self.name]['lot_size'])

        def get_fee(self, fee_type: Literal['taker', 'maker'], in_bps=False):
            if fee_type == 'taker':
                fee = self.taker_fee
            elif fee_type == 'maker':
                fee = self.maker_fee
            if not in_bps:
                fee /= 10000
            return fee
        
        def is_linear(self) -> bool:
            return (self.type in [CeFiProductType.PERP, CeFiProductType.FUT])
        
        def is_inverse(self) -> bool:
            return (self.type in [CeFiProductType.IPERP, CeFiProductType.IFUT])
        
        def is_perpetual(self) -> bool:
            return (self.type in [CeFiProductType.PERP, CeFiProductType.IPERP])
        
        def is_spot(self) -> bool:
            return (self.type == CeFiProductType.SPOT)
        
    return CryptoProduct
