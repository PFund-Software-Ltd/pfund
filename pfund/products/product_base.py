from typing import Any

from decimal import Decimal

from pydantic import model_validator, validate_call
from pydantic import BaseModel, ConfigDict, Field

from pfund.enums import ProductType, Broker


def get_product_class(product_basis: str):
    from pfund.products.product_stock import StockProduct
    from pfund.products.product_future import FutureProduct
    from pfund.products.product_option import OptionProduct
    from pfund.products.product_derivative import DerivativeProduct
    
    ptype = product_basis.split('_')[2]
    ptype = ProductType[ptype]
    if ptype in [ProductType.FUT, ProductType.IFUT]:
        return FutureProduct
    elif ptype == ProductType.OPT:
        return OptionProduct
    elif ptype == ProductType.STK:
        return StockProduct
    elif ptype in [ProductType.PERP, ProductType.IPERP]:
        return DerivativeProduct
    else:
        return BaseProduct
    
    
class BaseProduct(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bkr: Broker | str
    exch: str=''
    base_asset: str
    quote_asset: str
    type: ProductType
    asset_pair: str=''
    basis: str=''
    # specifications that make a product unique, e.g. for options, specs are strike_price, expiration_date, etc.
    specs: dict = Field(default_factory=dict)
    name: str=''
    # if not provided, might be derived from name. e.g. AAPL_USD_STK -> AAPL. For crypto, it's empty.
    symbol: str=''  

    # information that requires data fetching
    tick_size: Decimal | None = None
    lot_size: Decimal | None = None

    @classmethod
    def get_required_specs(cls) -> set[str]:
        '''
        Args:
            specs_only: if True, only specifications such as 'expiration', 'strike_price'
            will be returned, otherwise all required fields defined in BaseProduct will be returned
        '''
        non_specs_required_fields = {
            field_name for field_name, field in BaseProduct.model_fields.items()
            if field.is_required()
        }
        return {
            field_name for field_name, field in cls.model_fields.items()
            if field.is_required() and field_name not in non_specs_required_fields
        }
    
    def model_post_init(self, __context: Any):
        if isinstance(self.bkr, str):
            self.bkr = Broker[self.bkr.upper()]
        self.exch = self.exch.upper()
        self.base_asset = self.base_asset.upper()
        self.quote_asset = self.quote_asset.upper()
        self.asset_pair = '_'.join([self.base_asset, self.quote_asset])
        self.basis = '_'.join([self.base_asset, self.quote_asset, self.type.value])
        self.symbol = self._create_symbol()
        self.specs = self._create_specs()
        self.name = self._create_product_name()

    @model_validator(mode='before')
    @classmethod
    def assert_required_fields(cls, data: dict) -> dict:
        product_basis = '_'.join([data['base_asset'], data['quote_asset'], data['type'].value])
        required_specs: set[str] = cls.get_required_specs()
        missing_fields = []
        missing_fields_msg = f'"{product_basis}" is missing the following required fields:'
        for field_name in required_specs:
            if field_name not in data:
                missing_fields.append(field_name)
                missing_fields_msg += f'\n- "{field_name}"'
        missing_fields_msg += f'\nplease add them as kwargs, e.g. {"=..., ".join(missing_fields)+"=..."} \n'
        if missing_fields:
            raise ValueError('\n\033[1m' + missing_fields_msg + '\033[0m')
        return data
    
    def to_dict(self) -> dict:
        return self.model_dump()

    @validate_call
    def set_symbol(self, symbol: str):
        self.symbol = symbol
    
    def _create_symbol(self) -> str:
        return self.symbol
    
    def _create_specs(self) -> dict:
        return self.specs
    
    def _create_product_name(self) -> str:
        return self.basis
    
    def is_crypto(self) -> bool:
        return (self.type == ProductType.CRYPTO) or (self.bkr == Broker.CRYPTO)
     
    def is_future(self) -> bool:
        return (self.type in [ProductType.FUT, ProductType.IFUT])
    
    def is_option(self) -> bool:
        return (self.type == ProductType.OPT)
    
    def is_index(self) -> bool:
        return (self.type == ProductType.INDEX)

    def is_stock(self) -> bool:
        return (self.type == ProductType.STK)
    
    def is_etf(self) -> bool:
        return (self.type == ProductType.ETF)
    
    def is_fx(self) -> bool:
        return (self.type == ProductType.FX)
    
    def is_bond(self) -> bool:
        return (self.type == ProductType.BOND)
    
    def is_mutual_fund(self) -> bool:
        return (self.type == ProductType.MTF)
    
    def is_commodity(self) -> bool:
        return (self.type == ProductType.CMDTY)
    
    def __str__(self):
        if self.exch:
            return f'Broker={self.bkr}|Exchange={self.exch}|Product={self.name}'
        else:
            return f'Broker={self.bkr}|Product={self.name}'

    def __repr__(self):
        if self.exch:
            return f'{self.bkr}:{self.exch}:{self.name}'
        else:
            return f'{self.bkr}:{self.name}'
    
    def __eq__(self, other):
        if not isinstance(other, BaseProduct):
            return NotImplemented  # Allow other types to define equality with BaseProduct
        return (
            self.bkr == other.bkr
            and self.exch == other.exch
            and self.name == other.name
        )
        
    def __hash__(self):
        return hash((self.bkr, self.exch, self.name))
