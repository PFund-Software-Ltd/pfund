from __future__ import annotations
from typing import Any, ClassVar

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator

from pfeed.enums import DataSource
from pfund.entities.products.product_basis import ProductBasis, ProductAssetType
from pfund.enums import Broker, CryptoExchange


class BaseProduct(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    source: DataSource
    broker: Broker | None = None
    exchange: CryptoExchange | str | None = None
    basis: ProductBasis | str
    specs: dict[str, Any] = Field(default_factory=dict, description='specifications that make a product unique, e.g. for options, specs are strike_price, expiration_date, etc.')
    symbol: str = Field(
        default='', 
        description='''
            product symbol used by the trading venue.
            If not provided, it will be derived automatically based on conventions e.g. AAPL_USD_STK -> AAPL.
            Note that the derived symbol might not be correct, it would be better to provide it manually when it is wrong.
        '''
    )
    name: str = Field(default='', description='unique product name, if not provided, venue + symbol will be used')
    tick_size: Decimal | None=None
    lot_size: Decimal | None=None
    
    @field_validator('basis', mode='before')
    @classmethod
    def _create_product_basis(cls, basis: ProductBasis | str):
        if isinstance(basis, str):
            basis = ProductBasis(basis=basis.upper())
        return basis
    
    @model_validator(mode='after')
    def _check_if_source_is_trading_venue(self):
        from pfund.enums import TradingVenue
        if self.source.value in TradingVenue.__members__:
            assert self.broker is not None, f"broker must be provided for {self.source}"
        return self
    
    @model_validator(mode='before')
    @classmethod
    def _assert_required_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data   # already a built model instance - nothing to assert
        required_specs: set[str] = cls.get_required_specs()
        missing_fields = []
        missing_fields_msg = f'"{data["basis"]}" is missing the following required fields:'
        for field_name in required_specs:
            if field_name not in data['specs']:
                missing_fields.append(field_name)
                missing_fields_msg += f'\n- "{field_name}"'
            else:
                # bring fields such as "expiration" to the top level
                data[field_name] = data['specs'][field_name]
        missing_fields_msg += f'\nplease add them as kwargs, e.g. {"=..., ".join(missing_fields)+"=..."} \n'
        if missing_fields:
            raise ValueError('\n\033[1m' + missing_fields_msg + '\033[0m')
        return data
    
    def model_post_init(self, __context: Any):
        # calls __mixin_post_init__ in e.g. StockMixin if exists
        if hasattr(self, '__mixin_post_init__'):
            self.__mixin_post_init__()
        self.specs = self._create_specs()
        self.symbol = self.symbol or self._create_symbol()
        self.name = self.name or self._create_name()
    
    def _create_name(self) -> str:
        return '_'.join([str(self.source), self.symbol])
    
    @property
    def base_asset(self) -> str:
        return self.basis.base_asset
    base = base_asset
    
    @property
    def quote_asset(self) -> str:
        return self.basis.quote_asset
    quote = quote_asset
    
    @property
    def asset_type(self) -> ProductAssetType:
        return self.basis.asset_type
    type = asset_type
    
    @property
    def asset_pair(self) -> str:
        return self.basis.asset_pair
    
    @classmethod
    def get_required_specs(cls) -> set[str]:
        '''Gets required specifications'''
        return {
            field_name for field_name, field in cls.model_fields.items()
            if field.is_required() and field_name not in BaseProduct.model_fields
        }
        
    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def _create_specs(self) -> dict[str, Any]:
        '''Create specifications that make a product unique'''
        from pfund.entities.products.product_crypto import CryptoProduct
        # TODO: add DeFiProduct
        specification_fields = [
            field for field in self.__class__.model_fields 
            if field not in BaseProduct.model_fields
            and field not in CryptoProduct.model_fields
        ]
        return {
            field: getattr(self, field)
            for field in specification_fields
        } 
        
    def is_inverse(self) -> bool:
        return self.asset_type.is_inverse()
    
    def is_crypto(self) -> bool:
        return self.asset_type.is_crypto()
    
    def is_future(self) -> bool:
        return self.asset_type.is_future()
    
    def is_perpetual(self) -> bool:
        return self.asset_type.is_perpetual()
    
    def is_option(self) -> bool:
        return self.asset_type.is_option()
    
    def is_index(self) -> bool:
        return self.asset_type.is_index()

    def is_stock(self) -> bool:
        return self.asset_type.is_stock()
    
    def is_etf(self) -> bool:
        return self.asset_type.is_etf()
    
    def is_forex(self) -> bool:
        return self.asset_type.is_forex()
    
    def is_bond(self) -> bool:
        return self.asset_type.is_bond()
    
    def is_mutual_fund(self) -> bool:
        return self.asset_type.is_mutual_fund()
    
    def is_commodity(self) -> bool:
        return self.asset_type.is_commodity()
    
    def is_derivative(self) -> bool:
        from pfund.entities.products.mixins.derivative import DerivativeMixin
        return isinstance(self, DerivativeMixin)
    
    def __str__(self):
        return '|'.join([
            f'source={self.source}',
            f'basis={self.basis}',
            *[f'{k}={v}' for k, v in sorted(self.specs.items())]
        ])

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseProduct):
            return NotImplemented  # Allow other types to define equality with BaseProduct
        return (
            self.source == other.source 
            and self.symbol == other.symbol
            and self.asset_type == other.asset_type
        )
        
    def __hash__(self) -> int:
        return hash((self.source, self.symbol, self.asset_type))
