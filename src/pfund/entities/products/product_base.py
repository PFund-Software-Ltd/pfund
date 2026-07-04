from __future__ import annotations

from typing import Any, ClassVar, NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    from pfund_kit.utils.yaml import YAMLDocument

from pathlib import Path

from pfeed.enums import DataSource
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pfund.entities.markets.market_base import BaseMarket
from pfund.entities.products.asset_type import AssetType
from pfund.entities.products.product_basis import ProductBasis
from pfund.enums import TradingVenue


class ProductKey(NamedTuple):
    symbol: str
    asset_type: AssetType

    def __str__(self) -> str:
        return f"{self.symbol}.{self.asset_type.value}"


class BaseProduct(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid"
    )

    source: DataSource
    venue: TradingVenue | None = None
    exchange: str | None = None
    basis: ProductBasis
    specs: dict[str, Any] = Field(
        default_factory=dict,
        description="specifications that make a product unique, e.g. for options, specs are strike_price, expiration_date, etc.",
    )
    symbol: str = Field(
        default="",
        description="""
            product symbol used by the trading venue.
            If not provided, it will be derived automatically based on conventions e.g. AAPL_USD_STK -> AAPL.
            Note that the derived symbol might not be correct, it would be better to provide it manually when it is wrong.
        """,
    )
    name: str = Field(
        default="",
        description="unique product name, if not provided, venue + symbol will be used",
    )
    market: BaseMarket | None = Field(
        default=None, description="market this product belongs to"
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_specs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data  # already a built model instance - nothing to assert
        allowed_specs: set[str] = cls.get_allowed_specs()
        required_specs: set[str] = cls.get_required_specs()
        provided_specs: dict[str, Any] = data.get("specs") or {}

        unknown_specs = sorted(set(provided_specs) - allowed_specs)
        if unknown_specs:
            raise ValueError(
                f'"{data["basis"]}" got unexpected specs: {unknown_specs}.\n'
                + f"Allowed specs for {cls.__name__}: {sorted(allowed_specs) or 'none'}"
            )

        missing_fields = sorted(required_specs - set(provided_specs))
        if missing_fields:
            missing_fields_msg = (
                f'"{data["basis"]}" is missing the following required fields:'
            )
            for field_name in missing_fields:
                missing_fields_msg += f'\n- "{field_name}"'
            missing_fields_msg += f"\nplease add them as kwargs, e.g. {'=..., '.join(missing_fields) + '=...'}"
            raise ValueError(missing_fields_msg)

        # bring fields such as "expiration" to the top level so pydantic validates them
        for field_name in allowed_specs & set(provided_specs):
            data[field_name] = provided_specs[field_name]
        return data

    @model_validator(mode="after")
    def _validate_asset_type(self):
        if self.venue is None:
            return self
        VenueClass = self.venue.venue_class
        if str(self.asset_type) not in VenueClass.METADATA.asset_types:
            raise ValueError(f"Invalid asset type: {self.asset_type}")
        return self

    def model_post_init(self, __context: Any):
        # calls __mixin_post_init__ in e.g. StockMixin if exists
        if hasattr(self, "__mixin_post_init__"):
            self.__mixin_post_init__()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        self.specs = self._create_specs()
        self.name = self.name or self._create_name()
        self.symbol = self.symbol or self._create_symbol()
        if self.venue is None and self.source in TradingVenue.__members__:
            self.venue = TradingVenue[self.source]
        self.market = self._load_market()

    @property
    def key(self) -> ProductKey:
        return ProductKey(symbol=self.symbol, asset_type=self.asset_type)

    def _create_name(self) -> str:
        return "_".join([str(self.source), self.symbol])

    # Override this method to customize the symbol creation logic
    def _create_symbol(self) -> str:
        return self.symbol

    def _load_market(self, file_path: Path | None = None) -> BaseMarket | None:
        from pfund_kit.utils.yaml import load

        if self.venue is None:
            return None
        VenueClass = self.venue.venue_class
        if file_path is None:
            file_path = VenueClass._create_markets_yml_file_path()
        if not file_path.exists():
            return None
        document: YAMLDocument = load(file_path) or {}
        market_data = document.get(str(self.key))  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
        if market_data is None:
            return None
        return VenueClass.Market(**market_data)  # pyright: ignore[reportUnknownArgumentType]

    @property
    def base_asset(self) -> str:
        return self.basis.base_asset

    base = base_asset

    @property
    def quote_asset(self) -> str:
        return self.basis.quote_asset

    quote = quote_asset

    @property
    def asset_type(self) -> AssetType:
        assert self.basis.asset_type, "asset_type is None"
        return self.basis.asset_type

    type = asset_type

    @property
    def asset_pair(self) -> str:
        return self.basis.asset_pair

    @classmethod
    def get_allowed_specs(cls) -> set[str]:
        """
        Gets all spec field names (required + optional) contributed by asset-type mixins,
        i.e. all model fields except those defined by BaseProduct and the venue-specific
        Product subclass (e.g. BybitProduct's `category`). Mixins are plain classes, so any
        field owned by a BaseProduct subclass in the MRO is a non-spec field.
        """
        non_spec_fields: set[str] = set()
        for class_ in cls.__mro__:
            if (
                class_ is not cls
                and isinstance(class_, type)
                and issubclass(class_, BaseProduct)
            ):
                non_spec_fields |= set(class_.model_fields)
        return set(cls.model_fields) - non_spec_fields

    @classmethod
    def get_required_specs(cls) -> set[str]:
        """Gets spec fields that must be provided (no default)."""
        return {
            field_name
            for field_name in cls.get_allowed_specs()
            if cls.model_fields[field_name].is_required()
        }

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def _create_specs(self) -> dict[str, Any]:
        """Create specifications that make a product unique"""
        return {field: getattr(self, field) for field in self.get_allowed_specs()}

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

    def desc_str(self):
        return f"{self.source} product name={self.name} symbol={self.symbol}"

    def __str__(self):
        return "|".join(
            [
                f"source={self.source}",
                f"basis={self.basis}",
                f"name={self.name}",
                f"symbol={self.symbol}",
                *[f"{k}={v}" for k, v in sorted(self.specs.items())],
            ]
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseProduct):
            return False
        return self.source == other.source and self.key == other.key

    def __hash__(self) -> int:
        return hash((self.source, self.key))
