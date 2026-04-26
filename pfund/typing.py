from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Literal, TypeAlias, TypedDict, Any
if TYPE_CHECKING:
    from pfund.components.strategies.strategy_base import BaseStrategy
    from pfund.components.models.model_base import BaseModel
    from pfund.components.features.feature_base import BaseFeature
    from pfund.components.indicators.indicator_base import BaseIndicator
    from pfund.entities.products.product_base import BaseProduct

StrategyT = TypeVar('StrategyT', bound="BaseStrategy")
ModelT = TypeVar('ModelT', bound="BaseModel")
FeatureT = TypeVar('FeatureT', bound="BaseFeature")
IndicatorT = TypeVar('IndicatorT', bound="BaseIndicator")
ProductT = TypeVar('ProductT', bound="BaseProduct")

Component: TypeAlias = "BaseStrategy | BaseModel | BaseFeature | BaseIndicator"
ComponentT = TypeVar("ComponentT", bound="Component")


EngineName: TypeAlias = str
ComponentName: TypeAlias = str
ProductName: TypeAlias = str
AccountName: TypeAlias = str
ColumnName: TypeAlias = str
Currency: TypeAlias = str
# when user types in the full channel name, it is of type FullDataChannel
FullDataChannel: TypeAlias = str


class ParsedMessage(TypedDict):
    ts: float
    channel: str
    data: dict[str, Any]
    

# since Literal doesn't support variables as inputs, define variables in common.py here with prefix 't'
# DEPRECATED: to be removed
tEnvironment = Literal['BACKTEST', 'SANDBOX', 'PAPER', 'LIVE']
tTradingVenue = Literal['IB', 'BINANCE', 'BYBIT', 'OKX']
tBroker = Literal['CRYPTO', 'DEFI', 'IB']
tCryptoExchange = Literal['BINANCE', 'BYBIT', 'OKX']
tDatabase = Literal['DUCKDB', 'POSTGRESQL']
tOrderType = Literal['LIMIT', 'MARKET', 'STOP_MARKET', 'STOP_LIMIT']

EngineNameWithProxy: TypeAlias = EngineName
ComponentNameWithData: TypeAlias = ComponentName
ComponentNameWithLogger: TypeAlias = ComponentName
ZMQUrlKey = EngineName | Literal["data_engine"] | ComponentName
# NOTE: only senders need to set ports
ZMQPortKey = (
    ZMQUrlKey 
    | ComponentNameWithData  # {component_name}_data
    | ComponentNameWithLogger  # {component_name}_logger
)
