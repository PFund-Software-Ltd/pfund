from typing_extensions import TypedDict, Annotated
from typing import TypeVar, Literal, TypeAlias, Union

from pfund.datas.resolution import Resolution
from pfund.components.strategies.strategy_base import BaseStrategy
from pfund.components.models.model_base import BaseModel
from pfund.components.features.feature_base import BaseFeature
from pfund.components.indicators.indicator_base import BaseIndicator
from pfund.entities.products.product_base import BaseProduct


StrategyT = TypeVar('StrategyT', bound=BaseStrategy)
ModelT = TypeVar('ModelT', bound=BaseModel)
FeatureT = TypeVar('FeatureT', bound=BaseFeature)
IndicatorT = TypeVar('IndicatorT', bound=BaseIndicator)
ProductT = TypeVar('ProductT', bound=BaseProduct)

Component = BaseStrategy | BaseModel | BaseFeature | BaseIndicator
EngineName: TypeAlias = str
ComponentName: TypeAlias = str
ProductName: TypeAlias = str
ResolutionRepr: TypeAlias = str
AccountName: TypeAlias = str
Currency: TypeAlias = str
# when user types in the full channel name, it is of type FullDataChannel
FullDataChannel: TypeAlias = str

# since Literal doesn't support variables as inputs, define variables in common.py here with prefix 't'
tEnvironment = Literal['BACKTEST', 'SANDBOX', 'PAPER', 'LIVE']
tTradingVenue = Literal['IB', 'BINANCE', 'BYBIT', 'OKX']
tBroker = Literal['CRYPTO', 'DEFI', 'IB']
tCryptoExchange = Literal['BINANCE', 'BYBIT', 'OKX']
tDatabase = Literal['DUCKDB', 'POSTGRESQL']
tOrderType = Literal['LIMIT', 'MARKET', 'STOP_MARKET', 'STOP_LIMIT']

ComponentNameWithData: TypeAlias = ComponentName
ComponentNameWithLogger: TypeAlias = ComponentName
ZeroMQSenderName = Union[
    Literal[
        "data_engine",  # ZeroMQ data engine for pulling data from trading venues
        "proxy",  # ZeroMQ publisher for broadcasting internal states to external apps
    ],
    # each component has TWO ZeroMQ ports:
    # ComponentName is used for component's signals_zmq
    # ComponentNameWithData is used for component's data_zmq
    ComponentName,
    ComponentNameWithData,  # {component_name}_data
    ComponentNameWithLogger,  # {component_name}_logger
]


class DataConfigDict(TypedDict, total=False):  # total=False makes fields optional
    extra_resolutions: list[Resolution]
    resample: dict[Annotated[Resolution, "ResampleeResolution"], Annotated[Resolution, "ResamplerResolution"]]
    shift: dict[Resolution, int]
    skip_first_bar: dict[Resolution, bool]
    stale_bar_timeout: int
