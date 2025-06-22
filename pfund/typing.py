from typing_extensions import TypedDict, Annotated
from typing import TypeVar, Literal, TypeAlias

from pfund.datas.resolution import Resolution
from pfund.strategies.strategy_base import BaseStrategy
from pfund.models.model_base import BaseModel
from pfund.features.feature_base import BaseFeature
from pfund.indicators.indicator_base import BaseIndicator
from pfund.products.product_base import BaseProduct


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

# since Literal doesn't support variables as inputs, define variables in common.py here with prefix 't'
tEnvironment = Literal['BACKTEST', 'SANDBOX', 'PAPER', 'LIVE']
tTradingVenue = Literal['IB', 'BINANCE', 'BYBIT', 'OKX']
tBroker = Literal['CRYPTO', 'DEFI', 'IB']
tCryptoExchange = Literal['BINANCE', 'BYBIT', 'OKX']
tDatabase = Literal['DUCKDB', 'POSTGRESQL']
tOrderType = Literal['LIMIT', 'MARKET', 'STOP_MARKET', 'STOP_LIMIT']


class DatasetSplitsDict(TypedDict, total=True):
    train: float
    val: float
    test: float


class DataRangeDict(TypedDict, total=False):
    start_date: str
    end_date: str


class DataConfigDict(TypedDict, total=False):  # total=False makes fields optional
    extra_resolutions: list[Resolution]
    resample: dict[Annotated[Resolution, "ResampleeResolution"], Annotated[Resolution, "ResamplerResolution"]]
    shift: dict[Resolution, int]
    skip_first_bar: dict[Resolution, bool]
    stale_bar_timeout: int


class BaseEngineSettingsDict(TypedDict, total=False):
    zmq_urls: dict[EngineName | tTradingVenue | ComponentName, str]
    zmq_ports: dict[
        Literal[
            "proxy",  # ZeroMQ xsub-xpub proxy for messaging from trading venues -> engine -> components
            "router",  # ZeroMQ router-pull for pulling messages from components (e.g. strategies/models) -> engine -> trading venues
            "publisher",  # ZeroMQ publisher for broadcasting internal states to external apps
        ] 
        | tTradingVenue 
        | ComponentName,
        int
    ]


class TradeEngineSettingsDict(BaseEngineSettingsDict, total=False):
    cancel_all_at: dict[str, bool]
    # force refetching market configs
    refetch_market_configs: bool
    # Always use the WebSocket API for actions like placing or canceling orders, even if REST is available.
    websocket_first: bool


class BacktestEngineSettingsDict(BaseEngineSettingsDict, total=False):
    retention_period: int
    commit_to_git: bool


class ExternalListenersDict(TypedDict, total=False):
    notebooks: list[str] | bool
    dashboards: list[str] | bool
    monitor: bool
    recorder: bool
    profiler: bool