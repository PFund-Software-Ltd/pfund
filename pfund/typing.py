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
LocalComponent: TypeAlias = Component
EngineName: TypeAlias = str
ComponentName: TypeAlias = str
ProductName: TypeAlias = str
ResolutionRepr: TypeAlias = str

# since Literal doesn't support variables as inputs, define variables in common.py here with prefix 't'
tENVIRONMENT = Literal['BACKTEST', 'SANDBOX', 'PAPER', 'LIVE']
tTRADING_VENUE = Literal['IB', 'BINANCE', 'BYBIT', 'OKX']
tBROKER = Literal['CRYPTO', 'DEFI', 'IB']
tCRYPTO_EXCHANGE = Literal['BINANCE', 'BYBIT', 'OKX']
tTRADFI_PRODUCT_TYPE = Literal['STK', 'FUT', 'ETF', 'OPT', 'FX', 'CRYPTO', 'BOND', 'MTF', 'CMDTY']
tCEFI_PRODUCT_TYPE = Literal['SPOT', 'PERP', 'IPERP', 'FUT', 'IFUT', 'OPT']
tDATABASE = Literal['DUCKDB', 'POSTGRESQL', 'PGLITE', 'TIMESCALEDB']


class DatasetSplitsDict(TypedDict, total=True):
    train: float
    val: float
    test: float


class DataRangeDict(TypedDict, total=False):
    start_date: str
    end_date: str


class DataConfigDict(TypedDict, total=False):  # total=False makes fields optional
    extra_resolutions: list[Resolution]
    orderbook_depth: int
    fast_orderbook: bool
    resample: dict[Annotated[Resolution, "ResampleeResolution"], Annotated[Resolution, "ResamplerResolution"]]
    shift: dict[Resolution, int]
    skip_first_bar: dict[Resolution, bool]
    stale_bar_timeout: int


class BaseEngineSettingsDict(TypedDict, total=False):
    zmq_urls: dict[EngineName | tTRADING_VENUE | ComponentName, str]
    zmq_ports: dict[
        Literal[
            "proxy",  # ZeroMQ xsub-xpub proxy for messaging from trading venues -> engine -> components
            "router",  # ZeroMQ router-pull for pulling messages from components (e.g. strategies/models) -> engine -> trading venues
            "publisher",  # ZeroMQ publisher for broadcasting internal states to external apps
        ] 
        | tTRADING_VENUE 
        | ComponentName,
        int
    ]


class TradeEngineSettingsDict(BaseEngineSettingsDict, total=False):
    cancel_all_at: dict[str, bool]
    # TODO: bytewax_dataflow: dict


class BacktestEngineSettingsDict(BaseEngineSettingsDict, total=False):
    retention_period: int
    commit_to_git: bool


class ExternalListenersDict(TypedDict, total=False):
    notebooks: list[str] | bool
    dashboards: list[str] | bool
    monitor: bool
    recorder: bool
    profiler: bool