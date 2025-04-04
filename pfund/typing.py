from typing_extensions import TypedDict, Annotated
from typing import TypeVar, Literal

from pfeed.typing import tDATA_LAYER, tSTORAGE
from pfund.datas.resolution import Resolution
from pfund.strategies.strategy_base import BaseStrategy
from pfund.models.model_base import BaseModel, BaseFeature
from pfund.indicators.indicator_base import BaseIndicator
from pfund.products.product_base import BaseProduct


StrategyT = TypeVar('StrategyT', bound=BaseStrategy)
ModelT = TypeVar('ModelT', bound=BaseModel)
FeatureT = TypeVar('FeatureT', bound=BaseFeature)
IndicatorT = TypeVar('IndicatorT', bound=BaseIndicator)
ProductT = TypeVar('ProductT', bound=BaseProduct)


# since Literal doesn't support variables as inputs, define variables in common.py here with prefix 't'
tENVIRONMENT = Literal['BACKTEST', 'SANDBOX', 'PAPER', 'LIVE']
tTRADING_VENUE = Literal['IB', 'BINANCE', 'BYBIT', 'OKX']
tBROKER = Literal['CRYPTO', 'DEFI', 'IB']
tCRYPTO_EXCHANGE = Literal['BINANCE', 'BYBIT', 'OKX']
tTRADFI_PRODUCT_TYPE = Literal['STK', 'FUT', 'ETF', 'OPT', 'FX', 'CRYPTO', 'BOND', 'MTF', 'CMDTY']
tCEFI_PRODUCT_TYPE = Literal['SPOT', 'PERP', 'IPERP', 'FUT', 'IFUT', 'OPT']


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
    resample: dict[Annotated[Resolution, "ResampleeResolution"], Annotated[Resolution, "ResamplerResolution"]]
    shift: dict[Resolution, int]
    skip_first_bar: dict[Resolution, bool]
    stale_bar_timeout: int


class StorageConfigDict(TypedDict, total=False):
    pfeed_use_ray: bool  # if use_ray in pfeed
    retrieve_per_date: bool  # refer to `retrieve_per_date` in pfeed's get_historical_data()
    data_layer: tDATA_LAYER
    from_storage: tSTORAGE
    to_storage: tSTORAGE
    # configs specific to the storage type, for MinIO, it's access_key and secret_key etc.
    storage_options: dict


class TradeEngineSettingsDict(TypedDict, total=False):
    zmq_ports: dict
    cancel_all_at: dict[Literal['start', 'stop'], bool]
    # TODO: when use_ray=True, parallelism down to model level? feature level?


class BacktestEngineSettingsDict(TypedDict, total=False):
    retention_period: int
    commit_to_git: bool