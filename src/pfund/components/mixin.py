# pyright: reportUninitializedInstanceVariable=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportOptionalMemberAccess=false, reportUnknownArgumentType=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from narwhals.typing import IntoDataFrame

    from pfund.components.features.feature_base import BaseFeature
    from pfund.components.models.model_base import BaseModel, UnderlyingModel
    from pfund.datas import BarData, QuoteData, TickData
    from pfund.datas.data_base import BaseData
    from pfund.datas.data_market import MarketData
    from pfund.datas.stores.base_data_store import BaseDataStore
    from pfund.datas.stores.market_data_store import MarketDataStore
    from pfund.datas.timeframe import Timeframe
    from pfeed.storages.storage_config import StorageConfig
    from pfund.engines.contexts.trade_engine_context import TradeEngineContext
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings
    from pfund.entities.products.product_base import BaseProduct
    from pfund.typing import (
        Component,
        ComponentName,
        ComponentT,
        FeatureT,
        ModelT,
        ProductName,
        Signals,
    )

import datetime
import logging
from pathlib import Path

import narwhals as nw
from pfeed.enums import DataCategory
from pfund_kit.style import RichColor, TextStyle, cprint
from pfund_kit.utils import toml

from pfund.components.actor_proxy import ActorProxy
from pfund.datas.data_config import DataConfig
from pfund.datas.databoy import DataBoy
from pfund.datas.resolution import Resolution
from pfund.datas.stores.trading_store import TradingStore
from pfund.enums import ComponentType, Environment, RunMode, TradingVenue
from pfund.utils.decorators import ray_method


class ComponentMixin:
    config: ClassVar[dict[str, Any]] = {
        "max_rows": None,  # None means no limit, component will keep all data in memory
        "warmup_period": None,  # None means no warmup period, component will start right away
        "lookback_period": None,  # None means using the whole dataset
    }
    params: ClassVar[dict[str, Any]] = {}

    # custom post init for-common attributes of strategy and model
    def __mixin_post_init__(self, *args: Any, **kwargs: Any):
        self.__pfund_args__ = args
        self.__pfund_kwargs__ = kwargs

        self._name = self._get_default_name()
        self._run_mode: RunMode | None = None
        self._resolution: Resolution | None = None
        self._context: TradeEngineContext | None = None

        self.logger: logging.Logger = logging.getLogger("pfund")
        self.databoy = DataBoy(self)
        self.store = TradingStore(self)

        self._signal_cols: list[str] = []
        self.signals: Signals = {}  # latest signals

        self.products: dict[ProductName, BaseProduct] = {}
        self.models: dict[str, BaseModel | ActorProxy[BaseModel]] = {}
        self.features: dict[str, BaseFeature | ActorProxy[BaseFeature]] = {}

        self._is_running = False
        self._is_gathered = False
        self._is_top_component = False

    @property
    def env(self) -> Environment:
        assert self._context is not None, "context is not set"
        return self._context.env

    @property
    def name(self) -> ComponentName:
        return self._name

    @property
    def run_mode(self) -> RunMode:
        assert self._run_mode is not None, "run_mode is not set"
        return self._run_mode

    @property
    def context(self) -> TradeEngineContext:
        assert self._context is not None, "context is not set"
        return self._context

    @property
    def settings(self) -> TradeEngineSettings:
        return self.context.settings

    @property
    def market_data_store(self) -> MarketDataStore:
        return self.databoy.get_data_store(DataCategory.MARKET_DATA)

    @property
    def data_stores(self) -> dict[DataCategory, BaseDataStore[Any, Any]]:
        return self.databoy._data_stores

    # useful when user wants to set logger specific to the component. currently 'pfund' is the default logger.
    def set_logger(self, logger: logging.Logger):
        self.logger = logger
        if self.is_remote(direct_only=False):
            self._setup_logging()

    def is_top_component(self) -> bool:
        return self._is_top_component

    def _hydrate(
        self,
        name: ComponentName,
        run_mode: RunMode,
        resolution: Resolution | str,
        engine_context: TradeEngineContext,
        storage_config: StorageConfig,
        is_top_component: bool = False,
    ):
        """
        Hydrates the component with necessary attributes after initialization.

        Args:
            name (ComponentName): The name to assign to this component.
            run_mode (RunMode): The mode in which the component will run (e.g., local or remote).
            resolution (Resolution | str): The data resolution used by this component.
            engine_context (TradeEngineContext): The engine context associated with this component. It is None if the component is running in a remote process.
            storage_config (StorageConfig): where this component's artifacts are persisted.
                Always resolved by the caller (per-component override, else the engine
                default or parent component's config) — never None by the time it lands
                here, so the store receives a concrete config.
        """
        self._context = engine_context
        self._run_mode = run_mode
        self._set_name(name)
        self._set_resolution(resolution)
        self.set_logger(self.logger)
        self._is_top_component = is_top_component
        self.store.set_storage_config(storage_config)

    def _forward(self, data: BarData):
        child_signals: dict[ComponentName, Signals] = (
            self.databoy._wait_for_children_signals(data)
        )
        for signals in child_signals.values():
            self.store.update_df(signals, data=data)
        if self.is_ready(data=data):
            X = self.get_df(
                kind="features",
                window_size=self.config["lookback_period"],
                to_native=True,
            )
            self.signals = self.signalize(X)
            self.store.update_df(self.signals, data=data)
        # self.signals = latest signals: {} during warmup, last computed otherwise
        if self.databoy.is_using_zmq():
            self.databoy._publish_signals(self.signals, data)

    def _setup_scheduler(self):
        self.databoy._setup_scheduler()
        for component in self.components:
            component._setup_scheduler()

    def _setup_messaging(self):
        self.databoy._setup_messaging()
        for component in self.components:
            component._setup_messaging()
        self.databoy._subscribe()

    def _setup_logging(self):
        """Sets up logging for component running in remote process, uses zmq's PUSHHandler to send logs to engine"""
        from pfeed.streaming.zeromq import ZeroMQ
        from pfund_kit.logging.configurator import LoggingDictConfigurator
        from pfund_kit.utils import get_free_port

        from pfund.utils.zmq_pub_handler import ZMQPubHandler

        # configure logging based on pfund's logging config, e.g. log_level, log_file, log_format, etc.
        logging_configurator = LoggingDictConfigurator.create(
            log_path=self.context.pfund_config.log_path / self.env / self.context.name,
            logging_config=self.context.logging_config,
            lazy=True,
            use_colored_logger=True,
        )
        logging_configurator.configure()

        # remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()

        # add zmq PushHandler
        zmq_url = self.settings.zmq_urls.get(self.name, ZeroMQ.DEFAULT_URL)
        zmq_port = get_free_port()
        zmq_handler = ZMQPubHandler(f"{zmq_url}:{zmq_port}")
        zmq_formatter = logging.Formatter(
            fmt="%(message)s | from:%(filename)s fn:%(funcName)s ln:%(lineno)d (sent@%(asctime)s.%(msecs)03d)",
            datefmt="%H:%M:%S",
        )
        zmq_handler.setFormatter(zmq_formatter)
        self.logger.addHandler(zmq_handler)
        self.settings.zmq_urls.update({self.name: zmq_url})
        self.settings.zmq_ports.update({self.name + "_logger": zmq_port})

    @classmethod
    def load_config(cls, config: dict[str, Any] | None = None, file_path: str = ""):
        """Loads config from a dict or a TOML file, overriding the class-level defaults.
        Args:
            config: Config dict to set directly.
            file_path: Path to a TOML file to load config from.
        """
        if config and file_path:
            raise ValueError("config and file_path cannot be provided at the same time")
        if config:
            cls.config = config
        elif file_path:
            config = toml.load(file_path)
            if config is None:
                raise ValueError(f"Failed to load config from {file_path}")
            cls.config = config

    @classmethod
    def load_params(cls, params: dict[str, Any] | None = None, file_path: str = ""):
        """Loads params from a dict or a TOML file, overriding the class-level defaults.
        Args:
            params: Params dict to set directly.
            file_path: Path to a TOML file to load params from.
        """
        if params and file_path:
            raise ValueError("params and file_path cannot be provided at the same time")
        if params:
            cls.params = params
        elif file_path:
            params = toml.load(file_path)
            if params is None:
                raise ValueError(f"Failed to load params from {file_path}")
            cls.params = params

    def _check_input_sources(self):
        """Checks the component has at least one input source (data or child components)
        and that its X (features) will be non-empty."""
        if not self.get_datas():
            # only the top strategy executes (trade() -> orders need products/prices);
            # a sub-strategy is a non-executing signal generator, so like models/features
            # it just needs some input source: data or child components.
            if self.is_strategy() and self.is_top_component():
                raise ValueError(
                    f"{self.name} has no market data, did you forget to call add_data() / add_datas()?"
                )
            elif not self.components:
                raise ValueError(
                    f"{self.name} has no input source, did you forget to call add_data() / add_datas(), "
                    + "or add a child component via add_model() / add_feature()?"
                )

        # With no children, at least one populated data category must provide features.
        has_data_features = any(
            store.data_as_features and bool(store.get_datas())
            for store in self.data_stores.values()
        )
        if not self.components and not has_data_features:
            raise ValueError(
                f"{self.name} has no features: add a child component via add_model() / add_feature(), "
                + "or pass as_features=True when adding data"
            )

    def _check_config(self):
        """Validates config and writes normalized values back into self.config,
        so runtime readers (e.g. is_ready() comparing row count against warmup_period)
        never see None."""
        for key in ("max_rows", "warmup_period", "lookback_period"):
            if key not in self.config:
                raise ValueError(f"{key} is not set in {self.name} config")
        max_rows = self.config["max_rows"]
        warmup_period = self.config["warmup_period"]
        lookback_period = self.config["lookback_period"]
        if max_rows is None:
            cprint(
                f"'max_rows' is None. {self.name} data will be UNBOUNDED",
                style=TextStyle.BOLD + RichColor.YELLOW,
            )
        if lookback_period is None:
            cprint(
                f"'lookback_period' is None. {self.name} will use the WHOLE DATASET to compute signals",
                style=TextStyle.BOLD + RichColor.YELLOW,
            )
        else:
            assert lookback_period >= 1, (
                f"{self.name} {lookback_period=} is less than 1, please set lookback_period >= 1"
            )
        if warmup_period is None:
            # a lookback window of N rows needs at least N rows of history, so an unset
            # warmup_period defaults to lookback_period instead of tripping the
            # lookback <= warmup check below with a warmup the user never set
            warmup_period = lookback_period or 0
            if warmup_period:
                cprint(
                    f"'warmup_period' is None. defaulting to {lookback_period=} for {self.name}",
                    style=TextStyle.BOLD + RichColor.YELLOW,
                )
            else:
                cprint(
                    f"'warmup_period' is None. {self.name} will be ready to compute signals IMMEDIATELY",
                    style=TextStyle.BOLD + RichColor.YELLOW,
                )
            self.config["warmup_period"] = warmup_period
        assert warmup_period >= 0, (
            f"{self.name} {warmup_period=} is less than 0, please set warmup_period >= 0"
        )
        if lookback_period is not None and lookback_period > warmup_period:
            raise ValueError(
                f"{self.name} config: {lookback_period=} is greater than {warmup_period=}, please set lookback_period <= warmup_period"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.__class__.__name__,
            "component_name": self.name,
            "data_start": self.context.data_start,
            "data_end": self.context.data_end,
            "resolution": repr(self.resolution),
            "component_type": self.component_type,
            "df_form": self._df_form,
            "signal_cols": self._signal_cols,
            "run_mode": self.run_mode,
            "signature": (self.__pfund_args__, self.__pfund_kwargs__),
            "config": self.config,
            "params": self.params,
            "settings": self.settings.model_dump(),
            "datas": [data.to_dict() for data in self.get_datas()],
            "models": list(self.models),
            "features": list(self.features),
        }

    @property
    def components(self) -> list[Component | ActorProxy[Component]]:
        return self.get_components()

    def get_component(
        self, name: ComponentName
    ) -> Component | ActorProxy[Component] | None:
        return self.models.get(name, None) or self.features.get(name, None)

    def get_components(self) -> list[Component | ActorProxy[Component]]:
        return [
            *self.models.values(),
            *self.features.values(),
        ]

    @property
    def data_df(self) -> IntoDataFrame:
        return self.get_df(
            kind="data", data_category=None, window_size=None, to_native=True
        )

    @property
    def features_df(self) -> IntoDataFrame:
        return self.get_df(kind="features", window_size=None, to_native=True)

    X = features_df

    @property
    def X_pandas(self) -> pd.DataFrame:
        from pfeed._etl.base import convert_dataframe

        return convert_dataframe(self.X, data_tool="pandas")

    @property
    def X_polars(self) -> pl.DataFrame:
        from pfeed._etl.base import convert_dataframe

        return convert_dataframe(self.X, data_tool="polars").collect()

    @property
    def signals_df(self) -> IntoDataFrame:
        return self.get_df(kind="signals", window_size=None, to_native=True)

    @property
    def trading_df(self) -> IntoDataFrame:
        return self.get_df(kind="trading", window_size=None, to_native=True)

    @property
    def full_df(self) -> IntoDataFrame:
        return self.get_df(kind="full", window_size=None, to_native=True)

    df = full_df

    @property
    def df_pandas(self) -> pd.DataFrame:
        from pfeed._etl.base import convert_dataframe

        return convert_dataframe(self.df, data_tool="pandas")

    # mainly used to avoid type casting, since self.df will return IntoDataFrame,
    # which messes up with pl.DataFrame type hints.
    @property
    def df_polars(self) -> pl.DataFrame:
        from pfeed._etl.base import convert_dataframe

        return convert_dataframe(self.df, data_tool="polars").collect()

    def get_df(
        self,
        *,
        kind: Literal["data", "features", "signals", "trading", "full"] = "full",
        data_category: DataCategory | str | None = DataCategory.MARKET_DATA,
        window_size: int | None = None,
        pivot_data: bool = False,
        to_native: bool = False,
    ) -> nw.DataFrame[Any] | IntoDataFrame:
        """Returns one of the stored dataframes in either trading store or data stores.

        Args:
            kind: Which frame to return.
                - 'data': input dataframe from a data store (e.g. market data, news).
                - 'features': features used by this component (signals from child components), features part in trading_df
                - 'signals': signals generated by this component, signals part in trading_df
                - 'trading': trading_df = features + signals
                - 'full': full_df = data_df (merged data dfs from different categories) + trading_df (features + signals)
            data_category: For kind='data', which data category to return.
                If None, return a merged data_df from all data categories.
                Ignored when kind != 'data'.
                Defaults to market data.
            window_size: Number of most recent rows to return.
                Defaults to None, i.e. return the whole dataframe.
            pivot_data: pivot data dataframe from long form to wide form.
                Ignored when kind != 'data'.
                Defaults to False.
            to_native: If True, return the underlying backend frame (polars/pandas) instead
                of a Narwhals DataFrame. Defaults to True.
        """
        if kind == "data":
            if data_category is not None:
                store = self.databoy.get_data_store(data_category)
                df = store.get_df(
                    window_size=window_size, pivot=pivot_data, to_native=to_native
                )
            else:
                data_dfs: dict[DataCategory, IntoDataFrame | None] = {
                    category: store.get_df(window_size=window_size, to_native=True)
                    for category, store in self.data_stores.items()
                }
                df = self.merge_data_dfs(data_dfs)
                if not to_native:
                    df = nw.from_native(df)
        elif kind == "features":
            df = self.store.get_df(
                kind="features", window_size=window_size, to_native=to_native
            )
        elif kind == "signals":
            df = self.store.get_df(
                kind="signals", window_size=window_size, to_native=to_native
            )
        elif kind == "trading":
            df = self.store.get_df(
                kind="trading", window_size=window_size, to_native=to_native
            )
        elif kind == "full":
            trading_df = self.store.get_df(
                kind="trading", window_size=window_size, to_native=False
            )
            data_dfs_not_included_in_trading_df = {
                category: data_store.get_df(
                    window_size=window_size,
                    to_native=True,
                )
                for category, data_store in self.data_stores.items()
                if not data_store.data_as_features and data_store.get_datas()
            }
            if data_dfs_not_included_in_trading_df:
                data_df = nw.from_native(
                    self.merge_data_dfs(data_dfs_not_included_in_trading_df)
                )
                join_cols = [
                    col
                    for col in self.store.KEY_COLS
                    if col in data_df.columns and col in trading_df.columns
                ]
                if not join_cols:
                    raise ValueError(
                        f"No common key columns between {self.name}'s "
                        f"data_df {data_df.columns} and trading_df {trading_df.columns}"
                    )

                df = data_df.join(trading_df, on=join_cols, how="full")

                for key in join_cols:
                    right_key = f"{key}_right"
                    if right_key in df.columns:
                        df = df.with_columns(
                            nw.coalesce(key, right_key).alias(key)
                        ).drop(right_key)

                df = df.sort(join_cols)
            else:
                df = trading_df
            if to_native:
                df = df.to_native()
        else:
            raise ValueError(f"Invalid {kind=} for component {self.name}")
        return df

    @ray_method
    def get_df_form(self) -> Literal["wide", "long"]:
        return self._df_form

    @ray_method
    def get_trading_store(self) -> TradingStore:
        return self.store

    @ray_method
    def get_signals(self) -> Signals:
        return self.signals

    def _get_default_signal_cols(self, num_cols: int) -> list[str]:
        if num_cols == 1:
            columns = [self.name]
        else:
            columns = [f"{self.name}-{i}" for i in range(num_cols)]
        return columns

    def set_df_form(self, df_form: Literal["wide", "long"]):
        self._df_form = df_form.lower()

    def get_datas(self) -> list[BaseData]:
        datas = []
        for data_store in self.data_stores.values():
            datas.extend(data_store.get_datas())
        return datas

    def _get_supported_resolutions(
        self, product: BaseProduct
    ) -> dict[Timeframe, list[int]]:
        venue = product.venue
        assert venue is not None, "venue is not None"
        return venue.METADATA.supported_resolutions

    @staticmethod
    def dt(ts: float, tz: datetime.tzinfo = datetime.UTC) -> datetime.datetime:
        from pfund_kit.utils.temporal import convert_ts_to_dt

        return convert_ts_to_dt(ts, tz=tz)

    @staticmethod
    def now(tz: datetime.tzinfo = datetime.UTC) -> datetime.datetime:
        return datetime.datetime.now(tz=tz)

    @property
    def resolution(self) -> Resolution:
        assert self._resolution is not None, "resolution is not set"
        return self._resolution

    @property
    def _source_artifact(self) -> Path:
        import inspect

        source_file = inspect.getsourcefile(type(self))
        if source_file is None:
            raise ValueError(f"cannot locate source file for {type(self).__name__}")
        return Path(source_file)

    @property
    def _data_artifact(self) -> IntoDataFrame:
        # TODO: extract the latest rows (that have not been written to pfeed) from the trading store's df
        raise NotImplementedError

    def _set_resolution(self, resolution: Resolution | str):
        if self._resolution:
            raise ValueError(
                f"{self.name} already has a resolution {self._resolution}, cannot set to {resolution}"
            )
        resolution = Resolution(resolution)
        if not resolution.is_bar():
            raise ValueError(
                f"{self.component_type} must use a bar resolution (e.g. '1s', '1m', '1h', '1d'), got {resolution=}"
            )
        self._resolution = resolution

    def _set_name(self, name: str):
        if not name:
            return
        self._name = name
        if not self._name.lower().endswith(self.component_type):
            self._name += f"_{self.component_type}"

    def is_strategy(self) -> bool:
        return self.component_type == ComponentType.strategy

    def is_model(self) -> bool:
        return self.component_type == ComponentType.model

    def is_feature(self) -> bool:
        return self.component_type == ComponentType.feature

    def is_running(self) -> bool:
        return self._is_running

    def is_remote(self, direct_only: bool = True) -> bool:
        """
        Returns whether this component is running in a remote (Ray) process.

        Args:
            direct_only (bool):
                - If True (default), only checks the component's own `run_mode`.
                  Reflects whether the component *itself* is declared to be remote.
                  e.g. a model running inside a strategy (ray actor) is "local" relative to itself.
                - If False, also checks whether this component's code is currently
                  executing inside a Ray actor process via `ray.get_runtime_context()`.
                  This captures the case where a declaratively local component is
                  nested inside a remote parent and therefore runs in the parent's
                  actor process.

        Returns:
            bool: True if the component is declared remote, or (when `direct_only=False`)
                  is currently executing inside a Ray actor process.
        """
        assert self.run_mode is not None, f"{self.name} has no run mode"
        is_remote = self.run_mode == RunMode.REMOTE
        if is_remote or direct_only:
            return is_remote
        try:
            import ray

            if not ray.is_initialized():
                return False
            return ray.get_runtime_context().get_actor_id() is not None
        except Exception:
            return False
        return False

    def _add_product(
        self,
        venue: TradingVenue | str,
        basis: str,
        exchange: str = "",
        symbol: str = "",
        name: str = "",
        **specs: Any,
    ) -> BaseProduct:
        venue = TradingVenue[venue.upper()]
        VenueClass = venue.venue_class
        product = VenueClass.create_product(
            basis=basis, exchange=exchange, name=name, symbol=symbol, **specs
        )
        if product.name not in self.products:
            self.products[product.name] = product
            self.logger.debug(f"added {product.desc_str()}")
        else:
            existing_product = self.products[product.name]
            assert existing_product == product, (
                f"product name '{product.name}' is already used"
            )
        return self.products[product.name]

    def get_product(self, name: ProductName) -> BaseProduct | None:
        return self.products.get(name, None)

    def get_data(
        self, product: ProductName, resolution: Resolution | str
    ) -> MarketData | None:
        return self.market_data_store.get_data(product, resolution)

    def _get_default_name(self):
        return self.__class__.__name__

    def add_data(
        self,
        venue: TradingVenue | str,
        product: str,
        resolutions: list[Resolution | str] | None = None,
        exchange: str = "",
        symbol: str = "",
        product_name: str = "",
        config: DataConfig | None = None,
        as_features: bool = False,
        **product_specs: Any,
    ) -> list[MarketData]:
        """Adds market data to the component.

        Args:
            venue: trading venue, e.g. 'ibkr', 'bybit'.
            exchange: useful for TradFi brokers (e.g. IB), to specify the exchange (e.g. 'NASDAQ')
            symbol: useful for TradFi brokers (e.g. IB), to specify the symbol (e.g. 'AAPL')
            product: product basis, defined as {base_asset}_{quote_asset}_{product_type}, e.g. BTC_USDT_PERP
            resolutions: data resolutions in use, e.g. "1t" for tick data, "1q" for quote data
                if None, resolution of the component will be used
            product_name: A user-defined identifier for the product.
                If not provided, the default product symbol (e.g. 'BTC_USDT_PERP', 'TSLA241213C00075000') will be used.
                This is useful when you need to distinguish between similar instruments, such as options
                with different strike prices and expiration dates. Instead of using long identifiers like
                'BTC_USDT_OPTION_100000_20250620' and 'BTC_USDT_OPTION_101000_20250920', you can assign
                simpler names like 'BTC_OPT1' and 'BTC_OPT2'.
                Note:
                    It is the user's responsibility to manage and maintain these custom product names.
            config: Data Configuration
            as_features: Whether data columns (e.g. OHLCV in market data) are treated as "features"
                in the component's trading_df.
            product_specs: product specifications, e.g. expiration, strike_price etc.

        Returns:
            The market data objects added for the requested resolutions.
        """
        product: BaseProduct = self._add_product(
            venue=venue,
            basis=product,
            exchange=exchange,
            symbol=symbol,
            name=product_name,
            **product_specs,
        )
        datas: list[MarketData] = self.market_data_store.add_data(
            product=product,
            resolutions=resolutions,
            config=config,
            as_features=as_features,
        )
        return datas

    add_market_data = add_data

    def _add_component(
        self,
        component: ComponentT | ActorProxy[ComponentT],
        resolution: str = "",
        name: str = "",
        storage_config: StorageConfig | None = None,
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> ComponentT | ActorProxy[ComponentT] | None:
        """Adds a model component to the current component.
        A model component is a model or feature.
        Args:
            storage_config: per-component override for where this component's artifacts
                are persisted. Falls back to this parent's storage config (which itself
                came from the engine default or its own parent) when None.
            ray_kwargs: kwargs for ray actor, e.g. num_cpus
            ray_actor_options:
                Options for Ray actor.
                will be passed to ray actor like this: Actor.options(**ray_options).remote(**ray_kwargs)
        """
        Component = component.__class__
        ComponentName = Component.__name__
        if component.is_model():
            from pfund.components.models.model_base import BaseModel

            components = self.models
            BaseClass = BaseModel
        elif component.is_feature():
            from pfund.components.features.feature_base import BaseFeature

            components = self.features
            BaseClass = BaseFeature
        else:
            raise ValueError(
                f"{component.component_type} '{ComponentName}' is not a model or feature"
            )

        if not isinstance(component, ActorProxy):
            component_type = component.component_type
            assert isinstance(component, BaseClass), (
                f"{component_type} '{ComponentName}' is not an instance of {BaseClass.__name__}. Please create your {component_type} using 'class {ComponentName}(pf.{component_type.capitalize()})'"
            )
            component_name = name or component.name
            if component_name in components:
                raise ValueError(f"{component_name} already exists")

            # enforce GLOBAL name uniqueness (across other Ray actors too), not just this parent's dict
            if ray_kwargs:
                # upgrade BEFORE the actor is created, so the shared-registry context is what ships into it
                from pfund.engines.component_registry import to_registry_proxy

                self.context.component_registry = to_registry_proxy(
                    self.context.component_registry
                )
            # claim before spawning the actor, so a duplicate name aborts without leaking a live actor
            self.context.component_registry.claim(component_name)

            if ray_kwargs:
                component = ActorProxy(
                    component,
                    name=component_name,
                    resolution=resolution or self.resolution,
                    component_type=component_type,
                    engine_context=self.context,
                    ray_actor_options=ray_actor_options,
                    **ray_kwargs,
                )

            component._hydrate(
                name=component_name,
                run_mode=RunMode.REMOTE if ray_kwargs else RunMode.LOCAL,
                resolution=resolution or self.resolution,
                engine_context=self.context,
                storage_config=storage_config or self.store.storage_config,
            )

        components[component.name] = component
        self.logger.debug(f"{self.name} added {component.name}")

        # NOTE: returns None when adding a local component (not ActorProxy) to a remote component to avoid returning a serialized (copied) object
        if self.is_remote() and not isinstance(component, ActorProxy):
            return None
        return component

    def add_model(
        self,
        model: ModelT | ActorProxy[ModelT] | UnderlyingModel,
        resolution: str = "",
        name: str = "",
        storage_config: StorageConfig | None = None,
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> ModelT | ActorProxy[ModelT] | None:
        from pfund.components.models.wrap import wrap_model

        return self._add_component(
            component=wrap_model(model),
            resolution=resolution,
            name=name,
            storage_config=storage_config,
            ray_actor_options=ray_actor_options,
            **ray_kwargs,
        )

    def add_feature(
        self,
        feature: FeatureT | ActorProxy[FeatureT],
        resolution: str = "",
        name: str = "",
        storage_config: StorageConfig | None = None,
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> FeatureT | ActorProxy[FeatureT] | None:
        return self._add_component(
            component=feature,
            resolution=resolution,
            name=name,
            storage_config=storage_config,
            ray_actor_options=ray_actor_options,
            **ray_kwargs,
        )

    def set_signal_cols(self, signal_cols: list[str]):
        self._signal_cols = signal_cols

    def set_pivot_cols(self, pivot_cols: list[str]):
        self.store.set_pivot_cols(pivot_cols)

    def is_ready(self, data: BaseData) -> bool:
        if data.category == DataCategory.MARKET_DATA:
            df = self.get_df(kind="data", data_category=data.category, to_native=False)
            if df is None or len(df) < self.config["warmup_period"]:
                return False
            if self._df_form == "long":
                # long form: ready when this product's bar is closed
                return data.is_closed()
            elif self._df_form == "wide":
                # wide form: ready when all products' bars are closed
                return all(
                    cast(
                        "BarData", self.get_data(product.name, self.resolution)
                    ).is_closed()
                    for product in self.products.values()
                )
        # EXTEND:
        else:
            raise NotImplementedError(
                f"is_ready() is not implemented for {data.category=}"
            )

    def merge_data_dfs(
        self, data_dfs: dict[DataCategory, IntoDataFrame]
    ) -> IntoDataFrame:
        """Creates data_df by merging data_dfs per data category in long form
        Args:
            data_dfs: dataframes per data category in long form
        """
        dfs: list[nw.DataFrame[Any]] = []
        common_key_cols: set[str] | None = None
        for category, df in data_dfs.items():
            data_store = self.databoy.get_data_store(category)
            nw_df = nw.from_native(df)
            # strip provenance metadata: the merged data_df feeds featurize, where metadata is not a feature
            metadata_to_drop = [
                col for col in data_store.METADATA_COLS if col in nw_df.columns
            ]
            if metadata_to_drop:
                nw_df = nw_df.drop(metadata_to_drop)
            if self._df_form == "wide":
                # REVIEW: this requires dynamic pivoting for EACH call
                nw_df = data_store.pivot_df(nw_df)
            dfs.append(nw_df)
            # KEY_COLS that actually survive on this df (pivot_df folds PIVOT_COLS into column names)
            df_key_cols = set(
                col for col in data_store.KEY_COLS if col in nw_df.columns
            )
            if common_key_cols is None:
                common_key_cols = df_key_cols
            else:
                common_key_cols &= df_key_cols

        data_df = dfs[0]
        if len(dfs) == 1:
            return data_df.to_native()

        if not common_key_cols:
            raise ValueError(
                f"No common key columns found in data_dfs in {self._df_form} form for {self.name}. "
                + "Please define your own merge_data_dfs() to handle merging data_dfs manually."
            )

        for df in dfs[1:]:
            data_df = data_df.join(df, on=list(common_key_cols), how="full")
            data_df = data_df.with_columns(
                nw.coalesce(key, f"{key}_right").alias(key) for key in common_key_cols
            ).drop(*(f"{key}_right" for key in common_key_cols))
        return data_df.sort(common_key_cols).to_native()

    def merge_signals_dfs(
        self, signals_dfs: dict[ComponentName, IntoDataFrame]
    ) -> IntoDataFrame:
        """Creates features_df by merging signals_dfs (signals from other components).

        For the (self._df_form='long', component_df_form='wide') case the wide
        component signals cannot be melted back to long, so they are broadcast onto
        a long-form KEY_COLS scaffold built purely from this component's config
        (see _build_signals_key_grid) — no data_df is consulted, so no data value
        columns can leak into the returned features_df.

        Args:
            signals_dfs: dict of signals_df per component name
        Returns:
            features_df: features_df in {self._df_form} form
        """
        dfs: list[nw.DataFrame[Any]] = []
        # (self._df_form='long', component_df_form='wide') case:
        # cannot melt back to long, so merge them in afterwards by broadcasting on the index col only
        special_dfs: list[nw.DataFrame[Any]] = []
        common_index_col: str | None = None
        common_key_cols: set[str] | None = None
        for component_name, df in signals_dfs.items():
            component = self.get_component(component_name)
            component_df_form = component.get_df_form()
            trading_store = component.get_trading_store()
            if common_index_col is None:
                common_index_col = trading_store.INDEX_COL
            else:
                if common_index_col != trading_store.INDEX_COL:
                    raise ValueError(
                        f"Unhandled case: index column {common_index_col} is not the same as {trading_store.INDEX_COL} for {component_name}"
                    )
            # special case, cannot unpivot 'wide' to 'long'
            if self._df_form == "long" and component_df_form == "wide":
                special_dfs.append(nw.from_native(df))
                continue
            elif self._df_form == "wide" and component_df_form == "long":
                # REVIEW: this requires dynamic pivoting for EACH call
                nw_df = trading_store.pivot_df(nw.from_native(df))
            else:
                nw_df = nw.from_native(df)
            dfs.append(nw_df)
            # KEY_COLS that actually survive on this df (pivot_df folds PIVOT_COLS into column names)
            df_key_cols = set(
                col for col in trading_store.KEY_COLS if col in nw_df.columns
            )
            if common_key_cols is None:
                common_key_cols = df_key_cols
            else:
                common_key_cols &= df_key_cols

        if not dfs and not special_dfs:
            raise ValueError(
                f"Invalid input: {signals_dfs=} in merge_signals_dfs for {self.name}: no dataframe to merge"
            )

        # only special_dfs exist (e.g. self is strategy in long form, components are models in wide form):
        # build a long-form KEY_COLS scaffold from config + the dates present in special_dfs,
        # then broadcast the wide signals onto it below. data_df is NOT consulted.
        if not dfs:
            features_df = self._build_signals_key_grid(special_dfs, common_index_col)
            common_key_cols = set(
                col for col in self.store.KEY_COLS if col in features_df.columns
            )
        else:
            features_df = dfs[0]
            if len(dfs) > 1:
                if not common_key_cols:
                    raise ValueError(
                        f"No common key columns found in signals_dfs in {self._df_form} form for {self.name}. "
                        + "Please define your own merge_signals_dfs() to handle merging signals_dfs manually."
                    )
                for df in dfs[1:]:
                    features_df = features_df.join(
                        df, on=list(common_key_cols), how="full"
                    )
                    for key in common_key_cols:
                        right_key = f"{key}_right"
                        features_df = features_df.with_columns(
                            nw.coalesce(key, right_key).alias(key)
                        ).drop(right_key)

        # broadcast (self._df_form='long', component_df_form='wide') specials on index col only
        for df in special_dfs:
            features_df = features_df.join(df, on=common_index_col, how="full")
            right_key = f"{common_index_col}_right"
            features_df = features_df.with_columns(
                nw.coalesce(common_index_col, right_key).alias(common_index_col)
            ).drop(right_key)

        sort_keys = list(common_key_cols) if common_key_cols else [common_index_col]
        return features_df.sort(sort_keys).to_native()

    def _get_pivot_col_values(self, pivot_col: str) -> list[str]:
        """Returns the config-derived value list for one of the store's PIVOT_COLS,
        used to build the long-form key grid that wide component signals broadcast onto.

        The returned strings MUST match the values stamped into the long-form df for
        that column (product.name for 'product', str(resolution) for 'resolution'), so
        the resulting grid aligns with data_df on KEY_COLS.

        Args:
            pivot_col: a column name from self.store.PIVOT_COLS
        """
        # explicit name -> config-value-list dispatch: there is no generic getattr(self, pivot_col)
        # ('product' != attr 'products', values need .name; 'resolution' is a single object needing str())
        pivot_col_value_sources = {
            "product": lambda: [product.name for product in self.products.values()],
            "resolution": lambda: [str(self.resolution)],
        }
        if pivot_col not in pivot_col_value_sources:
            raise ValueError(
                f"Cannot build signals key grid for {self.name}: store PIVOT_COL "
                + f"{pivot_col!r} has no config value source "
                + f"(supported: {sorted(pivot_col_value_sources)}). "
                + "Please define your own merge_signals_dfs() to handle merging signals_dfs manually."
            )
        return pivot_col_value_sources[pivot_col]()

    def _build_signals_key_grid(
        self, special_dfs: list[nw.DataFrame[Any]], index_col: str
    ) -> nw.DataFrame[Any]:
        """Builds the long-form KEY_COLS scaffold that wide-form component signals
        broadcast onto, WITHOUT consulting data_df.

        The index (date) values come from the wide special_dfs; each PIVOT_COL's
        values come from config via _get_pivot_col_values. self.store.PIVOT_COLS is
        iterated in order and cross-joined onto the date column, so the grid columns
        are laid out as KEY_COLS = [INDEX_COL] + PIVOT_COLS.

        Args:
            special_dfs: wide-form signals dfs to be broadcast (source of index values)
            index_col: the shared index column name (e.g. 'date')
        """
        # keep every frame entering the cross joins on one backend (matching is required)
        backend = nw.get_native_namespace(special_dfs[0])
        # union of index (date) values across the wide special_dfs
        grid = nw.concat(
            [df.select(index_col) for df in special_dfs], how="vertical"
        ).unique()
        # cross-join each PIVOT_COL's config values in PIVOT_COLS order => [index_col] + PIVOT_COLS
        for pivot_col in self.store.PIVOT_COLS:
            values = self._get_pivot_col_values(pivot_col)
            values_df = nw.DataFrame.from_dict(
                data={pivot_col: values}, backend=backend
            )
            grid = grid.join(values_df, how="cross")
        # enforce the exact KEY_COLS layout [INDEX_COL] + PIVOT_COLS
        return grid.select([index_col, *self.store.PIVOT_COLS])

    def _reload_markets(self):
        """venues (at engine level) might have refetched the latest markets, reload markets from markets.yml"""
        # NOTE: must use pfund_config from context, config from get_config() could be different from what user has set in Ray Actor
        pfund_config = self.context.pfund_config
        for product in self.products.values():
            if product.venue is None:
                continue
            VenueClass = product.venue.venue_class
            file_path = VenueClass._create_markets_yml_file_path(pfund_config.data_path)
            product._load_market(file_path)

    def _materialize(self):
        for data_store in self.data_stores.values():
            data_store.materialize()
        self.store.materialize()

    def _gather(self) -> None:
        """Sets up everything before start"""
        # NOTE: use is_gathered to avoid a component being gathered multiple times when it's a shared component
        if not self._is_gathered:
            self.add_datas()
            self.add_models()
            self.add_features()
            self._check_input_sources()
            self._check_config()
            for component in self.components:
                component._gather()
            self._reload_markets()
            self._materialize()
            _ = self.store._save_source_artifact()
            self._is_gathered = True
            self.logger.info(f"'{self.name}' has gathered")
        else:
            self.logger.info(f"'{self.name}' has already gathered")

    def start(self) -> None:
        if not self.is_running():
            # set the ZMQPubHandler's receiver ready to flush the buffered log messages
            if self.is_remote(direct_only=False):
                self.logger.handlers[0].set_receiver_ready()
            for component in self.components:
                component.start()
            self._is_running = True
            self.on_start()
            self.databoy.start()
            self.logger.info(f"'{self.name}' has started")
        else:
            self.logger.info(f"'{self.name}' has already started")

    def stop(self, reason: str = "") -> None:
        """Stops the component, keep the internal states"""
        if self.is_running():
            for component in self.components:
                component.stop()
            self._is_running = False
            self.on_stop()
            self.databoy.stop()
            self.logger.info(f"'{self.name}' has stopped, ({reason=})")
        else:
            self.logger.info(f"'{self.name}' has already stopped")

    """
    ************************************************
    Override Methods
    Override these methods in your subclass to implement your custom behavior.
    ************************************************
    """

    def on_quote(self, data: QuoteData):
        pass

    def on_tick(self, data: TickData):
        pass

    def on_bar(self, data: BarData):
        pass

    def add_datas(self):
        pass

    def add_models(self):
        pass

    def add_features(self):
        pass

    def on_start(self):
        pass

    def on_stop(self):
        pass

    """
    ************************************************
    Sugar Functions
    ************************************************
    """

    def get_orderbook(self, product: ProductName) -> MarketData | None:
        return self.get_data(product, Resolution("1q"))

    def get_tradebook(self, product: ProductName) -> MarketData | None:
        return self.get_data(product, Resolution("1t"))
