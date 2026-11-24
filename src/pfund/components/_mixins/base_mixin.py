from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self, cast

if TYPE_CHECKING:
    from pfund_blocks.datas import BarData, BaseData, MarketData, QuoteData, TickData
    from pfund_blocks.entities import BaseVenue
    from pfund_blocks.entities.products.product_base import BaseProduct
    from pfund_blocks.typing import ProductName, TradingVenue
    from pfund_kit.logging.loggers import ColoredLogger

    from pfund.components.features.feature_base import BaseFeature
    from pfund.components.models.model_base import BaseModel
    from pfund.data_config import DataConfig
    from pfund.engines.contexts.base_engine_context import BaseEngineContext
    from pfund.engines.settings.trade_engine_settings import TradeEngineSettings
    from pfund.stores.base_data_store import BaseDataStore
    from pfund.stores.market_data_store import MarketDataStore
    from pfund.typing import Component, ComponentName, ComponentProxy, Signals

import copy
import datetime
import logging
from pathlib import Path

from pfund_blocks.enums import DataCategory
from pfund_blocks.enums.env import Environment
from pfund_blocks.resolution import Resolution
from pfund_kit.style import RichColor, TextStyle

from pfund.components.actor_proxy import ActorProxy
from pfund.databoy import DataBoy
from pfund.enums import ComponentType, RunMode
from pfund.stores.trading_store import TradingStore
from pfund.utils.decorators import ray_method


class ComponentBaseMixin:
    # NOTE: not ClassVar, class-level values are only read-only defaults,
    # __mixin_post_init__ gives each instance its own merged copy (see _merge_class_dicts),
    # and load_config()/load_params() only update that copy
    config: dict[str, Any] = {  # noqa: RUF012
        "max_rows": None,  # None means no limit, component will keep all data in memory
        "warmup_period": None,  # None means defaulting to lookback_period (or 0 if lookback_period is None)
        "lookback_period": None,  # None means using the whole dataset
    }
    params: dict[str, Any] = {}  # noqa: RUF012

    def __mixin_post_init__(self, component_type: ComponentType, *args: Any, **kwargs: Any):
        self.__pfund_args__ = args
        self.__pfund_kwargs__ = kwargs

        self.config = self._merge_class_dicts("config")
        self.params = self._merge_class_dicts("params")

        self._component_type = component_type
        self._name = self._get_default_name()
        self._run_mode: RunMode | None = None
        self._df_form: Literal["wide", "long"] | None = None
        self._resolution: Resolution | None = None
        self._context: BaseEngineContext[Any] | None = None
        self._source_artifact_path: Path | None = None

        # set in _setup_logging() during hydration, see the NOTE there
        self._logger: ColoredLogger | None = None
        self._products: dict[ProductName, BaseProduct] = {}
        # the mixin only ever runs as part of a strategy, model or feature
        component = cast("Component", self)
        self._databoy = DataBoy(component)
        self._store = TradingStore(component)

        self._signal_cols: list[str] = []
        # Last successfully computed signals; may not belong to the current bar.
        self._latest_signals: Signals = {}

        self.models: dict[str, BaseModel | ActorProxy[BaseModel]] = {}
        self.features: dict[str, BaseFeature | ActorProxy[BaseFeature]] = {}

        self._is_running = False
        self._is_gathered = False

    @classmethod
    def _merge_class_dicts(cls, attr: Literal["config", "params"]) -> dict[str, Any]:
        """Merges the class-level dicts along the MRO, base classes first,
        so a subclass only needs to declare the keys it overrides.
        Deep-copied so that instances never share (nested) values with the class defaults."""
        merged: dict[str, Any] = {}
        for klass in reversed(cls.__mro__):
            merged.update(vars(klass).get(attr, {}))
        return copy.deepcopy(merged)

    def load_config(
        self, config: dict[str, Any] | None = None, file_path: str | Path = ""
    ) -> Self:
        """Overrides this component's config with a dict or a TOML file.

        The given keys are merged over the current config, keys not given keep their values.
        Only this instance is affected, the class-level defaults are untouched.

        Args:
            config: Config dict to merge into the current config.
            file_path: Path to a TOML file to load the config from.

        Returns:
            The component itself, so the call can be chained, e.g. `MyModel().load_config(...)`.

        Raises:
            ValueError: If both or neither of config and file_path are given, if the file is not a
                TOML file or does not exist, if a key is not a valid config key,
                or if the component has already gathered.
        """
        new_config = self._load_dict_or_toml("config", config, file_path)
        if unknown_keys := new_config.keys() - ComponentBaseMixin.config.keys():
            raise ValueError(
                f"{self} config has unknown keys {sorted(unknown_keys)}, "
                + f"valid keys are {sorted(ComponentBaseMixin.config)}"
            )
        # deepcopy so nested values are not shared with the class defaults or the caller's dict
        self.config = copy.deepcopy({**self.config, **new_config})
        return self

    def load_params(
        self, params: dict[str, Any] | None = None, file_path: str | Path = ""
    ) -> Self:
        """Overrides this component's params with a dict or a TOML file.

        The given keys are merged over the current params, keys not given keep their values.
        Only this instance is affected, the class-level defaults are untouched.

        Args:
            params: Params dict to merge into the current params.
            file_path: Path to a TOML file to load the params from.

        Returns:
            The component itself, so the call can be chained, e.g. `MyModel().load_params(...)`.

        Raises:
            ValueError: If both or neither of params and file_path are given, if the file is not a
                TOML file or does not exist, or if the component has already gathered.
        """
        new_params = self._load_dict_or_toml("params", params, file_path)
        # deepcopy so nested values are not shared with the class defaults or the caller's dict
        self.params = copy.deepcopy({**self.params, **new_params})
        return self

    @ray_method
    def _load_via_proxy(
        self, kind: Literal["config", "params"], values: dict[str, Any]
    ) -> None:
        """Runs load_config()/load_params() inside this (remote) component's actor.
        Called by ActorProxy.load_config/load_params. Returns None instead of self,
        since returning the component through Ray would copy the whole component back to the caller.
        """
        if kind == "config":
            self.load_config(values)
        else:
            self.load_params(values)

    @ray_method
    def _set_config_and_params(
        self, config: dict[str, Any], params: dict[str, Any]
    ) -> None:
        """Sets the config and params of a remote component to the local instance's.
        Called by ActorProxy right after creating the actor, since the actor rebuilds the component
        from its constructor args only and would otherwise fall back to the class-level defaults.
        No merging or validation, the values were already merged/validated on the local instance."""
        self.config = config
        self.params = params

    def _load_dict_or_toml(
        self,
        kind: Literal["config", "params"],
        values: dict[str, Any] | None,
        file_path: str | Path,
    ) -> dict[str, Any]:
        from pfund_kit.utils import toml

        # config and params are part of component_id and are validated in _gather(),
        # changing them afterwards would leave them unchecked and mismatch the saved artifacts
        if self._is_gathered:
            raise ValueError(
                f"{self} has already gathered, {kind} must be loaded before running"
            )
        if values is not None and file_path:
            raise ValueError(
                f"{kind} and file_path cannot be provided at the same time"
            )
        if values is not None:
            return values
        if not file_path:
            raise ValueError(f"either {kind} or file_path must be provided")
        path = Path(file_path)
        if path.suffix != ".toml":
            raise ValueError(f"{kind} file must be a TOML file (.toml), got {path}")
        if not path.is_file():
            raise ValueError(f"{kind} file not found: {path}")
        loaded = toml.load(path)
        assert loaded is not None, f"failed to load {kind} from {path}"
        return loaded

    def _check_config(self):
        """Validates config and writes normalized values back into self.config,
        so runtime readers (e.g. is_ready() comparing row count against warmup_period)
        never see None."""
        for key in ("max_rows", "warmup_period", "lookback_period"):
            if key not in self.config:
                raise ValueError(f"{key} is not set in {self} config")
        max_rows = self.config["max_rows"]
        warmup_period = self.config["warmup_period"]
        lookback_period = self.config["lookback_period"]
        if max_rows is None:
            self.logger.warning(
                f"'max_rows' is None. {self} data will be UNBOUNDED",
                style=TextStyle.BOLD + RichColor.YELLOW,
            )
        if lookback_period is None:
            self.logger.warning(
                f"'lookback_period' is None. {self} will use the WHOLE DATASET to compute signals",
                style=TextStyle.BOLD + RichColor.YELLOW,
            )
        else:
            if lookback_period < 1:
                raise ValueError(
                    f"{self} {lookback_period=} is less than 1, please set lookback_period >= 1"
                )
        if warmup_period is None:
            # a lookback window of N rows needs at least N rows of history, so an unset
            # warmup_period defaults to lookback_period instead of tripping the
            # lookback <= warmup check below with a warmup the user never set
            warmup_period = lookback_period or 0
            if warmup_period:
                self.logger.warning(
                    f"'warmup_period' is None. defaulting to {lookback_period=} for {self}",
                    style=TextStyle.BOLD + RichColor.YELLOW,
                )
            else:
                self.logger.warning(
                    f"'warmup_period' is None. {self} will be ready to compute signals IMMEDIATELY",
                    style=TextStyle.BOLD + RichColor.YELLOW,
                )
            self.config["warmup_period"] = warmup_period
        if warmup_period < 0:
            raise ValueError(
                f"{self} {warmup_period=} is less than 0, please set warmup_period >= 0"
            )
        if lookback_period is not None and lookback_period > warmup_period:
            raise ValueError(
                f"{self} config: {lookback_period=} is greater than {warmup_period=}, please set lookback_period <= warmup_period"
            )
        # the trading store keeps at most max_rows rows, so a longer lookback window could never be filled
        if (
            max_rows is not None
            and lookback_period is not None
            and lookback_period > max_rows
        ):
            raise ValueError(
                f"{self} config: {lookback_period=} is greater than {max_rows=}, please set lookback_period <= max_rows"
            )

    @property
    def context(self) -> BaseEngineContext[TradeEngineSettings]:
        assert self._context is not None, "context is not set"
        # NOTE: backtesting narrows via BacktestMixin._backtest_context instead (see backtest_mixin.py)
        return cast("BaseEngineContext[TradeEngineSettings]", self._context)

    @property
    def logger(self) -> ColoredLogger:
        assert self._logger is not None, (
            f"{self} has no logger yet, it is set when the component is added to an engine"
        )
        return self._logger

    @property
    def settings(self) -> TradeEngineSettings:
        return self.context.settings

    @property
    def env(self) -> Environment:
        assert self._context is not None, "context is not set"
        return self._context.env

    @property
    def name(self) -> ComponentName:
        return self._name

    def __str__(self) -> str:
        """Display name, appends the component type unless the name already contains it, e.g. 'RSI (feature)'.
        For messages only; use `name` for identity (keys, signal columns, paths)."""
        if self.component_type.lower() in self.name.lower():
            return self.name
        return f"{self.name} ({self.component_type})"

    @property
    def run_mode(self) -> RunMode:
        assert self._run_mode is not None, "run_mode is not set"
        return self._run_mode

    @property
    def resolution(self) -> Resolution:
        assert self._resolution is not None, "resolution is not set"
        return self._resolution

    @property
    def df_form(self) -> Literal["wide", "long"]:
        assert self._df_form is not None, "df_form is not set"
        return self._df_form

    @property
    def component_type(self) -> ComponentType:
        return self._component_type

    @property
    def products(self) -> dict[ProductName, BaseProduct]:
        return self._products

    @property
    def store(self) -> TradingStore:
        return self._store

    def _setup_scheduler(self):
        self._databoy._setup_scheduler()
        for component in self.components:
            component._setup_scheduler()

    def _setup_messaging(self):
        self._databoy._setup_messaging()
        for component in self.components:
            component._setup_messaging()
        self._databoy._subscribe()

    def _setup_logging(self):
        """Sets up the component's logger, called in _hydrate().
        For a component running in a remote (Ray) process, also configures logging in that process
        and uses zmq's PUSHHandler to send logs to the engine."""
        # NOTE: the 'pfund' logger must be fetched here, after logging is set up, NOT at construction.
        # A logger's class is fixed when logging.getLogger() first creates it, setLoggerClass(ColoredLogger)
        # in setup_logging() only affects loggers created after it. Fetching 'pfund' earlier (e.g. a component
        # created before the engine, or constructed inside a Ray actor) would create it as a plain
        # logging.Logger for the whole process, and logger calls with `style=` would raise TypeError.
        # So nothing in pfund should call logging.getLogger("pfund") before the engine/remote logging is set up.
        if not self.is_remote(direct_only=False):
            # the engine has already set up logging, see BaseEngine.__init__()
            self._logger = cast("ColoredLogger", logging.getLogger("pfund"))
            return

        from pfeed.streaming.zeromq import ZeroMQ
        from pfund_kit.logging.configurator import LoggingDictConfigurator
        from pfund_kit.utils import get_free_port

        from pfund.utils.zmq_pub_handler import ZMQPubHandler

        # configure logging based on pfund's logging config, e.g. log_level, log_file, log_format, etc.
        logging_configurator = LoggingDictConfigurator.create(
            log_path=self.context.config.log_path / self.env,
            logging_config=self.context.logging_config,
            lazy=True,
            use_colored_logger=True,
        )
        logging_configurator.configure()
        self._logger = cast("ColoredLogger", logging.getLogger("pfund"))

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

    def set_signal_cols(self, signal_cols: list[str]) -> None:
        self._signal_cols = signal_cols

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
        self._name = name

    @ray_method
    def get_df_form(self) -> Literal["wide", "long"]:
        return self.df_form

    @ray_method
    def get_trading_store(self) -> TradingStore:
        return self.store

    def get_data(
        self, product: ProductName, resolution: Resolution | str
    ) -> MarketData | None:
        return self.market_data_store.get_data(product, resolution)

    def get_datas(self) -> list[BaseData]:
        datas = []
        for data_store in self.data_stores.values():
            datas.extend(data_store.get_datas())
        return datas

    @ray_method
    def get_data_configs(self) -> dict[BaseData, DataConfig]:
        """Returns every data of the component with its config.
        A method (not the stores' get_config()) so that the engine can also call it on a remote component."""
        return {
            data: data_store.get_config(data)
            for data_store in self.data_stores.values()
            for data in data_store.get_datas()
        }

    def _get_venue_class(self, venue: TradingVenue) -> type[BaseVenue]:
        if self.env.is_simulated():
            from pfund.simulated import SimulatedVenue

            # the per-venue subclass, which carries the real venue's metadata and adapter
            return SimulatedVenue.for_venue(venue)
        else:
            from pfund.registry import get_venue

            return get_venue(venue)

    def _get_streaming_resolutions(self, product: BaseProduct):
        Venue = self._get_venue_class(product.source)
        return Venue.get_streaming_resolutions(product)

    @property
    def components(self) -> list[Component | ComponentProxy]:
        return self.get_components()

    def get_component(self, name: ComponentName) -> Component | ComponentProxy | None:
        return self.models.get(name, None) or self.features.get(name, None)

    def get_components(self) -> list[Component | ComponentProxy]:
        return [
            *self.models.values(),
            *self.features.values(),
        ]

    @ray_method
    def get_latest_signals(self) -> Signals | None:
        return self._latest_signals

    @ray_method
    def get_signal_cols(self) -> list[str]:
        return self._signal_cols

    def _get_default_name(self):
        return self.__class__.__name__

    def _get_descendant_names(self) -> list[str]:
        """Names of every component below this one, at any depth.

        Works through `ActorProxy` too: the call is forwarded to the actor,
        so children added inside a remote component are included.
        """
        return [
            name
            for component in self.get_components()
            for name in (component.name, *component._get_descendant_names())
        ]

    def _get_default_signal_cols(self, num_cols: int) -> list[str]:
        # FIXME: the default signal column is the bare component name, so it can clash with a data column,
        # e.g. a feature named "close" vs the bar's "close" column, probably add the component type suffix
        # (e.g. "close_feature") only when it's a duplicate. same for sub-strategies in BaseStrategy._get_default_signal_cols
        if num_cols == 1:
            columns = [self.name]
        else:
            columns = [f"{self.name}-{i}" for i in range(num_cols)]
        return columns

    def _assert_unique_child_signal_cols(self) -> None:
        child_signal_cols = [
            signal_col
            for component in self.get_components()
            for signal_col in component.get_signal_cols()
        ]
        assert len(child_signal_cols) == len(set(child_signal_cols)), (
            f"'{self.name}' child components have duplicate signal columns: "
            f"{child_signal_cols}"
        )

    @property
    def market_data_store(self) -> MarketDataStore:
        return cast(
            "MarketDataStore", self._databoy.get_data_store(DataCategory.MARKET_DATA)
        )

    @property
    def data_stores(self) -> dict[DataCategory, BaseDataStore[Any, Any]]:
        return self._databoy._data_stores

    def get_data_store(self, category: DataCategory | str) -> BaseDataStore[Any, Any]:
        return self._databoy.get_data_store(category)

    @staticmethod
    def dt(ts: float, tz: datetime.tzinfo = datetime.UTC) -> datetime.datetime:
        from pfund_kit.utils.temporal import convert_ts_to_dt

        return convert_ts_to_dt(ts, tz=tz)

    @staticmethod
    def now(tz: datetime.tzinfo = datetime.UTC) -> datetime.datetime:
        return datetime.datetime.now(tz=tz)

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

    def is_running(self) -> bool:
        return self._is_running

    def is_strategy(self) -> bool:
        return self.component_type == ComponentType.strategy

    def is_model(self) -> bool:
        return self.component_type == ComponentType.model

    def is_feature(self) -> bool:
        return self.component_type == ComponentType.feature

    def get_orderbook(self, product: ProductName) -> MarketData | None:
        return self.get_data(product, Resolution("1q"))

    def get_tradebook(self, product: ProductName) -> MarketData | None:
        return self.get_data(product, Resolution("1t"))

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
