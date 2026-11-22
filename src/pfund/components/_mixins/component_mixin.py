from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame
    from pfund_blocks.datas import BarData, BaseData, MarketData
    from pfund_blocks.entities.products.product_base import BaseProduct
    from pfund_blocks.typing import TradingVenue

    from pfund.components.features.feature_base import BaseFeature
    from pfund.components.models.model_base import BaseModel, UnderlyingModel
    from pfund.engines.contexts.base_engine_context import BaseEngineContext
    from pfund.typing import (
        Component,
        ComponentName,
        Signals,
    )

import hashlib
import json
from functools import cached_property
from pathlib import Path

from pfeed.enums import DataCategory
from pfund_blocks.resolution import Resolution
from pfund_kit.style import RichColor, TextStyle, cprint
from pfund_kit.utils import toml

from pfund.components._mixins.base_mixin import ComponentBaseMixin
from pfund.components._mixins.dataframe_mixin import DataFrameMixin
from pfund.components.actor_proxy import ActorProxy
from pfund.data_config import DataConfig
from pfund.enums import RunMode
from pfund.utils.decorators import ray_method


class ComponentMixin(DataFrameMixin, ComponentBaseMixin):
    config: ClassVar[dict[str, Any]] = {
        "max_rows": None,  # None means no limit, component will keep all data in memory
        "warmup_period": None,  # None means no warmup period, component will start right away
        "lookback_period": None,  # None means using the whole dataset
    }
    params: ClassVar[dict[str, Any]] = {}

    def _hydrate(
        self,
        name: ComponentName,
        run_mode: RunMode,
        resolution: Resolution | str,
        engine_context: BaseEngineContext[Any],
        df_form: Literal["wide", "long"],
    ):
        """
        Hydrates the component with necessary attributes after initialization.

        Args:
            name (ComponentName): The name to assign to this component.
            run_mode (RunMode): The mode in which the component will run (e.g., local or remote).
            resolution (Resolution | str): The data resolution used by this component.
            engine_context (BaseEngineContext): The engine context associated with this component. It is None if the component is running in a remote process.
            df_form: DataFrame layout used by this component.
        """
        self._context = engine_context
        self._run_mode = run_mode
        self._df_form = df_form
        self._set_name(name)
        self._set_resolution(resolution)
        self.set_logger(self.logger)
        self.store.setup_lakehouse()

    def _update_data_df(self, data: BaseData) -> None:
        data_store = self.get_data_store(data.category)
        data_store.update_df(data)

    def _get_components_signals(self, data: BarData) -> dict[ComponentName, Signals]:
        return self._databoy._wait_for_components_signals(data)

    def step(self, data: BarData):
        """Forwards the given bar data to the child components, wait for their signals,
        compute the component's signals, and then updates its trading store.

        This function could be called manually.
        Manual step forward is triggered by the user, it could be useful in cases such as:
            e.g. user wants to forward 1m data a bit earlier (bar is incomplete)
                before the market close.

        Args:
            data (BarData): The bar data to step forward.
        """
        try:
            # NOTE: user might have manually called forward() before databoy does
            has_forwarded = self.store.has_updated(data)
            if has_forwarded:
                return
            if not self.is_ready(data=data):
                return
            signals: Signals = {}
            signals_per_component = self._get_components_signals(data)
            for component_signals in signals_per_component.values():
                self.store.update_df(component_signals, data=data)
            X = self.get_df(
                kind="features",
                window_size=self.config["lookback_period"],
                to_native=True,
            )
            signals = self.signalize(X)
            self._latest_signals = signals
            self.store.update_df(signals, data=data)
            self._publish_signals(signals, data)
        except Exception:
            self.logger.exception("Error forwarding data:")

    def _publish_signals(self, signals: Signals, data: BarData):
        if self._databoy.is_using_zmq():
            self._databoy._publish_signals(signals, data)

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
        """Check that the component can establish its spine and features.

        Spine invariant:
        - If the component has market data, that market data defines its spine.
        - Otherwise, a wide component needs a same-resolution child, while a
          long component needs a same-resolution long-form child so the child
          can provide ``(date, product, resolution)`` keys.
        - Same-resolution children collectively define the spine.
        - Coarser children may contribute values but cannot define the missing
          finer bars.
        """
        has_market_data = bool(self.market_data_store.get_datas())
        if not has_market_data:
            if self.is_strategy():
                raise ValueError(
                    f"{self.name} has no market data, did you forget to call add_data() / add_datas()?"
                )
            elif not self.components:
                raise ValueError(
                    f"{self.name} has no input source, did you forget to call add_data() / add_datas(), "
                    + "or add a child component via add_model() / add_feature()?"
                )
            elif not any(
                child.resolution.is_strict_equal(self.resolution)
                and (self.df_form == "wide" or child.df_form == "long")
                for child in self.components
            ):
                required_child = (
                    "same-resolution child"
                    if self.df_form == "wide"
                    else "same-resolution long-form child"
                )
                raise ValueError(
                    f"{self.name} has no market data and no {required_child} "
                    + "to establish its spine"
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
            "component_id": self.component_id,
            **self._identity_fields(),
            "signal_cols": self._signal_cols,
            "data_start": self.context.data_start,
            "data_end": self.context.data_end,
            "run_mode": self.run_mode,
            "settings": self.settings.model_dump(),
        }

    def _identity_fields(self) -> dict[str, Any]:
        return {
            "class_name": self.__class__.__name__,
            "engine_name": self.context.name,
            "component_name": self.name,
            "source_sha256": hashlib.sha256(
                self._source_artifact.read_bytes()
            ).hexdigest(),
            "resolution": repr(self.resolution),
            "component_type": self.component_type,
            "df_form": self._df_form,
            "signature": (self.__pfund_args__, self.__pfund_kwargs__),
            "config": self.config,
            "params": self.params,
            "datas": [data.to_dict() for data in self.get_datas()],
            "models": sorted(self.models),
            "features": sorted(self.features),
        }

    # Component identity must remain fixed after its first use. Runtime work can
    # mutate objects in the identity signature (e.g. materializing a PyTorch
    # LazyLinear), so recomputing this property could write one component's
    # artifacts under different IDs. Keep this cached so every artifact uses
    # the same ID for the lifetime of the component.
    @cached_property
    def component_id(self) -> str:
        identity = self._identity_fields()
        payload = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
        digest = hashlib.sha256(payload).hexdigest()[:12]
        return f"{identity['class_name']}-{digest}"

    @property
    def _source_artifact(self) -> Path:
        if self._source_artifact_path is not None:
            return self._source_artifact_path

        import inspect

        source_file = inspect.getsourcefile(type(self))
        if source_file is None:
            raise ValueError(f"cannot locate source file for {type(self).__name__}")
        return Path(source_file)

    @ray_method
    def _set_source_artifact_path(self, source_artifact_path: str) -> None:
        self._source_artifact_path = Path(source_artifact_path)

    @property
    def _data_artifact(self) -> IntoDataFrame:
        # TODO: extract the latest rows (that have not been written to pfeed) from the trading store's df
        raise NotImplementedError

    def _add_product(
        self,
        venue: TradingVenue,
        basis: str,
        exchange: str = "",
        symbol: str = "",
        name: str = "",
        **specs: Any,
    ) -> BaseProduct:
        Venue = self._get_venue_class(venue)
        product = Venue.create_product(
            basis=basis, exchange=exchange, name=name, symbol=symbol, **specs
        )
        if product.name not in self.products:
            self.products[product.name] = product
            self.logger.debug(f"added {product}")
        else:
            existing_product = self.products[product.name]
            assert existing_product == product, (
                f"product name '{product.name}' is already used"
            )
        return self.products[product.name]

    def add_data(
        self,
        venue: TradingVenue,
        product: str,
        resolutions: list[Resolution | str] | None = None,
        exchange: str = "",
        symbol: str = "",
        product_name: str = "",
        config: DataConfig | None = None,
        as_features: bool = True,
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

    def _add_component[ComponentT: Component](
        self,
        component: ComponentT | ActorProxy[ComponentT],
        resolution: str,
        name: str,
        df_form: Literal["wide", "long"],
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> ComponentT | ActorProxy[ComponentT] | None:
        """Adds a model component to the current component.
        A model component is a model or feature.
        Args:
            df_form: DataFrame layout used by this component.
            ray_kwargs: kwargs for ray actor, e.g. num_cpus
            ray_actor_options:
                Options for Ray actor.
                will be passed to ray actor like this: Actor.options(**ray_options).remote(**ray_kwargs)
        """
        Component = component.__class__
        ComponentName = Component.__name__
        if component.is_strategy():
            from pfund.components.strategies.strategy_base import BaseStrategy

            assert isinstance(self, BaseStrategy), (
                f"only a strategy can add a sub-strategy, '{self.name}' is a {self.component_type}"
            )
            components = self.strategies
        elif component.is_model():
            components = self.models
        elif component.is_feature():
            components = self.features
        else:
            raise ValueError(
                f"{component.component_type} '{ComponentName}' is not a strategy, model or feature"
            )

        component_resolution = (
            component.resolution
            if isinstance(component, ActorProxy)
            else Resolution(resolution or self.resolution)
        )
        if component_resolution > self.resolution:
            component_name = name or component.name
            raise ValueError(
                f"{component.component_type} '{component_name}' resolution "
                + f"{component_resolution!r} is finer than parent '{self.name}' resolution "
                + f"{self.resolution!r}; a child component's output resolution must "
                + "be equal to or coarser than its parent's resolution"
            )

        if not isinstance(component, ActorProxy):
            component_type = component.component_type
            component_name = name or component.name
            if any(
                component_name.casefold() == existing.casefold()
                for existing in components
            ):
                raise ValueError(f"{component_name} already exists")

            if ray_kwargs:
                component = ActorProxy(
                    component,
                    name=component_name,
                    resolution=component_resolution,
                    component_type=component_type,
                    engine_context=self.context,
                    ray_actor_options=ray_actor_options,
                    **ray_kwargs,
                )

            component._hydrate(
                name=component_name,
                run_mode=RunMode.REMOTE if ray_kwargs else RunMode.LOCAL,
                resolution=component_resolution,
                engine_context=self.context,
                df_form=df_form,
            )

        cast("dict[str, Any]", components)[component.name] = component
        self.logger.debug(f"{self.name} added {component.name}")

        # NOTE: returns None when adding a local component (not ActorProxy) to a remote component to avoid returning a serialized (copied) object
        if self.is_remote() and not isinstance(component, ActorProxy):
            return None
        return component

    def add_model[ModelT: BaseModel](
        self,
        model: ModelT | ActorProxy[ModelT] | UnderlyingModel,
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> ModelT | ActorProxy[ModelT] | None:
        """Add a model as a child of this component.

        Args:
            model: PFund model, remote model proxy, or supported underlying model
                to add.
            resolution: Resolution at which the model runs. Inherits this
                component's resolution when omitted.
            name: Optional name for the model.
            df_form: DataFrame layout used by the model.
            ray_actor_options: Options passed to the Ray actor.
            ray_kwargs: Ray actor constructor arguments. Providing these runs the
                model remotely.

        Returns:
            The added model or its remote proxy. Returns ``None`` when adding a
            local model to a remote parent component.
        """
        from pfund.components.models.wrap import wrap_model

        return self._add_component(
            component=wrap_model(model),
            resolution=resolution,
            name=name,
            df_form=df_form,
            ray_actor_options=ray_actor_options,
            **ray_kwargs,
        )

    def add_feature[FeatureT: BaseFeature](
        self,
        feature: FeatureT | ActorProxy[FeatureT],
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> FeatureT | ActorProxy[FeatureT] | None:
        """Add a feature as a child of this component.

        Args:
            feature: Feature instance or remote feature proxy to add.
            resolution: Resolution at which the feature runs. Inherits this
                component's resolution when omitted.
            name: Optional name for the feature.
            df_form: DataFrame layout used by the feature.
            ray_actor_options: Options passed to the Ray actor.
            ray_kwargs: Ray actor constructor arguments. Providing these runs the
                feature remotely.

        Returns:
            The added feature or its remote proxy. Returns ``None`` when adding a
            local feature to a remote parent component.
        """
        return self._add_component(
            component=feature,
            resolution=resolution,
            name=name,
            df_form=df_form,
            ray_actor_options=ray_actor_options,
            **ray_kwargs,
        )

    # FIXME
    # TODO: add a progress bar to it when its not ready
    def is_ready(self, data: BaseData) -> bool:
        if data.category == DataCategory.MARKET_DATA:
            df = self.get_df(kind="data", data_category=data.category, to_native=False)
            if len(df) < self.config["warmup_period"]:
                return False
            if self.df_form == "long":
                # long form: ready when this product's bar is closed
                return data.is_closed()
            elif self.df_form == "wide":
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

    def _reload_markets(self):
        """venues (at engine level) might have refetched the latest markets, reload markets from markets.yml"""
        # NOTE: must use pfund_config from context, config from get_config() could be different from what user has set in Ray Actor
        # .parent drops the engine name: markets.yml is venue data, shared by
        # every engine, and that is where the venues wrote it.
        data_path = self.context.config.data_path.parent
        for product in self.products.values():
            if product.source not in TradingVenue.__members__:
                continue
            VenueClass = TradingVenue(product.source).venue_class
            VenueClass.load_market(product, data_path)

    def _materialize(self):
        """Materialize historical frames only when a data range was requested."""
        if self.context.data_start is None and self.context.data_end is None:
            return
        for data_store in self.data_stores.values():
            data_store.materialize()
        self.store.materialize()

    def _gather(self) -> None:
        """Sets up everything before start"""
        # NOTE: use is_gathered to avoid a component being gathered multiple times when it's a shared component
        if not self._is_gathered:
            # 1. Let the component declare its complete graph.
            self.add_datas()
            self.add_models()
            self.add_features()

            # 2. Validate and normalize this component before expensive work.
            self._check_input_sources()
            self._check_config()
            self._reload_markets()

            # 3. Prepare dependencies completely, bottom-up.
            for component in self.components:
                component._gather()

            # 4. Validate state that child gathering establishes.
            self._assert_unique_child_signal_cols()

            # 5. Materialize this component from its prepared dependencies.
            self._materialize()

            # 6. Persist provenance only after successful setup.
            # NOTE: Keep this as the last setup step. During materialization,
            # pfeed normalizes data.config.storage_config.data_domain from "" to
            # "MARKET_DATA"; that changes the `datas` payload in component_id.
            # Save only after such mutations so every artifact uses the same ID.
            self.store._save_source_artifact()

            self._is_gathered = True
            self.logger.debug(f"'{self.name}' has gathered")
        else:
            self.logger.debug(f"'{self.name}' has already gathered")

    def start(self) -> None:
        if not self.is_running():
            # set the ZMQPubHandler's receiver ready to flush the buffered log messages
            if self.is_remote(direct_only=False):
                self.logger.handlers[0].set_receiver_ready()
            for component in self.components:
                component.start()
            self._is_running = True
            self.on_start()
            self._databoy.start()
            self.logger.debug(f"'{self.name}' has started")
        else:
            self.logger.debug(f"'{self.name}' has already started")

    def stop(self, reason: str = "") -> None:
        """Stops the component, keep the internal states"""
        if self.is_running():
            for component in self.components:
                component.stop()
            self._is_running = False
            self.on_stop()
            self._databoy.stop()
            self.logger.debug(f"'{self.name}' has stopped, ({reason=})")
        else:
            self.logger.debug(f"'{self.name}' has already stopped")
