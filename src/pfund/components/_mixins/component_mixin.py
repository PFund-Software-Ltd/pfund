from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, overload

if TYPE_CHECKING:
    import polars as pl
    from pfund_blocks.datas import BarData, BaseData, MarketData
    from pfund_blocks.entities.products.product_base import BaseProduct
    from pfund_blocks.typing import TradingVenue

    from pfund.components.features.feature_base import BaseFeature
    from pfund.components.models.model_base import BaseModel, UnderlyingModel
    from pfund.components.strategies.strategy_base import BaseStrategy
    from pfund.typing import (
        Component,
        ComponentName,
        Signals,
    )

from pathlib import Path

from pfund_blocks.enums import DataCategory
from pfund_blocks.resolution import Resolution

from pfund.components._mixins.base_mixin import ComponentBaseMixin
from pfund.components._mixins.dataframe_mixin import DataFrameMixin
from pfund.components.actor_proxy import ActorProxy
from pfund.data_config import DataConfig
from pfund.enums import ArtifactType, ComponentType, RunMode
from pfund.utils.decorators import ray_method


class ComponentMixin(DataFrameMixin, ComponentBaseMixin):
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
            )
            signals = self._signalize(X)
            self._latest_signals = signals
            self.store.update_df(signals, data=data)
            self._publish_signals(signals, data)
        except Exception:
            self.logger.exception("Error forwarding data:")

    def _signalize(self, X: pl.DataFrame) -> Signals:
        """Creates signals of this component

        Args:
            X: features df

        Returns:
            dict[ColumnName, Any]: The signals.
        """
        if self.is_strategy():
            strategy = cast("BaseStrategy", self)
            return strategy.to_signals(strategy.trade(X))
        elif self.is_model():
            model = cast("BaseModel", self)
            return model.to_signals(model.predict(X))
        elif self.is_feature():
            feature = cast("BaseFeature", self)
            return feature.to_signals(feature.transform(X))
        else:
            raise ValueError(f"unknown component type {self.component_type}")

    def _publish_signals(self, signals: Signals, data: BarData):
        if self._databoy.is_using_zmq():
            self._databoy._publish_signals(signals, data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            # the same snapshot component_id was hashed from, not recomputed, see _identity
            **self._identity,
            "signal_cols": self._signal_cols,
            "run_mode": self.run_mode,
            "settings": self.settings.model_dump(),
        }

    def _identity_fields(self) -> dict[str, Any]:
        return {
            "class_name": self.__class__.__name__,
            "engine_name": self.context.name,
            "component_name": self.name,
            "source_sha256": self._hash_source_files(),
            "resolution": repr(self.resolution),
            "component_type": self.component_type,
            "df_form": self._df_form,
            "signature": (self.__pfund_args__, self.__pfund_kwargs__),
            "config": self.config,
            "params": self.params,
            "datas": [data.to_dict() for data in self.get_datas()],
            "data_start": self.context.data_start,
            "data_end": self.context.data_end,
            # child ids, not names: a change in a child (e.g. params loaded from a TOML file)
            # must change the parent's id too, otherwise a non-top component reuses its
            # persisted df computed from the old child, see BacktestMixin._materialize()
            "models": {
                name: model.get_component_id()
                for name, model in sorted(self.models.items())
            },
            "features": {
                name: feature.get_component_id()
                for name, feature in sorted(self.features.items())
            },
        }

    @property
    def _source_file_path(self) -> Path:
        if self._source_file_path_provided is not None:
            return self._source_file_path_provided

        import inspect

        source_file = inspect.getsourcefile(type(self))
        if source_file is None:
            raise ValueError(f"cannot locate source file for {type(self).__name__}")
        return Path(source_file)

    @ray_method
    def _set_source_file_path(self, source_file_path: str) -> None:
        self._source_file_path_provided = Path(source_file_path)

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
            product: product basis, defined as {base_asset}_{quote_asset}_{product_type}, e.g. BTC_USDT_PERP
            resolutions: extra data resolutions in use, e.g. "1t" for tick data, "1q" for quote data.
                The component's resolution is always included.
            exchange: useful for TradFi brokers (e.g. IB), to specify the exchange (e.g. 'NASDAQ')
            symbol: useful for TradFi brokers (e.g. IB), to specify the symbol (e.g. 'AAPL')
            product_name: A user-defined identifier for the product.
                If not provided, the default product symbol (e.g. 'BTC_USDT_PERP', 'TSLA241213C00075000') will be used.
                This is useful when you need to distinguish between similar instruments, such as options
                with different strike prices and expiration dates. Instead of using long identifiers like
                'BTC_USDT_OPTION_100000_20250620' and 'BTC_USDT_OPTION_101000_20250920', you can assign
                simpler names like 'BTC_OPT1' and 'BTC_OPT2'.
                Note:
                    It is the user's responsibility to manage and maintain these custom product names.
            config: Data Configuration, shared by all the product's resolutions.
                A product can only be added once, so pass all its resolutions in one call.
            as_features: Whether data columns (e.g. OHLCV in market data) are treated as "features"
                in the component's trading_df. Must be the same across add_data() calls.
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
    ) -> ComponentT | ActorProxy[ComponentT]:
        """Adds a model component to the current component.
        A model component is a model or feature.
        Args:
            df_form: DataFrame layout used by this component.
            ray_kwargs: kwargs for ray actor, e.g. num_cpus
            ray_actor_options:
                Options for Ray actor.
                will be passed to ray actor like this: Actor.options(**ray_options).remote(**ray_kwargs)
        """
        # NOTE: use the component_type attribute, not is_strategy() etc., because on an
        # ActorProxy a method call goes through __getattr__ and creates the Ray actor
        component_type = component.component_type
        if component_type == ComponentType.strategy:
            from pfund.components.strategies.strategy_base import BaseStrategy

            assert isinstance(self, BaseStrategy), (
                f"only a strategy can add a sub-strategy, '{self.name}' is a {self.component_type}"
            )
            components = self.strategies
        elif component_type == ComponentType.model:
            components = self.models
        elif component_type == ComponentType.feature:
            components = self.features
        else:
            raise ValueError(
                f"{component_type} '{component.__class__.__name__}' is not a strategy, model or feature"
            )

        component_name = (
            component.name
            if isinstance(component, ActorProxy)
            else name or component.name
        )
        if any(
            component_name.casefold() == existing.casefold() for existing in components
        ):
            raise ValueError(f"{component_name} already exists")

        component_resolution = (
            component.resolution
            if isinstance(component, ActorProxy)
            else Resolution(resolution or self.resolution)
        )
        if component_resolution > self.resolution:
            raise ValueError(
                f"{component.component_type} '{component_name}' resolution "
                + f"{component_resolution!r} is finer than parent '{self.name}' resolution "
                + f"{self.resolution!r}; a child component's output resolution must "
                + "be equal to or coarser than its parent's resolution"
            )

        if not isinstance(component, ActorProxy):
            if ray_kwargs:
                component = ActorProxy(
                    component,
                    name=component_name,
                    resolution=component_resolution,
                    df_form=df_form,
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
                # NOTE: only strategies accept is_substrategy, see BaseStrategy._hydrate
                **(
                    {"is_substrategy": True}
                    if component_type == ComponentType.strategy
                    else {}
                ),
            )

        cast("dict[str, Any]", components)[component.name] = component
        self.logger.debug(f"{self.name} added {component.name}")
        return component

    @ray_method
    def _add_component_via_proxy(
        self,
        add_method: Literal["add_strategy", "add_model", "add_feature"],
        *args: Any,
        **kwargs: Any,
    ) -> ActorProxy[Any] | None:
        """Runs add_strategy/add_model/add_feature inside this (remote) component's actor.

        Called by ActorProxy.add_strategy/add_model/add_feature.
        A local child lives in this actor's process, returning it through Ray would only
        return a serialized copy, so None is returned. A remote child (ActorProxy) is
        returned since it is shared across processes.
        """
        component = getattr(self, add_method)(*args, **kwargs)
        # NOTE: a local child lives in this actor's process, so returning it through Ray
        # would hand the caller a serialized copy, not the child this actor runs;
        # return None instead. A remote child (ActorProxy) is safe to return, since every
        # copy of it points to the same Ray actor.
        return component if isinstance(component, ActorProxy) else None

    # remote model in -> remote model out, ray args are ignored (the proxy is already remote)
    @overload
    def add_model[ModelT: BaseModel](
        self,
        model: ActorProxy[ModelT],
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> ActorProxy[ModelT]: ...

    # local model in + num_cpus -> wrapped into a remote model (never None)
    @overload
    def add_model[ModelT: BaseModel](
        self,
        model: ModelT,
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        ray_actor_options: dict[str, Any] | None = None,
        *,
        num_cpus: float,
        **ray_kwargs: Any,
    ) -> ActorProxy[ModelT]: ...

    # local model in, no num_cpus -> local model out
    @overload
    def add_model[ModelT: BaseModel](
        self,
        model: ModelT,
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        ray_actor_options: dict[str, Any] | None = None,
        *,
        num_cpus: None = None,
        **ray_kwargs: Any,
    ) -> ModelT: ...

    # underlying model (e.g. sklearn/pytorch) in + num_cpus -> wrapped pfund model as a remote model
    @overload
    def add_model(
        self,
        model: UnderlyingModel,
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        ray_actor_options: dict[str, Any] | None = None,
        *,
        num_cpus: float,
        **ray_kwargs: Any,
    ) -> ActorProxy[BaseModel]: ...

    # underlying model in, no num_cpus -> wrapped pfund model
    @overload
    def add_model(
        self,
        model: UnderlyingModel,
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        ray_actor_options: dict[str, Any] | None = None,
        *,
        num_cpus: None = None,
        **ray_kwargs: Any,
    ) -> BaseModel: ...

    def add_model[ModelT: BaseModel](
        self,
        model: ModelT | ActorProxy[ModelT] | UnderlyingModel,
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> ModelT | ActorProxy[ModelT] | BaseModel:
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
            The added model or its remote proxy.
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

    # remote feature in -> remote feature out, ray args are ignored (the proxy is already remote)
    @overload
    def add_feature[FeatureT: BaseFeature](
        self,
        feature: ActorProxy[FeatureT],
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> ActorProxy[FeatureT]: ...

    # local feature in + num_cpus -> wrapped into a remote feature
    @overload
    def add_feature[FeatureT: BaseFeature](
        self,
        feature: FeatureT,
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        ray_actor_options: dict[str, Any] | None = None,
        *,
        num_cpus: float,
        **ray_kwargs: Any,
    ) -> ActorProxy[FeatureT]: ...

    # local feature in, no num_cpus -> local feature out
    @overload
    def add_feature[FeatureT: BaseFeature](
        self,
        feature: FeatureT,
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        ray_actor_options: dict[str, Any] | None = None,
        *,
        num_cpus: None = None,
        **ray_kwargs: Any,
    ) -> FeatureT: ...

    def add_feature[FeatureT: BaseFeature](
        self,
        feature: FeatureT | ActorProxy[FeatureT],
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> FeatureT | ActorProxy[FeatureT]:
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
            The added feature or its remote proxy.
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
            df = self.get_df(kind="data", data_category=data.category)
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

    # FIXME
    def _reload_markets(self):
        pass
        # """venues (at engine level) might have refetched the latest markets, reload markets from markets.yml"""
        # # NOTE: must use pfund_config from context, config from get_config() could be different from what user has set in Ray Actor
        # # .parent drops the engine name: markets.yml is venue data, shared by
        # # every engine, and that is where the venues wrote it.
        # data_path = self.context.config.data_path.parent
        # for product in self.products.values():
        #     if product.source not in TradingVenue.__members__:
        #         continue
        #     VenueClass = TradingVenue(product.source).venue_class
        #     VenueClass.load_market(product, data_path)

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
            self.store.setup_lakehouse()
            self._materialize()

            # 6. Persist provenance only after successful setup.
            # NOTE: Keep this as the last setup step. During materialization,
            # pfeed normalizes data.config.storage_config.data_domain from "" to
            # "MARKET_DATA"; that changes the `datas` payload in component_id.
            # Save only after such mutations so every artifact uses the same ID.
            self._feed.save_artifact(
                cast("Component", self),
                ArtifactType.source,
                artifact=self._source_file_path.read_bytes(),
            )

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
