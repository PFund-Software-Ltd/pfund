# ty: ignore[incompatible-property-override]
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, override

if TYPE_CHECKING:
    from pfund_blocks.datas import BarData, BaseData
    from pfund_blocks.entities.products.product_base import BaseProduct
    from pfund_blocks.resolution import Resolution

    from pfund.backtest._mixins.strategy_mixin import BacktestStrategyMixin
    from pfund.backtest.cv.cv_fold import CVFold
    from pfund.backtest.cv.holdout_split import HoldoutSplit
    from pfund.components.actor_proxy import ActorProxy
    from pfund.components.features.feature_base import BaseFeature
    from pfund.components.models.model_base import BaseModel
    from pfund.engines.contexts.backtest_engine_context import BacktestEngineContext
    from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
    from pfund.typing import Component, ComponentName, Signals

from pathlib import Path

import polars as pl
from pfund_kit.style import RichColor, TextStyle, cprint

from pfund.backtest._mixins.dataframe_mixin import BacktestDataFrameMixin
from pfund.components._mixins.component_mixin import ComponentMixin
from pfund.enums import BacktestMode


class BacktestMixin(BacktestDataFrameMixin, ComponentMixin):
    """Backtest behavior layered over a component.

    Subclassed per component type (BacktestModelMixin, BacktestStrategyMixin,
    BacktestFeatureMixin), which `_adopt()` mixes in as the first base of the
    component's class, e.g. ``(BacktestModelMixin, MyModel)``, so its overrides
    precede the component's and its ``super()`` calls resolve to the component
    class next in the MRO.
    """

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        # A backtest class never runs __init__: _adopt() swaps __class__ on an
        # already-constructed component, so an __init__ here would silently never run.
        # Backtest state belongs in _init_backtest_state() instead.
        assert "__init__" not in vars(cls), (
            f"{cls.__name__} must not define __init__(), put backtest state in _init_backtest_state()"
        )

    def __mixin_post_init__(self, *args: Any, **kwargs: Any):
        # A backtest component is never constructed directly: the user's component is
        # created (and post-initialized) under its own class, then adopted in place via
        # _adopt(), which swaps __class__ and calls _init_backtest_state() instead.
        # Reaching here means something constructed a backtest class directly,
        # bypassing _adopt().
        raise RuntimeError(
            f"{type(self).__name__} must be created via BacktestMixin._adopt(), not constructed directly"
        )

    def _init_backtest_state(self) -> None:
        # top components are the ones added by engine.add_strategy/model/feature()
        self._is_top_component = False
        self._holdout_split: HoldoutSplit | None = None
        self._folds: tuple[CVFold, ...] = ()

    @classmethod
    def _adopt[C: Component](cls, component: C) -> C:
        """Turns an already-constructed component into its backtesting version, in place.

        The component keeps its identity and state (name, source artifact path, etc.),
        only its class is swapped to a backtest class built from ``(cls, type(component))``,
        e.g. ``(BacktestModelMixin, MyModel)``. A new backtest class is built per adopted
        instance (no caching), so two adopted ``MyModel`` instances have different types.

        Args:
            component: the component to adopt. Returned as is if already adopted,
                e.g. the same instance shared by two parents.

        Returns:
            The same component object, now an instance of the backtest class.
        """
        if isinstance(component, BacktestMixin):
            return component
        Component = type(component)
        # built with the component's metaclass (MetaModel etc.), like a class statement would.
        # __name__/__qualname__ make it look like the component's own class in logs/errors,
        # __module__ marks it as a pfund class, so the metaclass doesn't treat it as user-defined
        Backtest = type(Component)(
            Component.__name__,
            (cls, Component),
            {
                "__module__": cls.__module__,
                "__qualname__": Component.__qualname__,
                "__wrapped__": Component,
            },
        )
        # a class swap can't be typed: ty types __class__ as type[C], while Backtest is a
        # subclass of type(component) built at runtime
        cast("Any", component).__class__ = Backtest
        cast("BacktestMixin", component)._init_backtest_state()
        return component

    def set_top_component(self):
        self._is_top_component = True

    @property
    def context(self) -> BacktestEngineContext:
        assert self._context is not None, "context is not set"
        return cast("BacktestEngineContext", self._context)

    @property
    def settings(self) -> BacktestEngineSettings:
        return self.context.settings

    @property
    def backtest_mode(self) -> BacktestMode:
        return self.context.mode

    @property
    def _source_artifact(self) -> Path:
        if self._source_artifact_path is not None:
            return self._source_artifact_path

        import inspect

        # the user's component class, stored by _adopt(); the MRO position isn't reliable
        # since BacktestMixin's own bases (e.g. BacktestDataFrameMixin) precede it
        source_class: type = getattr(type(self), "__wrapped__")  # noqa: B009
        source_file = inspect.getsourcefile(source_class)
        if source_file is None:
            raise ValueError(f"cannot locate source file for {source_class.__name__}")
        return Path(source_file)

    @property
    def _data_artifact(self) -> pl.DataFrame | None:
        # store._df is never trimmed in backtesting, so it holds every row produced during the run
        return self.store._df

    @property
    def _max_rows(self) -> None:
        # set max_rows to none so that trading_store._df can keep every row as data artifact.
        # trimming trading_df is done in get_df() to preserve the same output as live trading
        return None

    def to_dict(self) -> dict[str, Any]:
        """Adds how the component was backtested, e.g. so a chunked or vectorized df can't seed live trading."""
        return {
            **super().to_dict(),
            "backtest_mode": self.backtest_mode,
            "num_chunks": self.context.num_chunks,
        }

    @override
    def _get_streaming_resolutions(self, product: BaseProduct) -> list[Resolution]:
        """Gets supported resolutions for the product based on the trading venue.
        Overrides it in backtesting, supports only the primary resolution.
        """
        return [self.resolution]

    @override
    def _add_component[ComponentT: Component](
        self,
        component: ComponentT | ActorProxy[ComponentT],
        resolution: str,
        name: str,
        df_form: Literal["wide", "long"],
        # NOTE: ray args are ignored in backtesting, components always run locally
        ray_actor_options: dict[str, Any] | None = None,
        **ray_kwargs: Any,
    ) -> ComponentT:
        component = cast("ComponentT", component)
        if ray_actor_options or ray_kwargs:
            self.logger.warning(
                f"ray_actor_options and ray_kwargs are ignored in backtesting, {component.component_type} {name or component.name} will run locally"
            )
        if component.is_strategy():
            from pfund.backtest._mixins.strategy_mixin import BacktestStrategyMixin

            component = BacktestStrategyMixin._adopt(component)
        elif component.is_model():
            from pfund.backtest._mixins.model_mixin import BacktestModelMixin

            component = BacktestModelMixin._adopt(component)
        elif component.is_feature():
            from pfund.backtest._mixins.feature_mixin import BacktestFeatureMixin

            component = BacktestFeatureMixin._adopt(component)
        else:
            raise ValueError(f"Unsupported component type: {component}")
        return cast(
            "ComponentT",
            super()._add_component(
                component=component,
                resolution=resolution,
                name=name,
                df_form=df_form,
            ),
        )

    def _is_dummy_strategy(self) -> bool:
        from pfund.components.strategies._dummy_strategy import _DummyStrategy

        return isinstance(self, _DummyStrategy)

    def _validate_vectorized_backtest_inputs(self, df: pl.DataFrame) -> None:
        """Validates the strategy's backtest() and its input df before running a vectorized backtest.

        backtest() must exist and be callable with exactly one argument (df).
        Its return value can only be checked after the call, see _vectorized_backtest().

        The df must be in LONG form: one row per (date, product, resolution) bar
        with shared OHLC columns, unique keys, sorted by date:

            date | product | resolution | open | high | low | close | ...
            d1   | BTC     | 1d         | ...
            d1   | ETH     | 1d         | ...
            d2   | BTC     | 1d         | ...

        Wide form (e.g. 'BTC_close', 'ETH_close' columns) is not supported.
        """
        import inspect

        from pfund.components.bar_dataframe import KEY_COLS

        backtest = getattr(self, "backtest", None)
        if not callable(backtest):
            raise NotImplementedError(
                f"{self} has no backtest() method, which VECTORIZED backtesting requires. "
                + "Implement `def backtest(self, df: pl.DataFrame) -> pl.DataFrame` "
                + "that returns the backtested df, or backtest it in EVENT_DRIVEN mode instead."
            )
        # dry-runs the argument matching of backtest(df) without calling it
        try:
            inspect.signature(backtest).bind(None)
        except TypeError as err:
            raise TypeError(
                f"backtest() must be callable with exactly 1 argument (df): {err}"
            ) from None

        # NOTE: volume is not required, the kernels ignore it (fills never depend on volume)
        ohlc_cols = ["open", "high", "low", "close"]
        missing_ohlc_cols = [c for c in ohlc_cols if c not in df.columns]
        if missing_ohlc_cols:
            raise ValueError(
                f"missing column(s) {missing_ohlc_cols} — the dataframe must be in LONG form: "
                + "one row per (date, product, resolution) bar with shared OHLC columns "
                + f"({', '.join(ohlc_cols)}). "
                + "Wide form (e.g. 'BTC_close', 'ETH_close') is not supported."
            )
        missing_key_cols = [c for c in KEY_COLS if c not in df.columns]
        if missing_key_cols:
            raise ValueError(
                f"backtesting requires columns ({', '.join(KEY_COLS)}) "
                + f"to identify each bar — missing column(s) {missing_key_cols}"
            )
        if df.select(KEY_COLS).is_duplicated().any():
            raise ValueError(
                f"({', '.join(KEY_COLS)}) keys must be unique for backtesting"
            )
        # bars are processed positionally per (product, resolution) combo: an
        # unsorted df would silently produce garbage results
        if not df.get_column("date").is_sorted():
            raise ValueError("the dataframe must be sorted by 'date' (ascending)")

    @staticmethod
    def _setup_df_for_vectorized_backtest(df: pl.DataFrame) -> None:
        """Prepares a fresh vectorized backtest session for the df's class.

        Clears the product and portfolio registries of any previous configuration,
        then attaches the two ENTRY methods to the df's class so they survive native
        operations (filter/join/... construct plain dataframes of the same class):
            create_signal — product backtesting
            create_weight — portfolio backtesting

        The rest of each mode's methods are deliberately NOT attached here:
        backtest()'s name is shared by both modes, so the entry method attaches
        its mode's implementation when called — create_signal() attaches
        open_position/close_position and ProductBacktestMixin's backtest,
        create_weight() attaches PortfolioBacktestMixin's backtest. (Consequence:
        chains must run sequentially — finish one chain before starting one of
        the other mode.)
        """
        from pfund.backtest._mixins import (
            portfolio_backtest_mixin,
            product_backtest_mixin,
        )
        from pfund.backtest._mixins.portfolio_backtest_mixin import (
            PortfolioBacktestMixin,
        )
        from pfund.backtest._mixins.product_backtest_mixin import ProductBacktestMixin

        product_backtest_mixin._clear_registry()
        portfolio_backtest_mixin._clear_registry()

        df_class = type(df)
        df_class.create_signal = ProductBacktestMixin.create_signal  # ty: ignore[missing-attribute]
        df_class.create_weight = PortfolioBacktestMixin.create_weight  # ty: ignore[missing-attribute]

    def _vectorized_backtest(self, X: pl.DataFrame) -> pl.DataFrame:
        """Runs a vectorized backtest and completes the feature-only trading dataframe with the resulting signals."""
        if self.is_strategy():
            self._validate_vectorized_backtest_inputs(X)
            self._setup_df_for_vectorized_backtest(X)
            backtest_df: Any = cast("BacktestStrategyMixin", self).backtest(X)

            # a lazy chain is fine, the backtest only needs its result
            if isinstance(backtest_df, pl.LazyFrame):
                backtest_df = backtest_df.collect()
            # check if returns a df
            if not isinstance(backtest_df, pl.DataFrame):
                raise TypeError(
                    f"{self.name}.backtest() must return the backtested polars DataFrame (or LazyFrame), got {type(backtest_df).__name__}. Did you forget to return df?"
                )
            # check if returns a new df
            if backtest_df is X:
                cprint(
                    f"WARNING: {self.name} backtest() returned the same df unchanged.\n"
                    + "This is fine if you only used native e.g. Polars/Pandas operations on the original df.\n"
                    + "However, [italic]this is an ERROR[/italic] if you called backtest methods such as "
                    + "create_signal(), open_position(), or close_position() —\n"
                    + "these return a new df, so you must reassign: df = df.create_signal(...) and return the new df",
                    style=TextStyle.BOLD + RichColor.RED,
                )
            return backtest_df
        else:
            if callable(getattr(self, "backtest", None)):
                raise TypeError(
                    f"{self} defines a backtest() method, but only strategies can have one."
                )
            signals = cast("Component", self).signalize(X)
            signals_df = pl.DataFrame(signals)
            return pl.concat([X, signals_df], how="horizontal")

    # EXTEND: support non-bar data types:
    # brainstorm: heapq.merge(market_data_df, news_data_df), with a dispatcher to create data updates for each data type
    # TODO: write trading_df (trading_df.delta) and compare it with the one generated by vectorized backtesting
    # TODO: need to build a clock? coz looping bars has time gaps
    # TODO: support dataset_splits
    def _event_driven_backtest(self, df: pl.DataFrame) -> pl.DataFrame:
        from pfund_kit.style import RichColor
        from pfund_kit.utils.progress_bar import track

        # OPTIMIZE: critical loop
        for row in track(
            df.iter_rows(named=False),
            total=df.shape[0],
            description=...,
            bar_style=RichColor.BRIGHT_YELLOW,
            disable=not self.context.config.show_progress_bar,
        ):
            ts, resolution, product_name, source_type, o, h, l, c, v = row
            data = cast("BarData", self.get_data(product_name, resolution))
            update: BarUpdate = {
                "ts": ts,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
                "is_incremental": False,
                "msg_ts": None,
                "extra": {},
            }
            self._databoy.collect(...)

        # TODO: return the backtested dataframe, how?
        return df

    @property
    def _reuse_child_component_signals(self):
        return not self._is_top_component and self.settings.reuse_signals

    @override
    def _get_components_signals(self, data: BarData) -> dict[ComponentName, Signals]:
        return {
            component.name: component.store.get_signals(data)
            for component in self.components
        }

    # data df has already been loaded in full in backtesting, so no need to update
    @override
    def _update_data_df(self, data: BaseData) -> None:
        pass

    @override
    def step(self, data: BarData):
        try:
            if self._reuse_child_component_signals:
                # NOTE: child component's signals are already available (loaded during materialization), reuse them
                self._latest_signals = self.store.get_signals(data)
                return
            return super().step(data)
        except Exception:
            self.logger.exception("Error forwarding data:")

    # no-op in backtesting, zeromq is not in use
    @override
    def _publish_signals(self, signals: Signals, data: BarData):
        pass

    @override
    def _materialize(self):
        self.store.setup_lakehouse()
        for data_store in self.data_stores.values():
            data_store.materialize()
        if self.backtest_mode == BacktestMode.VECTORIZED:
            # child component will try to load its trading_df
            if not self._is_top_component:
                self.store.materialize()
            if self.store._df is None:
                X = self._merge_signals_dfs().to_native()
                if not self._is_top_component:
                    cprint(
                        f"No persisted trading dataframe found for '{self.name}'. "
                        + "Running its vectorized backtest on the fly...",
                        style=TextStyle.BOLD + RichColor.YELLOW,
                    )
                    self.store._df = self._vectorized_backtest(X)
                    self.store.persist_to_datalake()
                else:
                    self.store._df = X
            self._resolve_dataset_splits()
        elif self.backtest_mode == BacktestMode.EVENT_DRIVEN:
            if self._reuse_child_component_signals:
                self.store.materialize()
        else:
            raise ValueError(f"Unknown backtest mode: {self.backtest_mode}")

    @override
    def _gather(self) -> None:
        if self._is_dummy_strategy():
            assert len(self.components) == 1, (
                "dummy strategy must have exactly one component"
            )
            component = cast("BaseModel | BaseFeature", self.components[0])
            component._gather()
            self._is_gathered = True
        else:
            return super()._gather()
