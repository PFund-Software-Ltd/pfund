# ty: ignore[incompatible-property-override]
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self, cast, override

if TYPE_CHECKING:
    from pfund_blocks.datas import BarData, BaseData
    from pfund_blocks.entities.products.product_base import BaseProduct
    from pfund_blocks.resolution import Resolution

    from pfund.backtest.cv.dataset_splits import DatasetSplits, SplitPart
    from pfund.components.actor_proxy import ActorProxy
    from pfund.components.features.feature_base import BaseFeature
    from pfund.components.models.model_base import BaseModel
    from pfund.engines.contexts.backtest_engine_context import BacktestEngineContext
    from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
    from pfund.typing import Component, ComponentName, Signals

from pathlib import Path

import polars as pl
from pfund_kit.style import RichColor, TextStyle

from pfund.backtest._mixins.portfolio_backtest_mixin import PortfolioBacktestMixin
from pfund.backtest._mixins.product_backtest_mixin import ProductBacktestMixin
from pfund.backtest.cv.indexing import SplittableData, take_rows
from pfund.components._mixins.component_mixin import ComponentMixin
from pfund.enums import BacktestMode
from pfund.errors import StaleArtifactError


class BacktestDataFrame(ProductBacktestMixin, PortfolioBacktestMixin, pl.DataFrame):
    """Annotation-only type: at runtime the df is a plain native DataFrame
    whose class has the backtest methods attached by _setup_backtest_df_for_vectorized_mode()."""

    # Both mixins define backtest() with different signatures; at runtime only
    # one is attached (by create_signal() or create_weight()). This merged stub
    # accepts both call shapes: product mode takes no args, portfolio mode uses them.
    def backtest(
        self,
        initial_capital: float = 1_000_000,
        compound: bool = True,
        check: bool = True,
    ) -> Self:
        raise NotImplementedError(
            "BacktestDataFrame is an annotation-only type."
        )


class BacktestMixin(ComponentMixin):
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
        self._is_backtestee = False
        # the trading df over the whole data range, kept by _use_split() while X holds one split
        self._unsplit_trading_df: pl.DataFrame | None = None

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

    def set_as_backtestee(self):
        self._is_backtestee = True

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
    def dataset_splits(self) -> DatasetSplits | None:
        return self.context.dataset_splits

    def _assert_split_resolution(self) -> None:
        """Raises if split boundaries could fall inside one of this component's bars."""
        from pfund_blocks.resolution import Resolution

        if self.dataset_splits is None:
            return
        split_resolution = Resolution(self.settings.split_resolution)
        if split_resolution > self.resolution:
            raise ValueError(
                f"split_resolution {split_resolution!r} is finer than {self.component_type} "
                + f"'{self.name}' resolution {self.resolution!r}; a split boundary would fall "
                + "inside a bar, set split_resolution in the backtest engine settings "
                + "to be equal to or coarser than every component's resolution"
            )

    def _get_trading_df(self) -> pl.DataFrame:
        """The trading df over the whole data range, whether or not use_split() has sliced it."""
        if self._unsplit_trading_df is not None:
            return self._unsplit_trading_df
        return self.get_df(kind="trading")

    def _take_split[DataT: SplittableData](
        self, data: DataT, part: SplitPart, fold: int | None = None
    ) -> DataT | None:
        """Rows of ``data`` in one split part, None if empty.

        ``data`` is row-aligned with the unsplit trading df, whose dates map its rows onto the split grid,
        so the result doesn't depend on which split use_split() has set.
        """
        dataset_splits = self.dataset_splits
        assert dataset_splits is not None, "no dataset_splits on the backtest engine"
        rows = dataset_splits.row_indices(self._get_trading_df(), part, fold)
        return None if rows.size == 0 else take_rows(data, rows)

    def _use_split(self, part: SplitPart, fold: int | None = None) -> None:
        """Sets the trading df of this component and its children to their rows in one split part, so X (its features) holds that split.

        Internal, called by BacktestEngine.use_split(): the split is engine state,
        so a component never sits on a different split from its parent.
        """
        if self.dataset_splits is None:
            raise ValueError(
                "no dataset_splits on the backtest engine, pass dataset_splits=... to BacktestEngine()"
            )
        # raises if the trading df is not ready, i.e. the engine hasn't run yet
        trading_df = self._get_trading_df()
        split_df = self._take_split(trading_df, part, fold)
        if split_df is None:
            raise ValueError(f"{self.name} has no rows in the {part!r} set")
        self._unsplit_trading_df = trading_df
        self.store.set_df(split_df)
        in_fold = "" if fold is None else f" (fold {fold})"
        self.logger.warning(
            f"{self.name}.X now only holds the {part!r} set{in_fold}",
            style=TextStyle.BOLD + RichColor.YELLOW,
        )
        for component in self.components:
            # components always run locally in backtesting, never as ActorProxy
            cast("BacktestMixin", component)._use_split(part, fold)

    def _clear_split(self) -> None:
        """Sets the trading df of this component and its children back to the whole data range, see _use_split()."""
        if self._unsplit_trading_df is not None:
            self.store.set_df(self._unsplit_trading_df)
            self._unsplit_trading_df = None
        for component in self.components:
            cast("BacktestMixin", component)._clear_split()

    @property
    def _source_file_path(self) -> Path:
        if self._source_file_path_provided is not None:
            return self._source_file_path_provided

        import inspect

        # the user's component class, stored by _adopt(); the MRO position isn't reliable
        # since BacktestMixin's own bases (e.g. ComponentMixin) precede it
        source_class: type = getattr(type(self), "__wrapped__")  # noqa: B009
        source_file = inspect.getsourcefile(source_class)
        if source_file is None:
            raise ValueError(f"cannot locate source file for {source_class.__name__}")
        return Path(source_file)

    def to_dict(self) -> dict[str, Any]:
        """Adds how the component was backtested, e.g. so a vectorized df can't seed live trading."""
        return {
            **super().to_dict(),
            "backtest_mode": self.backtest_mode,
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

    @property
    def _reuse_child_component_signals(self):
        return not self._is_backtestee and self.settings.reuse_signals

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

    def _compute_signals_on_the_fly(self, reason: str):
        """Computes the trading df of a child whose persisted one can't be reused, and persists it.

        Args:
            reason: why the persisted trading df can't be reused, e.g. it isn't stored or it is stale.
        """
        self.logger.warning(
            f"{reason}. Running its vectorized backtest on the fly...",
            style=TextStyle.BOLD + RichColor.YELLOW,
        )
        X = self.X
        signals = cast("Component", self)._signalize(X)
        signals_df = pl.DataFrame(signals)
        trading_df = pl.concat([X, signals_df], how="horizontal")
        self.store.set_df(trading_df)
        self.save_df()

    @override
    def _materialize(self):
        for data_store in self.data_stores.values():
            data_store.materialize()
        if self.backtest_mode == BacktestMode.VECTORIZED:
            if self._is_backtestee:
                # backtestee: only prepare its input, engine._backtest() runs its backtest later
                self.store.set_df(self._build_features_df().to_native())
            else:
                # child component will load its trading_df so that its signals can be merged into its parent's trading df
                try:
                    self.store.materialize()
                except (FileNotFoundError, StaleArtifactError) as err:
                    self.store.set_df(self._build_features_df().to_native())
                    self._compute_signals_on_the_fly(reason=str(err))
            self._assert_split_resolution()
        # FIXME: event-driven should also call self._assert_split_resolution(),
        # brainstorm: let it own trading df the same way as VECTORIZED, empty it before looping
        # blocker: building X needs child signals up front (_build_features_df), but without reuse_signals children only compute them during the loop
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

    @override
    def _build(self) -> None:
        if self._is_dummy_strategy():
            # the dummy strategy only hosts its component, nothing of its own to build
            component = cast("BaseModel | BaseFeature", self.components[0])
            component._build()
            self._is_built = True
        else:
            return super()._build()
