from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    import polars as pl
    from mtflow.tracking.run import MTFlowRun
    from pfund_blocks.resolution import Resolution

    from pfund.backtest._mixins.backtest_mixin import BacktestMixin
    from pfund.backtest.cv.dataset_splits import (
        DatasetSplits,
        DatasetSplitsInput,
        FoldNum,
        SplitPart,
    )
    from pfund.backtest.cv.indexing import SplittableData
    from pfund.components.features.feature_base import BaseFeature
    from pfund.components.models.model_base import BaseModel, UnderlyingModel
    from pfund.components.strategies.strategy_base import BaseStrategy
    from pfund.engines.base_engine import DataRangeDict
    from pfund.typing import Component

    type BacktesteeName = str

from pfund_blocks.enums import DataCategory, Environment

from pfund.components.strategies._dummy_strategy import _DummyStrategy
from pfund.engines.base_engine import BaseEngine
from pfund.engines.contexts.backtest_engine_context import BacktestEngineContext
from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
from pfund.enums import BacktestMode


class _SplitScope:
    """Returned by BacktestEngine.use_split(): the split is already set, exiting a `with` block clears it."""

    def __init__(self, engine: BacktestEngine):
        self._engine = engine

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info: object) -> None:
        self._engine.clear_split()


class BacktestEngine(BaseEngine[BacktestEngineSettings, BacktestEngineContext]):
    Context = BacktestEngineContext

    def __init__(
        self,
        *,
        name: str = "engine",
        data_range: str
        | Resolution
        | DataRangeDict
        | tuple[str, str]
        | Literal["ytd"] = "1mo",
        settings: BacktestEngineSettings | None = None,
        mode: BacktestMode
        | Literal["vectorized", "event_driven"] = BacktestMode.VECTORIZED,
        dataset_splits: DatasetSplitsInput | None = None,
        # TODO:
        # profiling: bool=False,
    ):
        """
        Args:
            name: engine name
            data_range: range of data to be used for the engine,
                when it is a string, it is a resolution, e.g. '1m', '1d', '1w', '1mo', '1y'
                when it is a dict, it is a dict with keys 'start_date' and 'end_date',
                    e.g. {'start_date': '2024-01-01', 'end_date': '2024-12-31'}
                when it is a tuple, it is (start_date, end_date),
                    e.g. ('2024-01-01', '2024-12-31')
            settings:
                if not provided, settings.toml will be used.
                if provided, will override the settings in settings.toml.
            dataset_splits:
                how to split the dataset into train/val/test sets, e.g. a
                three-digit ratio such as 721, a ratio dict, Holdout, or an
                sklearn-compatible cross-validator. Defaults to None, i.e. 100% train
                (same as 100), where the train set is in use without engine.use_split().
                With any other split, select a split with engine.use_split() before
                accessing a component's X or saving a model.
                Dataset splitting is only available in vectorized mode.
        """
        env = Environment.BACKTEST
        super().__init__(
            env=env,
            name=name,
            data_range=data_range,
            settings=settings,
            mode=mode,
            dataset_splits=dataset_splits,
        )
        if self.dataset_splits.is_training_set_only():
            self.context.set_active_split(('train', None))

    @property
    def backtest_mode(self) -> BacktestMode:
        return self.context.mode

    @property
    def dataset_splits(self) -> DatasetSplits:
        return self.context.dataset_splits

    @property
    def _dummy_strategy_name(self) -> str:
        return _DummyStrategy.__name__

    def add_strategy[StrategyT: BaseStrategy](
        self,
        strategy: StrategyT,
        resolution: str,
        name: str = "",
        # NOTE: ray args are ignored in backtesting, same as BacktestMixin._add_component
        ray_actor_options: dict[str, Any] | None = None,
        *,
        num_chunks: int = 1,
        num_cpus: float | None = None,
        **ray_kwargs: Any,
    ) -> StrategyT:
        """Add a strategy as the top-level backtesting component.

        Backtesting supports one top-level strategy per engine.

        Args:
            strategy: Strategy instance to backtest.
            resolution: Resolution at which the strategy runs.
            name: Optional name for the strategy. Defaults to its class name.
            num_chunks:
                Number of standalone backtests strategy.backtest(X) splits X into.
                if <= 1, process the whole df all at once (values below 1 are treated as 1).
                if > 1, use Ray for parallel processing.
                When chunked, the strategy's dfs (e.g. self.data_df, self.df) are None inside its backtest(),
                use the df passed in.
            num_cpus:
                Maximum number of chunks (Ray tasks) in flight at once, i.e. how many chunks run in parallel.
                if None, defaults to os.cpu_count(). A float is floored, e.g. 2.5 -> 2 (min. 1).
                Typed as float to match BaseEngine.add_strategy(), where it is the Ray actor's num_cpus.
                This will be ignored if Ray is not used (i.e. num_chunks = 1).
                Unlike in TradeEngine.add_strategy(), it does NOT run the strategy as a Ray actor.

        Returns:
            The strategy configured for backtesting.
        """
        from pfund.backtest._mixins.strategy_mixin import BacktestStrategyMixin

        if dummy_strategy := self._get_dummy_strategy():
            raise Exception(
                f"{next(iter({**dummy_strategy.models, **dummy_strategy.features}))} is already being backtested, "
                + "cannot add a strategy to the engine. "
            )

        if self._strategies:
            raise Exception(
                f"strategy {next(iter(self._strategies))} already exists, only one strategy is allowed in backtesting"
            )

        if ray_actor_options or ray_kwargs:
            self._logger.warning(
                f"ray_actor_options and ray_kwargs are ignored in backtesting, strategy {name or strategy.name} will run locally"
            )

        strategy = BacktestStrategyMixin._adopt(strategy)
        cast("BacktestStrategyMixin", strategy)._set_chunking(num_chunks, num_cpus)
        strategy = super().add_strategy(strategy=strategy, resolution=resolution, name=name)
        cast("BacktestMixin", strategy).set_as_backtestee()
        return strategy

    def _create_dummy_strategy(self, resolution: str) -> _DummyStrategy:
        if self._strategies:
            raise ValueError(
                f"strategy {next(iter(self._strategies))} is already being backtested, cannot add a top-level model/feature to the engine. "
                + "Did you mean strategy.add_model/add_feature(...) instead of engine.add_model/add_feature(...)?"
            )
        return self.add_strategy(
            _DummyStrategy(),
            resolution,
            name=self._dummy_strategy_name,
        )

    @property
    def _backtestee(self) -> Component:
        """The one component being backtested: the strategy, or the model/feature hosted by the dummy strategy."""
        if not self._strategies:
            raise ValueError(
                "Nothing to backtest, add a strategy/model/feature to the engine first"
            )
        if dummy_strategy := self._get_dummy_strategy():
            # the dummy strategy has exactly one model or one feature
            components = {**dummy_strategy.models, **dummy_strategy.features}
            if not components:
                raise ValueError("No model or feature to backtest")
            return cast("Component", next(iter(components.values())))
        # NOTE: only one strategy exists in backtesting
        return cast("Component", next(iter(self._strategies.values())))

    def _get_dummy_strategy(self) -> _DummyStrategy | None:
        if self._dummy_strategy_name in self._strategies:
            dummy_strategy = cast(
                "_DummyStrategy", self.get_strategy(self._dummy_strategy_name)
            )
            return dummy_strategy
        else:
            return None

    def _assert_no_component_in_dummy_strategy(self):
        if (dummy_strategy := self._get_dummy_strategy()) and (
            dummy_strategy.models or dummy_strategy.features
        ):
            raise ValueError(
                f"{next(iter({**dummy_strategy.models, **dummy_strategy.features}))} is already being backtested, "
                + "only one top-level model/feature is allowed in backtesting. "
                + "Backtest (train) and save your models/features one by one, "
                + "or add them to a strategy via strategy.add_model/add_feature(...)"
            )

    def add_model[ModelT: BaseModel](
        self,
        model: ModelT | UnderlyingModel,
        resolution: str,
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
    ) -> ModelT:
        """Add a model as the top-level backtesting component.

        The model is hosted by an internal dummy strategy. Only one top-level
        model or feature may be added per engine.

        Args:
            model: Model instance, or supported underlying model, to backtest.
            resolution: Resolution at which the model runs.
            name: Optional name for the model.
            df_form: Dataframe layout used by the model.

        Returns:
            The model configured for backtesting.
        """
        dummy_strategy = self._get_dummy_strategy() or self._create_dummy_strategy(
            resolution
        )
        self._assert_no_component_in_dummy_strategy()
        model = cast(
            "ModelT",
            dummy_strategy.add_model(
                model=model,
                resolution=resolution,
                name=name,
                df_form=df_form,
            ),
        )
        cast("BacktestMixin", model).set_as_backtestee()
        return model

    def add_feature[FeatureT: BaseFeature](
        self,
        feature: FeatureT,
        resolution: str,
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
    ) -> FeatureT:
        """Add a feature as the top-level backtesting component.

        The feature is hosted by an internal dummy strategy. Only one top-level
        model or feature may be added per engine.

        Args:
            feature: Feature instance to backtest.
            resolution: Resolution at which the feature runs.
            name: Optional name for the feature.
            df_form: Dataframe layout used by the feature.

        Returns:
            The feature configured for backtesting.
        """
        dummy_strategy = self._get_dummy_strategy() or self._create_dummy_strategy(
            resolution
        )
        self._assert_no_component_in_dummy_strategy()
        feature = dummy_strategy.add_feature(
            feature=feature,
            resolution=resolution,
            name=name,
            df_form=df_form,
        )
        cast("BacktestMixin", feature).set_as_backtestee()
        return feature

    def use_split(self, part: SplitPart, fold: FoldNum | None = None) -> _SplitScope:
        """Sets the trading df of every component to its rows in one split part, so X (its features) holds that split.

        pfund's internal flow reads X (features df, backtest()), so it then works on that
        split without passing X_train etc. around. Any other data row-aligned with X
        (a target y, sample weights, ...) is split with apply_split().
        The split is recorded, so model.save() stamps what the model was trained on.

        Use it as a context manager, so the split can't leak into the code after it:

            engine.build()
            with engine.use_split("train", fold=k):
                model.fit(model.X, engine.apply_split(y))
                model.save()
            with engine.use_split("val", fold=k):
                model.predict(model.X)

        or call it plainly (e.g. across notebook cells) and end it with clear_split().

        Args:
            part: ``train``/``val`` of a fold, ``trainval`` (everything before test) or ``test``.
            fold: required for ``train``/``val`` under cross-validation,
                optional (only 0) under holdout.
        """
        if not self._is_built:
            raise RuntimeError(
                f"{self.name} has not built yet, call engine.build() first"
            )
        try:
            cast("BacktestMixin", self._backtestee)._use_split(part, fold)
        except Exception:
            # e.g. a child has no rows in the part: undo the components already split
            self.clear_split()
            raise
        self.context.set_active_split((part, fold))
        return _SplitScope(self)

    def clear_split(self) -> None:
        """Sets the trading df of every component back to the whole data range, ending use_split().

        The split in use goes back to the default: train if it's the only part, otherwise none.
        A no-op if use_split() isn't in use.
        """
        cast("BacktestMixin", self._backtestee)._clear_split()
        if self.dataset_splits.is_training_set_only():
            self.context.set_active_split(('train', None))
        else:
            self.context.set_active_split(None)

    def apply_split[DataT: SplittableData](self, data: DataT) -> DataT:
        """The rows of `data` in the split set by use_split(), i.e. row-aligned with X.

        `data` needs no `date` column of its own: it only has to be row-aligned 1:1 with
        the backtestee's trading df over the whole data range (same length and order),
        e.g. a target y, class labels or sample weights built from model.df before use_split().

            y = model.df.get_column(close).shift(-1) / model.df.get_column(close) - 1.0
            with engine.use_split("train"):
                model.fit(model.X, engine.apply_split(y))

        Args:
            data: a DataFrame, Series, or ndarray row-aligned 1:1 with the whole trading df.
        """
        from pfund.backtest.cv.indexing import validate_aligned

        part, fold = self.context.active_split
        backtestee = cast("BacktestMixin", self._backtestee)
        validate_aligned(data, backtestee._get_trading_df())
        split_data = backtestee._take_split(data, part, fold)
        # use_split() has already rejected an empty split part
        assert split_data is not None
        return split_data

    def run(
        self,
        *,
        run: MTFlowRun | None = None,
    ) -> pl.DataFrame | None:
        is_event_driven = self.backtest_mode == BacktestMode.EVENT_DRIVEN
        try:
            # NOTE: inside the try, so the engine still ends if the component setup fails
            super().run(run=run)

            # # EXTEND: support non-bar data types:
            # brainstorm: heapq.merge(market_data_df, news_data_df), with a dispatcher to create data updates for each data type
            # TODO: write trading_df (trading_df.delta) and compare it with the one generated by vectorized backtesting
            # TODO: need to build a clock? coz looping bars has time gaps
            # TODO: support dataset_splits, pass in df into run()?
            from pfund_kit.style import RichColor
            from pfund_kit.utils.progress_bar import track

            if is_event_driven:
                backtestee = self._backtestee
                market_data_df = backtestee.get_df(kind="data", data_category=DataCategory.MARKET_DATA)

                # OPTIMIZE: critical loop
                for row in track(
                    market_data_df.iter_rows(named=False),
                    total=market_data_df.shape[0],
                    description="",
                    bar_style=RichColor.BRIGHT_YELLOW,
                    disable=not self.context.config.show_progress_bar,
                ):
                    ts, resolution, product_name, source_type, o, h, l, c, v = row
                    # data = cast("BarData", self.get_data(product_name, resolution))
                    # update: BarUpdate = {
                    #     "ts": ts,
                    #     "open": o,
                    #     "high": h,
                    #     "low": l,
                    #     "close": c,
                    #     "volume": v,
                    #     "is_incremental": False,
                    #     "msg_ts": None,
                    #     "extra": {},
                    # }
                    # self._databoy.collect(...)
        except Exception:
            self._logger.exception("Error in backtesting:")
            raise
        finally:
            self.end()
        return self._backtestee.df if is_event_driven else None
