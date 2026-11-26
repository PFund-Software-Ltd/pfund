from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from mtflow.tracking.run import MTFlowRun
    from pfund_blocks.resolution import Resolution

    from pfund.backtest._mixins.backtest_mixin import BacktestMixin
    from pfund.backtest.cv.dataset_splits import DatasetSplitsInput, SplitPart
    from pfund.components.features.feature_base import BaseFeature
    from pfund.components.models.model_base import BaseModel, UnderlyingModel
    from pfund.components.strategies.strategy_base import BaseStrategy
    from pfund.engines.base_engine import DataRangeDict

    type BacktesteeName = str

from pfund_blocks.enums.env import Environment

from pfund.components.strategies._dummy_strategy import _DummyStrategy
from pfund.engines.base_engine import BaseEngine
from pfund.engines.contexts.backtest_engine_context import BacktestEngineContext
from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
from pfund.enums import BacktestMode


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
                sklearn-compatible cross-validator. Defaults to None (no split),
                i.e. the complete dataset is the training data. Dataset
                splitting is only available in vectorized mode.
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

    @property
    def backtest_mode(self) -> BacktestMode:
        return self.context.mode

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
        **ray_kwargs: Any,
    ) -> StrategyT:
        """Add a strategy as the top-level backtesting component.

        Backtesting supports one top-level strategy per engine.

        Args:
            strategy: Strategy instance to backtest.
            resolution: Resolution at which the strategy runs.
            name: Optional name for the strategy. Defaults to its class name.

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
        strategy = cast(
            "StrategyT",
            super().add_strategy(strategy=strategy, resolution=resolution, name=name),
        )
        cast("BacktestMixin", strategy).set_top_component()
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
        cast("BacktestMixin", model).set_top_component()
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
        cast("BacktestMixin", feature).set_top_component()
        return feature

    def use_split(self, part: SplitPart, fold: int | None = None) -> None:
        """Sets X (the trading df) of every component to its rows in one split part.

        pfund's internal flow reads X (features df, save_df(), backtest()), so it then works on that
        split without passing X_train etc. around. Data from component.split(data) has to be
        row-aligned with the current X.

            engine.use_split("train", fold=k)
            model.fit(model.X, y_train)
            engine.use_split("val", fold=k)
            model.save_df(model.predict(model.X))

        Args:
            part: ``train``/``val`` of a fold, ``trainval`` (everything before test) or ``test``.
            fold: required for ``train``/``val`` under cross-validation,
                optional (only 0) under holdout.
        """
        if not self._strategies:
            raise ValueError("Nothing to split, add a strategy/model/feature first")
        # NOTE: only one strategy exists in backtesting, it carries the whole component tree
        backtestee: Any = next(iter(self._strategies.values()))
        if dummy_strategy := self._get_dummy_strategy():
            # the dummy strategy has no trading df of its own, split the one model/feature it hosts
            backtestee = next(
                iter({**dummy_strategy.models, **dummy_strategy.features}.values())
            )
        cast("BacktestMixin", backtestee)._use_split(part, fold)

    def run(
        self,
        *,
        run: MTFlowRun | None = None,
    ) -> None:
        """
        Args:
            run:
                mtflow run to track this backtest, each run is written to its own folder (e.g. run001, run002).
                if None, the reused default_run/ folder is overwritten.

        Returns:
            The backtest df.
            None if there is nothing to return, e.g. model training.
        """
        try:
            # NOTE: inside the try, so the engine still ends if the component setup fails
            super().run(run=run)
            # NOTE: in vectorized backtesting, engine only prepares the df for users, nothing to run
            if self.backtest_mode != BacktestMode.EVENT_DRIVEN:
                return
            # NOTE: only one strategy exists in backtesting
            strategy = cast("BaseStrategy", next(iter(self._strategies.values())))
            is_dummy_strategy = strategy.name == self._dummy_strategy_name
            if is_dummy_strategy:
                # dummy strategy has exactly one model or one feature
                if strategy.models:
                    model: BaseModel = cast(
                        "BaseModel", next(iter(strategy.models.values()))
                    )
                    if model.is_training():
                        raise ValueError(
                            f"No trained model found for '{model.name}', event-driven backtesting needs a trained model, "
                            + "train it in a vectorized backtest first and call save()"
                        )
                    backtestee = model
                elif strategy.features:
                    feature: BaseFeature = cast(
                        "BaseFeature", next(iter(strategy.features.values()))
                    )
                    backtestee = feature
                else:
                    raise ValueError("No model or feature to backtest")
            else:
                backtestee = strategy
            # # EXTEND: support non-bar data types:
            # brainstorm: heapq.merge(market_data_df, news_data_df), with a dispatcher to create data updates for each data type
            # TODO: write trading_df (trading_df.delta) and compare it with the one generated by vectorized backtesting
            # TODO: need to build a clock? coz looping bars has time gaps
            # TODO: support dataset_splits, pass in df into run()?
        #         from pfund_kit.style import RichColor
        #         from pfund_kit.utils.progress_bar import track
        #
        #         # OPTIMIZE: critical loop
        #         for row in track(
        #             df.iter_rows(named=False),
        #             total=df.shape[0],
        #             description=...,
        #             bar_style=RichColor.BRIGHT_YELLOW,
        #             disable=not self.context.config.show_progress_bar,
        #         ):
        #             ts, resolution, product_name, source_type, o, h, l, c, v = row
        #             data = cast("BarData", self.get_data(product_name, resolution))
        #             update: BarUpdate = {
        #                 "ts": ts,
        #                 "open": o,
        #                 "high": h,
        #                 "low": l,
        #                 "close": c,
        #                 "volume": v,
        #                 "is_incremental": False,
        #                 "msg_ts": None,
        #                 "extra": {},
        #             }
        #             self._databoy.collect(...)
        except Exception:
            self._logger.exception("Error in backtesting:")
            raise
        finally:
            self.end()
