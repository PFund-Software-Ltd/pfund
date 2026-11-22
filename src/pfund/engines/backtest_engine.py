from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

if TYPE_CHECKING:
    from mtflow.tracking.run import MTFlowRun
    from narwhals.typing import IntoDataFrame
    from pfund_blocks.resolution import Resolution

    from pfund.backtest._mixins.backtest_mixin import BacktestMixin
    from pfund.backtest.cv.dataset_splits import DatasetSplits
    from pfund.components.features.feature_base import BaseFeature
    from pfund.components.models.model_base import BaseModel, UnderlyingModel
    from pfund.components.strategies.strategy_base import BaseStrategy
    from pfund.engines.base_engine import DataRangeDict

    type BacktesteeName = str

import os

import narwhals as nw
import numpy as np
from pfund_blocks.enums.env import Environment
from pfund_kit.utils.progress_bar import ProgressBar

from pfund.components.strategies._dummy_strategy import _DummyStrategy
from pfund.engines.base_engine import BaseEngine
from pfund.engines.contexts.backtest_engine_context import BacktestEngineContext
from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
from pfund.enums import BacktestMode


def _split_df_into_time_chunks(
    df: nw.DataFrame[Any],
    num_chunks: int,
) -> list[tuple[nw.DataFrame[Any], int]]:
    """Split a sorted dataframe into contiguous, timestamp-safe chunks."""
    if num_chunks < 1:
        raise ValueError("num_chunks must be greater than 0")
    if "date" not in df.columns:
        raise ValueError("chunked backtesting requires a 'date' column")

    date = df.get_column("date")
    if len(date) == 0:
        raise ValueError("chunked backtesting requires at least one row")
    if date.is_null().any():
        raise ValueError("chunked backtesting does not support null dates")
    if not date.is_sorted():
        raise ValueError("the dataframe must be sorted by 'date' before chunking")

    row_dates = date.to_numpy()
    starts_new_time = np.empty(len(row_dates), dtype=bool)
    starts_new_time[0] = True
    starts_new_time[1:] = row_dates[1:] != row_dates[:-1]
    time_starts = np.flatnonzero(starts_new_time)
    num_times = len(time_starts)
    if num_chunks > num_times:
        raise ValueError(
            f"num_chunks ({num_chunks}) cannot exceed the number of unique "
            + f"timestamps ({num_times})"
        )

    row_boundaries = np.append(time_starts, len(row_dates))
    time_chunks = np.array_split(np.arange(num_times), num_chunks)
    return [
        (
            df[
                int(row_boundaries[time_chunk[0]]) : int(
                    row_boundaries[time_chunk[-1] + 1]
                )
            ],
            chunk_num,
        )
        for chunk_num, time_chunk in enumerate(time_chunks)
    ]


class BacktestEngine(BaseEngine[BacktestEngineSettings, BacktestEngineContext]):
    Context = BacktestEngineContext
    # column telling apart the standalone backtests (chunks) in a strategy's backtest df, 0 for an unchunked run
    CHUNK_COL: ClassVar[str] = "chunk_num"

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
        dataset_splits: DatasetSplits | None = None,
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
        self.results: list[IntoDataFrame] | None = None

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
        feature = cast(
            "FeatureT",
            dummy_strategy.add_feature(
                feature=feature,
                resolution=resolution,
                name=name,
                df_form=df_form,
            ),
        )
        cast("BacktestMixin", feature).set_top_component()
        return feature

    def run(
        self,
        *,
        overwrite: bool = False,
        run: MTFlowRun | None = None,
        num_chunks: int = 1,
        num_cpus: int | None = None,
    ) -> list[IntoDataFrame]:
        """
        Args:
            overwrite: If True, clear the reused run path (the default_run/ folder) before running.
                Ignored when an mtflow run is active, since each mtflow run gets its own folder.
            num_chunks:
                Number of standalone strategy backtests to split the dataset into.
                if = 1, process the whole dataset all at once.
                if > 1, use Ray for parallel processing.
                Chunking is not supported for model or feature backtesting.
            num_cpus:
                Maximum number of CPUs (Ray workers) to use per batch, i.e. how many chunks run in parallel at once.
                if None, defaults to os.cpu_count().
                This will be ignored if Ray is not used (i.e. num_chunks = 1).
        """
        if num_chunks < 1:
            raise ValueError("num_chunks must be greater than 0")
        if num_cpus is not None:
            if num_cpus < 1:
                raise ValueError("num_cpus must be greater than 0")
            available_cpus = os.cpu_count()
            if available_cpus is not None:
                num_cpus = min(num_cpus, available_cpus)
        if num_chunks > 1 and _DummyStrategy.__name__ in self._strategies:
            raise ValueError("Chunking is only supported for strategy backtesting")

        # before super().run(), which sets up the components, so their metadata records it, see BacktestMixin.to_dict()
        self.context.num_chunks = num_chunks
        super().run(overwrite=overwrite, run=run)

        backtest_results: list[IntoDataFrame] = []

        try:
            # NOTE: only one strategy exists in backtesting
            strategy = cast("BaseStrategy", next(iter(self._strategies.values())))
            is_dummy_strategy = strategy.name == _DummyStrategy.__name__
            if is_dummy_strategy:
                # dummy strategy has exactly one model or one feature
                if strategy.models:
                    model: BaseModel = cast(
                        "BaseModel", next(iter(strategy.models.values()))
                    )
                    if (
                        self.backtest_mode == BacktestMode.VECTORIZED
                        and model.is_training()
                    ):
                        self.result = backtest_results
                        return backtest_results
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
            backtest_dfs = self._backtest(
                backtestee,
                num_chunks=num_chunks,
                num_cpus=num_cpus,
            )
            backtest_results = backtest_dfs

        except Exception:
            self._logger.exception("Error in backtesting:")
        finally:
            self.end()

        self.results = backtest_results
        return backtest_results

    def _backtest(
        self,
        backtestee: BaseStrategy | BaseModel | BaseFeature,
        num_chunks: int = 1,
        num_cpus: int | None = None,
    ) -> list[IntoDataFrame]:
        ### Pre-Backtest ###
        is_using_ray = num_chunks > 1
        backtest_dfs: list[IntoDataFrame] = []

        if self.backtest_mode == BacktestMode.EVENT_DRIVEN:
            df = nw.from_native(backtestee.data_df)
        else:
            if not backtestee.is_strategy():
                df = nw.from_native(backtestee.trading_df)
            else:
                df = nw.from_native(backtestee.full_df)

        def _run_backtest(
            backtestee: BaseStrategy | BaseModel | BaseFeature,
            df_chunk: nw.DataFrame[Any],
            chunk_num: int | None = None,
            batch_num: int | None = None,
        ) -> IntoDataFrame:
            backtest_df = backtestee.backtest(df_chunk.to_native())
            backtest_df.chunk_num = chunk_num
            backtest_df.batch_num = batch_num
            # NOTE: written by Claude (Claude Code)
            # FIXME: chunk_num above is a df attribute, not a column: it is lost on any new df
            # (e.g. _postprocess_backtest_df() below, a Ray round-trip), and PFundComponentDataHandler
            # needs it as a column to tell the chunks apart in the stored data artifact.
            # an unchunked run is chunk 0, so every BACKTEST strategy df has the same schema.
            # remove the two attribute lines above once nothing reads them (nothing does today).
            backtest_df = (
                nw.from_native(backtest_df)
                .with_columns(nw.lit(chunk_num or 0).alias(BacktestEngine.CHUNK_COL))
                .to_native()
            )
            return backtest_df

        ### Backtest ###
        if not is_using_ray:
            backtest_df: IntoDataFrame = _run_backtest(
                backtestee=backtestee, df_chunk=df
            )
            backtest_dfs.append(backtest_df)
        else:
            import ray
            from pfund_kit.utils.ray import (
                ray_logging_context,
                setup_logger_in_ray_task,
                setup_ray,
            )
            from ray.util.queue import Queue

            @ray.remote
            def ray_task(
                log_queue: Queue,
                backtestee_ref: ray.ObjectRef[BaseStrategy | BaseModel],
                df_chunk: nw.DataFrame[Any],
                chunk_num: int,
                batch_num: int,
            ):
                backtestee = ray.get(backtestee_ref)
                logger = setup_logger_in_ray_task(backtestee.logger.name, log_queue)
                try:
                    backtest_df: IntoDataFrame = _run_backtest(
                        backtestee=backtestee,
                        df_chunk=df_chunk,
                        chunk_num=chunk_num,
                        batch_num=batch_num,
                    )
                    return backtest_df
                except Exception:
                    logger.exception(
                        f"Error in Backtest-Chunk{chunk_num}-Batch{batch_num}:"
                    )
                    return None

            df_chunks = _split_df_into_time_chunks(df, num_chunks)

            logger = backtestee.logger
            self._logger.debug("setting up ray...")
            setup_ray()
            backtestee_ref: ray.ObjectRef[BaseStrategy | BaseModel] = ray.put(
                backtestee
            )
            with ray_logging_context(logger) as log_queue:
                try:
                    num_cpus = num_cpus or os.cpu_count()
                    if num_cpus is None:
                        raise ValueError("num_cpus must be set when using Ray")
                    batch_size: int = min(num_cpus, num_chunks)
                    batches = [
                        df_chunks[i : i + batch_size]
                        for i in range(0, len(df_chunks), batch_size)
                    ]
                    with ProgressBar(
                        total=len(batches),
                        description=f"Backtesting {backtestee.name} ({batch_size} chunks per batch)",
                    ) as pbar:
                        for batch_num, batch in enumerate(batches):
                            futures = [
                                ray_task.remote(
                                    log_queue=log_queue,
                                    backtestee_ref=backtestee_ref,
                                    df_chunk=df_chunk,
                                    chunk_num=chunk_num,
                                    batch_num=batch_num,
                                )
                                for df_chunk, chunk_num in batch
                            ]
                            backtest_dfs_in_batch: list[IntoDataFrame | None] = ray.get(
                                futures
                            )
                            backtest_dfs_in_batch_not_none: list[IntoDataFrame] = [
                                backtest_df
                                for backtest_df in backtest_dfs_in_batch
                                if backtest_df is not None
                            ]
                            backtest_dfs.extend(backtest_dfs_in_batch_not_none)
                            if len(backtest_dfs_in_batch_not_none) != len(batch):
                                logger.warning(
                                    f"Some backtesting tasks in batch-{batch_num} failed, check {logger.name}.log for details"
                                )
                            pbar.advance(1)
                except KeyboardInterrupt:
                    self._logger.warning(
                        f"KeyboardInterrupt received, stopping {backtestee.name} backtesting..."
                    )
                except Exception:
                    logger.exception("Error in backtesting:")

        # ### Post-Backtest ###
        if backtestee.is_strategy():
            backtest_dfs: list[IntoDataFrame] = [
                backtestee._postprocess_backtest_df(backtest_df)
                for backtest_df in backtest_dfs
            ]
            # NOTE: written by Claude (Claude Code)
            # FIXME: the top strategy's backtest results are never persisted.
            # PFundComponentDataHandler writes component._data_artifact, i.e. store._df on THIS (driver's) component,
            # but with chunking, each chunk ran on a Ray worker against a copy of the component (ray.put(backtestee)),
            # so the results only exist in backtest_dfs, store._df never sees them;
            # and BacktestMixin.backtest() only persists for components without a user backtest(), not the top strategy.
            # so: combine the chunks into one df (ordered by (chunk_num, date): chunks are contiguous date ranges,
            # ray.get() keeps the submission order) and persist it as the strategy's data artifact.
            # failed chunks are already dropped from backtest_dfs, so chunk_num can have gaps, e.g. 0, 1, 3.
            # only vectorized: in event-driven mode, BaseStrategy._data_artifact raises NotImplementedError.
            if self.backtest_mode == BacktestMode.VECTORIZED and backtest_dfs:
                backtestee.store._df = nw.concat(
                    [nw.from_native(backtest_df) for backtest_df in backtest_dfs],
                    how="vertical",
                ).to_native()
                backtestee.store.persist_to_datalake()
        return backtest_dfs
