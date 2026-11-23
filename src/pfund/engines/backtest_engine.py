from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Generator

    from mtflow.tracking.run import MTFlowRun
    from pfund_blocks.resolution import Resolution

    from pfund.backtest._mixins.backtest_mixin import BacktestMixin
    from pfund.backtest._mixins.strategy_mixin import BacktestStrategyMixin
    from pfund.backtest.cv.dataset_splits import DatasetSplits
    from pfund.components.features.feature_base import BaseFeature
    from pfund.components.models.model_base import BaseModel, UnderlyingModel
    from pfund.components.strategies.strategy_base import BaseStrategy
    from pfund.engines.base_engine import DataRangeDict
    from pfund.typing import Component

    type BacktesteeName = str
    type ChunkNum = int

import os
from contextlib import contextmanager

import polars as pl
from pfund_blocks.enums.env import Environment
from pfund_kit.utils.progress_bar import ProgressBar

from pfund.components.strategies._dummy_strategy import _DummyStrategy
from pfund.engines.base_engine import BaseEngine
from pfund.engines.contexts.backtest_engine_context import BacktestEngineContext
from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
from pfund.enums import BacktestMode


def _split_df_into_time_chunks(
    df: pl.DataFrame,
    num_chunks: int,
) -> list[tuple[pl.DataFrame, ChunkNum]]:
    """Splits a date-sorted df into contiguous, timestamp-safe chunks.

    The unique dates are split like np.array_split (the extra dates go to the first chunks),
    so the rows of one date never end up in two chunks, e.g. in long form.
    """
    if num_chunks < 1:
        raise ValueError("num_chunks must be greater than 0")
    if "date" not in df.columns:
        raise ValueError("chunked backtesting requires a 'date' column")

    date = df.get_column("date")
    if date.is_empty():
        raise ValueError("chunked backtesting requires at least one row")
    if date.has_nulls():
        raise ValueError("chunked backtesting does not support null dates")
    if not date.is_sorted():
        raise ValueError("the dataframe must be sorted by 'date' before chunking")

    # 0, 0, 1, 1, 2, ...: the index of each row's date among the unique dates
    time_ids = date.rle_id()
    num_times = cast("int", time_ids[-1]) + 1
    if num_chunks > num_times:
        raise ValueError(
            f"num_chunks ({num_chunks}) cannot exceed the number of unique "
            + f"timestamps ({num_times})"
        )

    # chunk k covers the dates [time_bounds[k], time_bounds[k + 1])
    q, r = divmod(num_times, num_chunks)
    time_bounds = [k * q + min(k, r) for k in range(num_chunks + 1)]
    # the first row of each bound's date, the last bound (num_times) maps to len(df)
    row_bounds = time_ids.search_sorted(
        pl.Series(time_bounds, dtype=time_ids.dtype), side="left"
    ).to_list()
    return [
        (df.slice(row_bounds[k], row_bounds[k + 1] - row_bounds[k]), k)
        for k in range(num_chunks)
    ]


@contextmanager
def _without_dfs(component: Component) -> Generator[None]:
    """Temporarily detaches the dfs of the component tree, i.e. its data stores' dfs and store._df, children included.

    Used around ray.put() in chunked backtesting: otherwise every Ray worker in flight holds a copy of the
    full dataset (~67MB per product-year of 1m bars, ~4GB at 1s), though a chunk task only uses its df_chunk.
    So inside a chunk task, the backtestee's own dfs are None,
    i.e. a user's backtest() must use the df passed in, not self.data_df/self.df.
    The dfs are restored on exit.
    """
    detached: list[tuple[Any, pl.DataFrame | None]] = []

    def _detach(comp: Component) -> None:
        for store in [comp.store, *comp.data_stores.values()]:
            detached.append((store, store._df))
            store._df = None
        for child in comp.components:
            # components always run locally in backtesting, never as ActorProxy
            _detach(cast("Component", child))

    try:
        _detach(component)
        yield
    finally:
        for store, df in detached:
            store._df = df


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
        self.result: pl.DataFrame | None = None

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
        run: MTFlowRun | None = None,
        num_chunks: int = 1,
        num_cpus: int | None = None,
    ) -> pl.DataFrame | None:
        """
        Args:
            run:
                mtflow run to track this backtest, each run is written to its own folder (e.g. run001, run002).
                if None, the reused default_run/ folder is overwritten.
            num_chunks:
                Number of standalone strategy backtests to split the dataset into.
                if <= 1, process the whole dataset all at once (values below 1 are treated as 1).
                if > 1, use Ray for parallel processing.
                Chunking is not supported for model or feature backtesting,
                num_chunks is set to 1 with a warning.
                When chunked, the strategy's dfs (e.g. self.data_df, self.df) are None inside its backtest(),
                use the df passed in.
            num_cpus:
                Maximum number of chunks (Ray tasks) in flight at once, i.e. how many chunks run in parallel.
                if None, defaults to os.cpu_count().
                This will be ignored if Ray is not used (i.e. num_chunks = 1).

        Returns:
            The backtest df, all chunks concatenated in chunk order, with each row's chunk in the CHUNK_COL column.
            Use df.partition_by(CHUNK_COL, as_dict=True) to get the per-chunk dfs.
            None if there is nothing to return, e.g. model training.
        """
        num_cpus = (os.cpu_count() or 1) if num_cpus is None else max(1, num_cpus)
        if self._get_dummy_strategy() and num_chunks > 1:
            self._logger.warning(
                f"Chunking is only supported for strategy backtesting, num_chunks={num_chunks} is set to 1"
            )
            num_chunks = 1
        # NOTE: before super().run(), which sets up the components, so their metadata records it, see BacktestMixin.to_dict()
        self.context.set_num_chunks(num_chunks)

        super().run(run=run)
        backtest_result: pl.DataFrame | None = None
        try:
            # NOTE: only one strategy exists in backtesting
            strategy = cast("BaseStrategy", next(iter(self._strategies.values())))
            is_dummy_strategy = (strategy.name == self._dummy_strategy_name)
            if is_dummy_strategy:
                # dummy strategy has exactly one model or one feature
                if strategy.models:
                    model: BaseModel = cast(
                        "BaseModel", next(iter(strategy.models.values()))
                    )
                    if model.is_training():
                        if self.backtest_mode == BacktestMode.EVENT_DRIVEN:
                            raise ValueError(
                                f"No trained model found for '{model.name}', event-driven backtesting needs a trained model, "
                                + "train it in a vectorized backtest first and call save()"
                            )
                        self.result = backtest_result
                        return backtest_result
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
            backtest_result = self._backtest(backtestee, num_cpus=num_cpus)

        except Exception:
            self._logger.exception("Error in backtesting:")
            raise
        finally:
            self.end()

        self.result = backtest_result
        return backtest_result

    def _backtest(self, backtestee: Component, num_cpus: int) -> pl.DataFrame:
        ### Pre-Backtest ###
        num_chunks = self.context.num_chunks
        is_using_ray = num_chunks > 1
        is_strategy = backtestee.is_strategy()
        # each chunk starts from a fresh state, so a feature/model would warm up again at every chunk boundary,
        # and without the chunk_num column (strategy-only) its artifact would look like one continuous run
        if is_using_ray and not is_strategy:
            raise ValueError(
                f"{backtestee} cannot be backtested in chunks (num_chunks={num_chunks}), only strategies can. "
                + "Run it with num_chunks=1."
            )

        def _run_backtest(
            backtestee: Component,
            df_chunk: pl.DataFrame,
            chunk_num: ChunkNum = 0,
        ) -> pl.DataFrame:
            _backtestee = cast("BacktestMixin", backtestee)
            if _backtestee.backtest_mode == BacktestMode.VECTORIZED:
                backtest_df = _backtestee._vectorized_backtest(df_chunk)
            elif _backtestee.backtest_mode == BacktestMode.EVENT_DRIVEN:
                backtest_df = _backtestee._event_driven_backtest(df_chunk)
            else:
                raise ValueError(f"Unknown backtest mode: {_backtestee.backtest_mode}")
            # only strategy BACKTEST data may carry the column (see data_handler._validate_schema),
            # chunks are standalone backtests, the column tells them apart in the stored data artifact;
            # an unchunked run is chunk 0, so every BACKTEST strategy df has the same schema
            if not backtestee.is_strategy():
                return backtest_df
            return backtest_df.with_columns(
                pl.lit(chunk_num).alias(BacktestEngine.CHUNK_COL)
            )

        ### Backtest ###
        backtest_dfs: list[pl.DataFrame] = []
        if self.backtest_mode == BacktestMode.EVENT_DRIVEN:
            df = backtestee.data_df
        else:
            df = backtestee.trading_df if not is_strategy else backtestee.full_df

        if not is_using_ray:
            backtest_dfs.append(_run_backtest(backtestee=backtestee, df_chunk=df))
        else:
            from itertools import islice

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
                # passed in as an ObjectRef, Ray resolves top-level ObjectRef args before the task runs
                backtestee: Component,
                df_chunk: pl.DataFrame,
                chunk_num: ChunkNum,
            ) -> pl.DataFrame:
                logger = setup_logger_in_ray_task(backtestee.logger.name, log_queue)
                try:
                    return _run_backtest(
                        backtestee=backtestee,
                        df_chunk=df_chunk,
                        chunk_num=chunk_num,
                    )
                except Exception:
                    # log it to the log file, then re-raise: ray.get() raises it in the driver,
                    # which cancels the other chunks, so a partial set of chunks is never persisted
                    logger.exception(f"Error in Backtest-Chunk{chunk_num}:")
                    raise

            df_chunks: list[tuple[pl.DataFrame, ChunkNum]] = (
                _split_df_into_time_chunks(df, num_chunks)
            )

            self._logger.debug("setting up ray...")
            setup_ray()
            # serialize the backtestee into the object store once, instead of once per chunk,
            # each task still deserializes its own copy, so chunks don't share state.
            # its dfs are left out, see _without_dfs()
            with _without_dfs(backtestee):
                backtestee_ref: ray.ObjectRef[Component] = ray.put(backtestee)

            logger = backtestee.logger
            with ray_logging_context(logger) as log_queue:
                unfinished: list[ray.ObjectRef] = []
                try:
                    pending = iter(df_chunks)
                    # chunks finish in any order, keep the results in chunk order,
                    # chunks are contiguous date ranges, so the combined df stays sorted by (chunk_num, date)
                    ref_to_chunk_num: dict[ray.ObjectRef, ChunkNum] = {}
                    # placeholders, every slot is filled before the chunks are combined
                    results: list[pl.DataFrame] = cast(
                        "list[pl.DataFrame]", [None] * len(df_chunks)
                    )

                    def submit_next(
                        df_chunk: pl.DataFrame, chunk_num: ChunkNum
                    ) -> ray.ObjectRef:
                        # Ray's .remote() only takes positional arguments (for type checkers)
                        ref = ray_task.remote(
                            log_queue, backtestee_ref, df_chunk, chunk_num
                        )
                        ref_to_chunk_num[ref] = chunk_num
                        return ref

                    # sliding window: keep num_cpus chunks in flight, start the next chunk
                    # as soon as one finishes, so a slow chunk doesn't leave the other workers idle
                    unfinished = [
                        submit_next(*item) for item in islice(pending, num_cpus)
                    ]
                    with ProgressBar(
                        total=len(df_chunks),
                        description=f"Backtesting {backtestee.name} ({len(df_chunks)} chunks, {min(num_cpus, len(df_chunks))} in parallel)",
                    ) as pbar:
                        while unfinished:
                            finished, unfinished = ray.wait(unfinished, num_returns=1)
                            results[ref_to_chunk_num[finished[0]]] = ray.get(
                                finished[0]
                            )
                            pbar.advance(1)
                            if (item := next(pending, None)) is not None:
                                unfinished.append(submit_next(*item))
                except BaseException:
                    # e.g. KeyboardInterrupt or a crashed worker: cancel the in-flight chunks,
                    # otherwise they keep running after the backtest returns,
                    # and re-raise, so a partial set of chunks is never persisted as the backtestee's results.
                    # force=True kills the worker, a non-forced cancel doesn't stop a task that's already running
                    for ref in unfinished:
                        ray.cancel(ref, force=True)
                    raise
            backtest_dfs = results

        ### Post-Backtest ###
        if is_strategy:
            backtest_dfs = [
                cast("BacktestStrategyMixin", backtestee)._postprocess_backtest_df(backtest_df)
                for backtest_df in backtest_dfs
            ]
        # combine the chunks into one df, ordered by (chunk_num, date): chunks are contiguous date ranges,
        # and the results are kept in chunk order.
        backtest_df = pl.concat(backtest_dfs, how="vertical")
        if self.backtest_mode == BacktestMode.VECTORIZED:
            trading_store = backtestee.store
            trading_store._df = backtest_df
            trading_store.persist_to_datalake()
        return backtest_df
