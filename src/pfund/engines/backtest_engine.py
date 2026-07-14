# pyright: reportUninitializedInstanceVariable=false, reportUnsafeMultipleInheritance=false, reportIncompatibleVariableOverride=false, reportAssignmentType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast, ClassVar

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame
    from sklearn.model_selection import TimeSeriesSplit
    from mtflow.contexts.backtest_context import BacktestContext

    from pfund.components.models.model_base import BaseModel, UnderlyingModel
    from pfund.components.strategies.strategy_base import BaseStrategy
    from pfund.components.features.feature_base import BaseFeature
    from pfund.datas.resolution import Resolution
    from pfund.engines.base_engine import DataRangeDict
    from pfund.typing import FeatureT, ModelT, StrategyT
    from pfund._backtest.cv.base import CrossValidatorDatasetPeriods
    from pfund._backtest.dataset_splitter import DatasetPeriods, DatasetSplitsDict

    BacktesteeName: TypeAlias = str

import os

import narwhals as nw
from pfund_kit.utils.progress_bar import ProgressBar
from pfeed.storages.storage_config import StorageConfig

from pfund.engines.base_engine import BaseEngine
from pfund.engines.contexts.backtest_engine_context import BacktestEngineContext
from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
from pfund.enums import BacktestMode, Environment


class BacktestEngine(BaseEngine[BacktestEngineSettings, BacktestEngineContext]):
    Context: ClassVar[type[BacktestEngineContext]] = BacktestEngineContext

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
        storage_config: StorageConfig | None = None,
        mode: BacktestMode
        | Literal["vectorized", "event_driven"] = BacktestMode.VECTORIZED,
        dataset_splits: int | DatasetSplitsDict | TimeSeriesSplit = 721,
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
            storage_config:
                where the engine persists its own state storage (e.g. pfund.db), and
                the default inherited by every component added under this engine for
                their artifacts. Overridable per-component via
                add_strategy(..., storage_config=...) / add_model(...).
                If not provided, a default StorageConfig() (local storage) is used.
        """
        env = Environment.BACKTEST
        # NOTE: create context first to set up config by engine name before super().__init__()
        self._context = self._create_context(
            env=env,
            name=name,
            data_range=data_range,
            settings=settings,
            storage_config=storage_config,
            mode=mode,
            dataset_splits=dataset_splits,
        )
        super().__init__(env=self.env, name=self.name)
        self.results: list[IntoDataFrame] | None = None

    @property
    def backtest_mode(self) -> BacktestMode:
        return self._context.mode

    @property
    def dataset_periods(self) -> DatasetPeriods | list[CrossValidatorDatasetPeriods]:
        return self._context.dataset_periods

    def add_strategy(
        self,
        strategy: StrategyT,
        resolution: str,
        name: str = "",
        df_form: Literal["wide", "long"] = "long",
        storage_config: StorageConfig | None = None,
    ) -> StrategyT:
        from pfund.components.strategies._dummy_strategy import _DummyStrategy
        from pfund.components.strategies.strategy_backtest import BacktestStrategy

        dummy_strategy_name = _DummyStrategy.__name__
        if dummy_strategy_name in self._strategies:
            raise Exception(
                "adding another strategy is not allowed during model/feature backtesting"
            )
        elif self._strategies:
            raise Exception(
                f"strategy {list(self._strategies)[0]} already exists, only one strategy is allowed in backtesting"
            )
        Strategy = type(strategy)
        if Strategy is not _DummyStrategy:
            if name == dummy_strategy_name:
                raise ValueError(
                    f'strategy name "{dummy_strategy_name}" is reserved, please use another name'
                )
        strategy: StrategyT = BacktestStrategy(
            Strategy, *strategy.__pfund_args__, **strategy.__pfund_kwargs__
        )
        return cast(
            "StrategyT",
            super().add_strategy(
                strategy=strategy,
                resolution=resolution,
                name=name or Strategy.__name__,
                df_form=df_form,
                storage_config=storage_config,
            ),
        )

    def _add_component(
        self,
        component: ModelT | FeatureT,
        resolution: str,
        name: str,
        df_form: Literal["wide", "long"],
        storage_config: StorageConfig | None,
    ) -> ModelT | FeatureT:
        """Add model without creating a strategy (using dummy strategy)"""
        from pfund.components.strategies._dummy_strategy import _DummyStrategy

        dummy_strategy_name = _DummyStrategy.__name__
        only_dummy_strategy_exists = dummy_strategy_name in self._strategies
        assert not only_dummy_strategy_exists, (
            "Please use strategy.add_model/add_feature(...) instead of engine.add_model/add_feature(...) when a strategy is already created"
        )
        if dummy_strategy_name not in self._strategies:
            strategy = self.add_strategy(
                _DummyStrategy(),
                resolution,
                name=dummy_strategy_name,
            )
        strategy = self.get_strategy(dummy_strategy_name)
        if strategy.models or strategy.features:
            raise ValueError(
                "Adding more than 1 model/feature in backtesting is not supported, "
                + "you should (train) and save your models/features one by one"
            )
        component = strategy._add_component(
            component=component,
            resolution=resolution,
            name=name,
            df_form=df_form,
            storage_config=storage_config,
        )
        return component

    def add_model(
        self,
        model: ModelT | UnderlyingModel,
        resolution: str,
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        storage_config: StorageConfig | None = None,
    ) -> ModelT:
        from pfund.components.models.wrap import wrap_model

        return self._add_component(
            component=cast("ModelT", wrap_model(model)),
            resolution=resolution,
            name=name,
            df_form=df_form,
            storage_config=storage_config,
        )

    def add_feature(
        self,
        feature: FeatureT,
        resolution: str = "",
        name: str = "",
        df_form: Literal["wide", "long"] = "wide",
        storage_config: StorageConfig | None = None,
    ) -> FeatureT:
        return self._add_component(
            component=feature,
            resolution=resolution,
            name=name,
            df_form=df_form,
            storage_config=storage_config,
        )

    def run(
        self,
        ctx: BacktestContext | None = None,
        num_chunks: int = 1,
        num_cpus: int | None = None,
    ) -> list[IntoDataFrame]:
        """
        Args:
            num_chunks:
                Number of chunks to split the dataset into.
                if = 1, process the whole dataset all at once.
                if > 1, use Ray for parallel processing.
            num_cpus:
                Maximum number of CPUs (Ray workers) to use per batch, i.e. how many chunks run in parallel at once.
                if None, defaults to os.cpu_count().
                This will be ignored if Ray is not used (i.e. num_chunks = 1).
        """
        from pfund.components.strategies._dummy_strategy import _DummyStrategy

        if num_chunks < 1:
            raise ValueError("num_chunks must be greater than 0")
        if num_cpus:
            num_cpus = min(num_cpus, cast(int, os.cpu_count()))
            if num_cpus < 1:
                raise ValueError("num_cpus must be greater than 0")

        super().run(ctx=ctx)

        backtest_results: list[IntoDataFrame] = []

        try:
            # NOTE: only one strategy exists in backtesting
            strategy = cast("BaseStrategy", list(self._strategies.values())[0])
            is_dummy_strategy = strategy.name == _DummyStrategy.__name__
            if is_dummy_strategy:
                # dummy strategy has exactly one model or one feature
                if strategy.models:
                    model: BaseModel = cast(
                        "BaseModel", list(strategy.models.values())[0]
                    )
                    backtestee = model
                elif strategy.features:
                    feature: BaseFeature = cast(
                        "BaseFeature", list(strategy.features.values())[0]
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
            return backtest_df

        ### Backtest ###
        if not is_using_ray:
            backtest_df: IntoDataFrame = _run_backtest(
                backtestee=backtestee, df_chunk=df
            )
            backtest_dfs.append(backtest_df)
        else:
            import ray
            from pfeed.utils.ray import (
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

            df_chunks: list[tuple[nw.DataFrame[Any], int]] = []
            total_rows = df.shape[0]
            chunk_size: int = total_rows // num_chunks
            for chunk_num, row_offset in enumerate(range(0, total_rows, chunk_size)):
                df_chunk: nw.DataFrame[Any] = df[row_offset : row_offset + chunk_size]
                df_chunks.append((df_chunk, chunk_num))

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
                                    log_queue=log_queue,  # pyright: ignore[reportCallIssue]
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
        return backtest_dfs
