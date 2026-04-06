# pyright: reportUninitializedInstanceVariable=false, reportUnsafeMultipleInheritance=false, reportIncompatibleVariableOverride=false, reportAssignmentType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Any, cast, TypeAlias
if TYPE_CHECKING:
    from narwhals.typing import Frame
    from sklearn.model_selection import TimeSeriesSplit
    from pfund.typing import StrategyT, ModelT, FeatureT, IndicatorT
    from pfund.components.strategies.strategy_base import BaseStrategy
    from pfund.components.models.model_base import BaseModel
    from pfund.datas.data_bar import BarData
    from pfund.utils.dataset_splitter import DatasetSplitsDict, CrossValidatorDatasetPeriods, DatasetPeriods
    from pfund._backtest.backtest_mixin import BacktestMixin
    from pfund.engines.engine_context import DataRangeDict
    from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
    from pfund.engines.engine_context import EngineContext
    from pfund.brokers.crypto.broker import CryptoBroker
    from pfund.brokers.ibkr.broker import InteractiveBrokers
    from pfund.brokers.broker_simulated import SimulatedBroker
    from pfund.datas.stores.market_data_store import BarUpdate
    class SimulatedCryptoBroker(SimulatedBroker, CryptoBroker): ...
    class SimulatedInteractiveBrokers(SimulatedBroker, InteractiveBrokers): ...
    class BacktestEngineContext(EngineContext):
        backtest: BacktestContext
    BacktesteeName: TypeAlias = str

import os
import importlib
from dataclasses import dataclass

import narwhals as nw
from pfeed.enums import DataTool
from pfund_kit.utils.progress_bar import track, ProgressBar
from pfund_kit.style import cprint, RichColor, TextStyle
import pfund as pf
from pfund.enums import BacktestMode, Environment
from pfund.engines.base_engine import BaseEngine
from pfund.utils.dataset_splitter import DatasetSplitter


@dataclass(frozen=True)
class BacktestContext:
    backtest_mode: BacktestMode
    dataset_splitter: DatasetSplitter


class BacktestEngine(BaseEngine):
    _context: BacktestEngineContext
    strategies: dict[str, BaseStrategy]

    def __init__(
        self,
        name: str='engine',
        data_range: str | DataRangeDict | Literal['ytd']='1mo',
        mode: BacktestMode | Literal['vectorized', 'hybrid', 'event_driven']=BacktestMode.VECTORIZED,
        dataset_splits: int | DatasetSplitsDict | TimeSeriesSplit=721,
        cv_test_ratio: float=0.1,
        settings: BacktestEngineSettings | None=None,
        # TODO: add profiling option for event-driven backtesting?
        # profiling: bool=False,
    ):
        '''
        Args:
            cv_test_ratio:
                if passing in a cross-validator in dataset_splits, 
                this is the ratio of the entire dataset to be reserved as a final hold-out test set.
            settings:
                if not provided, settings.toml will be used.
                if provided, will override the settings in settings.toml.
        '''
        from pfund.utils.dataset_splitter import DatasetSplitter
        super().__init__(env=Environment.BACKTEST, name=name, data_range=data_range, settings=settings)
        self._context.backtest = BacktestContext(
            backtest_mode=BacktestMode[mode.upper()],
            dataset_splitter=DatasetSplitter(
                dataset_start=self.data_start,
                dataset_end=self.data_end,
                dataset_splits=dataset_splits, 
                cv_test_ratio=cv_test_ratio
            )
        )
        if self.backtest_mode == BacktestMode.EVENT_DRIVEN:
            # REVIEW:
            if self.settings.reuse_signals:
                if self.settings.assert_signals:
                    raise ValueError('reuse_signals must be False when assert_signals=True in event-driven backtesting')
                cprint(
                    'Warning: Reusing precomputed signals to speed up event-driven backtesting,\n' +
                    'i.e. computing signals on the fly will be skipped',
                    style='bold'
                )
    
    @property
    def settings(self) -> BacktestEngineSettings:
        return cast("BacktestEngineSettings", self._context.settings)
    
    @property
    def backtest_mode(self) -> BacktestMode:
        return self._context.backtest.backtest_mode
    
    @property
    def dataset_periods(self) -> DatasetPeriods | list[CrossValidatorDatasetPeriods]:
        return self._context.backtest.dataset_splitter.dataset_periods
    
    @property
    def _dummy(self) -> str:
        '''gets the name of the dummy strategy'''
        from pfund.components.strategies._dummy_strategy import DummyStrategy
        return DummyStrategy.name
        
    def add_strategy(
        self, 
        strategy: StrategyT, 
        resolution: str, 
        name: str='',
        min_data: int | None=None,
        max_data: int | None=None,
    ) -> StrategyT:
        '''
        Args:
            min_data (int | None): Minimum number of data rows required before the strategy can produce signals.
                When `preload_min_data` is enabled in engine settings, these rows are pre-loaded during materialization
                for event-driven backtesting so the strategy starts warm.
            max_data (int | None): Maximum number of data rows kept in memory.
                Once exceeded, oldest rows are dropped (sliding window). Useful for bounding memory usage.
                If None, all rows are kept (unlimited).
        '''
        from pfund.components.strategies._dummy_strategy import DummyStrategy
        from pfund.components.strategies.strategy_backtest import BacktestStrategy

        if self._dummy in self.strategies:
            raise Exception('adding another strategy is not allowed during model backtesting (i.e. engine.add_model(...) has been called)')
        Strategy = type(strategy)
        if Strategy is not DummyStrategy:
            if name == self._dummy:
                raise ValueError(f'strategy name "{self._dummy}" is reserved, please use another name')
        strategy: StrategyT = BacktestStrategy(Strategy, *strategy.__pfund_args__, **strategy.__pfund_kwargs__)
        return cast("StrategyT", super().add_strategy(
            strategy=strategy,
            resolution=resolution,
            name=name or Strategy.__name__,
            min_data=min_data,
            max_data=max_data,
        ))

    def add_model(
        self, 
        model: ModelT, 
        resolution: str,
        name: str='',
        min_data: int | None=None,
        max_data: int | None=None,
        group_data: bool=True,
        signal_cols: list[str] | None=None,
    ) -> BacktestMixin | ModelT:
        '''Add model without creating a strategy (using dummy strategy)'''
        from pfund.components.strategies._dummy_strategy import DummyStrategy
        only_dummy_strategy_exists = self._dummy in self.strategies and len(self.strategies) == 1
        assert not only_dummy_strategy_exists, 'Please use strategy.add_model(...) instead of engine.add_model(...) when a strategy is already created'
        if not (strategy := self.get_strategy(self._dummy)):
            strategy = self.add_strategy(DummyStrategy(), resolution, name=self._dummy)
            strategy.set_flags(True)
        assert not strategy.models, 'Adding more than 1 model to dummy strategy in backtesting is not supported, you should train and dump your models one by one'
        model = strategy.add_model(
            model,
            name=name,
            min_data=min_data,
            max_data=max_data,
            group_data=group_data,
            signal_cols=signal_cols,
        )
        model.set_flags(True)
        return model
    
    def add_feature(
        self, 
        feature: FeatureT, 
        resolution: str,
        name: str='',
        min_data: int | None=None,
        max_data: int | None=None,
        group_data: bool=True,
        signal_cols: list[str] | None=None,
    ) -> BacktestMixin | FeatureT:
        return self.add_model(
            feature, 
            resolution,
            name=name, 
            min_data=min_data, 
            max_data=max_data, 
            group_data=group_data,
            signal_cols=signal_cols,
        )
    
    def add_indicator(
        self, 
        indicator: IndicatorT, 
        resolution: str,
        name: str='',
        min_data: int | None=None,
        max_data: int | None=None,
        group_data: bool=True,
        signal_cols: list[str] | None=None,
    ) -> BacktestMixin | IndicatorT:
        return self.add_model(
            indicator, 
            resolution,
            name=name, 
            min_data=min_data, 
            max_data=max_data, 
            group_data=group_data,
            signal_cols=signal_cols,
        )
    
    def run(self, num_chunks: int=1, num_cpus: int | None=None) -> dict[str, Any]:
        '''
        num_chunks:
            Number of chunks to split the dataset into.
            if = 1, process the whole dataset all at once.
            if > 1, use Ray for parallel processing.
        num_cpus:
            Maximum number of CPUs (Ray workers) to use per batch, i.e. how many chunks run in parallel at once.
            if None, defaults to os.cpu_count().
            This will be ignored if Ray is not used (i.e. num_chunks = 1).
        '''
        if num_chunks < 1:
            raise ValueError('num_chunks must be greater than 0')
        if num_cpus:
            num_cpus = min(num_cpus, cast(int, os.cpu_count()))
            if num_cpus < 1:
                raise ValueError('num_cpus must be greater than 0')

        super().run()

        backtest_results: dict[BacktesteeName, list[pf.BacktestDataFrame]] = {}

        try:
            for strategy in self.strategies.values():
                if strategy.name == self._dummy:
                    # nothing to run when backtesting model in vectorized or hybrid mode
                    if self.backtest_mode in [BacktestMode.VECTORIZED, BacktestMode.HYBRID]:
                        continue
                    elif self.backtest_mode == BacktestMode.EVENT_DRIVEN:
                        # dummy strategy has exactly one model
                        model: BaseModel = cast("BaseModel", list(strategy.models.values())[0])
                        backtestee = model
                        backtest_dfs = self._backtest(backtestee, num_chunks=num_chunks, num_cpus=num_cpus)
                        backtest_results[backtestee.name] = backtest_dfs
                    else:
                        raise NotImplementedError(f'Backtesting mode {self.backtest_mode} is not supported for model backtesting')
                else:
                    backtestee = strategy
                    backtest_dfs = self._backtest(backtestee, num_chunks=num_chunks, num_cpus=num_cpus)
                    backtest_results[backtestee.name] = backtest_dfs

        except Exception:
            self._logger.exception('Error in backtesting:')
        finally:
            self.end()
        
        self.results = backtest_results
        return backtest_results
    
    def _backtest(
        self, 
        backtestee: BaseStrategy | BaseModel, 
        num_chunks: int=1,
        num_cpus: int | None=None,
    ) -> list[pf.BacktestDataFrame]:       
        ### Pre-Backtest ###
        data_tool: DataTool = self._context.pfeed_config.data_tool
        is_using_ray = num_chunks > 1
        backtest_dfs: list[pf.BacktestDataFrame] = []

        df: Frame = backtestee.get_df(to_native=False)
        if isinstance(df, nw.LazyFrame):
            df = df.collect()
        # REVIEW: still needed for event-driven backtesting?
        if self.backtest_mode == BacktestMode.EVENT_DRIVEN:
            # NOTE: clear dfs so that strategies/models don't know anything about the incoming data
            # FIXME
            # backtestee.clear_dfs()
            pass
    

        def _run_backtest(
            backtestee: BaseStrategy | BaseModel,
            df_chunk: nw.DataFrame[Any],
            backtest_mode: BacktestMode,
            data_tool: DataTool,
            chunk_num: int | None=None,
            batch_num: int | None=None,
        ) -> pf.BacktestDataFrame:
            if backtest_mode in [BacktestMode.VECTORIZED, BacktestMode.HYBRID]:
                BacktestDataFrame = importlib.import_module(f'pfund._backtest.{data_tool.lower()}').BacktestDataFrame
                backtest_df_original = BacktestDataFrame(df_chunk.to_native(), backtest_mode=backtest_mode)
                backtest_df = backtestee.backtest(backtest_df_original)
                if backtestee.is_strategy() and backtest_df is backtest_df_original:
                    cprint(
                        f"WARNING: {backtestee.name} backtest() returned the same df unchanged.\n" +
                        "This is fine if you only used native e.g. Polars/Pandas operations on the original df.\n" +
                        "However, [italic]this is an ERROR[/italic] if you called BacktestDataFrame methods like " +
                        "create_signal(), open_position(), or close_position() —\n" +
                        "these return a new df, so you must reassign: df = df.create_signal(...) and return the new df",
                        style=TextStyle.BOLD + RichColor.RED,
                    )
            elif backtest_mode == BacktestMode.EVENT_DRIVEN:
                backtest_df = BacktestEngine._event_driven_loop(
                    backtestee=backtestee, 
                    df_chunk=df_chunk, 
                    chunk_num=chunk_num, 
                    batch_num=batch_num
                )
            else:
                raise ValueError(f'{backtest_mode} is not supported')
            return backtest_df
        
        
        ### Backtest ###
        if not is_using_ray:
            backtest_df: pf.BacktestDataFrame = _run_backtest(
                backtestee=backtestee,
                df_chunk=df, 
                backtest_mode=self.backtest_mode,
                data_tool=data_tool,
            )
            backtest_dfs.append(backtest_df)
        else:
            import ray
            from ray.util.queue import Queue
            from pfeed.utils.ray import shutdown_ray, setup_ray, setup_logger_in_ray_task, ray_logging_context

            @ray.remote
            def ray_task(
                log_queue: Queue,
                backtestee_ref: ray.ObjectRef[BaseStrategy | BaseModel],
                df_chunk: nw.DataFrame[Any],
                backtest_mode: BacktestMode,
                data_tool: DataTool,
                chunk_num: int,
                batch_num: int,
            ):
                backtestee = ray.get(backtestee_ref)
                logger = setup_logger_in_ray_task(backtestee.logger.name, log_queue)
                try:
                    backtest_df: pf.BacktestDataFrame = _run_backtest(
                        backtestee=backtestee,
                        df_chunk=df_chunk, 
                        backtest_mode=backtest_mode,
                        data_tool=data_tool,
                        chunk_num=chunk_num,
                        batch_num=batch_num,
                    )
                    return backtest_df
                except Exception:
                    logger.exception(f'Error in Backtest-Chunk{chunk_num}-Batch{batch_num}:')
                    return None
            
            df_chunks: list[tuple[nw.DataFrame[Any], int]] = []
            total_rows = df.shape[0]
            chunk_size: int = total_rows // num_chunks
            for chunk_num, row_offset in enumerate(range(0, total_rows, chunk_size)):
                df_chunk: nw.DataFrame[Any] = df[row_offset : row_offset + chunk_size]
                df_chunks.append((df_chunk, chunk_num))
            
            logger = backtestee.logger
            self._logger.debug('setting up ray...')
            setup_ray()
            backtestee_ref: ray.ObjectRef[BaseStrategy | BaseModel] = ray.put(backtestee)
            with ray_logging_context(logger) as log_queue:
                try:
                    num_cpus = num_cpus or os.cpu_count()
                    if num_cpus is None:
                        raise ValueError('num_cpus must be set when using Ray')
                    batch_size: int = min(num_cpus, num_chunks)
                    batches = [
                        df_chunks[i: i + batch_size] 
                        for i in range(0, len(df_chunks), batch_size)
                    ]
                    with ProgressBar(
                        total=len(batches),
                        description=f'Backtesting {backtestee.name} ({batch_size} chunks per batch)', 
                    ) as pbar:
                        for batch_num, batch in enumerate(batches):
                            futures = [
                                ray_task.remote(
                                    log_queue=log_queue,   # pyright: ignore[reportCallIssue]
                                    backtestee_ref=backtestee_ref,
                                    df_chunk=df_chunk,
                                    backtest_mode=self.backtest_mode,
                                    data_tool=data_tool,
                                    chunk_num=chunk_num,
                                    batch_num=batch_num
                                )
                                for df_chunk, chunk_num in batch
                            ]
                            backtest_dfs_in_batch: list[pf.BacktestDataFrame | None] = ray.get(futures)
                            backtest_dfs_in_batch_not_none: list[pf.BacktestDataFrame] = [backtest_df for backtest_df in backtest_dfs_in_batch if backtest_df is not None]
                            backtest_dfs.extend(backtest_dfs_in_batch_not_none)
                            if len(backtest_dfs_in_batch_not_none) != len(batch):
                                logger.warning(f'Some backtesting tasks in batch-{batch_num} failed, check {logger.name}.log for details')
                            pbar.advance(1)
                except KeyboardInterrupt:
                    self._logger.warning(f"KeyboardInterrupt received, stopping {backtestee.name} backtesting...")
                except Exception:
                    logger.exception('Error in backtesting:')
            self._logger.debug('shutting down ray...')
            shutdown_ray()

        # ### Post-Backtest ###
        if backtestee.is_strategy():
            backtest_dfs: list[pf.BacktestDataFrame] = [
                backtestee._postprocess_backtest_df(backtest_df)
                for backtest_df in backtest_dfs
            ]
        return backtest_dfs

    @staticmethod
    def _event_driven_loop(
        backtestee: BaseStrategy | BaseModel,
        df_chunk: nw.DataFrame[Any], 
        chunk_num: int | None=None, 
        batch_num: int | None=None,
    ) -> pf.BacktestDataFrame:
        if chunk_num is not None and batch_num is not None:
            description = f'Backtest-Chunk{chunk_num}-Batch{batch_num}'
        else:
            description = ''

        # OPTIMIZE: critical loop
        for row in track(
            df_chunk.iter_rows(named=False),
            total=df_chunk.shape[0],
            description=description,
            bar_style=RichColor.BRIGHT_YELLOW,
        ):
            ts, resolution, product_name, symbol, source_type, o, h, l, c, v = row  # pyright: ignore[reportGeneralTypeIssues, reportUnusedVariable]  # noqa: E741
            databoy = backtestee.databoy
            data = cast("BarData", backtestee.get_data(product_name, resolution))
            update: BarUpdate = {
                'ts': ts,
                'open': o,
                'high': h,
                'low': l,
                'close': c,
                'volume': v,
                'is_incremental': False,
                'msg_ts': None,
                'extra_data': {}
            }
            databoy._update_bar(data, update)
        
        # TODO: return the backtested dataframe, how?
        # return ...
