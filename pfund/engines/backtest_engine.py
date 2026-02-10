# pyright: reportUninitializedInstanceVariable=false, reportUnsafeMultipleInheritance=false, reportIncompatibleVariableOverride=false
from __future__ import annotations
from typing import TYPE_CHECKING, Literal, cast
if TYPE_CHECKING:
    import pandas as pd
    from sklearn.model_selection import TimeSeriesSplit
    from pfund.typing import StrategyT, ModelT, FeatureT, IndicatorT
    from pfund.brokers.broker_base import BaseBroker
    from pfund.components.strategies.strategy_base import BaseStrategy
    from pfund.components.models.model_base import BaseModel
    from pfund.utils.dataset_splitter import DatasetSplitsDict, CrossValidatorDatasetPeriods, DatasetPeriods
    from pfund._backtest.backtest_mixin import BacktestMixin
    from pfund.engines.engine_context import DataRangeDict
    from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
    from pfund.engines.engine_context import EngineContext
    from pfund.entities.accounts.account_base import BaseAccount
    from pfund.brokers.crypto.broker import CryptoBroker
    from pfund.brokers.ibkr.broker import InteractiveBrokers
    from pfund.brokers.broker_simulated import SimulatedBroker
    class SimulatedCryptoBroker(SimulatedBroker, CryptoBroker): ...
    class SimulatedInteractiveBrokers(SimulatedBroker, InteractiveBrokers): ...
    class BacktestEngineContext(EngineContext):
        backtest: BacktestContext


import os
import inspect
import time
from dataclasses import dataclass

import polars as pl

from pfund_kit.utils.progress_bar import track, ProgressBar
from pfund_kit.style import cprint, RichColor
from pfund.enums import BacktestMode, ComponentType, Environment
from pfund.engines.base_engine import BaseEngine
from pfund.utils.dataset_splitter import DatasetSplitter


@dataclass(frozen=True)
class BacktestContext:
    backtest_mode: BacktestMode
    dataset_splitter: DatasetSplitter


class BacktestEngine(BaseEngine):
    def __init__(
        self,
        data_range: str | DataRangeDict | Literal['ytd']='1mo',
        mode: Literal['vectorized', 'hybrid', 'event_driven']='vectorized',
        dataset_splits: int | DatasetSplitsDict | TimeSeriesSplit=721,
        cv_test_ratio: float=0.1,
    ):
        '''
        Args:
            cv_test_ratio:
                if passing in a cross-validator in dataset_splits, 
                this is the ratio of the entire dataset to be reserved as a final hold-out test set.
        '''
        from pfund.utils.dataset_splitter import DatasetSplitter
        super().__init__(env=Environment.BACKTEST, data_range=data_range)
        cast("BacktestEngineContext", self._context).backtest = BacktestContext(
            backtest_mode=BacktestMode[mode.lower()],
            dataset_splitter=DatasetSplitter(
                dataset_start=self.data_start,
                dataset_end=self.data_end,
                dataset_splits=dataset_splits, 
                cv_test_ratio=cv_test_ratio
            )
        )
        if self.backtest_mode == BacktestMode.event_driven and self.settings.reuse_signals:
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
        return cast("BacktestEngineContext", self._context).backtest.backtest_mode
    
    @property
    def dataset_periods(self) -> DatasetPeriods | list[CrossValidatorDatasetPeriods]:
        return cast("BacktestEngineContext", self._context).backtest.dataset_splitter.dataset_periods
    
    @property
    def _dummy(self) -> str:
        '''gets the name of the dummy strategy'''
        from pfund.components.strategies._dummy_strategy import _DummyStrategy
        return _DummyStrategy.name
        
    def add_strategy(self, strategy: StrategyT, resolution: str, name: str='') -> StrategyT:
        '''
        Args:
            ray_actor_options:
                Options for Ray actor.
                will be passed to ray actor like this: Actor.options(**ray_options).remote(**ray_kwargs)
        '''
        from pfund.components.strategies._dummy_strategy import _DummyStrategy
        from pfund.components.strategies.strategy_backtest import BacktestStrategy

        if self._dummy in self.strategies:
            raise Exception('adding another strategy is not allowed during model backtesting (i.e. engine.add_model(...) has been called)')
        Strategy = type(strategy)
        if Strategy is not _DummyStrategy:
            if name == self._dummy:
                raise ValueError(f'strategy name "{self._dummy}" is reserved, please use another name')
        name = name or Strategy.__name__
        strategy: StrategyT = BacktestStrategy(Strategy, *strategy.__pfund_args__, **strategy.__pfund_kwargs__)
        return cast("StrategyT", super().add_strategy(strategy=strategy, resolution=resolution, name=name))

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
        from pfund.components.strategies._dummy_strategy import _DummyStrategy
        only_dummy_strategy_exists = self._dummy in self.strategies and len(self.strategies) == 1
        assert not only_dummy_strategy_exists, 'Please use strategy.add_model(...) instead of engine.add_model(...) when a strategy is already created'
        if not (strategy := self.get_strategy(self._dummy)):
            strategy = self.add_strategy(_DummyStrategy(), resolution, name=self._dummy)
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
    
    def _assert_backtest_function(self, backtestee: BaseStrategy | BaseModel):
        assert self.backtest_mode == BacktestMode.vectorized, 'assert_backtest_function() is only for vectorized backtesting'
        if not hasattr(backtestee, 'backtest'):
            raise Exception(f'class "{backtestee.name}" does not have backtest() method, cannot run vectorized backtesting')
        sig = inspect.signature(backtestee.backtest)
        params = list(sig.parameters.values())
        if not params or params[0].name != 'df':
            raise Exception(f'{backtestee.name} backtest() must have "df" as its first arg, i.e. backtest(self, df)')
    
    def run(self, num_chunks: int=1):
        '''
        num_chunks:
            number of chunks to split the dataset into for parallel processing.
            if > 1, will use ray for parallel processing and num_chunks = num_cpus in ray's task.
            if = 1, will process the dataset sequentially.
        '''
        from pfund.components.strategies._dummy_strategy import _DummyStrategy
        
        super().run()
        assert num_chunks > 0, 'num_chunks must be greater than 0'
        # TODO: to be refactored
        # if num_chunks > 1 and not self._use_ray:
        #     self._logger.warning('num_chunks > 1 but ray is not enabled, chunks will be processed sequentially')
        # for broker in self.brokers.values():
        #     broker.start()
        # self.strategy_manager.start()
        # backtest_results = {}
        # error = ''
        # try:
        #     for strat, strategy in self.strategies.items():
        #         backtestee = strategy
        #         if strat == _DummyStrategy.name:
        #             if self.backtest_mode == BacktestMode.vectorized:
        #                 continue
        #             elif self.backtest_mode == BacktestMode.event_driven:
        #                 # dummy strategy has exactly one model
        #                 model = list(strategy.models.values())[0]
        #                 backtestee = model
        #         backtest_result: dict = self._backtest(backtestee, num_chunks=num_chunks, ray_kwargs=ray_kwargs)
        #         backtest_results.update(backtest_result)
        #     # if only one backtest is run, return the backtest result without backtestee's name
        #     if len(backtest_results) == 1:
        #         backtest_results = backtest_results[backtestee.name]
        # except Exception as err:
        #     error = str(err)
        #     self._logger.exception('Error in backtesting:')
        # finally:
        #     self.end(reason=error)
        
        # return backtest_results

    def _backtest(self, backtestee: BaseStrategy | BaseModel, num_chunks: int=1) -> dict:       
        backtest_result = {}
        dtl = backtestee.dtl
        df = backtestee.get_df(copy=True)
        
        # Pre-Backtesting
        if self.backtest_mode == BacktestMode.vectorized:
            self._assert_backtest_function(backtestee)
            df_chunks = []
        elif self.backtest_mode == BacktestMode.event_driven:
            # NOTE: clear dfs so that strategies/models don't know anything about the incoming data
            backtestee.clear_dfs()
        else:
            raise NotImplementedError(f'Backtesting mode {self.backtest_mode} is not supported')
        
        
        # Backtesting
        if not self._use_ray:
            progress_desc = f'Backtesting {backtestee.name} (per chunk)'
            progress_bar = track(total=num_chunks, description=progress_desc)
        else:
            ray_tasks = []
            if 'num_cpus' not in ray_kwargs:
                ray_kwargs['num_cpus'] = os.cpu_count()
            num_cpus = ray_kwargs['num_cpus']
            if num_cpus > num_chunks:
                num_chunks = num_cpus
                print(f'num_chunks is adjusted to {num_cpus} because {num_cpus=}')
        start_time = time.time()
        for chunk_num, df_chunk in enumerate(dtl.iterate_df_by_chunks(df, num_chunks=num_chunks)):
            if self._use_ray:
                ray_tasks.append((df_chunk, chunk_num))
            else:
                if self.backtest_mode == BacktestMode.vectorized:
                    df_chunk = dtl.preprocess_vectorized_df(df_chunk)
                    backtestee.backtest(df_chunk)
                    df_chunks.append(df_chunk)
                elif self.backtest_mode == BacktestMode.event_driven:
                    df_chunk = dtl.preprocess_event_driven_df(df_chunk)
                    self._event_driven_backtest(df_chunk, chunk_num=chunk_num)
                progress_bar.advance(1)
            
        if self._use_ray:
            import ray
            from ray.util.queue import Queue
            from pfeed.utils.ray import shutdown_ray, setup_logger_in_ray_task, ray_logging_context
            
            @ray.remote
            def _run_task(log_queue: Queue,  _df_chunk: pd.DataFrame | pl.LazyFrame, _chunk_num: int, _batch_num: int):
                try:
                    logger =setup_logger_in_ray_task(backtestee.logger.name, log_queue)
                    if self.backtest_mode == BacktestMode.vectorized:
                        _df_chunk = dtl.preprocess_vectorized_df(_df_chunk, backtestee)
                        backtestee.backtest(_df_chunk)
                    elif self.backtest_mode == BacktestMode.event_driven:
                        _df_chunk = dtl.preprocess_event_driven_df(_df_chunk)
                        self._event_driven_backtest(_df_chunk, chunk_num=_chunk_num, batch_num=_batch_num)
                except Exception:
                    logger.exception(f'Error in backtest-chunk{_chunk_num}-batch{_batch_num}:')
                    return False
                return True

            logger = backtestee.logger
            with ray_logging_context(logger) as log_queue:
                try:
                    batch_size = ray_kwargs['num_cpus']  # FIXME: replace with num_chunks
                    batches = [ray_tasks[i: i + batch_size] for i in range(0, len(ray_tasks), batch_size)]
                    with ProgressBar(
                        total=len(batches),
                        description=f'Backtesting {backtestee.name} ({batch_size} chunks per batch)', 
                    ) as progress_bar:
                        for batch_num, batch in enumerate(batches):
                            futures = [_run_task.remote(log_queue, *task, batch_num) for task in batch]
                            results = ray.get(futures)
                            if not all(results):
                                logger.warning(f'Some backtesting tasks in batch{batch_num} failed, check {logger.name}.log for details')
                            progress_bar.advance(1)
                except Exception:
                    logger.exception('Error in backtesting:')
            self._logger.debug('shutting down ray...')
            shutdown_ray()
        end_time = time.time()
        cprint(f'Backtest elapsed time: {end_time - start_time:.3f}(s)', style='bold')
        
        
        # Post-Backtesting
        if backtestee.component_type == ComponentType.strategy:
            if self.backtest_mode == BacktestMode.vectorized:
                df = dtl.postprocess_vectorized_df(df_chunks)
            # TODO
            elif self.backtest_mode == BacktestMode.event_driven:
                pass
            backtest_history: dict = self.history.create(backtestee, df, start_time, end_time)
            backtest_result[backtestee.name] = backtest_history
        return backtest_result

    def _event_driven_backtest(self, df_chunk, chunk_num=0, batch_num=0):
        COMMON_COLS = ['ts', 'product', 'resolution', 'broker', 'is_quote', 'is_tick']
        if isinstance(df_chunk, pl.LazyFrame):
            df_chunk = df_chunk.collect().to_pandas()
        
        # OPTIMIZE: critical loop
        for row in track(
            df_chunk.itertuples(index=False), 
            total=df_chunk.shape[0], 
            description=f'Backtest-Chunk{chunk_num}-Batch{batch_num} (per row)', 
            bar_style=RichColor.BRIGHT_YELLOW.value,
        ):
            # TODO: don't use product objects, use product name instead
            # users should use self.product to get the product object
            # i.e. self.product will create the product object when needed?
            
            ts, product, resolution = row.ts, row.product, row.resolution
            # FIXME: broker is not in row anymore, find a way to get broker, product, resolution from the data path instead
            data_manager = self.brokers[row.broker].data_manager
            # TODO: move quote and tick to separate functions
            if row.is_quote:
                # TODO
                raise NotImplementedError('Quote data is not supported in event-driven backtesting yet')
                quote = {}
                data_manager._update_quote(product, quote)
            elif row.is_tick:
                # TODO
                raise NotImplementedError('Tick data is not supported in event-driven backtesting yet')
                tick = {}
                data_manager._update_tick(product, tick)
            else:
                bar_cols = ['open', 'high', 'low', 'close', 'volume']
                bar = {
                    'resolution': resolution,
                    'data': {
                        'ts': ts,
                        'open': row.open,
                        'high': row.high,
                        'low': row.low,
                        'close': row.close,
                        'volume': row.volume,
                    },
                    'extra_data': {
                        col: getattr(row, col) for col in row._fields
                        if col not in COMMON_COLS + bar_cols
                    },
                }
                data_manager._update_bar(product, bar, is_incremental=False)
    
    def end(self, reason: str=''):
        for strat in list(self.strategies):
            self.strategy_manager.stop(strat, reason=reason or 'finished backtesting')
            self.remove_strategy(strat)
        for broker in list(self.brokers.values()):
            broker.stop()
            self.remove_broker(broker.name)
