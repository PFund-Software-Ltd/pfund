from __future__ import annotations

import os
import hashlib
import inspect
import time
import datetime
import json
import uuid
import logging
from logging.handlers import QueueHandler, QueueListener

from tqdm import tqdm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.types.common_literals import tSUPPORTED_BACKTEST_MODES, tSUPPORTED_DATA_TOOLS
    from pfund.types.core import tStrategy, tModel, tFeature, tIndicator
    from pfund.models.model_base import BaseModel
    
try:
    import pandas as pd
    import polars as pl
except ImportError:
    pass

import pfund as pf
from pfund.git_controller import GitController
from pfund.engines.base_engine import BaseEngine
from pfund.brokers.broker_backtest import BacktestBroker
from pfund.strategies.strategy_base import BaseStrategy
from pfund.strategies.strategy_backtest import BacktestStrategy
from pfund.config_handler import ConfigHandler
from pfund.utils import utils
from pfund.mixins.backtest import BacktestMixin


class BacktestEngine(BaseEngine):
    def __new__(
        cls, *, env: str='BACKTEST', data_tool: tSUPPORTED_DATA_TOOLS='pandas', mode: tSUPPORTED_BACKTEST_MODES='vectorized', 
        config: ConfigHandler | None=None,
        use_signal_df=True,
        auto_git_commit=False,
        save_backtests=False,
        num_chunks=1,
        use_ray=False,
        num_cpus=8,
        **settings
    ):
        if not hasattr(cls, 'mode'):
            cls.mode = mode.lower()
        # NOTE: use_signal_df=True means model's prepared signals will be reused in model.next()
        # instead of recalculating the signals. This will make event-driven backtesting faster but less consistent with live trading
        if not hasattr(cls, 'use_signal_df'):
            cls.use_signal_df = use_signal_df
        if not hasattr(cls, 'auto_git_commit'):
            cls.auto_git_commit = auto_git_commit
        if not hasattr(cls, 'save_backtests'):
            cls.save_backtests = save_backtests
        if not hasattr(cls, 'num_chunks'):
            cls.num_chunks = num_chunks
        if not hasattr(cls, 'use_ray'):
            cls.use_ray = use_ray
            if use_ray:
                logical_cpus = os.cpu_count()
                cls.num_cpus = min(num_cpus, logical_cpus)
                if cls.num_cpus > cls.num_chunks:
                    cls.num_chunks = cls.num_cpus
                    print(f'num_chunks is adjusted to {num_cpus} because {num_cpus=}')
        return super().__new__(cls, env, data_tool=data_tool, config=config, **settings)

    def __init__(
        self, *, env: str='BACKTEST', data_tool: tSUPPORTED_DATA_TOOLS='pandas', mode: tSUPPORTED_BACKTEST_MODES='vectorized', 
        config: ConfigHandler | None=None,
        use_signal_df=True,
        auto_git_commit=False,
        save_backtests=False,
        num_chunks=1,
        use_ray=False,
        num_cpus=8,
        **settings
    ):
        # avoid re-initialization to implement singleton class correctly
        if not hasattr(self, '_initialized'):
            # Get the current frame and then the outer frame (where the engine instance is created)
            caller_frame = inspect.currentframe().f_back
            file_path = caller_frame.f_code.co_filename  # Extract the file path from the frame
            self._git = GitController(os.path.abspath(file_path))
            super().__init__(env, data_tool=data_tool)
    
    # HACK: since python doesn't support dynamic typing, true return type should be subclass of BacktestMixin and tStrategy
    # write -> BacktestMixin | tStrategy for better intellisense in IDEs
    def add_strategy(self, strategy: tStrategy, name: str='', is_parallel=False) -> BacktestMixin | tStrategy:
        is_dummy_strategy_exist = '_dummy' in self.strategy_manager.strategies
        assert not is_dummy_strategy_exist, 'dummy strategy is being used for model backtesting, adding another strategy is not allowed'
        if is_parallel:
            is_parallel = False
            self.logger.warning(f'Parallel strategy is not supported in backtesting, {strategy.__class__.__name__} will be run in sequential mode')
        if type(strategy) is not BaseStrategy:
            assert name != '_dummy', 'dummy strategy is reserved for model backtesting, please use another name'
        name = name or strategy.__class__.__name__
        strategy = BacktestStrategy(type(strategy), *strategy._args, **strategy._kwargs)
        return super().add_strategy(strategy, name=name, is_parallel=is_parallel)

    def add_model(
        self, 
        model: tModel, 
        name: str='',
        min_data: int=1,
        max_data: None | int=None,
        group_data: bool=True,
        signal_cols: list[str] | None=None,
    ) -> BacktestMixin | tModel:
        '''Add model without creating a strategy (using dummy strategy)'''
        is_non_dummy_strategy_exist = bool([strat for strat in self.strategy_manager.strategies if strat != '_dummy'])
        assert not is_non_dummy_strategy_exist, 'Please use strategy.add_model(...) instead of engine.add_model(...) when a strategy is already created'
        if not (strategy := self.strategy_manager.get_strategy('_dummy')):
            strategy = self.add_strategy(BaseStrategy(), name='_dummy')
            # add event driven functions to dummy strategy to avoid NotImplementedError in backtesting
            empty_function = lambda *args, **kwargs: None
            for func in strategy.REQUIRED_FUNCTIONS:
                setattr(strategy, func, empty_function)
        assert not strategy.models, 'Adding more than 1 model to dummy strategy in backtesting is not supported, you should train and dump your models one by one'
        model = strategy.add_model(
            model, 
            name=name, 
            min_data=min_data, 
            max_data=max_data, 
            group_data=group_data,
            signal_cols=signal_cols,
        )
        return model
    
    def add_feature(
        self, 
        feature: tFeature, 
        name: str='',
        min_data: int=1,
        max_data: None | int=None,
        group_data: bool=True,
        signal_cols: list[str] | None=None,
    ) -> BacktestMixin | tFeature:
        return self.add_model(
            feature, 
            name=name, 
            min_data=min_data, 
            max_data=max_data, 
            group_data=group_data,
            signal_cols=signal_cols,
        )
    
    def add_indicator(
        self, 
        indicator: tIndicator, 
        name: str='',
        min_data: int=1,
        max_data: None | int=None,
        group_data: bool=True,
        signal_cols: list[str] | None=None,
    ) -> BacktestMixin | tIndicator:
        return self.add_model(
            indicator, 
            name=name, 
            min_data=min_data, 
            max_data=max_data, 
            group_data=group_data,
            signal_cols=signal_cols,
        )
    
    def add_broker(self, bkr: str):
        bkr = bkr.upper()
        if bkr in self.brokers:
            return self.get_broker(bkr)
        Broker = self.get_Broker(bkr)
        broker = BacktestBroker(Broker)
        bkr = broker.name
        self.brokers[bkr] = broker
        self.logger.debug(f'added {bkr=}')
        return broker
    
    @staticmethod 
    def _generate_backtest_id() -> str:
        return uuid.uuid4().hex
    
    def _create_backtest_name(self, strat: str, backtest_id: str, backtest_id_length: int=12):
        local_tz = utils.get_local_timezone()
        utcnow = datetime.datetime.now(tz=local_tz).strftime('%Y-%m-%d_%H:%M:%S_UTC%z')
        trimmed_backtest_id = backtest_id[:backtest_id_length]
        return '.'.join([strat, utcnow, trimmed_backtest_id])
    
    @staticmethod 
    def _generate_backtest_hash(strategy: BaseStrategy):
        '''Generate hash based on strategy for backtest traceability
        backtest_hash is used to identify if the backtests are generated by the same strategy.
        Useful for avoiding overfitting the strategy on the same dataset.
        '''
        # REVIEW: currently only use strategy to generate hash, may include other settings in the future
        strategy_dict = strategy.to_dict()
        # since conceptually backtest_hash should be the same regardless of the 
        # strategy_signature (params) and data_signatures (e.g. backtest_kwargs, train_kwargs, data_source, resolution etc.)
        # remove them
        del strategy_dict['strategy_signature']
        del strategy_dict['data_signatures']
        strategy_str = json.dumps(strategy_dict)
        return hashlib.sha256(strategy_str.encode()).hexdigest()
    
    def read_json(self, file_name: str) -> dict:
        '''Reads json file from backtest_path'''
        file_path = os.path.join(self.config.backtest_path, file_name)
        backtest_json = {}
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    backtest_json = json.load(f)
        except:
            self.logger.exception(f"Error reading from {file_path}:")
        return backtest_json
    
    def _write_json(self, file_name: str, json_file: dict) -> None:
        '''Writes json file to backtest_path'''
        file_path = os.path.join(self.config.backtest_path, file_name)
        try:
            with open(file_path, 'w') as f:
                json.dump(json_file, f, indent=4)
        except:
            self.logger.exception(f"Error writing to {file_path}:")
    
    def _generate_backtest_iteration(self, backtest_hash: str) -> int:
        '''Generate backtest iteration number for the same backtest_hash.
        Read the existing backtest.json file to get the iteration number for the same strategy hash
        If the backtest hash is not found, create a new entry with iteration number 1
        else increment the iteration number by 1.
        '''
        file_name = 'backtest.json'
        backtest_json = self.read_json(file_name)
        backtest_json[backtest_hash] = backtest_json.get(backtest_hash, 0) + 1
        self._write_json(file_name, backtest_json)
        return backtest_json[backtest_hash]
    
    def _commit_strategy(self, strategy: BaseStrategy) -> str | None:
        engine_name = self.__class__.__name__
        strat = strategy.name
        commit_hash: str | None = self._git.commit(strategy._file_path, f'[PFund] {engine_name}: auto-commit strategy "{strat}"')
        if commit_hash:
            self.logger.debug(f"Strategy {strat} committed. {commit_hash=}")
        else:
            commit_hash = self._git.get_last_n_commit(n=1)[0]
            self.logger.debug(f"Strategy {strat} has no changes to commit, return the last {commit_hash=}")
        return commit_hash
    
    def _create_backtest_history(self, strategy: BaseStrategy, start_time: float, end_time: float):
        initial_balances = {bkr: broker.get_initial_balances() for bkr, broker in self.brokers.items()}
        backtest_id = self._generate_backtest_id()
        backtest_hash = self._generate_backtest_hash(strategy)
        backtest_name = self._create_backtest_name(strategy.name, backtest_id)
        backtest_iter = self._generate_backtest_iteration(backtest_hash)
        if self.auto_git_commit and self._git.is_git_repo():
            commit_hash = self._commit_strategy(strategy)
        else:
            commit_hash = None
        local_tz = utils.get_local_timezone()
        duration = end_time - start_time
        backtest_history = {
            'metadata': {
                'pfund_version': pf.__version__,
                'backtest_id': backtest_id,
                'backtest_hash': backtest_hash,
                'backtest_name': backtest_name,
                'backtest_iteration': backtest_iter,
                'commit_hash': commit_hash,
                'duration': f'{duration:.2f}s' if duration > 1 else f'{duration*1000:.2f}ms',
                'start_time': datetime.datetime.fromtimestamp(start_time, tz=local_tz).strftime('%Y-%m-%dT%H:%M:%S%z'),
                'end_time': datetime.datetime.fromtimestamp(end_time, tz=local_tz).strftime('%Y-%m-%dT%H:%M:%S%z'),
                'settings': self.settings,
            },
            'initial_balances': initial_balances,
            'strategy': strategy.to_dict(),
        }
        return backtest_history
        
    def _output_backtest_results(self, strategy: BaseStrategy, df: pd.DataFrame | pl.LazyFrame, backtest_history: dict) -> dict:
        backtest_name = backtest_history['metadata']['backtest_name']
        if self.mode == 'vectorized':
            output_file_path = os.path.join(self.config.backtest_path, f'{backtest_name}.parquet')
            strategy.dtl.output_df_to_parquet(df, output_file_path)
        elif self.mode == 'event_driven':
            # TODO: output trades? or orders? or df?
            output_file_path = ...
        backtest_history['result'] = output_file_path
        self._write_json(f'{backtest_name}.json', backtest_history)
        return backtest_history
        
    def run(self):
        for broker in self.brokers.values():
            broker.start()
        self.strategy_manager.start()
        for strat, strategy in self.strategy_manager.strategies.items():
            backtestee = strategy
            if strat == '_dummy':
                if self.mode == 'vectorized':
                    continue
                elif self.mode == 'event_driven':
                    # dummy strategy has exactly one model
                    model = list(strategy.models.values())[0]
                    backtestee = model
            backtests: dict = self._backtest(backtestee)
        self.strategy_manager.stop(reason='finished backtesting')
        return backtests

    def _backtest(self, backtestee: BaseStrategy | BaseModel) -> dict:
        backtests = {}
        dtl = backtestee.dtl
        df = backtestee.get_df(copy=True)
        
        if not self.use_ray:
            tqdm_bar = tqdm(
                total=self.num_chunks, 
                desc=f'Backtesting {backtestee.name} (per chunk)', 
                colour='green'
            )
        else:
            ray_tasks = []
        
        # Pre-Backtesting
        if self.mode == 'vectorized':
            if not hasattr(backtestee, 'backtest'):
                raise Exception(f'{backtestee.name} does not have backtest() method, cannot run vectorized backtesting')
            sig = inspect.signature(backtestee.backtest)
            params = list(sig.parameters.values())
            if not params or params[0].name != 'df':
                raise Exception(f'{backtestee.name} backtest() must have "df" as its first arg, i.e. backtest(self, df)')
            df_chunks = []
        elif self.mode == 'event_driven':
            # NOTE: clear dfs so that strategies/models don't know anything about the incoming data
            backtestee.clear_dfs()
        else:
            raise NotImplementedError(f'Backtesting mode {self.mode} is not supported')
        
        
        # Backtesting
        start_time = time.time()
        for chunk_num, df_chunk in enumerate(dtl.iterate_df_by_chunks(df, num_chunks=self.num_chunks)):
            if self.use_ray:
                ray_tasks.append((df_chunk, chunk_num))
            else:
                if self.mode == 'vectorized':
                    df_chunk = dtl.preprocess_vectorized_df(df_chunk, backtestee)
                    backtestee.backtest(df_chunk)
                    df_chunks.append(df_chunk)
                elif self.mode == 'event_driven':
                    df_chunk = dtl.preprocess_event_driven_df(df_chunk)
                    self._event_driven_backtest(df_chunk, chunk_num=chunk_num)
                tqdm_bar.update(1)
            
        if self.use_ray:
            import ray
            from ray.util.queue import Queue
            
            ray.init(num_cpus=self.num_cpus)
            print(f"Ray's num_cpus is set to {self.num_cpus}")
            
            @ray.remote
            def _run_task(log_queue: Queue,  _df_chunk: pd.DataFrame | pl.LazyFrame, _chunk_num: int, _batch_num: int):
                logger = backtestee.logger
                if not logger.handlers:
                    logger.addHandler(QueueHandler(log_queue))
                    logger.setLevel(logging.DEBUG)
                if self.mode == 'vectorized':
                    _df_chunk = dtl.preprocess_vectorized_df(_df_chunk, backtestee)
                    backtestee.backtest(_df_chunk)
                elif self.mode == 'event_driven':
                    _df_chunk = dtl.preprocess_event_driven_df(_df_chunk)
                    self._event_driven_backtest(_df_chunk, chunk_num=_chunk_num, batch_num=_batch_num)

            batch_size = self.num_cpus
            log_queue = Queue()
            QueueListener(log_queue, *backtestee.logger.handlers, respect_handler_level=True).start()
            batches = [ray_tasks[i: i + batch_size] for i in range(0, len(ray_tasks), batch_size)]
            with tqdm(
                total=len(batches),
                desc=f'Backtesting {backtestee.name} ({batch_size} chunks per batch)', 
                colour='green'
            ) as tqdm_bar:
                for batch_num, batch in enumerate(batches):
                    futures = [_run_task.remote(log_queue, *task, batch_num) for task in batch]
                    ray.get(futures)
                    tqdm_bar.update(1)
        end_time = time.time()
        print(f'Backtest elapsed time: {end_time - start_time:.2f}(s)')
        
        
        # Post-Backtesting
        if backtestee.type == 'strategy':
            if self.mode == 'vectorized':
                df = dtl.postprocess_vectorized_df(df_chunks)
            # TODO
            elif self.mode == 'event_driven':
                pass
            backtest_history: dict = self._create_backtest_history(backtestee, start_time, end_time)
            if self.save_backtests:
                backtest_history = self._output_backtest_results(backtestee, df, backtest_history)
            backtests[backtestee.name] = backtest_history
        else:
            self.assert_consistent_signals()
            
        return backtests

    def _event_driven_backtest(self, df_chunk, chunk_num=0, batch_num=0):
        common_cols = ['ts', 'product', 'resolution', 'broker', 'is_quote', 'is_tick']
        if isinstance(df_chunk, pl.LazyFrame):
            df_chunk = df_chunk.collect().to_pandas()
        
        # OPTIMIZE: critical loop
        for row in tqdm(
            df_chunk.itertuples(index=False), 
            total=df_chunk.shape[0], 
            desc=f'Backtest-Chunk{chunk_num}-Batch{batch_num} (per row)', 
            colour='yellow'
        ):
            ts, product, resolution = row.ts, row.product, row.resolution
            broker = self.brokers[row.broker]
            data_manager = broker.dm
            if row.is_quote:
                # TODO
                raise NotImplementedError('Quote data is not supported in event-driven backtesting yet')
                quote = {}
                data_manager.update_quote(product, quote)
            elif row.is_tick:
                # TODO
                raise NotImplementedError('Tick data is not supported in event-driven backtesting yet')
                tick = {}
                data_manager.update_tick(product, tick)
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
                    'other_info': {
                        col: getattr(row, col) for col in row._fields
                        if col not in common_cols + bar_cols
                    },
                }
                data_manager.update_bar(product, bar, now=ts)
    
    def end(self):
        self.strategy_manager.stop(reason='finished backtesting')
        for broker in self.brokers.values():
            broker.stop()
