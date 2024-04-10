from __future__ import annotations

import hashlib
import os
import time
import datetime
import json
import uuid

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.types.common_literals import tSUPPORTED_BACKTEST_MODES, tSUPPORTED_DATA_TOOLS
    from pfund.types.core import tStrategy, tModel, tFeature, tIndicator
    
import pfund as pf
from pfund.engines.base_engine import BaseEngine
from pfund.brokers.broker_backtest import BacktestBroker
from pfund.strategies.strategy_base import BaseStrategy
from pfund.strategies.strategy_backtest import BacktestStrategy
from pfund.config_handler import ConfigHandler
from pfund.utils import utils
from pfund.mixins.backtest import BacktestMixin


class BacktestEngine(BaseEngine):
    def __new__(cls, *, env: str='BACKTEST', data_tool: tSUPPORTED_DATA_TOOLS='pandas', mode: tSUPPORTED_BACKTEST_MODES='vectorized', append_to_strategy_df=False, use_prepared_signals=True, config: ConfigHandler | None=None, **settings):
        if not hasattr(cls, 'mode'):
            cls.mode = mode.lower()
        if not hasattr(cls, 'append_to_strategy_df'):
            cls.append_to_strategy_df = append_to_strategy_df
        # NOTE: use_prepared_signals=True means model's prepared signals will be reused in model.next()
        # instead of recalculating the signals. This will make event-driven backtesting faster but less consistent with live trading
        if not hasattr(cls, 'use_prepared_signals'):
            cls.use_prepared_signals = use_prepared_signals
        return super().__new__(cls, env, data_tool=data_tool, config=config, **settings)

    def __init__(self, *, env: str='BACKTEST', data_tool: tSUPPORTED_DATA_TOOLS='pandas', mode: tSUPPORTED_BACKTEST_MODES='vectorized', append_to_strategy_df=False, use_prepared_signals=True, config: ConfigHandler | None=None, **settings):
        super().__init__(env, data_tool=data_tool)
        # avoid re-initialization to implement singleton class correctly
        # if not hasattr(self, '_initialized'):
        #     pass
    
    # HACK: since python doesn't support dynamic typing, true return type should be subclass of BacktestMixin and tStrategy
    # write -> BacktestMixin | tStrategy for better intellisense in IDEs
    def add_strategy(self, strategy: tStrategy, name: str='', is_parallel=False) -> BacktestMixin | tStrategy:
        is_dummy_strategy_exist = '_dummy' in self.strategy_manager.strategies
        assert not is_dummy_strategy_exist, 'dummy strategy is being used for model backtesting, adding another strategy is not allowed'
        if is_parallel:
            is_parallel = False
            self.logger.warning(f'Parallel strategy is not supported in backtesting, {strategy.__class__.__name__} will be run in sequential mode')
        name = name or strategy.__class__.__name__
        strategy = BacktestStrategy(type(strategy), *strategy._args, **strategy._kwargs)
        return super().add_strategy(strategy, name=name, is_parallel=is_parallel)

    def add_model(self, model: tModel, name: str='', model_path: str='', is_load: bool=True) -> BacktestMixin | tModel:
        '''Add model without creating a strategy (using dummy strategy)'''
        is_non_dummy_strategy_exist = bool([strat for strat in self.strategy_manager.strategies if strat != '_dummy'])
        assert not is_non_dummy_strategy_exist, 'Please use strategy.add_model(...) instead of engine.add_model(...) when a strategy is already created'
        if not (strategy := self.strategy_manager.get_strategy('_dummy')):
            strategy = self.add_strategy(BaseStrategy(), name='_dummy')
            # add event driven functions to dummy strategy to avoid NotImplementedError in backtesting
            empty_function = lambda *args, **kwargs: None
            event_driven_funcs = ['on_quote', 'on_tick', 'on_bar', 'on_position', 'on_balance', 'on_order', 'on_trade']
            for func in event_driven_funcs:
                setattr(strategy, func, empty_function)
        assert not strategy.models, 'Adding more than 1 model to dummy strategy in backtesting is not supported, you should train and dump your models one by one'
        model = strategy.add_model(model, name=name, model_path=model_path, is_load=is_load)
        return model
    
    def add_feature(self, feature: tFeature, name: str='', feature_path: str='', is_load: bool=True) -> BacktestMixin | tFeature:
        return self.add_model(feature, name=name, model_path=feature_path, is_load=is_load)
    
    def add_indicator(self, indicator: tIndicator, name: str='', indicator_path: str='', is_load: bool=True) -> BacktestMixin | tIndicator:
        return self.add_model(indicator, name=name, model_path=indicator_path, is_load=is_load)
    
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
    
    def _create_backtest_name(self, strat: str):
        local_tz = utils.get_local_timezone()
        utcnow = datetime.datetime.now(tz=local_tz).strftime('%Y-%m-%d_%H:%M:%S_UTC%z')
        backtest_id = self._generate_backtest_id()
        return '.'.join([strat, utcnow, backtest_id])
    
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
    
    def _generate_backtest_iteration(self, strategy: BaseStrategy) -> int:
        '''Generate backtest iteration number for the same backtest_hash.
        Read the existing backtest.json file to get the iteration number for the same strategy hash
        If the backtest hash is not found, create a new entry with iteration number 1
        else increment the iteration number by 1.
        '''
        backtest_hash = self._generate_backtest_hash(strategy)
        file_name = 'backtest.json'
        backtest_json = self.read_json(file_name)
        backtest_json[backtest_hash] = backtest_json.get(backtest_hash, 0) + 1
        self._write_json(file_name, backtest_json)
        return backtest_json[backtest_hash]
    
    def _write_backtest_history(self, backtest_name: str, backtest_name_trimmed: str, start_time: float, end_time: float):
        splits = backtest_name.split('.')
        strat, backtest_id = splits[0], splits[-1]
        strategy = self.get_strategy(strat)
        local_tz = utils.get_local_timezone()
        duration = end_time - start_time
        backtest_history = {
            'settings': self.settings,
            'metadata': {
                'pfund_version': pf.__version__,
                'backtest_name': backtest_name,
                'backtest_id': backtest_id,
                'backtest_iteration': self._generate_backtest_iteration(strategy),
                'duration': f'{duration:.2f}s' if duration > 1 else f'{duration*1000:.2f}ms',
                'start_time': datetime.datetime.fromtimestamp(start_time, tz=local_tz).strftime('%Y-%m-%dT%H:%M:%S%z'),
                'end_time': datetime.datetime.fromtimestamp(end_time, tz=local_tz).strftime('%Y-%m-%dT%H:%M:%S%z'),
            },
            'strategy': strategy.to_dict(),
            'results': {
                'df_file_path': os.path.join(self.config.backtest_path, f'{backtest_name_trimmed}.parquet'),
            }
        }
        self._write_json(f'{backtest_name_trimmed}.json', backtest_history)
    
    def trim_backtest_name(self, backtest_name: str) -> str:
        splits = backtest_name.split('.')
        backtest_id_len = 12
        splits[-1] = splits[-1][:backtest_id_len]
        return '.'.join(splits)
    
    def output_backtest_results(self, strat: str, df, start_time: float, end_time: float):
        backtest_name = self._create_backtest_name(strat)
        backtest_name_trimmed = self.trim_backtest_name(backtest_name)
        self.data_tool.output_df_to_parquet(backtest_name_trimmed, df, self.config.backtest_path)
        self._write_backtest_history(backtest_name, backtest_name_trimmed, start_time, end_time)
        
    def run(self):
        for broker in self.brokers.values():
            broker.start()
        self.strategy_manager.start()

        if self.mode == 'vectorized':
            for strat, strategy in self.strategy_manager.strategies.items():
                # _dummy strategy is only created for model training, do nothing
                if strat == '_dummy':
                    continue
                if not hasattr(strategy, 'backtest'):
                    raise Exception(f'Strategy {strat} does not have backtest() method, cannot run vectorized backtesting')
                start_time = time.time()
                strategy.backtest()
                end_time = time.time()
                df = strategy.get_df()
                self.output_backtest_results(strat, df, start_time, end_time)
        elif self.mode == 'event_driven':
            for strat, strategy in self.strategy_manager.strategies.items():
                if strat == '_dummy':
                    # dummy strategy has exactly one model
                    model = list(strategy.models.values())[0]
                    strategy_or_model = model
                else:
                    strategy_or_model = strategy
                df = strategy_or_model.get_df()
                df = strategy_or_model._transform_df_for_event_driven_backtesting(df.copy(deep=True))
                # clear df so that strategy/model doesn't know anything about the incoming data
                strategy_or_model._clear_df()
                
                # OPTIMIZE: critical loop
                # FIXME: pandas specific, if BacktestEngine.data_tool == 'pandas':
                for row in df.itertuples(index=False):
                    resolution = row.resolution
                    product = row.product
                    broker = self.brokers[product.bkr]
                    data_manager = broker.dm
                    if resolution.is_quote():
                        # TODO
                        quote = {}
                        data_manager.update_quote(product, quote)
                    elif resolution.is_tick():
                        # TODO
                        tick = {}
                        data_manager.update_tick(product, tick)
                    else:
                        bar = {
                            'resolution': repr(resolution),
                            'data': {
                                'ts': row.ts,
                                'open': row.open,
                                'high': row.high,
                                'low': row.low,
                                'close': row.close,
                                'volume': row.volume
                            },
                            'other_info': {
                                col: getattr(row, col) for col in row._fields
                                if col not in ['product', 'resolution', 'ts', 'open', 'high', 'low', 'close', 'volume']
                            },
                        }
                        data_manager.update_bar(product, bar, now=row.ts)
                
                if strat == '_dummy':
                    model.assert_consistent_signals()
        else:
            raise NotImplementedError(f'Backtesting mode {self.mode} is not supported')
        self.strategy_manager.stop(reason='finished backtesting')

    def end(self):
        self.strategy_manager.stop(reason='finished backtesting')
        for broker in self.brokers.values():
            broker.stop()
