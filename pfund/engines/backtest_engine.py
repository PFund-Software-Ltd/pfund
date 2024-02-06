from typing import Literal

from pfund.data_tools.data_tool_base import DataTool
from pfund.engines.base_engine import BaseEngine
from pfund.models.model_base import BaseModel
from pfund.brokers.broker_backtest import BacktestBroker
from pfund.strategies.strategy_base import BaseStrategy
from pfund.strategies.strategy_backtest import BacktestStrategy
from pfund.const.commons import *
from pfund.config_handler import ConfigHandler


BacktestMode = Literal['vectorized', 'event_driven']
        

class BacktestEngine(BaseEngine):
    def __new__(cls, *, env: str='BACKTEST', data_tool: DataTool='pandas', mode: BacktestMode='vectorized', append_to_strategy_df=False, use_prepared_signals=True, config: ConfigHandler | None=None, **settings):
        if not hasattr(cls, 'mode'):
            cls.mode = mode.lower()
        if not hasattr(cls, 'append_to_strategy_df'):
            cls.append_to_strategy_df = append_to_strategy_df
        # NOTE: use_prepared_signals=True means model's prepared signals will be reused in model.next()
        # instead of recalculating the signals. This will make event-driven backtesting faster but less consistent with live trading
        if not hasattr(cls, 'use_prepared_signals'):
            cls.use_prepared_signals = use_prepared_signals
        return super().__new__(cls, env, data_tool=data_tool, config=config, **settings)

    def __init__(self, *, env: str='BACKTEST', data_tool: DataTool='pandas', mode: BacktestMode='vectorized', append_to_strategy_df=False, use_prepared_signals=True, config: ConfigHandler | None=None, **settings):
        super().__init__(env, data_tool=data_tool)
        # avoid re-initialization to implement singleton class correctly
        # if not hasattr(self, '_initialized'):
        #     pass
    
    def add_strategy(self, strategy: BaseStrategy, name: str='', is_parallel=False) -> BaseStrategy:
        is_dummy_strategy_exist = '_dummy' in self.strategy_manager.strategies
        assert not is_dummy_strategy_exist, 'dummy strategy is being used for model backtesting, adding another strategy is not allowed'
        if is_parallel:
            is_parallel = False
            self.logger.warning(f'Parallel strategy is not supported in backtesting, {strategy.__class__.__name__} will be run in sequential mode')
        name = name or strategy.__class__.__name__
        strategy = BacktestStrategy(type(strategy), *strategy._args, **strategy._kwargs)
        return super().add_strategy(strategy, name=name, is_parallel=is_parallel)

    def add_model(self, model: BaseModel, name: str='', model_path: str='', is_load: bool=True) -> BaseModel:
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
                df = strategy.get_df()
                strategy.backtest(df.copy(deep=True))
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
